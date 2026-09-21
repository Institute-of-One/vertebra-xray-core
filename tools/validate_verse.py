"""Run the whole pipeline over VerSe and report what it gets right.

Usage::

    python tools/validate_verse.py --root D:/tmp/verse/dataset-verse19training
    python tools/validate_verse.py --root ... --noise-mm 1.0 --out results/verse.csv

What this validates, and what it does not
-----------------------------------------
Landmarks come from projecting the CT-derived model, so **landmark detection
from radiographic pixels is not under test**. That is by design: this package
is the measurement layer, detector error is handled analytically by
:mod:`vertebra_xray_core.uncertainty`, and ``--noise-mm`` injects a stated
amount of it to check the conclusions survive.

What *is* under test is everything that depends on real spinal anatomy rather
than on a phantom: whether the sagittal curvature reversal sits where the
method assumes on real spines, whether the level count holds across real
anatomical variation, whether the biplanar reconstruction recovers real
orientations including real axial rotation, and whether the three-dimensional
Cobb angles computed from two projections match those computed directly from
the CT.

VerSe is supine and axial CT, radiographs are upright. Lumbar lordosis is
smaller supine, so the thoracolumbar anchor is weaker here than it would be
on a standing film. That makes this a conservative test rather than an easy
one, but it is a real difference and the anchor strengths are reported so it
can be seen.
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from vertebra_xray_core import cobb, cobb3d, labeling, uncertainty
from vertebra_xray_core import nomenclature as nom
from vertebra_xray_core.datasets import verse
from vertebra_xray_core.spine3d import Projection, reconstruct_from_biplanar

PA = Projection(view="pa")
LAT = Projection(view="lateral")


@dataclass
class Row:
    """One scan's outcome, flat enough to write straight to CSV."""

    subject: str
    status: str = "ok"
    detail: str = ""
    n_vertebrae: int = 0
    cranial: str = ""
    caudal: str = ""
    contiguous: bool = False
    has_anomaly: bool = False

    single_film_correct: bool | None = None
    single_film_offset: int | None = None
    single_film_confidence: str = ""
    single_film_flagged: bool | None = None

    biplanar_correct: bool | None = None
    biplanar_offset: int | None = None
    biplanar_confidence: str = ""

    anchor_reversal_deg: float = float("nan")
    anchor_position_error: float = float("nan")

    max_abs_psi_deg: float = float("nan")
    theta_error_deg: float = float("nan")
    phi_error_deg: float = float("nan")
    normal_error_deg: float = float("nan")
    theta_error_zero_psi_deg: float = float("nan")
    normal_error_zero_psi_deg: float = float("nan")

    coronal_cobb_deg: float = float("nan")
    pmc_cobb_deg: float = float("nan")
    pmc_minus_coronal_deg: float = float("nan")
    pmc_error_deg: float = float("nan")

    extra: dict = field(default_factory=dict)


def _offset(predicted: tuple[str, ...], truth: tuple[str, ...]) -> int | None:
    """How many levels the prediction is shifted, or ``None`` if not a pure shift."""
    if len(predicted) != len(truth):
        return None
    shifts = {nom.index_of(p) - nom.index_of(t) for p, t in zip(predicted, truth, strict=True)}
    return shifts.pop() if len(shifts) == 1 else None


def _labelling_outcome(result, truth: tuple[str, ...]) -> tuple[bool, int | None, str]:
    return result.labels == truth, _offset(result.labels, truth), result.confidence


def evaluate(sample: verse.VerseSample, *, noise_mm: float, seed: int) -> Row:
    """Measure one scan end to end."""
    row = Row(subject=sample.subject, has_anomaly=sample.has_enumeration_anomaly())

    model, _ = verse.load_model(sample)
    cover = verse.coverage(model)
    row.n_vertebrae = int(cover["n_vertebrae"])
    row.cranial = str(cover["cranial"])
    row.caudal = str(cover["caudal"])
    row.contiguous = bool(cover["contiguous"])

    if not cover["spans_thoracolumbar"]:
        row.status = "skipped"
        row.detail = (
            f"field of view {row.cranial}-{row.caudal} does not span the "
            "thoracolumbar junction with enough bodies on both sides"
        )
        return row
    if not cover["contiguous"]:
        row.status = "skipped"
        row.detail = "levels are not contiguous after dropping unusable vertebrae"
        return row

    truth = tuple(model.labels)
    row.max_abs_psi_deg = float(np.abs(model.psi_deg).max())

    frontal = model.project(PA).with_labels(None)
    lateral = model.project(LAT).with_labels(None)
    if noise_mm > 0:
        rng = np.random.default_rng(seed)
        frontal = uncertainty.jitter(frontal, noise_mm, rng)
        lateral = uncertainty.jitter(lateral, noise_mm, rng)

    # -- labelling ---------------------------------------------------------
    try:
        single = labeling.label_by_sagittal_inflection(lateral)
        row.single_film_correct, row.single_film_offset, row.single_film_confidence = (
            _labelling_outcome(single, truth)
        )
        row.single_film_flagged = single.confidence == "low" or bool(single.warnings)
        anchor = next((a for a in single.anchors if a.kind == "thoracolumbar"), None)
        if anchor is not None:
            row.anchor_reversal_deg = anchor.reversal_deg
            if nom.THORACOLUMBAR_JUNCTION in truth:
                row.anchor_position_error = float(
                    anchor.index - truth.index(nom.THORACOLUMBAR_JUNCTION)
                )
    except ValueError as exc:
        row.single_film_correct = False
        row.single_film_flagged = True
        row.single_film_confidence = "refused"
        row.detail = f"single film: {exc.args[0][:80]}"

    try:
        both = labeling.label_biplanar(frontal, lateral, axial_rotation_deg=model.psi_deg)
        row.biplanar_correct, row.biplanar_offset, row.biplanar_confidence = _labelling_outcome(
            both, truth
        )
    except ValueError as exc:
        row.biplanar_correct = False
        row.biplanar_confidence = "refused"
        row.detail = (row.detail + " | biplanar: " + exc.args[0][:80]).strip(" |")

    # -- reconstruction, with the true axial rotation and without ----------
    labelled_pa = frontal.with_labels(truth)
    labelled_lat = lateral.with_labels(truth)
    informed = reconstruct_from_biplanar(
        labelled_pa, labelled_lat, axial_rotation_deg=model.psi_deg
    )
    naive = reconstruct_from_biplanar(labelled_pa, labelled_lat)

    row.theta_error_deg = float(np.abs(informed.theta_deg - model.theta_deg).max())
    row.phi_error_deg = float(np.abs(informed.phi_deg - model.phi_deg).max())
    row.normal_error_deg = float(
        max(
            _angle(a, b)
            for a, b in zip(informed.endplate_normals, model.endplate_normals, strict=True)
        )
    )
    row.theta_error_zero_psi_deg = float(np.abs(naive.theta_deg - model.theta_deg).max())
    row.normal_error_zero_psi_deg = float(
        max(
            _angle(a, b)
            for a, b in zip(naive.endplate_normals, model.endplate_normals, strict=True)
        )
    )

    # -- angles ------------------------------------------------------------
    curves = cobb.cobb_angles(labelled_pa).curves
    if curves:
        major = max(curves, key=lambda c: c.angle_deg)
        truth3d = cobb3d.measure_between_levels(model, major.upper_label, major.lower_label)
        rebuilt3d = cobb3d.measure_between_levels(informed, major.upper_label, major.lower_label)
        row.coronal_cobb_deg = major.angle_deg
        row.pmc_cobb_deg = truth3d.pmc_deg
        row.pmc_minus_coronal_deg = truth3d.pmc_deg - truth3d.coronal_deg
        row.pmc_error_deg = abs(rebuilt3d.pmc_deg - truth3d.pmc_deg)
    return row


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    return float(
        np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), float(np.dot(u, v))))
    )


def summarise(rows: list[Row]) -> str:
    """Human-readable summary of the run."""
    done = [r for r in rows if r.status == "ok"]
    skipped = [r for r in rows if r.status == "skipped"]
    failed = [r for r in rows if r.status == "failed"]
    lines = [
        f"scans found              {len(rows)}",
        f"  usable                 {len(done)}",
        f"  skipped (coverage)     {len(skipped)}",
        f"  failed                 {len(failed)}",
    ]
    if not done:
        return "\n".join(lines)

    normal = [r for r in done if not r.has_anomaly]
    anomalous = [r for r in done if r.has_anomaly]

    def rate(rows_, attribute):
        values = [getattr(r, attribute) for r in rows_ if getattr(r, attribute) is not None]
        return f"{sum(values)}/{len(values)}" if values else "n/a"

    lines += [
        "",
        "labelling, spines with normal segmentation",
        f"  single lateral film    {rate(normal, 'single_film_correct')}",
        f"  biplanar two-pass      {rate(normal, 'biplanar_correct')}",
        "",
        f"labelling, spines with an enumeration anomaly ({len(anomalous)} scans)",
        f"  single lateral film    {rate(anomalous, 'single_film_correct')}",
        f"  flagged as unreliable  {rate(anomalous, 'single_film_flagged')}",
    ]

    def stat(rows_, attribute):
        values = np.array(
            [getattr(r, attribute) for r in rows_ if np.isfinite(getattr(r, attribute))]
        )
        if not len(values):
            return "n/a"
        return f"median {np.median(values):6.2f}  p90 {np.quantile(values, 0.9):6.2f}  max {values.max():6.2f}"

    lines += [
        "",
        "biplanar reconstruction, degrees of error in the endplate normal",
        f"  axial rotation known   {stat(done, 'normal_error_deg')}",
        f"  axial rotation assumed {stat(done, 'normal_error_zero_psi_deg')}",
        "",
        "three-dimensional Cobb",
        f"  PMC minus coronal      {stat(done, 'pmc_minus_coronal_deg')}",
        f"  PMC error vs CT truth  {stat(done, 'pmc_error_deg')}",
        "",
        f"axial rotation present   {stat(done, 'max_abs_psi_deg')}",
        f"thoracolumbar anchor     {stat(done, 'anchor_reversal_deg')}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="directory holding an extracted VerSe release")
    parser.add_argument("--noise-mm", type=float, default=0.0, help="landmark jitter to inject")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many scans")
    parser.add_argument("--out", default="", help="write a per-scan CSV here")
    args = parser.parse_args(argv)

    samples = verse.find_samples(args.root)
    if not samples:
        print(f"no VerSe segmentation masks found under {args.root}", file=sys.stderr)
        return 1
    if args.limit:
        samples = samples[: args.limit]

    rows: list[Row] = []
    for index, sample in enumerate(samples, 1):
        try:
            row = evaluate(sample, noise_mm=args.noise_mm, seed=args.seed + index)
        except Exception as exc:  # a dataset this varied will have surprises
            row = Row(subject=sample.subject, status="failed", detail=f"{type(exc).__name__}: {exc}")
            if "--traceback" in sys.argv:
                traceback.print_exc()
        rows.append(row)
        print(f"[{index:3d}/{len(samples)}] {row.subject:28s} {row.status:8s} {row.detail[:60]}")

    print()
    print(summarise(rows))

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [f for f in asdict(rows[0]) if f != "extra"]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({f: getattr(row, f) for f in fields})
        print(f"\nper-scan results written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
