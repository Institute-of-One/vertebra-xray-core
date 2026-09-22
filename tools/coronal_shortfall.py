"""What the coronal projection omits, measured on real vertebral orientations.

Every number in section 3.6 of the manuscript comes from this script, so that
none of them has to be copied by hand. It reports the four angle definitions
over every structural curve, the same figures stratified by the size of the
coronal curve, and where the measurement plane actually sits.

The stratification is the point. A three-dimensional angle that exceeds the
coronal angle is unremarkable in a spine that is nearly straight in the
coronal plane and normally kyphotic in the sagittal one: the plane of maximum
curvature simply rotates towards the sagittal and reports the kyphosis. The
question the table has to answer is whether the shortfall survives where the
coronal curve is large, and in this cohort it is not possible to say, because
the cohort contains no such curves.

    python tools/coronal_shortfall.py --root D:/tmp/verse/dataset-verse19training
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from vertebra_xray_core import cobb, cobb3d
from vertebra_xray_core import ct as ctmod
from vertebra_xray_core import geometry as geo
from vertebra_xray_core.datasets import verse
from vertebra_xray_core.spine3d import Projection, SpineModel3D

PA = Projection(view="pa")
BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 1e9))


def _curve_row(model: SpineModel3D, curve) -> dict[str, float]:
    measured = cobb3d.measure_between_levels(model, curve.upper_label, curve.lower_label)
    labels = list(model.labels)
    upper = model.endplate_normals[labels.index(curve.upper_label)]
    lower = model.endplate_normals[labels.index(curve.lower_label)]

    row = {
        "coronal": measured.coronal_deg,
        "sagittal": measured.sagittal_deg,
        "pmc": measured.pmc_deg,
        "plane": abs(measured.pmc_from_coronal_deg),
        # The dihedral angle between the two endplates: the largest angle any
        # plane can show, and the definition that needs no plane at all.
        "dihedral": abs(
            float(geo.wrap_to_signed_right_angle(geo.angle_between(upper, lower)))
        ),
    }
    try:
        normal = cobb3d.centroid_plane_normal(
            model, curve.upper_label, curve.apex_label, curve.lower_label
        )
    except ValueError:
        row["election"] = float("nan")
        row["election_tilt"] = float("nan")
    else:
        row["election"] = cobb3d.angle_in_plane(upper, lower, normal)
        # How far that plane leans out of vertical. A rotating radiograph can
        # only realise vertical planes, so a tilted one is not a view anyone
        # can take.
        row["election_tilt"] = abs(90.0 - float(geo.angle_between(normal, np.array([0.0, 0.0, 1.0]))))
    return row


def gather(root: str) -> tuple[list[dict[str, float]], dict[str, int]]:
    import nibabel

    names = {k: v for k, v in verse.VERSE_LEVELS.items() if k != 25}
    rows: list[dict[str, float]] = []
    funnel = {"found": 0, "read": 0, "covered": 0, "plausible": 0}
    for sample in verse.find_samples(root):
        funnel["found"] += 1
        try:
            canonical = nibabel.as_closest_canonical(nibabel.load(str(sample.mask_path)))
            mask = np.asarray(canonical.dataobj).astype(np.int16)
            model, _ = ctmod.model_from_segmentation(mask, canonical.affine, names)
        except (ValueError, OSError):
            continue
        funnel["read"] += 1
        cover = verse.coverage(model)
        if not (cover["spans_thoracolumbar"] and cover["contiguous"]):
            continue
        funnel["covered"] += 1
        # Scans carrying a vertebra the dimension check rejected are dropped
        # here as they are everywhere else. Included, two of them contribute
        # single-segment curves of 86 degrees that are segmentation failures,
        # not deformities.
        if model.meta.get("implausible"):
            continue
        funnel["plausible"] += 1
        for curve in cobb.cobb_angles(model.project(PA)).curves:
            row = _curve_row(model, curve)
            row["subject"] = sample.subject
            rows.append(row)
    return rows, funnel


def _stat(rows: Sequence[dict], key: str) -> tuple[float, float, float]:
    values = np.array([r[key] for r in rows if np.isfinite(r[key])])
    return float(np.median(values)), float(np.percentile(values, 90)), float(values.max())


def report(rows: Sequence[dict], funnel: dict[str, int]) -> None:
    subjects = {r["subject"] for r in rows}
    print(f"\n{len(rows)} curves from {len(subjects)} spines\n")
    print(
        f"scans found {funnel['found']}, read {funnel['read']}, "
        f"spanning the thoracolumbar junction and contiguous {funnel['covered']}, "
        f"passing the dimension check {funnel['plausible']}\n"
    )

    print("angle definitions over every curve (deg)")
    print(f"  {'':<38} {'median':>8} {'p90':>8} {'max':>8}")
    for key, label in (
        ("coronal", "coronal"),
        ("pmc", "maximum over vertical planes"),
        ("election", "plan d'election (centroid plane)"),
        ("dihedral", "dihedral angle between endplates"),
    ):
        m, p, x = _stat(rows, key)
        print(f"  {label:<38} {m:>8.2f} {p:>8.2f} {x:>8.2f}")

    excess = np.array([r["pmc"] - r["coronal"] for r in rows])
    print(f"\ncurves where the coronal angle is the larger: {int((excess < -1e-9).sum())}")
    m, p, x = _stat(rows, "plane")
    print(f"measurement plane, |tilt| from coronal:  median {m:.1f}  p90 {p:.1f}  max {x:.1f}")
    m, p, x = _stat(rows, "election_tilt")
    print(f"centroid plane, |tilt| from vertical:    median {m:.1f}  p90 {p:.1f}  max {x:.1f}")
    over = sum(1 for r in rows if np.isfinite(r["election"]) and r["election"] > r["pmc"] + 1e-9)
    print(f"curves where the centroid plane exceeds the vertical-plane maximum: {over}")
    m, _, _ = _stat(rows, "sagittal")
    print(f"sagittal angle over the same levels:     median {m:.2f}")

    print("\nstratified by the size of the coronal curve")
    print(f"  {'band':>12} {'n':>4} {'coronal':>9} {'3-D':>9} {'excess':>9} {'ratio':>7} {'plane':>7}")
    for low, high in BANDS:
        keep = [r for r in rows if low <= r["coronal"] < high]
        if not keep:
            print(f"  {f'>= {low:.0f} deg':>12} {0:>4}")
            continue
        label = f"{low:.0f}-{high:.0f} deg" if high < 1e8 else f">= {low:.0f} deg"
        cor = float(np.median([r["coronal"] for r in keep]))
        pmc = float(np.median([r["pmc"] for r in keep]))
        plane = float(np.median([r["plane"] for r in keep]))
        print(f"  {label:>12} {len(keep):>4} {cor:>9.2f} {pmc:>9.2f} {pmc - cor:>9.2f}"
              f" {pmc / max(cor, 1e-9):>7.2f} {plane:>7.1f}")

    large = [r for r in rows if r["coronal"] >= 20.0]
    if large:
        print(f"\nevery curve at or above 20 deg (n={len(large)})")
        for r in sorted(large, key=lambda r: -r["coronal"]):
            print(f"  coronal {r['coronal']:6.2f}  3-D {r['pmc']:6.2f}  "
                  f"sagittal {r['sagittal']:6.2f}  plane {r['plane']:5.1f}  "
                  f"ratio {r['pmc'] / r['coronal']:.2f}  {r['subject']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="an extracted VerSe release")
    args = parser.parse_args()
    rows, funnel = gather(args.root)
    if not rows:
        raise SystemExit("no usable curves")
    report(rows, funnel)


if __name__ == "__main__":
    main()
