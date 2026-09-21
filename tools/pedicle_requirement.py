"""What a landmark detector must deliver for an unbiased three-dimensional angle.

    python tools/pedicle_requirement.py --root D:/tmp/verse/dataset-verse19training

Four corners per vertebral body per view leave axial rotation undetermined, so
the three-dimensional angle built from them carries whatever bias the
unmeasured rotation imposes. Two more landmarks — the pedicles, on the frontal
view the detector is already processing — make the system determined.

This asks the question that turns that observation into a specification: given
that a detector places the pedicles with some error, and that their geometry
has to be taken from a normative table rather than measured on the patient,
how much bias is left?

Everything is run twice: on phantoms, where the answer is exact, and on VerSe,
where the axial rotation is real and measured from the segmentation.
"""

from __future__ import annotations

import argparse

import numpy as np

from vertebra_xray_core import ct as ctmod
from vertebra_xray_core import geometry as geo
from vertebra_xray_core.datasets import verse
from vertebra_xray_core.pedicles import (
    PedicleGeometry,
    normative_pedicles,
    project_pedicle_offsets,
    solve_orientation_with_pedicles,
)
from vertebra_xray_core.spine3d import _measured_tilts, endplate_normal, solve_orientation

LATERAL_SIGN = -1.0  # anterior on the image left


def residual_bias(
    theta: float,
    phi: float,
    psi: float,
    truth: PedicleGeometry,
    assumed: PedicleGeometry,
    pedicle_error_mm: float,
    rng: np.random.Generator,
) -> float:
    """Endplate-normal error left after solving with noisy, mis-sized pedicles."""
    alpha, beta = _measured_tilts(theta, phi, psi, LATERAL_SIGN)
    offsets = np.array(project_pedicle_offsets(theta, phi, psi, truth))
    if pedicle_error_mm:
        offsets = offsets + rng.normal(0.0, pedicle_error_mm, 2)
    got_theta, got_phi, _ = solve_orientation_with_pedicles(
        alpha, beta, (float(offsets[0]), float(offsets[1])), assumed
    )
    return geo.angle_between(
        endplate_normal(got_theta, got_phi), endplate_normal(theta, phi)
    )


def bias_without_pedicles(theta: float, phi: float, psi: float) -> float:
    alpha, beta = _measured_tilts(theta, phi, psi, LATERAL_SIGN)
    got_theta, got_phi = solve_orientation(alpha, beta, 0.0)
    return geo.angle_between(
        endplate_normal(got_theta, got_phi), endplate_normal(theta, phi)
    )


def phantom_sweep(resamples: int = 400, seed: int = 0) -> None:
    """Requirement curve over the anatomical range of orientations."""
    rng = np.random.default_rng(seed)
    pedicles = normative_pedicles("T8")

    print("\nPhantom: orientations drawn uniformly over the anatomical range")
    print("  coronal tilt +-35 deg, sagittal +-30 deg, axial rotation +-30 deg")
    print(f"  pedicle geometry exact, {resamples} draws per row\n")
    print(f"  {'pedicle error':>14}  {'median':>7} {'p90':>7} {'max':>7}   endplate normal error (deg)")

    baseline = [
        bias_without_pedicles(
            rng.uniform(-35, 35), rng.uniform(-30, 30), rng.uniform(-30, 30)
        )
        for _ in range(resamples)
    ]
    print(
        f"  {'no pedicles':>14}  {np.median(baseline):7.2f} "
        f"{np.quantile(baseline, 0.9):7.2f} {max(baseline):7.2f}   (axial rotation assumed zero)"
    )

    for error in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
        values = []
        for _ in range(resamples):
            theta, phi, psi = (
                rng.uniform(-35, 35),
                rng.uniform(-30, 30),
                rng.uniform(-30, 30),
            )
            values.append(
                residual_bias(theta, phi, psi, pedicles, pedicles, error, rng)
            )
        print(
            f"  {error:11.2f} mm  {np.median(values):7.2f} "
            f"{np.quantile(values, 0.9):7.2f} {max(values):7.2f}"
        )


def geometry_sweep(resamples: int = 400, seed: int = 1) -> None:
    """Cost of taking the pedicle geometry from a table rather than the patient."""
    rng = np.random.default_rng(seed)
    truth_base = normative_pedicles("T8")

    print("\nPhantom: pedicle geometry taken from a table, with the patient's differing")
    print("  (the table's own spread on VerSe was 12.3 to 15.4 mm in separation,")
    print("   25.0 to 29.4 mm in posterior offset, interquartile)\n")
    print(f"  {'geometry error':>14}  {'median':>7} {'p90':>7} {'max':>7}")

    for error in (0.0, 1.0, 2.0, 4.0, 6.0):
        values = []
        for _ in range(resamples):
            theta, phi, psi = (
                rng.uniform(-35, 35),
                rng.uniform(-30, 30),
                rng.uniform(-30, 30),
            )
            truth = PedicleGeometry(
                truth_base.half_separation + rng.normal(0.0, error),
                truth_base.posterior_offset + rng.normal(0.0, error),
            )
            values.append(residual_bias(theta, phi, psi, truth, truth_base, 0.0, rng))
        print(
            f"  {error:11.2f} mm  {np.median(values):7.2f} "
            f"{np.quantile(values, 0.9):7.2f} {max(values):7.2f}"
        )


def verse_sweep(root: str, seed: int = 2) -> None:
    """The same question with real orientations and real pedicle geometry."""
    import nibabel

    rng = np.random.default_rng(seed)
    names = {k: v for k, v in verse.VERSE_LEVELS.items() if k != 25}

    cases = []
    for sample in verse.find_samples(root):
        try:
            canonical = nibabel.as_closest_canonical(nibabel.load(str(sample.mask_path)))
            mask = np.asarray(canonical.dataobj).astype(np.int16)
            model, measured = ctmod.model_from_segmentation(mask, canonical.affine, names)
        except (ValueError, OSError):
            continue
        cover = verse.coverage(model)
        if not (cover["spans_thoracolumbar"] and cover["contiguous"]):
            continue
        if model.meta.get("implausible"):
            continue
        by_label = {names[label]: geometry for label, geometry in measured.items()}
        for level, theta, phi, psi in zip(
            model.labels, model.theta_deg, model.phi_deg, model.psi_deg, strict=True
        ):
            geometry = by_label.get(level)
            if geometry is None or not np.isfinite(geometry.pedicle_half_separation_mm):
                continue
            cases.append(
                (
                    theta,
                    phi,
                    psi,
                    PedicleGeometry(
                        geometry.pedicle_half_separation_mm,
                        geometry.pedicle_posterior_offset_mm,
                    ),
                    normative_pedicles(level),
                )
            )

    if not cases:
        print("\nVerSe: no usable vertebrae found")
        return

    rotations = np.abs([c[2] for c in cases])
    print(f"\nVerSe: {len(cases)} vertebrae with measured pedicles from real spines")
    print(
        f"  axial rotation present: median {np.median(rotations):.1f}, "
        f"p90 {np.quantile(rotations, 0.9):.1f}, max {rotations.max():.1f} deg\n"
    )

    baseline = [bias_without_pedicles(t, p, s) for t, p, s, _, _ in cases]
    print(f"  {'pedicle error':>14}  {'median':>7} {'p90':>7} {'max':>7}   endplate normal error (deg)")
    print(
        f"  {'no pedicles':>14}  {np.median(baseline):7.2f} "
        f"{np.quantile(baseline, 0.9):7.2f} {max(baseline):7.2f}   (axial rotation assumed zero)"
    )

    for error in (0.0, 0.25, 0.5, 1.0, 2.0):
        for assumed_from_table in (False, True):
            values = [
                residual_bias(t, p, s, truth, table if assumed_from_table else truth, error, rng)
                for t, p, s, truth, table in cases
            ]
            label = "table geometry" if assumed_from_table else "measured geometry"
            print(
                f"  {error:11.2f} mm  {np.median(values):7.2f} "
                f"{np.quantile(values, 0.9):7.2f} {max(values):7.2f}   {label}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="", help="an extracted VerSe release")
    parser.add_argument("--resamples", type=int, default=400)
    args = parser.parse_args(argv)

    phantom_sweep(args.resamples)
    geometry_sweep(args.resamples)
    if args.root:
        verse_sweep(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
