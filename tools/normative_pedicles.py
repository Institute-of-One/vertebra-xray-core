"""Rebuild the normative pedicle table from VerSe, and check the shipped one.

Table 2 of the manuscript is a contribution, so it has to be regenerable
rather than asserted. This script measures the pedicle half-separation and
posterior offset of every vertebra in the usable VerSe scans, reports the
per-level medians and the spread behind them, and prints the difference from
:data:`vertebra_xray_core.pedicles.NORMATIVE_PEDICLES_BY_LEVEL` so that the
constant in the package can be shown to match what the data say.

    python tools/normative_pedicles.py --root D:/tmp/verse/dataset-verse19training
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from vertebra_xray_core import ct as ctmod
from vertebra_xray_core.datasets import verse
from vertebra_xray_core.nomenclature import ALL_LEVELS
from vertebra_xray_core.pedicles import NORMATIVE_PEDICLES_BY_LEVEL


def measure(root: str) -> tuple[dict[str, list[tuple[float, float]]], dict[str, int]]:
    import nibabel

    names = {k: v for k, v in verse.VERSE_LEVELS.items() if k != 25}
    by_level: dict[str, list[tuple[float, float]]] = defaultdict(list)
    counts = {"scans": 0, "vertebrae": 0, "with_pedicles": 0}
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
        counts["scans"] += 1
        by_label = {names[key]: value for key, value in measured.items()}
        for level in model.labels:
            counts["vertebrae"] += 1
            geometry = by_label.get(level)
            if geometry is None or not np.isfinite(geometry.pedicle_half_separation_mm):
                continue
            counts["with_pedicles"] += 1
            by_level[level].append(
                (
                    float(geometry.pedicle_half_separation_mm),
                    float(geometry.pedicle_posterior_offset_mm),
                )
            )
    return by_level, counts


def report(by_level: dict[str, list[tuple[float, float]]], counts: dict[str, int]) -> None:
    print(
        f"\n{counts['with_pedicles']} vertebrae with measurable pedicles, "
        f"of {counts['vertebrae']} in {counts['scans']} usable spines\n"
    )
    print(
        f"{'level':>6} {'n':>4} {'half-sep':>9} {'IQR':>13} "
        f"{'offset':>9} {'IQR':>13} {'shipped':>16} {'diff':>13}"
    )
    separations, offsets = [], []
    worst = 0.0
    for level in ALL_LEVELS:
        rows = by_level.get(level)
        if not rows:
            continue
        values = np.array(rows)
        sep, off = np.median(values[:, 0]), np.median(values[:, 1])
        separations.append(sep)
        offsets.append(off)
        sq = np.percentile(values[:, 0], [25, 75])
        oq = np.percentile(values[:, 1], [25, 75])
        shipped = NORMATIVE_PEDICLES_BY_LEVEL.get(level)
        if shipped is None:
            print(
                f"{level:>6} {len(values):>4} {sep:>9.1f} {sq[0]:>6.1f}-{sq[1]:<6.1f}"
                f" {off:>9.1f} {oq[0]:>6.1f}-{oq[1]:<6.1f} {'--':>16} {'--':>13}"
            )
            continue
        d_sep = sep - shipped.half_separation
        d_off = off - shipped.posterior_offset
        worst = max(worst, abs(d_sep), abs(d_off))
        print(
            f"{level:>6} {len(values):>4} {sep:>9.1f} {sq[0]:>6.1f}-{sq[1]:<6.1f}"
            f" {off:>9.1f} {oq[0]:>6.1f}-{oq[1]:<6.1f}"
            f" {shipped.half_separation:>7.1f}/{shipped.posterior_offset:<8.1f}"
            f" {d_sep:>+6.1f}/{d_off:<+6.1f}"
        )

    print(f"\nlargest disagreement with the shipped table: {worst:.2f} mm")
    if separations:
        print(
            f"across levels, half-separation median {np.median(separations):.1f} mm "
            f"(interquartile {np.percentile(separations, 25):.1f}-"
            f"{np.percentile(separations, 75):.1f}), "
            f"posterior offset median {np.median(offsets):.1f} mm "
            f"(interquartile {np.percentile(offsets, 25):.1f}-"
            f"{np.percentile(offsets, 75):.1f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="an extracted VerSe release")
    parser.add_argument(
        "--emit",
        action="store_true",
        help="print the table as a Python literal, for pasting into pedicles.py",
    )
    args = parser.parse_args()
    by_level, counts = measure(args.root)
    if not by_level:
        raise SystemExit("no measurable pedicles")
    report(by_level, counts)
    if args.emit:
        print()
        print("# paste into vertebra_xray_core/pedicles.py")
        for level in ALL_LEVELS:
            rows = by_level.get(level)
            if not rows or level not in NORMATIVE_PEDICLES_BY_LEVEL:
                continue
            values = np.array(rows)
            print(
                f'    "{level}": PedicleGeometry'
                f"({np.median(values[:, 0]):.1f}, {np.median(values[:, 1]):.1f}),"
            )


if __name__ == "__main__":
    main()
