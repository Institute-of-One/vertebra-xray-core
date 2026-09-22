"""The results figure: what the coronal projection omits, on real spines.

    python tools/make_verse_figure.py --root D:/tmp/verse/dataset-verse19training

Recomputes from the segmentations rather than from a summary file, so the
figure cannot drift away from the numbers in the text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vertebra_xray_core import cobb, cobb3d  # noqa: E402
from vertebra_xray_core import ct as ctmod  # noqa: E402
from vertebra_xray_core import geometry as geo  # noqa: E402
from vertebra_xray_core.datasets import verse  # noqa: E402
from vertebra_xray_core.spine3d import (  # noqa: E402
    Projection,
    axial_rotation_bias,
    reconstruct_from_biplanar,
)

PA = Projection(view="pa")
LAT = Projection(view="lateral")

plt.rcParams.update(
    {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150}
)


def gather(root: str):
    """Per-curve and per-scan measurements from every usable VerSe scan."""
    import nibabel

    names = {k: v for k, v in verse.VERSE_LEVELS.items() if k != 25}
    curves, scans = [], []
    for sample in verse.find_samples(root):
        try:
            canonical = nibabel.as_closest_canonical(nibabel.load(str(sample.mask_path)))
            mask = np.asarray(canonical.dataobj).astype(np.int16)
            model, _ = ctmod.model_from_segmentation(mask, canonical.affine, names)
        except (ValueError, OSError):
            continue
        cover = verse.coverage(model)
        if not (cover["spans_thoracolumbar"] and cover["contiguous"]):
            continue
        if model.meta.get("implausible"):
            continue

        frontal = model.project(PA)
        lateral = model.project(LAT)
        for curve in cobb.cobb_angles(frontal).curves:
            measured = cobb3d.measure_between_levels(
                model, curve.upper_label, curve.lower_label, name=curve.name
            )
            curves.append(
                {
                    "subject": sample.subject,
                    "coronal": measured.coronal_deg,
                    "pmc": measured.pmc_deg,
                    "plane": measured.pmc_from_coronal_deg,
                    "region": curve.apex_label,
                }
            )

        naive = reconstruct_from_biplanar(frontal, lateral)
        errors = [
            geo.angle_between(a, b)
            for a, b in zip(naive.endplate_normals, model.endplate_normals, strict=True)
        ]
        scans.append(
            {
                "subject": sample.subject,
                "psi": np.abs(model.psi_deg),
                "normal_error": np.asarray(errors),
            }
        )
    return curves, scans


def figure(curves, scans, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    coronal = np.array([c["coronal"] for c in curves])
    pmc = np.array([c["pmc"] for c in curves])
    plane = np.array([c["plane"] for c in curves])

    top = max(pmc.max(), coronal.max()) * 1.05
    points = axes[0].scatter(
        coronal,
        pmc,
        c=np.abs(plane),
        cmap="magma_r",
        s=26,
        edgecolor="#333333",
        linewidth=0.3,
        vmin=0,
        vmax=90,
    )
    axes[0].plot([0, top], [0, top], color="#888888", lw=1.0, ls="--")
    axes[0].set_xlim(0, top)
    axes[0].set_ylim(0, top)
    axes[0].set_xlabel("coronal Cobb angle (deg)")
    axes[0].set_ylabel("angle in the plane of maximum curvature (deg)")
    axes[0].set_title(f"a  every curve lies above the identity line (n={len(curves)})")
    bar = fig.colorbar(points, ax=axes[0], fraction=0.045, pad=0.02)
    bar.set_label("|plane orientation| from coronal (deg)", fontsize=7)
    bar.ax.tick_params(labelsize=6)

    gap = pmc - coronal
    axes[1].hist(gap, bins=np.arange(0, gap.max() + 3, 3), color="#c2703d", edgecolor="#333333")
    for value, style, label in (
        (np.median(gap), "-", f"median {np.median(gap):.1f} deg"),
        (np.quantile(gap, 0.9), "--", f"p90 {np.quantile(gap, 0.9):.1f} deg"),
    ):
        axes[1].axvline(value, color="#1a1a1a", lw=1.4, ls=style, label=label)
    axes[1].set_xlabel("understatement by the coronal projection (deg)")
    axes[1].set_ylabel("curves")
    axes[1].set_title("b  how much the coronal film omits")
    axes[1].legend(fontsize=8, frameon=False)

    psi = np.concatenate([s["psi"] for s in scans])
    err = np.concatenate([s["normal_error"] for s in scans])
    axes[2].scatter(psi, err, s=14, alpha=0.5, color="#2f6f4f", edgecolor="none")
    grid = np.linspace(0, max(psi.max(), 1.0), 60)
    predicted = [axial_rotation_bias(20.0, -15.0, float(p))["normal_error_deg"] for p in grid]
    axes[2].plot(grid, predicted, color="#b03030", lw=1.8, label="phantom prediction at 20/-15 deg")
    axes[2].set_xlabel("axial rotation present (deg)")
    axes[2].set_ylabel("endplate normal error (deg)")
    axes[2].set_title("c  the cost of assuming no axial rotation")
    axes[2].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=400)  # QIMS asks for 300 dpi or better
    plt.close(fig)
    print(f"wrote {out}")

    carrying = len({c["subject"] for c in curves})
    print(
        f"\ncurves {len(curves)} from {carrying} of {len(scans)} usable spines "
        f"({len(scans) - carrying} carry no structural curve)"
    )
    print(
        f"PMC minus coronal: median {np.median(gap):.2f}, "
        f"p90 {np.quantile(gap, 0.9):.2f}, max {gap.max():.2f}"
    )
    print(f"curves where the coronal angle is the larger: {int((gap < -1e-9).sum())}")
    print(
        f"normal error with psi assumed zero: median {np.median(err):.2f}, "
        f"p90 {np.quantile(err, 0.9):.2f}, max {err.max():.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="paper/figures/fig3_verse.png")
    args = parser.parse_args(argv)
    curves, scans = gather(args.root)
    if not curves:
        print("no usable scans found")
        return 1
    figure(curves, scans, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
