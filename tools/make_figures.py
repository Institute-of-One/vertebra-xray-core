"""Generate the manuscript figures from phantoms alone.

    python tools/make_figures.py --out paper/figures

Every panel here comes from a spine built from its own specification, so the
figures contain no patient data, carry no dataset licence, and are byte-stable
across machines. That is a deliberate choice rather than a convenience: a
phantom's angles are known exactly, so a figure about a measurement can show
the measurement against its truth instead of against another measurement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vertebra_xray_core import (  # noqa: E402
    cobb,
    cobb3d,
    labeling,
    phantom,
    uncertainty,
    viz,
)
from vertebra_xray_core.spine3d import (  # noqa: E402
    Projection,
    axial_rotation_bias,
)

PA = Projection(view="pa")
LAT = Projection(view="lateral")
CONE_PA = Projection(view="pa", sod_mm=1000.0, sdd_mm=1200.0)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9,
        "figure.dpi": 150,
    }
)


def _case():
    return phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=50.0, lumbar_deg=32.0)


def figure_overview(out: Path) -> None:
    """The method end to end on one spine."""
    model = _case()
    frontal = model.project(PA)
    lateral = model.project(LAT).with_labels(None)
    curves = cobb.cobb_angles(frontal).curves
    three_d = cobb3d.cobb3d_for_curves(model, curves)
    major = max(three_d, key=lambda c: c.pmc_deg)
    levels = labeling.label_by_sagittal_inflection(lateral)
    anchor = levels.anchors[0]

    fig = plt.figure(figsize=(13.0, 8.2))
    grid = fig.add_gridspec(
        2, 4, width_ratios=[1.0, 0.95, 0.95, 2.0], height_ratios=[1.0, 0.5], wspace=0.3, hspace=0.35
    )

    ax = fig.add_subplot(grid[:, 0])
    viz.plot_spine_3d(
        model,
        ax,
        curves=curves,
        show_normals=True,
        pmc_plane_for=major,
        label_every=3,
        scale_bar_mm=100,
        view=(12.0, 30.0),
    )
    ax.set_title("a  three-dimensional arrangement,\nwith the plane of maximum curvature")

    ax = fig.add_subplot(grid[:, 1])
    viz.plot_projection(frontal, ax, curves=curves)
    ax.set_title("b  frontal projection\nand the Cobb construction")

    ax = fig.add_subplot(grid[:, 2])
    viz.plot_tilt_profile(
        lateral.with_labels(levels.labels),
        ax,
        anchor_index=anchor.index,
        anchor_position=anchor.position,
        title="c  sagittal tilt; the curvature\nreversal anchors T12",
    )

    ax = fig.add_subplot(grid[0, 3])
    viz.plot_pmc_profile(model, three_d, ax)
    ax.set_title("d  Cobb angle against measurement-plane orientation")

    ax = fig.add_subplot(grid[1, 3])
    viz.plot_axial_path(
        model,
        ax,
        measurements=three_d,
        title="e  the centreline seen from above, and each curve's measurement plane",
    )

    _save(fig, out / "fig1_overview.png")


def figure_axial_rotation(out: Path) -> None:
    """What assuming no axial rotation costs, by three independent routes."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    psis = np.linspace(0.0, 30.0, 31)

    bias = [axial_rotation_bias(20.0, -15.0, float(p))["normal_error_deg"] for p in psis]
    axes[0].plot(psis, bias, color="#b03030", lw=1.8)
    axes[0].set_xlabel("axial rotation assumed to be zero (deg)")
    axes[0].set_ylabel("error in the endplate normal (deg)")
    axes[0].set_title("a  biased reconstruction")

    divergence = []
    for p in psis[::3]:
        model = phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=float(p))
        divergence.append(
            float(np.abs(model.project(PA).body_tilt() - model.project(CONE_PA).body_tilt()).max())
        )
    axes[1].plot(psis[::3], divergence, color="#2f6f4f", lw=1.8, marker="o", ms=3)
    axes[1].set_xlabel("axial rotation (deg)")
    axes[1].set_ylabel("coronal tilt error (deg)")
    axes[1].set_title("b  cone-beam divergence\nis harmless until the spine rotates")

    single, both = [], []
    for p in psis[::3]:
        model = phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=float(p))
        frontal = model.project(PA).with_labels(None)
        lateral = model.project(LAT).with_labels(None)
        try:
            single.append(labeling.label_by_sagittal_inflection(lateral).labels == model.labels)
        except ValueError:
            single.append(False)
        both.append(
            labeling.label_biplanar(
                frontal, lateral, axial_rotation_deg=np.full(len(model), float(p))
            ).labels
            == model.labels
        )
    axes[2].step(psis[::3], np.array(single, dtype=float), where="mid", lw=1.8, label="one lateral film")
    axes[2].step(psis[::3], np.array(both, dtype=float), where="mid", lw=1.8, label="biplanar, two-pass")
    axes[2].set_ylim(-0.1, 1.15)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["wrong", "correct"])
    axes[2].set_xlabel("axial rotation (deg)")
    axes[2].set_title("c  vertebral levels")
    axes[2].legend(fontsize=8, frameon=False, loc="center left")

    _save(fig, out / "fig2_axial_rotation.png")


def figure_kyphosis(out: Path) -> None:
    """A lateral film under-reads kyphosis in proportion to the coronal curve."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    mains = np.arange(0.0, 86.0, 5.0)
    truth, film = [], []
    for main in mains:
        model = (
            phantom.normal_adult_spine()
            if main == 0.0
            else phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=float(main))
        )
        truth.append(cobb3d.measure_between_levels(model, "T1", "T12").sagittal_deg)
        film.append(cobb3d.as_seen_on_film(model, "T1", "T12", LAT))
    ax.plot(mains, truth, lw=1.8, label="sagittal-plane truth")
    ax.plot(mains, film, lw=1.8, ls="--", label="as read on a lateral film")
    ax.fill_between(mains, film, truth, alpha=0.15, color="#b03030")
    ax.set_xlabel("main thoracic Cobb angle (deg)")
    ax.set_ylabel("T1-T12 sagittal angle (deg)")
    ax.set_title("kyphosis is under-read exactly where it matters")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out / "fig3_kyphosis.png")


def figure_uncertainty(out: Path) -> None:
    """Where the spread in a Cobb angle comes from."""
    frontal = _case().project(PA)
    sigmas = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    sampled = {
        sigma: uncertainty.bootstrap_cobb(frontal, sigma, resamples=600) for sigma in sigmas
    }

    for name in ("PT", "MT", "L"):
        widths = [
            next(u for u in sampled[sigma] if u.name == name).redetected.high
            - next(u for u in sampled[sigma] if u.name == name).redetected.low
            for sigma in sigmas
        ]
        axes[0].plot(sigmas, widths, marker="o", ms=3.5, lw=1.6, label=name)
    axes[0].set_xlabel("corner localisation error (mm)")
    axes[0].set_ylabel("width of the 95% interval (deg)")
    axes[0].set_title("a  a Cobb angle is an interval, not a number")
    axes[0].legend(fontsize=8, frameon=False)

    # Decompose the spread on the main thoracic curve. The two mechanisms add
    # in quadrature, so the shaded bands are directly comparable.
    endplate = [next(u for u in sampled[s] if u.name == "MT").fixed.sd for s in sigmas]
    total = [next(u for u in sampled[s] if u.name == "MT").redetected.sd for s in sigmas]
    stability = [
        next(u for u in sampled[s] if u.name == "MT").end_vertebra_stability for s in sigmas
    ]
    axes[1].fill_between(sigmas, 0, endplate, alpha=0.65, label="drawing the endplate lines")
    axes[1].fill_between(sigmas, endplate, total, alpha=0.65, label="choosing the end vertebrae")
    axes[1].plot(sigmas, total, color="#222222", lw=1.4)
    axes[1].set_xlabel("corner localisation error (mm)")
    axes[1].set_ylabel("standard deviation of the main thoracic angle (deg)")
    axes[1].set_title("b  where the spread comes from")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")

    twin = axes[1].twinx()
    twin.plot(sigmas, stability, color="#7a5b9a", lw=1.2, ls=":", marker="s", ms=3)
    twin.set_ylabel("end vertebrae unchanged", fontsize=8, color="#7a5b9a")
    twin.set_ylim(0, 1.05)
    twin.tick_params(axis="y", labelsize=7, colors="#7a5b9a")
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color("#7a5b9a")

    _save(fig, out / "fig4_uncertainty.png")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


FIGURES = {
    "overview": figure_overview,
    "axial": figure_axial_rotation,
    "kyphosis": figure_kyphosis,
    "uncertainty": figure_uncertainty,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="paper/figures")
    parser.add_argument("--only", choices=sorted(FIGURES), action="append")
    args = parser.parse_args(argv)

    out = Path(args.out)
    for name in args.only or sorted(FIGURES):
        FIGURES[name](out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
