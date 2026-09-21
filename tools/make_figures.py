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
    phantom,
    simulate,
    uncertainty,
    viz,
)
from vertebra_xray_core.pedicles import (  # noqa: E402
    PedicleGeometry,
    normative_pedicles,
    project_pedicle_offsets,
    solve_orientation_with_pedicles,
)
from vertebra_xray_core.spine3d import (  # noqa: E402
    Projection,
    _measured_tilts,
    axial_rotation_bias,
    endplate_normal,
)

NL = chr(10)

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


def figure_planes(out: Path) -> None:
    """The angle depends on the plane it is measured in, and by how much."""
    model = _case()
    curves = cobb.cobb_angles(model.project(PA)).curves
    three_d = cobb3d.cobb3d_for_curves(model, curves)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), width_ratios=[1.3, 1.0])
    viz.plot_pmc_profile(model, three_d, axes[0])
    axes[0].set_title("a  the angle against its measurement plane", fontsize=9)
    viz.plot_axial_path(
        model, axes[1], measurements=three_d, title="b  the centreline from above",
    )
    fig.subplots_adjust(wspace=0.32)
    _save(fig, out / "fig4_planes.png")


def figure_axial_rotation(out: Path) -> None:
    """The two ways unmeasured axial rotation corrupts a biplanar measurement."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    psis = np.linspace(0.0, 30.0, 31)

    bias = [axial_rotation_bias(20.0, -15.0, float(p))["normal_error_deg"] for p in psis]
    axes[0].plot(psis, bias, color="#b03030", lw=1.8)
    axes[0].set_xlabel("axial rotation assumed to be zero (deg)")
    axes[0].set_ylabel("error in the endplate normal (deg)")
    axes[0].set_title("a  the orientation solve is biased")

    divergence = []
    for p in psis[::3]:
        model = phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=float(p))
        divergence.append(
            float(np.abs(model.project(PA).body_tilt() - model.project(CONE_PA).body_tilt()).max())
        )
    axes[1].plot(psis[::3], divergence, color="#2f6f4f", lw=1.8, marker="o", ms=3)
    axes[1].set_xlabel("axial rotation (deg)")
    axes[1].set_ylabel("coronal tilt error (deg)")
    axes[1].set_title("b  and cone-beam divergence, harmless at zero rotation, is let in")

    _save(fig, out / "fig5_axial_rotation.png")


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
    _save(fig, out / "fig6_kyphosis.png")


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

    _save(fig, out / "fig7_uncertainty.png")


def figure_landmarks(out: Path) -> None:
    """What a detector is being asked to produce, and the spine it comes from.

    Both panels are rendered from the same phantom, so the pedicle shadows in
    the radiograph and the pedicle landmarks drawn on it are the same points.
    No patient data and no licensed image is involved.
    """
    model = phantom.adolescent_idiopathic_scoliosis(
        main_thoracic_deg=50.0, lumbar_deg=32.0, axial_rotation_deg=14.0
    )
    radiograph = simulate.simulate_radiograph(model, PA)
    frontal = model.project(PA)
    pedicle_points = viz.project_pedicles_to_view(model, PA, normative_pedicles)
    curves = cobb.cobb_angles(frontal).curves
    three_d = cobb3d.cobb3d_for_curves(model, curves)
    major = max(three_d, key=lambda c: c.pmc_deg)

    # Crop the films to the trunk. Left at the full rendered extent they are
    # mostly air, and with equal aspect that makes them squat beside the
    # three-dimensional panel.
    trunk = float(np.median(model.centroids[:, 0]))
    crop = (trunk - 175.0, trunk + 175.0)

    fig = plt.figure(figsize=(11.0, 7.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.04)

    ax = fig.add_subplot(grid[0, 0])
    viz.plot_radiograph(radiograph, ax)
    ax.set_xlim(*crop)
    ax.set_title("a  simulated frontal radiograph", fontsize=9)

    ax = fig.add_subplot(grid[0, 1])
    viz.plot_radiograph(radiograph, ax, gamma=1.3)
    ax.set_xlim(*crop)
    viz.plot_landmark_overlay(
        frontal, ax, pedicles=pedicle_points,
        curves=tuple(c for c in curves if c.is_major),
    )
    ax.set_title(
        "b  four corners (blue) leave rotation undetermined;"
        + NL
        + "two pedicles (gold) close it",
        fontsize=9,
    )

    ax = fig.add_subplot(grid[0, 2])
    viz.plot_spine_3d(
        model, ax, curves=curves, show_normals=True, pmc_plane_for=major,
        label_every=3, scale_bar_mm=100, view=(14.0, 34.0),
    )
    ax.set_title(
        "c  the same spine in three dimensions," + NL + "with its measurement plane",
        fontsize=9,
    )

    _save(fig, out / "fig1_landmarks.png")


def figure_pedicle_requirement(out: Path) -> None:
    """What a detector must deliver for the three-dimensional angle to be unbiased."""
    rng = np.random.default_rng(0)
    pedicles = normative_pedicles("T8")
    draws = 600

    def sample():
        return rng.uniform(-35, 35), rng.uniform(-30, 30), rng.uniform(-30, 30)

    def residual(theta, phi, psi, truth, assumed, noise):
        alpha, beta = _measured_tilts(theta, phi, psi, -1.0)
        offsets = np.array(project_pedicle_offsets(theta, phi, psi, truth))
        if noise:
            offsets = offsets + rng.normal(0.0, noise, 2)
        got = solve_orientation_with_pedicles(
            alpha, beta, (float(offsets[0]), float(offsets[1])), assumed
        )
        return geo_angle(endplate_normal(got[0], got[1]), endplate_normal(theta, phi))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    rotations = np.arange(0.0, 31.0, 2.0)
    without = [axial_rotation_bias(20.0, -15.0, float(r))["normal_error_deg"] for r in rotations]
    axes[0].plot(rotations, without, lw=2.0, color="#b03030", label="four corners only")
    for noise, style in ((0.5, "-"), (1.0, "--"), (2.0, ":")):
        curve = []
        for r in rotations:
            values = [residual(20.0, -15.0, float(r), pedicles, pedicles, noise) for _ in range(60)]
            curve.append(np.median(values))
        axes[0].plot(rotations, curve, lw=1.6, ls=style, color="#2f6f4f",
                     label=f"plus pedicles at {noise:.1f} mm")
    axes[0].set_xlabel("axial rotation present (deg)")
    axes[0].set_ylabel("endplate normal error (deg)")
    axes[0].set_title("a  two more landmarks remove the bias")
    axes[0].legend(fontsize=7.5, frameon=False)

    errors = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    medians, p90s = [], []
    for noise in errors:
        values = [residual(*sample(), pedicles, pedicles, noise) for _ in range(draws)]
        medians.append(np.median(values))
        p90s.append(np.quantile(values, 0.9))
    baseline = [axial_rotation_bias(*sample()[:2], sample()[2])["normal_error_deg"]
                for _ in range(draws)]
    axes[1].axhline(np.median(baseline), color="#b03030", lw=1.6, ls="--",
                    label=f"no pedicles, median {np.median(baseline):.1f} deg")
    axes[1].plot(errors, medians, marker="o", ms=4, lw=1.8, color="#2f6f4f", label="median")
    axes[1].plot(errors, p90s, marker="s", ms=4, lw=1.4, ls="--", color="#2f6f4f", label="90th pct")
    axes[1].axhline(1.0, color="#999999", lw=0.9, ls=":")
    axes[1].set_xlabel("pedicle localisation error (mm)")
    axes[1].set_ylabel("endplate normal error (deg)")
    axes[1].set_title("b  the requirement: about 1 mm")
    axes[1].legend(fontsize=7.5, frameon=False)

    geometry_errors = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
    medians, p90s = [], []
    for spread in geometry_errors:
        values = []
        for _ in range(draws):
            truth = PedicleGeometry(
                pedicles.half_separation + rng.normal(0.0, spread),
                pedicles.posterior_offset + rng.normal(0.0, spread),
            )
            values.append(residual(*sample(), truth, pedicles, 0.0))
        medians.append(np.median(values))
        p90s.append(np.quantile(values, 0.9))
    axes[2].plot(geometry_errors, medians, marker="o", ms=4, lw=1.8, color="#7a5b9a", label="median")
    axes[2].plot(geometry_errors, p90s, marker="s", ms=4, lw=1.4, ls="--", color="#7a5b9a",
                 label="90th pct")
    axes[2].axvspan(0.0, 2.5, color="#cccccc", alpha=0.35)
    axes[2].text(1.25, axes[2].get_ylim()[1] * 0.85, "observed spread", ha="center", fontsize=7)
    axes[2].axhline(1.0, color="#999999", lw=0.9, ls=":")
    axes[2].set_xlabel("error in the assumed pedicle geometry (mm)")
    axes[2].set_ylabel("endplate normal error (deg)")
    axes[2].set_title("c  a normative table is enough")
    axes[2].legend(fontsize=7.5, frameon=False)

    _save(fig, out / "fig2_pedicles.png")


def geo_angle(u, v):
    from vertebra_xray_core import geometry as geo

    return geo.angle_between(u, v)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


FIGURES = {
    "landmarks": figure_landmarks,
    "pedicles": figure_pedicle_requirement,
    "planes": figure_planes,
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
