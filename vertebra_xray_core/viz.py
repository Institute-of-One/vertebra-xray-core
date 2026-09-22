"""Figures: the spine in three dimensions, and the plots behind each claim.

Matplotlib only, and an optional dependency at that -- the core measures
without it. This is the figure layer for the manuscript and for checking a
result by eye; the interactive product front-end renders the same geometry
through VTK and Cornerstone3D and does not import this module.

Everything here draws from a :class:`~vertebra_xray_core.spine3d.SpineModel3D`
or from landmarks, never from an image, so a figure can be produced from a
phantom with no patient data in it at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import cobb3d
from . import geometry as geo
from . import nomenclature as nom
from ._profile import prune, turning_points
from .cobb import Curve
from .landmarks import LL, LR, UL, UR, SpineLandmarks
from .spine3d import SpineModel3D

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

__all__ = [
    "REGION_COLOURS",
    "view_basis",
    "plot_spine_3d",
    "plot_projection",
    "plot_tilt_profile",
    "plot_pmc_profile",
    "plot_axial_path",
    "plot_radiograph",
    "plot_landmark_overlay",
    "project_pedicles_to_view",
    "angle_between_normals",
    "UL",
    "UR",
    "LL",
    "LR",
]

#: One colour per spinal region, used consistently across every figure so a
#: reader can carry the mapping from one panel to the next.
REGION_COLOURS = {
    "cervical": "#7e9bb8",
    "thoracic": "#c2703d",
    "lumbar": "#4f8a6b",
    "sacral": "#8a7b9e",
}


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - environment dependent
        raise ImportError(
            "figures need matplotlib; install it with pip install 'vertebra-xray-core[figures]'"
        ) from None
    return plt


def _hex_to_rgb(colour: str) -> tuple[float, float, float]:
    colour = colour.lstrip("#")
    return tuple(int(colour[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


# --------------------------------------------------------------------------
# the spine in space
# --------------------------------------------------------------------------

#: The six faces of a body box, as indices into the corner order
#: :meth:`SpineModel3D.body_corners` produces.
_BOX_FACES = (
    (0, 1, 3, 2),  # towards the patient's right
    (4, 5, 7, 6),  # towards the patient's left
    (0, 1, 5, 4),  # posterior
    (2, 3, 7, 6),  # anterior
    (0, 2, 6, 4),  # inferior endplate
    (1, 3, 7, 5),  # superior endplate
)


def view_basis(elev_deg: float, azim_deg: float) -> np.ndarray:
    """Camera axes for an orthographic view, as rows ``(right, up, towards)``.

    ``azim`` orbits the patient about the cranio-caudal axis and ``elev``
    raises the camera above the transverse plane. At ``(0, 0)`` the camera
    looks along the patient's anterior axis from in front, which is the
    frontal radiograph's own viewpoint, so a figure and a projection can be
    put side by side without the reader having to re-orient.
    """
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    towards = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), -np.sin(e)])
    right = np.cross(towards, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-8:  # looking straight down the spine
        right = np.array([1.0, 0.0, 0.0])
    right = geo.unit(right)
    return np.stack([right, geo.unit(np.cross(right, towards)), towards])


def plot_spine_3d(
    model: SpineModel3D,
    ax: Axes | None = None,
    *,
    show_bodies: bool = True,
    show_centreline: bool = True,
    show_normals: bool = False,
    show_labels: bool = True,
    label_every: int = 1,
    curves: tuple[Curve, ...] = (),
    pmc_plane_for: cobb3d.Cobb3D | None = None,
    body_alpha: float = 1.0,
    view: tuple[float, float] = (12.0, 28.0),
    scale_bar_mm: float | None = None,
    light: tuple[float, float, float] = (0.45, -0.75, 0.5),
) -> Axes:
    """Render the vertebral bodies and the spinal centreline in three dimensions.

    The scene is projected orthographically and drawn into an ordinary 2-D
    axis with equal aspect, not through Matplotlib's 3-D axes. That is not a
    workaround for its own sake. A 3-D axis fits a cube to the panel, so a
    spine -- half a metre tall and a hand's breadth wide -- comes out as a
    sliver with the rest of the panel empty, and the remedies for that either
    distort the angles or clip the drawing. Projecting directly costs a few
    dozen lines and gives exact control of scale, so the angles in the picture
    are the angles in the table beside it.

    Faces are drawn far-to-near and shaded by their orientation to ``light``,
    which is what makes the boxes read as solids and makes axial rotation
    visible at a glance rather than only in a number.

    Parameters
    ----------
    view
        ``(elevation, azimuth)`` in degrees. ``(0, 0)`` is the frontal
        radiograph's viewpoint; the default is a slight oblique so that
        coronal and sagittal deformity are both visible rather than one of
        them being edge-on.
    curves
        Structural coronal curves. Their end vertebrae and apices are outlined
        and named, so the measurement can be read off the anatomy.
    pmc_plane_for
        Also draw the plane of maximum curvature for this measurement, which
        is the one thing a coronal radiograph cannot show.
    scale_bar_mm
        Length of the scale bar drawn at the left. The axes carry no ticks --
        a grid around a spine says nothing and costs most of the panel.
    """
    plt = _require_matplotlib()
    from matplotlib.patches import Polygon

    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 8.5))

    basis = view_basis(*view)
    highlighted = _curve_roles(model, curves)
    corners = model.body_corners()

    faces: list[tuple[float, np.ndarray, dict]] = []
    if show_bodies:
        light_dir = geo.unit(np.asarray(light, dtype=float))
        for index, label in enumerate(model.labels):
            role = highlighted.get(label)
            base = np.array(_hex_to_rgb(REGION_COLOURS.get(nom.region_of(label), "#888888")))
            for face in _BOX_FACES:
                quad = corners[index][list(face)]
                normal = geo.unit(np.cross(quad[1] - quad[0], quad[2] - quad[1]))
                shade = 0.55 + 0.45 * abs(float(np.dot(normal, light_dir)))
                faces.append(
                    (
                        float((quad @ basis[2]).mean()),
                        quad @ basis[:2].T,
                        {
                            "facecolor": tuple(np.clip(base * shade, 0.0, 1.0)),
                            "edgecolor": "#101010" if role else "#4a4a4a",
                            "linewidth": 1.1 if role else 0.35,
                            "alpha": body_alpha,
                        },
                    )
                )

    if pmc_plane_for is not None:
        quad = _pmc_quad(model, pmc_plane_for)
        faces.append(
            (
                float((quad @ basis[2]).mean()),
                quad @ basis[:2].T,
                {
                    "facecolor": "#c8b45a",
                    "edgecolor": "#8a7a2a",
                    "linewidth": 0.9,
                    "alpha": 0.30,
                },
            )
        )

    for _, polygon, style in sorted(faces, key=lambda item: -item[0]):
        ax.add_patch(Polygon(polygon, closed=True, **style))

    flat = model.centroids @ basis[:2].T
    if show_centreline:
        ax.plot(flat[:, 0], flat[:, 1], color="#111111", lw=1.5, zorder=50)
        ax.plot(flat[:, 0], flat[:, 1], "o", ms=3.0, color="#111111", zorder=51)

    if show_normals:
        length = 1.15 * float(np.median(model.height_mm))
        tips = (model.centroids + length * model.endplate_normals) @ basis[:2].T
        for start, end in zip(flat, tips, strict=True):
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#a02020", lw=1.0, zorder=52)

    corner_2d = corners.reshape(-1, 3) @ basis[:2].T
    label_margin = 0.0
    if show_labels:
        label_margin = 0.62 * float(np.max(model.width_mm))
        for index, label in enumerate(model.labels):
            role = highlighted.get(label)
            if index % label_every and role is None:
                continue
            ax.text(
                flat[index, 0] + label_margin,
                flat[index, 1],
                label if role is None else f"{label} {role}",
                fontsize=7,
                va="center",
                zorder=60,
                color="#101010" if role else "#666666",
                weight="bold" if role else "normal",
            )

    pad = 0.04 * float(corner_2d[:, 1].max() - corner_2d[:, 1].min())
    left = corner_2d[:, 0].min() - pad - (0.9 * label_margin if scale_bar_mm else pad)
    ax.set_xlim(left, corner_2d[:, 0].max() + pad + 2.4 * label_margin)
    ax.set_ylim(corner_2d[:, 1].min() - pad, corner_2d[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    if scale_bar_mm:
        x = left + 0.35 * pad
        y = corner_2d[:, 1].min()
        ax.plot([x, x], [y, y + scale_bar_mm], color="#111111", lw=2.2, zorder=60)
        ax.text(
            x + 0.45 * pad,
            y + scale_bar_mm / 2,
            f"{scale_bar_mm:.0f} mm",
            fontsize=7,
            rotation=90,
            va="center",
            zorder=60,
        )
    return ax


def _curve_roles(model: SpineModel3D, curves: tuple[Curve, ...]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for curve in curves:
        for label, role in (
            (curve.upper_label, "end"),
            (curve.lower_label, "end"),
            (curve.apex_label, "apex"),
        ):
            if label in model.labels:
                roles[label] = role
    return roles


def _pmc_quad(model: SpineModel3D, measurement: cobb3d.Cobb3D) -> np.ndarray:
    """Corners of the plane of maximum curvature, sized to the curve it belongs to."""
    labels = list(model.labels)
    i, j = labels.index(measurement.upper_label), labels.index(measurement.lower_label)
    centre = model.centroids[[i, j]].mean(axis=0)
    direction = np.array(
        [
            np.cos(np.radians(measurement.pmc_from_coronal_deg)),
            np.sin(np.radians(measurement.pmc_from_coronal_deg)),
            0.0,
        ]
    )
    half_width = 0.95 * float(np.max(model.width_mm))
    half_height = 0.62 * abs(model.centroids[i, 2] - model.centroids[j, 2])
    up = np.array([0.0, 0.0, 1.0])
    return np.array(
        [
            centre + s * half_width * direction + t * half_height * up
            for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )


# --------------------------------------------------------------------------
# projections and profiles
# --------------------------------------------------------------------------


def plot_projection(
    landmarks: SpineLandmarks,
    ax: Axes | None = None,
    *,
    curves: tuple[Curve, ...] = (),
    show_endplates: bool = True,
    show_labels: bool = True,
) -> Axes:
    """Draw one projection's landmarks with the Cobb construction over them.

    Coordinates are the ``math`` frame, so the figure is the right way up
    without any further flipping.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 8.0))

    quads = landmarks.corners
    for index in range(len(landmarks)):
        outline = quads[index][[UL, UR, LR, LL, UL]]
        ax.plot(outline[:, 0], outline[:, 1], color="#8a8a8a", lw=0.7)
    ax.plot(
        landmarks.centroids[:, 0],
        landmarks.centroids[:, 1],
        color="#1a1a1a",
        lw=1.2,
        marker="o",
        markersize=2.5,
    )

    if show_endplates:
        for curve in curves:
            for index, which in ((curve.upper_index, "superior"), (curve.lower_index, "inferior")):
                pair = (UL, UR) if which == "superior" else (LL, LR)
                a, b = quads[index][pair[0]], quads[index][pair[1]]
                mid = 0.5 * (a + b)
                extend = 1.9
                ax.plot(
                    *zip(mid + extend * (a - mid), mid + extend * (b - mid), strict=True),
                    color="#b03030",
                    lw=1.4,
                )

    if show_labels and landmarks.labels is not None:
        offset = 0.8 * float(np.median(np.linalg.norm(landmarks.body_axis, axis=1)))
        for index, label in enumerate(landmarks.labels):
            ax.text(
                landmarks.centroids[index, 0] + offset,
                landmarks.centroids[index, 1],
                label,
                fontsize=6.5,
                va="center",
                color="#555555",
            )

    ax.set_aspect("equal")
    ax.set_xlabel("image x (mm)")
    ax.set_ylabel("image y, cranial up (mm)")
    return ax


def plot_tilt_profile(
    landmarks: SpineLandmarks,
    ax: Axes | None = None,
    *,
    min_amplitude: float = 5.0,
    anchor_index: int | None = None,
    anchor_position: float | None = None,
    title: str | None = None,
) -> Axes:
    """The per-vertebra tilt profile with its turning points marked.

    This is the signal every structural decision in the package is made from:
    coronal curves are its monotone runs, and the thoracolumbar anchor is the
    reversal in its sagittal counterpart.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(3.4, 8.0))

    tilt = landmarks.body_tilt()
    rows = np.arange(len(tilt))
    ax.plot(tilt, rows, color="#1a1a1a", lw=1.4, marker="o", markersize=3)
    ax.axvline(0.0, color="#999999", lw=0.8, ls=":")

    for point in prune(tilt, turning_points(tilt), min_amplitude):
        ax.plot(tilt[point], point, marker="s", markersize=7, mfc="none", color="#b03030")

    if anchor_position is not None:
        ax.axhline(anchor_position, color="#2f6f4f", lw=1.2, ls="--")
    if anchor_index is not None:
        ax.plot(tilt[anchor_index], anchor_index, marker="*", markersize=14, color="#2f6f4f")

    if landmarks.labels is not None:
        ax.set_yticks(rows)
        ax.set_yticklabels(landmarks.labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("endplate tilt (deg)")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_pmc_profile(
    model: SpineModel3D,
    measurements: tuple[cobb3d.Cobb3D, ...],
    ax: Axes | None = None,
) -> Axes:
    """Cobb angle as a function of the measurement plane's orientation.

    The coronal radiograph samples this curve at a single point, chosen for
    convenience rather than for where the deformity is. Drawing the whole
    function is the clearest statement of what a three-dimensional Cobb angle
    adds: the circle marks what the radiograph reports, the star what the
    deformity actually is.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.0))

    labels = list(model.labels)
    normals = model.endplate_normals
    for measurement in measurements:
        i = labels.index(measurement.upper_label)
        j = labels.index(measurement.lower_label)
        omega, angle = cobb3d.pmc_profile(normals[i], normals[j], step_deg=0.25)
        shifted = np.where(omega > 90.0, omega - 180.0, omega)
        order = np.argsort(shifted)
        name = measurement.name or f"{measurement.upper_label}-{measurement.lower_label}"
        (line,) = ax.plot(shifted[order], angle[order], lw=1.6, label=name)
        ax.plot(
            measurement.pmc_from_coronal_deg,
            measurement.pmc_deg,
            marker="*",
            markersize=13,
            color=line.get_color(),
        )
        ax.plot(0.0, measurement.coronal_deg, marker="o", markersize=6, color=line.get_color())

    for position in (-90.0, 0.0, 90.0):
        ax.axvline(position, color="#999999", lw=0.8, ls=":")
    ax.set_xlim(-90.0, 90.0)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel("measurement plane, degrees from coronal")
    ax.set_ylabel("Cobb angle (deg)")
    ax.legend(fontsize=8, frameon=False)
    return ax


def plot_axial_path(
    model: SpineModel3D,
    ax: Axes | None = None,
    *,
    measurements: tuple[cobb3d.Cobb3D, ...] = (),
    label_every: int = 3,
    title: str | None = None,
) -> Axes:
    """The spinal centreline seen from above, with each curve's measurement plane.

    Looking down the patient collapses the cranio-caudal axis and leaves the
    two deviations that matter: lateral, which the frontal radiograph shows,
    and antero-posterior, which the lateral one shows. A scoliotic spine
    traces a loop here, and the direction that loop is longest in is the plane
    of maximum curvature. Rendering the spine itself from this angle does not
    work -- every vertebra lands on top of every other -- but the path does,
    and it is the view that makes a three-dimensional deformity legible.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.0))

    path = model.centroids
    points = ax.scatter(
        path[:, 0],
        path[:, 1],
        c=path[:, 2],
        cmap="viridis",
        s=26,
        zorder=20,
        edgecolor="#333333",
        linewidth=0.3,
    )
    ax.plot(path[:, 0], path[:, 1], color="#555555", lw=1.0, zorder=10)
    bar = ax.figure.colorbar(points, ax=ax, fraction=0.045, pad=0.03)
    bar.set_label("cranial position (mm)", fontsize=7)
    bar.ax.tick_params(labelsize=6)

    for index, label in enumerate(model.labels):
        if index % label_every:
            continue
        ax.annotate(
            label,
            (path[index, 0], path[index, 1]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=6.5,
            color="#444444",
        )

    reach = float(np.abs(path[:, :2] - path[:, :2].mean(axis=0)).max()) * 1.35
    centre = path[:, :2].mean(axis=0)
    for measurement in measurements:
        angle = np.radians(measurement.pmc_from_coronal_deg)
        direction = np.array([np.cos(angle), np.sin(angle)])
        ends = np.stack([centre - reach * direction, centre + reach * direction])
        name = measurement.name or measurement.upper_label
        ax.plot(
            ends[:, 0],
            ends[:, 1],
            lw=1.4,
            ls="--",
            zorder=5,
            label=f"{name} plane, {measurement.pmc_deg:.0f} deg",
        )

    ax.axhline(centre[1], color="#cccccc", lw=0.7, zorder=1)
    ax.axvline(centre[0], color="#cccccc", lw=0.7, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlabel("patient left (mm)")
    ax.set_ylabel("anterior (mm)")
    if measurements:
        ax.legend(fontsize=7, frameon=False, loc="best")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def angle_between_normals(model: SpineModel3D, upper: str, lower: str) -> float:
    """Convenience for captions: the 3-D angle between two endplates."""
    labels = list(model.labels)
    normals = model.endplate_normals
    return geo.angle_between(normals[labels.index(upper)], normals[labels.index(lower)])


# --------------------------------------------------------------------------
# radiographs and the landmarks a detector should place on them
# --------------------------------------------------------------------------


def plot_radiograph(
    radiograph,
    ax: Axes | None = None,
    *,
    gamma: float = 1.0,
    polarity: str = "film",
) -> Axes:
    """Draw a simulated radiograph in millimetre coordinates.

    The extent is in the same ``math`` frame the landmarks are projected into,
    so anything from
    :meth:`~vertebra_xray_core.spine3d.SpineModel3D.project` overlays on top
    without further transformation.

    ``Radiograph.image`` holds transmitted intensity, which is the physical
    quantity; a radiograph as anyone reads one is its negative, bone white and
    air black. ``polarity="film"`` shows that, ``"transmission"`` shows the
    array as it stands.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 8.5))
    if polarity not in {"film", "transmission"}:
        raise ValueError(f"polarity must be 'film' or 'transmission', got {polarity!r}")
    image = radiograph.image ** gamma if gamma != 1.0 else radiograph.image
    ax.imshow(
        image,
        cmap="gray_r" if polarity == "film" else "gray",
        origin="lower", extent=radiograph.extent, aspect="equal",
        interpolation="bilinear",
    )
    ax.set_axis_off()
    return ax


def plot_landmark_overlay(
    landmarks: SpineLandmarks,
    ax: Axes,
    *,
    pedicles: np.ndarray | None = None,
    curves: tuple[Curve, ...] = (),
    label_every: int = 2,
    corner_colour: str = "#3aa0ff",
    pedicle_colour: str = "#ffd23a",
    cobb_colour: str = "#ff5c5c",
) -> Axes:
    """Draw the landmark set a detector is being asked to produce.

    Four corners per vertebral body is what current detectors output and what
    leaves axial rotation undetermined. The two pedicle centroids are the
    addition this package argues for, drawn in a second colour so the ask is
    legible at a glance.

    ``pedicles`` is ``(N, 2, 2)`` in the same frame as the landmarks.
    """
    quads = landmarks.corners
    for index in range(len(landmarks)):
        outline = quads[index][[UL, UR, LR, LL, UL]]
        ax.plot(outline[:, 0], outline[:, 1], color=corner_colour, lw=0.8, alpha=0.75, zorder=10)
        ax.plot(
            quads[index][:, 0], quads[index][:, 1], "o", ms=3.2,
            mfc=corner_colour, mec="#0b2b4a", mew=0.4, zorder=12,
        )

    if pedicles is not None:
        flat = np.asarray(pedicles, dtype=float).reshape(-1, 2)
        ax.plot(
            flat[:, 0], flat[:, 1], "o", ms=4.2,
            mfc=pedicle_colour, mec="#5a4400", mew=0.5, zorder=13,
        )

    for curve in curves:
        for index, which in ((curve.upper_index, "superior"), (curve.lower_index, "inferior")):
            pair = (UL, UR) if which == "superior" else (LL, LR)
            a, b = quads[index][pair[0]], quads[index][pair[1]]
            mid = 0.5 * (a + b)
            extend = 2.6
            ax.plot(
                *zip(mid + extend * (a - mid), mid + extend * (b - mid), strict=True),
                color=cobb_colour, lw=1.6, zorder=14,
            )

    if landmarks.labels is not None:
        offset = 0.95 * float(np.median(np.linalg.norm(landmarks.body_axis, axis=1)))
        for index, label in enumerate(landmarks.labels):
            if index % label_every:
                continue
            ax.text(
                landmarks.centroids[index, 0] - offset,
                landmarks.centroids[index, 1],
                label,
                fontsize=6.5, va="center", ha="right", zorder=15,
                color="#f2f2f2",
                bbox={"facecolor": "#1a1a1a", "alpha": 0.55, "pad": 0.8, "edgecolor": "none"},
            )
    return ax


def project_pedicles_to_view(model: SpineModel3D, projection, pedicle_table) -> np.ndarray:
    """``(N, 2, 2)`` projected pedicle positions for overlaying on a view."""
    out = np.empty((len(model), 2, 2))
    rotations = model.rotations
    for index, label in enumerate(model.labels):
        geometry = pedicle_table(label)
        world = model.centroids[index] + geometry.as_body_frame() @ rotations[index].T
        out[index] = projection.project_points(world)
    return out
