"""Three-dimensional spine model, its radiographic projections, and the
inverse problem of rebuilding it from a biplanar pair.

Patient frame
-------------
Right-handed, origin anywhere:

``X``
    towards the patient's **left**
``Y``
    towards the patient's **anterior**
``Z``
    towards the patient's **head**

Vertebral orientation
---------------------
Each vertebral body carries a rotation ``R`` taking the patient frame to its
own frame, composed as

.. math:: R = R_y(\\theta)\\, R_x(\\phi)\\, R_z(\\psi)

``theta``
    coronal tilt, about the antero-posterior axis. Positive raises the
    patient's left side. This is what a frontal radiograph measures.
``phi``
    sagittal tilt, about the left-right axis. Positive tips the superior
    endplate normal posteriorly.
``psi``
    axial rotation, about the vertebra's own cranio-caudal axis.

The order matters and is not arbitrary. Because ``psi`` is applied **first**,
about ``z``, it leaves the endplate normal ``n = R e_z`` untouched:

.. math:: R e_z = R_y R_x R_z e_z = R_y R_x e_z

So the endplate *normal* -- and therefore every 3-D Cobb angle -- is
independent of axial rotation. The endplate *lateral* and *antero-posterior*
directions, which are what a radiograph actually shows, are not. That
asymmetry is the whole difficulty of the inverse problem below.

The inverse problem
-------------------
A frontal radiograph yields ``alpha``, the tilt of the line joining the left
and right edges of an endplate. A lateral radiograph yields ``beta``, the tilt
of the line joining its anterior and posterior edges. Two measurements,
three unknowns: ``(theta, phi, psi)`` is underdetermined by a biplanar corner
annotation alone.

:func:`solve_orientation` therefore takes ``psi`` as an input. Supplying it --
from pedicle landmarks, from a CT segmentation, or as a torsion prior -- gives
an exact solution. Leaving it at zero, which is what published biplanar Cobb
pipelines do implicitly, gives a biased one;
:func:`axial_rotation_bias` measures how biased.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from . import geometry as geo
from .landmarks import SpineLandmarks

__all__ = [
    "Projection",
    "SpineModel3D",
    "rot_x",
    "rot_y",
    "rot_z",
    "orientation_matrix",
    "endplate_normal",
    "solve_orientation",
    "reconstruct_from_biplanar",
    "axial_rotation_bias",
]


# --------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------


def rot_x(deg: float) -> np.ndarray:
    """Rotation about the patient's left-right axis (sagittal tilt)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    """Rotation about the antero-posterior axis (coronal tilt).

    Signed so that a positive angle raises the patient's left side, which
    makes the coronal tilt of an unrotated vertebra equal to the tilt its
    endplate shows on a frontal radiograph.
    """
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def rot_z(deg: float) -> np.ndarray:
    """Rotation about the cranio-caudal axis (axial rotation)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def orientation_matrix(theta: float, phi: float, psi: float) -> np.ndarray:
    """``R_y(theta) R_x(phi) R_z(psi)`` for one vertebra, angles in degrees."""
    return rot_y(theta) @ rot_x(phi) @ rot_z(psi)


def endplate_normal(theta: float, phi: float) -> np.ndarray:
    """Unit endplate normal for a vertebra at ``(theta, phi)``, pointing cranially.

    Axial rotation is absent from the signature on purpose: it cannot change
    the normal. See the module docstring.
    """
    return (rot_y(theta) @ rot_x(phi))[:, 2]


# --------------------------------------------------------------------------
# projection geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Projection:
    """How a 3-D patient is turned into one radiograph.

    Attributes
    ----------
    view
        ``pa``, ``ap`` or ``lateral``.
    sod_mm, sdd_mm
        Source-to-object and source-to-detector distances. Both ``None``
        selects parallel (orthographic) projection, which is the usual
        idealisation and the default. Supplying them switches to the true
        point-source geometry, so the magnification of structures away from
        the mid-plane is reproduced and its effect on the measured angles can
        be quantified rather than assumed away.
    right_on_image_left
        Frontal views only. The usual display convention, patient's right on
        the viewer's left, is ``True``.
    anterior_on_image_left
        Lateral views only. ``True`` means the patient faces the left edge of
        the image, the common scoliosis-series convention.
    """

    view: Literal["pa", "ap", "lateral"] = "pa"
    sod_mm: float | None = None
    sdd_mm: float | None = None
    right_on_image_left: bool = True
    anterior_on_image_left: bool = True

    def __post_init__(self) -> None:
        if (self.sod_mm is None) != (self.sdd_mm is None):
            raise ValueError("give both sod_mm and sdd_mm, or neither")
        if self.sod_mm is not None and not 0 < self.sod_mm < self.sdd_mm:
            raise ValueError("need 0 < sod_mm < sdd_mm")

    @property
    def is_orthographic(self) -> bool:
        return self.sod_mm is None

    @property
    def beam_axis(self) -> np.ndarray:
        """Unit vector from source towards detector, in the patient frame."""
        if self.view == "pa":
            return np.array([0.0, 1.0, 0.0])  # enters the back, exits the front
        if self.view == "ap":
            return np.array([0.0, -1.0, 0.0])
        return np.array([1.0, 0.0, 0.0])  # lateral: beam runs right to left

    @property
    def image_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Patient-frame directions of the image ``+x`` and ``math``-frame ``+y``."""
        up = np.array([0.0, 0.0, 1.0])
        if self.view in ("pa", "ap"):
            sign = 1.0 if self.right_on_image_left else -1.0
            return sign * np.array([1.0, 0.0, 0.0]), up
        sign = -1.0 if self.anterior_on_image_left else 1.0
        return sign * np.array([0.0, 1.0, 0.0]), up

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """Project patient-frame points to ``math``-frame image coordinates."""
        p = np.asarray(points, dtype=float)
        ex, ey = self.image_axes
        u = p @ ex
        v = p @ ey
        if self.is_orthographic:
            return np.stack([u, v], axis=-1)
        depth = p @ self.beam_axis  # signed distance along the beam from isocentre
        scale = self.sdd_mm / (self.sod_mm + depth)
        return np.stack([u * scale, v * scale], axis=-1)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpineModel3D:
    """A stack of oriented vertebral bodies in the patient frame."""

    labels: tuple[str, ...]
    centroids: np.ndarray  # (N, 3) mm
    theta_deg: np.ndarray  # (N,) coronal tilt
    phi_deg: np.ndarray  # (N,) sagittal tilt
    psi_deg: np.ndarray  # (N,) axial rotation
    width_mm: np.ndarray  # (N,) left-right body width
    depth_mm: np.ndarray  # (N,) antero-posterior body depth
    height_mm: np.ndarray  # (N,) body height
    source: str = "unknown"
    meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        n = len(self.labels)
        for name in ("centroids", "theta_deg", "phi_deg", "psi_deg", "width_mm", "depth_mm", "height_mm"):
            arr = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, arr)
            if len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} rows for {n} labels")
        if self.centroids.shape != (n, 3):
            raise ValueError(f"centroids must be (N, 3), got {self.centroids.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def rotations(self) -> np.ndarray:
        """``(N, 3, 3)`` orientation matrices."""
        return np.stack(
            [
                orientation_matrix(t, p, s)
                for t, p, s in zip(self.theta_deg, self.phi_deg, self.psi_deg, strict=True)
            ]
        )

    @property
    def endplate_normals(self) -> np.ndarray:
        """``(N, 3)`` unit superior-endplate normals, pointing cranially."""
        return np.stack(
            [endplate_normal(t, p) for t, p in zip(self.theta_deg, self.phi_deg, strict=True)]
        )

    def body_corners(self) -> np.ndarray:
        """``(N, 8, 3)`` corners of each vertebral body as a rectangular box.

        Order along axis 1 is ``(+-lateral, +-anterior, +-cranial)`` with the
        lateral index varying slowest, which the projection helpers rely on.
        """
        out = np.empty((len(self), 8, 3))
        rots = self.rotations
        for i, R in enumerate(rots):
            half = np.array([self.width_mm[i], self.depth_mm[i], self.height_mm[i]]) / 2.0
            signs = np.array(
                [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
            )
            out[i] = self.centroids[i] + (signs * half) @ R.T
        return out

    def project(self, projection: Projection, *, source: str | None = None) -> SpineLandmarks:
        """Render the model as the four annotated corners of one radiograph.

        On a frontal view the four landmarks are the left and right ends of
        the superior and inferior endplates. On a lateral view they are the
        anterior and posterior ends. That is what an annotator marks, and it
        is why the two views constrain different components of the
        orientation.
        """
        corners = self.body_corners()  # (N, 8, 3)
        # Pick the four box corners an annotator would see, per view.
        if projection.view in ("pa", "ap"):
            # vary the lateral sign and the cranial sign, keep the body mid-depth
            picks = [(-1, 0, +1), (+1, 0, +1), (-1, 0, -1), (+1, 0, -1)]
        else:
            picks = [(0, -1, +1), (0, +1, +1), (0, -1, -1), (0, +1, -1)]
        rots = self.rotations
        pts = np.empty((len(self), 4, 3))
        for i, R in enumerate(rots):
            half = np.array([self.width_mm[i], self.depth_mm[i], self.height_mm[i]]) / 2.0
            for k, sign in enumerate(picks):
                pts[i, k] = self.centroids[i] + (np.array(sign, dtype=float) * half) @ R.T
        del corners
        projected = projection.project_points(pts.reshape(-1, 3)).reshape(len(self), 4, 2)
        return SpineLandmarks(
            corners=_order_quad(projected),
            view=projection.view,
            labels=self.labels,
            source=source or f"projection({projection.view}) of {self.source}",
            meta={"projection": projection},
        )


def _order_quad(quads: np.ndarray) -> np.ndarray:
    """Canonical ``UL, UR, LL, LR`` ordering for already-``math``-frame quads."""
    from .landmarks import canonicalise_corners

    return canonicalise_corners(quads)


# --------------------------------------------------------------------------
# the inverse problem
# --------------------------------------------------------------------------


def _measured_tilts(theta: float, phi: float, psi: float, lateral_sign: float) -> tuple[float, float]:
    """Endplate tilts a biplanar pair would show for a given orientation.

    ``lateral_sign`` is ``-1`` when the lateral view puts anterior on the left.
    """
    R = orientation_matrix(theta, phi, psi)
    e = R[:, 0]  # left-right body direction
    a = R[:, 1]  # antero-posterior body direction
    alpha = float(geo.tilt_deg(np.array([e[0], e[2]])))
    beta = float(geo.tilt_deg(np.array([lateral_sign * a[1], a[2]])))
    return alpha, beta


def solve_orientation(
    alpha_deg: float,
    beta_deg: float,
    psi_deg: float = 0.0,
    *,
    anterior_on_image_left: bool = True,
) -> tuple[float, float]:
    """Recover ``(theta, phi)`` from the two measured endplate tilts.

    ``psi_deg`` must be supplied from outside -- pedicle landmarks, a CT
    segmentation, or a prior. With ``psi_deg=0`` the solution reduces to the
    assumption every published biplanar pipeline makes implicitly.

    The forward map is smooth and, near the anatomically plausible range,
    close to the identity in ``theta``, so a two-parameter least-squares solve
    from the obvious starting guess converges in a handful of iterations.
    """
    lateral_sign = -1.0 if anterior_on_image_left else 1.0

    def residual(x: np.ndarray) -> np.ndarray:
        a, b = _measured_tilts(float(x[0]), float(x[1]), psi_deg, lateral_sign)
        return np.array(
            [
                float(geo.wrap_to_signed_right_angle(a - alpha_deg)),
                float(geo.wrap_to_signed_right_angle(b - beta_deg)),
            ]
        )

    guess = np.array([alpha_deg, -beta_deg], dtype=float)
    fit = least_squares(residual, guess, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return float(fit.x[0]), float(fit.x[1])


def reconstruct_from_biplanar(
    frontal: SpineLandmarks,
    lateral: SpineLandmarks,
    *,
    axial_rotation_deg: np.ndarray | None = None,
    spacing_mm: float | None = None,
    anterior_on_image_left: bool = True,
    match: Literal["label", "row"] = "label",
) -> SpineModel3D:
    """Rebuild a :class:`SpineModel3D` from a matched frontal/lateral pair.

    Both inputs must be labelled; vertebrae are matched by level, not by row
    order, so a lateral view that omits a level the frontal view resolved is
    handled by intersecting the two label sets.

    Parameters
    ----------
    axial_rotation_deg
        Per-level axial rotation in the *matched* level order. ``None`` means
        zero everywhere, which is the conventional assumption and introduces
        the bias :func:`axial_rotation_bias` quantifies.
    match
        How the two views are put into correspondence. ``label`` intersects
        the two label sets, which is right whenever the levels are known and
        handles a view cropped short at either end. ``row`` pairs them by
        position and needs equal row counts; it exists for the chicken-and-egg
        case where the levels are not known yet, because reconstruction only
        needs to know which row is which row, not what either is called.
    spacing_mm
        Isotropic millimetres per pixel, applied to both views. Angles are
        scale-free, so this only affects the reported centroid positions and
        body dimensions.

    Notes
    -----
    The cranio-caudal coordinate is observed twice, once per view. They are
    reconciled by a single similarity fit of the lateral height axis onto the
    frontal one over the shared levels, rather than by averaging raw
    coordinates, because the two exposures are rarely at the same table height
    or magnification.
    """
    if frontal.view == "lateral":
        raise ValueError("the first argument must be the frontal view")
    if lateral.view != "lateral":
        raise ValueError("the second argument must be the lateral view")

    if match == "row":
        if len(frontal) != len(lateral):
            raise ValueError(
                f"row matching needs equal row counts, got {len(frontal)} and {len(lateral)}"
            )
        fi = li = list(range(len(frontal)))
        shared = list(frontal.labels or lateral.labels or [f"#{i}" for i in fi])
    elif match == "label":
        if frontal.labels is None or lateral.labels is None:
            raise ValueError(
                "both views must be labelled before they can be matched by label; "
                "pass match='row' to pair them by position instead"
            )
        if len(set(frontal.labels)) != len(frontal.labels):
            raise ValueError(f"the frontal view has repeated labels: {frontal.labels}")
        if len(set(lateral.labels)) != len(lateral.labels):
            raise ValueError(f"the lateral view has repeated labels: {lateral.labels}")
        shared = [lab for lab in frontal.labels if lab in set(lateral.labels)]
        if len(shared) < 2:
            raise ValueError("the two views share fewer than two labelled vertebrae")
        fi = [frontal.index_of_label(lab) for lab in shared]
        li = [lateral.index_of_label(lab) for lab in shared]
    else:
        raise ValueError(f"unknown match mode {match!r}")

    scale = 1.0 if spacing_mm is None else float(spacing_mm)
    f_centroids = frontal.centroids[fi] * scale
    l_centroids = lateral.centroids[li] * scale

    # Reconcile the two height axes: fit l_z = m * f_z + c over shared levels.
    m, c = np.polyfit(l_centroids[:, 1], f_centroids[:, 1], 1)
    z = 0.5 * (f_centroids[:, 1] + (m * l_centroids[:, 1] + c))
    x = f_centroids[:, 0]
    y = -l_centroids[:, 0] / m if anterior_on_image_left else l_centroids[:, 0] / m
    y = y - y.mean()
    centroids = np.stack([x - x.mean(), y, z], axis=1)

    alpha = frontal.body_tilt()[fi]
    beta = lateral.body_tilt()[li]
    psi = (
        np.zeros(len(shared))
        if axial_rotation_deg is None
        else np.asarray(axial_rotation_deg, dtype=float)
    )
    if len(psi) != len(shared):
        raise ValueError(f"axial_rotation_deg has {len(psi)} entries for {len(shared)} levels")

    theta = np.empty(len(shared))
    phi = np.empty(len(shared))
    for k in range(len(shared)):
        theta[k], phi[k] = solve_orientation(
            float(alpha[k]),
            float(beta[k]),
            float(psi[k]),
            anterior_on_image_left=anterior_on_image_left,
        )

    width = np.linalg.norm(frontal.body_axis[fi], axis=1) * scale
    depth = np.linalg.norm(lateral.body_axis[li], axis=1) * scale
    height = 0.5 * (frontal.body_height[fi] + lateral.body_height[li]) * scale

    return SpineModel3D(
        labels=tuple(shared),
        centroids=centroids,
        theta_deg=theta,
        phi_deg=phi,
        psi_deg=psi,
        width_mm=width,
        depth_mm=depth,
        height_mm=height,
        source=f"biplanar({frontal.source} + {lateral.source})",
        meta={
            "height_axis_scale": float(m),
            "assumed_zero_axial_rotation": axial_rotation_deg is None,
            "spacing_mm": spacing_mm,
        },
    )


def axial_rotation_bias(
    theta_deg: float,
    phi_deg: float,
    psi_deg: float,
    *,
    anterior_on_image_left: bool = True,
) -> dict[str, float]:
    """Error incurred by reconstructing a rotated vertebra as if ``psi`` were 0.

    Returns the recovered tilts, their errors, and the angle between the true
    and recovered endplate normals. Mapping this over the plausible range of
    ``psi`` is what turns "we assume no axial rotation" into a stated
    uncertainty.
    """
    lateral_sign = -1.0 if anterior_on_image_left else 1.0
    alpha, beta = _measured_tilts(theta_deg, phi_deg, psi_deg, lateral_sign)
    theta_hat, phi_hat = solve_orientation(
        alpha, beta, 0.0, anterior_on_image_left=anterior_on_image_left
    )
    n_true = endplate_normal(theta_deg, phi_deg)
    n_hat = endplate_normal(theta_hat, phi_hat)
    return {
        "alpha_deg": alpha,
        "beta_deg": beta,
        "theta_true_deg": theta_deg,
        "phi_true_deg": phi_deg,
        "theta_recovered_deg": theta_hat,
        "phi_recovered_deg": phi_hat,
        "theta_error_deg": theta_hat - theta_deg,
        "phi_error_deg": phi_hat - phi_deg,
        "normal_error_deg": geo.angle_between(n_true, n_hat),
    }
