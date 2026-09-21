"""Three-dimensional Cobb angles and the plane of maximum curvature.

A Cobb angle is not a property of a spine; it is a property of a spine *and a
projection*. The coronal radiograph is one particular projection, chosen
because it is easy to acquire, not because it is where the deformity is
largest. Rotating the measurement plane about the cranio-caudal axis changes
the number continuously, and the plane where it peaks -- the plane of maximum
curvature, PMC -- is the one that describes the deformity.

Three distinct numbers are reported for every curve and they are routinely
conflated:

``coronal_deg``
    what a frontal radiograph measures, the projection at the coronal plane.
``pmc_deg``
    the maximum over all measurement planes containing the cranio-caudal
    axis, with ``pmc_from_coronal_deg`` saying where that maximum sits. This
    is the SRS three-dimensional Cobb angle. It is always at least
    ``coronal_deg``.
``normal_angle_deg``
    the plain angle between the two endplate normals in space. This is *not*
    a Cobb angle and is generally smaller than ``pmc_deg``, because projection
    onto a plane can open an angle rather than close it.

All three fall out of the endplate normals, which
:mod:`vertebra_xray_core.spine3d` shows to be independent of axial rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from . import geometry as geo
from .cobb import Curve
from .spine3d import SpineModel3D

__all__ = [
    "Cobb3D",
    "projected_tilt_deg",
    "projected_angle_deg",
    "plane_of_maximum_curvature",
    "pmc_profile",
    "measure_between_levels",
    "cobb3d_for_curves",
    "as_seen_on_film",
    "KYPHOSIS_LEVELS",
    "LORDOSIS_LEVELS",
]

#: Levels conventionally used to report thoracic kyphosis and lumbar lordosis.
KYPHOSIS_LEVELS = ("T4", "T12")
LORDOSIS_LEVELS = ("L1", "L5")


@dataclass(frozen=True)
class Cobb3D:
    """One curve measured in three dimensions."""

    upper_label: str
    lower_label: str
    coronal_deg: float
    sagittal_deg: float
    pmc_deg: float
    pmc_from_coronal_deg: float
    normal_angle_deg: float
    name: str | None = None

    @property
    def pmc_from_sagittal_deg(self) -> float:
        """Orientation of the plane of maximum curvature from the sagittal plane."""
        return float(geo.wrap_to_signed_right_angle(90.0 - self.pmc_from_coronal_deg))

    @property
    def coronal_underestimate_deg(self) -> float:
        """How much the frontal radiograph understates the deformity."""
        return self.pmc_deg - self.coronal_deg

    def describe(self) -> str:
        """One-line summary in the order a report would state it."""
        name = self.name or "curve"
        return (
            f"{name} {self.upper_label}-{self.lower_label}: "
            f"coronal {self.coronal_deg:.1f} deg, "
            f"3-D {self.pmc_deg:.1f} deg in a plane {self.pmc_from_coronal_deg:+.1f} deg "
            f"from coronal (sagittal {self.sagittal_deg:.1f} deg)"
        )


# --------------------------------------------------------------------------
# projection of an endplate normal into an arbitrary measurement plane
# --------------------------------------------------------------------------


def _plane_direction(omega_deg: float | np.ndarray) -> np.ndarray:
    """In-plane horizontal direction of the measurement plane at ``omega``.

    ``omega`` is measured from the patient's left axis, so ``0`` is the
    coronal plane and ``90`` the sagittal plane. Every measurement plane
    contains the cranio-caudal axis, which is what makes the result a Cobb
    angle rather than an arbitrary dihedral.
    """
    w = np.radians(np.asarray(omega_deg, dtype=float))
    return np.stack([np.cos(w), np.sin(w), np.zeros_like(w)], axis=-1)


def projected_tilt_deg(normal: np.ndarray, omega_deg: float | np.ndarray) -> np.ndarray:
    """Tilt of an endplate, in degrees, as seen in the measurement plane ``omega``.

    Endplate normals point cranially, so the vertical component is positive
    and the tilt stays inside ``(-90, 90)``. That makes the tilt a continuous
    function of ``omega`` with no branch cut, which is what lets the maximum
    below be found by a plain one-dimensional search.
    """
    n = np.asarray(normal, dtype=float)
    u = _plane_direction(omega_deg)
    return np.degrees(np.arctan2(u @ n, n[2]))


def projected_angle_deg(
    upper_normal: np.ndarray, lower_normal: np.ndarray, omega_deg: float | np.ndarray
) -> np.ndarray:
    """Cobb angle between two endplates in the measurement plane ``omega``."""
    return np.abs(
        projected_tilt_deg(upper_normal, omega_deg) - projected_tilt_deg(lower_normal, omega_deg)
    )


def pmc_profile(
    upper_normal: np.ndarray, lower_normal: np.ndarray, *, step_deg: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """``(omega, angle)`` sampled over a half turn, for plotting.

    The profile is 180-degree periodic because a measurement plane and its
    opposite are the same plane.
    """
    omega = np.arange(0.0, 180.0, step_deg)
    return omega, projected_angle_deg(upper_normal, lower_normal, omega)


def plane_of_maximum_curvature(
    upper_normal: np.ndarray, lower_normal: np.ndarray
) -> tuple[float, float]:
    """``(angle_deg, omega_deg)`` at the plane of maximum curvature.

    A coarse sweep brackets the maximum and Brent's method refines it, which
    is both faster and more accurate than a fine grid and, unlike a bare
    optimiser, cannot settle on the wrong one of the two local maxima that a
    curve with combined coronal and sagittal deformity has.
    """
    omega, angle = pmc_profile(upper_normal, lower_normal, step_deg=0.5)
    k = int(np.argmax(angle))
    lo, hi = omega[k] - 0.5, omega[k] + 0.5
    fit = minimize_scalar(
        lambda w: -float(projected_angle_deg(upper_normal, lower_normal, w)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10},
    )
    best_omega = float(fit.x)
    best_angle = float(projected_angle_deg(upper_normal, lower_normal, best_omega))
    return best_angle, float(geo.wrap_to_signed_right_angle(best_omega))


# --------------------------------------------------------------------------
# measuring a model
# --------------------------------------------------------------------------


def measure_between_levels(
    model: SpineModel3D, upper_label: str, lower_label: str, *, name: str | None = None
) -> Cobb3D:
    """Full 3-D measurement between two named levels.

    Uses the superior endplate of ``upper_label`` and the inferior endplate of
    ``lower_label``, following the SRS definition. Both endplates of a
    vertebral body share its orientation in this model, so the two normals are
    the vertebrae's normals.
    """
    labels = list(model.labels)
    try:
        i, j = labels.index(upper_label), labels.index(lower_label)
    except ValueError as exc:
        raise ValueError(f"{exc.args[0]}; model spans {model.labels}") from None
    if i > j:
        i, j = j, i
        upper_label, lower_label = lower_label, upper_label

    normals = model.endplate_normals
    n_upper, n_lower = normals[i], normals[j]
    pmc_deg, omega = plane_of_maximum_curvature(n_upper, n_lower)
    return Cobb3D(
        upper_label=upper_label,
        lower_label=lower_label,
        coronal_deg=float(projected_angle_deg(n_upper, n_lower, 0.0)),
        sagittal_deg=float(projected_angle_deg(n_upper, n_lower, 90.0)),
        pmc_deg=pmc_deg,
        pmc_from_coronal_deg=omega,
        normal_angle_deg=geo.angle_between(n_upper, n_lower),
        name=name,
    )


def as_seen_on_film(
    model: SpineModel3D, upper_label: str, lower_label: str, projection
) -> float:
    """Angle a reader would measure between the same two endplates on a film.

    This is deliberately *not* the same quantity as the corresponding field of
    :func:`measure_between_levels`, and the gap between them is worth
    reporting.

    In a plane, the trace of an endplate is ``n x d`` for viewing direction
    ``d``, which is what :func:`projected_tilt_deg` uses. On a real
    radiograph, what a reader marks is the pair of visible body corners: the
    left and right cortical margins on a frontal view, the anterior and
    posterior margins on a lateral one. Those two lines coincide on a frontal
    view, because coronal tilt leaves the body's left-right axis in the
    coronal plane. They do **not** coincide on a lateral view of a coronally
    tilted vertebra, because coronal tilt swings the body's antero-posterior
    axis out of the sagittal plane.

    The practical consequence is that a lateral radiograph understates the
    sagittal endplate angle in proportion to the coronal deformity, so
    kyphosis is under-read in exactly the patients whose kyphosis matters.
    """
    film = model.project(projection)
    i, j = film.index_of_label(upper_label), film.index_of_label(lower_label)
    tilts = film.body_tilt()
    return float(abs(geo.wrap_to_signed_right_angle(tilts[i] - tilts[j])))


def cobb3d_for_curves(model: SpineModel3D, curves: tuple[Curve, ...]) -> tuple[Cobb3D, ...]:
    """Lift coronally detected curves into three dimensions.

    The curves come from :func:`vertebra_xray_core.cobb.cobb_angles` on the
    frontal view, so the end vertebrae are the ones a clinician would pick off
    that radiograph. Each is then re-measured in three dimensions, which is
    how the coronal and 3-D numbers stay comparable: same levels, different
    plane.
    """
    out = []
    for curve in curves:
        if curve.upper_label is None or curve.lower_label is None:
            raise ValueError("curves must be labelled before they can be lifted to 3-D")
        out.append(
            measure_between_levels(model, curve.upper_label, curve.lower_label, name=curve.name)
        )
    return tuple(out)
