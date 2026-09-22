"""Coronal Cobb angles from vertebral corner landmarks.

Two families of algorithm live here and they answer different questions.

``detect_curves`` / ``cobb_angles`` -- *structural*
    Reproduces what a clinician does. The signed endplate-tilt profile of the
    spine is decomposed into monotone runs; each run is one structural curve,
    its two turning points are the end vertebrae, and the Cobb angle is the
    angle between the superior endplate of the upper end vertebra and the
    inferior endplate of the lower one (the SRS definition). Curves are then
    named PT / MT / TL / L from the anatomical level of their apex.

``all_pairs_max`` / ``aasce_triplet`` -- *challenge-compatible*
    The convention used by the AASCE 2019 benchmark: take the maximum angle
    over all pairs of vertebral body axes, then read a proximal and a distal
    angle off the same matrix. It ignores anatomy and cannot name a curve,
    but it is what the published SMAPE numbers are computed against, so it is
    needed to compare against the literature.

Reporting both, and the difference between them, is deliberate: a sizeable
part of the spread between published automatic Cobb systems is definitional
rather than algorithmic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import geometry as geo
from . import nomenclature as nom
from ._profile import prune as _prune
from ._profile import turning_points as _turning_points
from .landmarks import SpineLandmarks

__all__ = [
    "Curve",
    "CobbResult",
    "Definition",
    "detect_curves",
    "cobb_angles",
    "all_pairs_max",
    "aasce_triplet",
    "SCOLIOSIS_THRESHOLD_DEG",
]

#: A coronal curve is conventionally called scoliosis at or above this angle.
SCOLIOSIS_THRESHOLD_DEG = 10.0

Definition = Literal["srs", "body_axis"]


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Curve:
    """One structural coronal curve.

    ``angle_deg`` is always non-negative; the side the curve bulges towards is
    in :attr:`convexity`, not in the sign of the angle.
    """

    upper_index: int
    lower_index: int
    apex_index: int
    angle_deg: float
    upper_tilt_deg: float
    lower_tilt_deg: float
    apex_deviation_px: float
    convexity: Literal["left", "right"]
    name: str | None = None
    upper_label: str | None = None
    lower_label: str | None = None
    apex_label: str | None = None
    is_major: bool = False

    @property
    def is_scoliotic(self) -> bool:
        """Whether the curve reaches the conventional 10 deg threshold."""
        return self.angle_deg >= SCOLIOSIS_THRESHOLD_DEG

    def describe(self) -> str:
        """One-line clinical-style summary."""
        name = self.name or "curve"
        upper = self.upper_label or f"#{self.upper_index}"
        lower = self.lower_label or f"#{self.lower_index}"
        apex = self.apex_label or f"#{self.apex_index}"
        return (
            f"{name} {self.angle_deg:.1f} deg {upper}-{lower}, apex {apex}, convex {self.convexity}"
        )


@dataclass(frozen=True)
class CobbResult:
    """Every coronal curve found in one projection, plus how it was measured."""

    curves: tuple[Curve, ...]
    definition: Definition
    min_curve_deg: float
    tilt_profile_deg: tuple[float, ...]
    view: str
    labelled: bool

    @property
    def major(self) -> Curve | None:
        """The largest curve, or ``None`` when the spine is straight."""
        return max(self.curves, key=lambda c: c.angle_deg, default=None)

    @property
    def max_angle_deg(self) -> float:
        """Largest structural curve angle; ``0.0`` for a straight spine."""
        major = self.major
        return 0.0 if major is None else major.angle_deg

    def by_name(self, name: str) -> Curve | None:
        """First curve named ``name`` (``PT``, ``MT``, ``TL`` or ``L``)."""
        return next((c for c in self.curves if c.name == name), None)


# --------------------------------------------------------------------------
# turning-point decomposition of the tilt profile
# --------------------------------------------------------------------------


def _pair_angle(lm: SpineLandmarks, upper: int, lower: int, definition: Definition) -> float:
    """Angle in degrees between the two endplates delimiting a curve."""
    if definition == "srs":
        u = lm.superior_endplate[upper]
        v = lm.inferior_endplate[lower]
    elif definition == "body_axis":
        u = lm.body_axis[upper]
        v = lm.body_axis[lower]
    else:
        raise ValueError(f"unknown Cobb definition {definition!r}")
    return float(geo.wrap_to_signed_right_angle(geo.tilt_deg(u) - geo.tilt_deg(v)).__abs__())


def _apex(lm: SpineLandmarks, upper: int, lower: int) -> tuple[int, float]:
    """Most laterally deviated vertebra between the end vertebrae.

    Deviation is measured perpendicular to the chord joining the two end
    vertebra centroids, which is the clinical definition of the apex and is
    more stable than looking for the zero crossing of the tilt profile.
    Returns the index and its signed deviation (positive towards ``+x``).
    """
    c = lm.centroids
    a, b = c[upper], c[lower]
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length == 0.0:
        return upper, 0.0
    idx = np.arange(upper, lower + 1)
    rel = c[idx] - a
    dev = (axis[0] * rel[:, 1] - axis[1] * rel[:, 0]) / length
    k = int(np.argmax(np.abs(dev)))
    return int(idx[k]), float(dev[k])


def _patient_side(
    deviation: float, view: str, right_on_image_left: bool
) -> Literal["left", "right"]:
    """Translate a ``+x`` deviation in the ``math`` frame into a patient side.

    Frontal radiographs are conventionally displayed as though facing the
    patient, so increasing image ``x`` is the patient's *left*. Data that does
    not follow that convention sets ``right_on_image_left=False``.
    """
    if view == "lateral":
        raise ValueError("convexity is a coronal concept; a lateral view has none")
    towards_patient_left = deviation > 0 if right_on_image_left else deviation < 0
    return "left" if towards_patient_left else "right"


# --------------------------------------------------------------------------
# structural detection
# --------------------------------------------------------------------------


def detect_curves(
    lm: SpineLandmarks,
    *,
    definition: Definition = "srs",
    min_curve_deg: float = 5.0,
    right_on_image_left: bool = True,
) -> tuple[Curve, ...]:
    """Decompose the spine into structural coronal curves.

    Parameters
    ----------
    lm
        Frontal (``pa`` or ``ap``) landmarks. A lateral view is rejected:
        the same decomposition applied sagittally measures kyphosis and
        lordosis, which :mod:`vertebra_xray_core.sagittal` handles separately.
    definition
        Which vectors delimit a curve. ``srs`` uses the superior endplate of
        the upper end vertebra against the inferior endplate of the lower one;
        ``body_axis`` uses the mid-body axis at both ends, as AASCE does.
    min_curve_deg
        Runs of the tilt profile shallower than this are absorbed into their
        neighbours rather than reported as curves.
    right_on_image_left
        Whether the image follows the usual frontal display convention.

    Notes
    -----
    Segmentation always runs on the mid-body tilt profile, whichever
    ``definition`` is used to report the angle, because the mid-body tilt
    averages the two endplates and is the more stable of the two for deciding
    *where* a curve begins. The reported angle then uses the requested
    definition, so the choice affects the number but not the anatomy.
    """
    if lm.view == "lateral":
        raise ValueError("detect_curves expects a frontal view, got 'lateral'")

    tilt = lm.body_tilt()
    points = _prune(tilt, _turning_points(tilt), min_curve_deg)

    curves: list[Curve] = []
    for k in range(len(points) - 1):
        upper, lower = points[k], points[k + 1]
        angle = _pair_angle(lm, upper, lower, definition)
        if angle < min_curve_deg:
            continue
        apex_index, deviation = _apex(lm, upper, lower)
        curves.append(
            Curve(
                upper_index=upper,
                lower_index=lower,
                apex_index=apex_index,
                angle_deg=angle,
                upper_tilt_deg=float(tilt[upper]),
                lower_tilt_deg=float(tilt[lower]),
                apex_deviation_px=deviation,
                convexity=_patient_side(deviation, lm.view, right_on_image_left),
                upper_label=None if lm.labels is None else lm.labels[upper],
                lower_label=None if lm.labels is None else lm.labels[lower],
                apex_label=None if lm.labels is None else lm.labels[apex_index],
            )
        )
    return tuple(curves)


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------

#: Apex level ranges defining each curve name, following SRS terminology.
#: A thoracic apex sits between T2 and the T11/T12 disc, a thoracolumbar apex
#: at T12 or L1, and a lumbar apex between L1/L2 and L4.
_APEX_REGION = {
    **{level: "thoracic" for level in nom.span("T1", "T11")},
    "T12": "thoracolumbar",
    "L1": "thoracolumbar",
    **{level: "lumbar" for level in nom.span("L2", "L5")},
    **{level: "cervicothoracic" for level in nom.CERVICAL},
    "S1": "lumbar",
}


def _name_curves(curves: tuple[Curve, ...], labelled: bool) -> tuple[Curve, ...]:
    """Attach PT / MT / TL / L names and flag the major curve."""
    from dataclasses import replace as _replace

    if not curves:
        return curves

    major_at = max(range(len(curves)), key=lambda i: curves[i].angle_deg)
    names: list[str | None] = [None] * len(curves)

    if labelled and all(c.apex_label is not None for c in curves):
        apex_labels = [c.apex_label or "" for c in curves]
        regions = [_APEX_REGION.get(label, "thoracic") for label in apex_labels]
        thoracic = [i for i, r in enumerate(regions) if r in ("thoracic", "cervicothoracic")]
        # With two or more thoracic curves the caudal one is the main
        # thoracic curve and everything above it is proximal thoracic.
        for rank, i in enumerate(thoracic):
            names[i] = "MT" if rank == len(thoracic) - 1 else "PT"
        for i, r in enumerate(regions):
            if r == "thoracolumbar":
                names[i] = "TL"
            elif r == "lumbar":
                names[i] = "L"
    else:
        # Positional fallback, the AASCE convention: the largest curve is the
        # main thoracic one, its neighbours are proximal thoracic and
        # thoracolumbar/lumbar.
        names[major_at] = "MT"
        for i in range(len(curves)):
            if i < major_at:
                names[i] = "PT"
            elif i > major_at:
                names[i] = "TL/L"

    return tuple(_replace(c, name=names[i], is_major=(i == major_at)) for i, c in enumerate(curves))


def cobb_angles(
    lm: SpineLandmarks,
    *,
    definition: Definition = "srs",
    min_curve_deg: float = 5.0,
    right_on_image_left: bool = True,
) -> CobbResult:
    """Full structural coronal analysis of one frontal projection."""
    curves = detect_curves(
        lm,
        definition=definition,
        min_curve_deg=min_curve_deg,
        right_on_image_left=right_on_image_left,
    )
    return CobbResult(
        curves=_name_curves(curves, lm.labels is not None),
        definition=definition,
        min_curve_deg=min_curve_deg,
        tilt_profile_deg=tuple(float(v) for v in lm.body_tilt()),
        view=lm.view,
        labelled=lm.labels is not None,
    )


# --------------------------------------------------------------------------
# AASCE-compatible measurement
# --------------------------------------------------------------------------


def _pairwise_angle_matrix(lm: SpineLandmarks, definition: Definition) -> np.ndarray:
    """``(N, N)`` unsigned angles between every pair of vertebral vectors."""
    vectors = lm.body_axis if definition == "body_axis" else lm.superior_endplate
    tilts = np.asarray(geo.tilt_deg(vectors), dtype=float)
    diff = tilts[:, None] - tilts[None, :]
    return np.abs(np.asarray(geo.wrap_to_signed_right_angle(diff), dtype=float))


def all_pairs_max(
    lm: SpineLandmarks, *, definition: Definition = "body_axis"
) -> tuple[float, int, int]:
    """Largest angle between any two vertebral vectors, and the pair achieving it.

    This is the unambiguous half of the AASCE convention and the number most
    often quoted as "the Cobb angle" by automatic systems. It has no anatomy
    in it: on a spine with two similar curves the maximising pair can straddle
    the junction between them, which is why :func:`cobb_angles` exists.
    """
    matrix = _pairwise_angle_matrix(lm, definition)
    flat = int(np.argmax(np.triu(matrix, k=1)))
    upper, lower = divmod(flat, matrix.shape[1])
    return float(matrix[upper, lower]), int(upper), int(lower)


def aasce_triplet(
    lm: SpineLandmarks, *, definition: Definition = "body_axis"
) -> tuple[float, float, float]:
    """``(PT, MT, TL)`` in the AASCE 2019 benchmark convention.

    ``MT`` is the all-pairs maximum. ``PT`` is the largest angle entirely
    cranial to the maximising pair's upper vertebra and ``TL`` the largest
    entirely caudal to its lower vertebra; each is ``0.0`` when no vertebrae
    remain on that side.

    This is a reconstruction of the published convention rather than a port of
    the organisers' script, so treat small disagreements with challenge
    leaderboard numbers as definitional. :func:`cobb_angles` is the
    clinically faithful measurement and is what this package reports by
    default.
    """
    matrix = _pairwise_angle_matrix(lm, definition)
    mt, upper, lower = all_pairs_max(lm, definition=definition)
    pt = float(matrix[: upper + 1, : upper + 1].max()) if upper >= 1 else 0.0
    tl = float(matrix[lower:, lower:].max()) if lower <= len(lm) - 2 else 0.0
    return pt, mt, tl
