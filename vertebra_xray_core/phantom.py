"""Analytic spine phantoms with exactly known angles.

Every published automatic Cobb system is validated against human annotation,
which carries several degrees of its own inter-observer spread. That makes it
impossible to tell an algorithm's error from the reference's. A phantom whose
curvature is prescribed in closed form has no such floor: the measured value
can be compared with the number that was put in, so an implementation bug of a
tenth of a degree is visible.

The phantoms here are also the reason this package can be published without
the retrospective clinical DICOM series it was originally developed on. They
are synthetic, contain no patient data, and are reproducible from the seed and
the specification alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import nomenclature as nom
from .spine3d import SpineModel3D, endplate_normal

__all__ = ["CurveSpec", "synthetic_spine", "normal_adult_spine", "adolescent_idiopathic_scoliosis"]


@dataclass(frozen=True)
class CurveSpec:
    """One prescribed curve between two levels.

    ``angle_deg`` is the Cobb angle the curve is built to have: the tilt runs
    from ``+angle/2`` at ``upper`` to ``-angle/2`` at ``lower``, so the angle
    between the two end vertebrae's endplates is exactly ``angle_deg``.
    """

    upper: str
    lower: str
    angle_deg: float
    direction: Literal["left", "right", "kyphosis", "lordosis"]

    @property
    def plane(self) -> Literal["coronal", "sagittal"]:
        return "coronal" if self.direction in ("left", "right") else "sagittal"

    @property
    def sign(self) -> float:
        """Sign of the tilt at the upper end vertebra.

        Walking caudally, each vertebra is stacked along its own endplate
        normal, so a positive tilt at the top drives the spine towards ``+X``
        (the patient's left) before the tilt reverses. A left-convex curve is
        therefore ``+``. In the sagittal plane the same reasoning with ``+Y``
        anterior makes kyphosis, a posterior convexity, the ``-`` case.
        """
        return {"left": +1.0, "right": -1.0, "kyphosis": -1.0, "lordosis": +1.0}[self.direction]


# Body dimensions in millimetres at C1 and at S1, linearly interpolated in
# between. The trend, not the exact figures, is what matters: bodies widen and
# deepen steadily from the upper thoracic spine to the lumbosacral junction,
# and a phantom that ignores that produces unrealistically uniform
# foreshortening under a divergent beam.
_DIMENSIONS = {
    "width": (18.0, 52.0),
    "depth": (16.0, 36.0),
    "height": (14.0, 30.0),
}
_DISC_FRACTION = 0.30  # disc height as a fraction of the adjacent body height


def _dimension(name: str, levels: tuple[str, ...]) -> np.ndarray:
    lo, hi = _DIMENSIONS[name]
    idx = np.array([nom.index_of(lab) for lab in levels], dtype=float)
    span = len(nom.ALL_LEVELS) - 1
    return lo + (hi - lo) * idx / span


def _tilt_profile(levels: tuple[str, ...], specs: tuple[CurveSpec, ...], plane: str) -> np.ndarray:
    """Chain the prescribed curves into one tilt profile over ``levels``.

    The end vertebrae are *knots* carrying a fixed tilt, and each curve fixes
    the difference between its two knots to be its Cobb angle. Adjacent curves
    share a knot, so a double major pattern comes out with both angles exactly
    as prescribed.

    Superposing the curves instead -- the obvious implementation -- adds the
    two tilts at the shared end vertebra and silently inflates both angles; a
    45 deg thoracic curve abutting a 30 deg lumbar one measures 60 deg.

    Outside the outermost knot the tilt is held constant rather than relaxed
    to zero. Relaxing would introduce an extra monotone run at each end, which
    is anatomically reasonable but would mean the phantom contains curves
    nobody asked for, and a phantom is only useful if it contains exactly what
    was specified.
    """
    n = len(levels)
    position = {lab: i for i, lab in enumerate(levels)}
    chosen = [s for s in specs if s.plane == plane]
    if not chosen:
        return np.zeros(n)

    for spec in chosen:
        for end in (spec.upper, spec.lower):
            if end not in position:
                raise ValueError(f"{end!r} is not among the phantom's levels")
        if position[spec.upper] >= position[spec.lower]:
            raise ValueError(f"{spec.upper!r} must be cranial to {spec.lower!r}")
    chosen.sort(key=lambda s: position[s.upper])

    knots: dict[int, float] = {}
    for spec in chosen:
        i, j = position[spec.upper], position[spec.lower]
        if i not in knots:
            # A curve starting immediately below where the previous one ended
            # inherits its tilt, because the tilt profile of a spine is
            # continuous: thoracic kyphosis does not stop and lumbar lordosis
            # restart at some other angle. The inherited value makes the two
            # adjacent knots equal, which puts the curvature reversal on the
            # intervening disc -- at T12/L1 for the thoracolumbar junction --
            # exactly where anatomy puts it.
            knots[i] = knots.get(i - 1, spec.sign * 0.5 * spec.angle_deg)
        knots[j] = knots[i] - spec.sign * spec.angle_deg

    ordered = sorted(knots)
    profile = np.empty(n)
    profile[: ordered[0]] = knots[ordered[0]]
    profile[ordered[-1] :] = knots[ordered[-1]]
    for a, b in zip(ordered, ordered[1:], strict=False):
        # Linear in tilt between knots, so curvature -- the derivative of the
        # tilt -- is constant within each region. That makes every region a
        # circular arc, which is the standard idealisation of the sagittal
        # profile, and it puts a genuine corner in the tilt profile at each
        # junction.
        #
        # A smooth half-cosine ramp looks more realistic and is worse: its
        # derivative vanishes at both knots, so curvature dies away to zero on
        # *both* sides of a junction and the reversal becomes a flat region
        # instead of a point. Any estimator of where the curvature reverses is
        # then unidentifiable, which is an artefact of the phantom rather than
        # a property of spines.
        profile[a : b + 1] = np.linspace(knots[a], knots[b], b - a + 1)

    if plane == "coronal":
        # Offset the whole profile so the top and bottom vertebrae are tilted
        # equally and oppositely. Cobb angles are differences, so this changes
        # none of them; what it fixes is realism. Chaining knots outward from
        # a symmetric first curve lets the absolute tilt drift, and a phantom
        # with a 90 deg curve ends up with an end vertebra tilted 80 deg from
        # horizontal, which no spine does. A balanced coronal profile is also
        # what compensated scoliosis actually looks like.
        profile -= 0.5 * (profile[0] + profile[-1])
    return profile


def synthetic_spine(
    *,
    levels: tuple[str, ...] = nom.span("T1", "L5"),
    curves: tuple[CurveSpec, ...] = (),
    axial_rotation_deg: np.ndarray | float = 0.0,
    landmark_noise_mm: float = 0.0,
    seed: int | None = None,
    source: str = "synthetic",
) -> SpineModel3D:
    """Build a spine whose curve angles are exactly the ones prescribed.

    Parameters
    ----------
    levels
        Which vertebrae to include, cranial to caudal.
    curves
        Coronal and sagittal curves to superpose. Overlapping curves add,
        which is how a genuine double major pattern arises: the shared end
        vertebra carries the sum of both curves' tilts.
    axial_rotation_deg
        A scalar applied to every level, or a per-level array. Axial rotation
        does not change any endplate normal, so it leaves every 3-D angle in
        the phantom untouched while visibly changing both projections -- which
        is exactly what makes it the right probe for the bias in
        :func:`~vertebra_xray_core.spine3d.reconstruct_from_biplanar`.
    landmark_noise_mm
        Standard deviation of isotropic Gaussian jitter added to the vertebral
        centroids. Orientation is left clean so that the effect of positional
        and angular error can be separated.
    seed
        Seeds the jitter. With ``landmark_noise_mm=0`` the phantom is fully
        deterministic and ``seed`` is irrelevant.
    """
    levels = tuple(levels)
    if len(levels) < 2:
        raise ValueError("a phantom needs at least two levels")

    theta = _tilt_profile(levels, curves, "coronal")
    phi = _tilt_profile(levels, curves, "sagittal")
    psi = np.broadcast_to(np.asarray(axial_rotation_deg, dtype=float), (len(levels),)).astype(float)

    width = _dimension("width", levels)
    depth = _dimension("depth", levels)
    height = _dimension("height", levels)

    # Stack the bodies along their own endplate normals, caudally from the top.
    normals = np.stack([endplate_normal(t, p) for t, p in zip(theta, phi, strict=True)])
    centroids = np.zeros((len(levels), 3))
    for k in range(len(levels) - 1):
        gap = _DISC_FRACTION * 0.5 * (height[k] + height[k + 1])
        step = 0.5 * height[k] + gap + 0.5 * height[k + 1]
        direction = normals[k] + normals[k + 1]
        direction /= np.linalg.norm(direction)
        centroids[k + 1] = centroids[k] - step * direction

    if landmark_noise_mm > 0:
        rng = np.random.default_rng(seed)
        centroids = centroids + rng.normal(0.0, landmark_noise_mm, centroids.shape)

    return SpineModel3D(
        labels=levels,
        centroids=centroids - centroids.mean(axis=0),
        theta_deg=theta,
        phi_deg=phi,
        psi_deg=psi,
        width_mm=width,
        depth_mm=depth,
        height_mm=height,
        source=source,
        meta={
            "curves": curves,
            "landmark_noise_mm": landmark_noise_mm,
            "seed": seed,
        },
    )


def normal_adult_spine(**kwargs) -> SpineModel3D:
    """T1-L5 with physiological kyphosis and lordosis and no coronal curve.

    Thoracic kyphosis is prescribed over T4-T12 and lumbar lordosis over
    L1-L5, at the middle of the usual adult ranges.
    """
    curves = (
        CurveSpec("T1", "T12", 35.0, "kyphosis"),
        CurveSpec("T12", "L5", 45.0, "lordosis"),
    )
    kwargs.setdefault("curves", curves)
    kwargs.setdefault("source", "normal_adult_spine")
    return synthetic_spine(**kwargs)


def adolescent_idiopathic_scoliosis(
    *,
    main_thoracic_deg: float = 45.0,
    lumbar_deg: float = 30.0,
    proximal_thoracic_deg: float = 20.0,
    axial_rotation_deg: float = 0.0,
    **kwargs,
) -> SpineModel3D:
    """A right-convex main thoracic curve with counter-curves above and below.

    This is the commonest adolescent idiopathic pattern (Lenke 1 / King III)
    and exercises everything at once: three coronal curves sharing end
    vertebrae, physiological sagittal curves underneath them, and optional
    axial rotation.

    The mild proximal thoracic curve is not decoration. Without it the tilt
    profile is flat from T1 to T5, and in a flat region the end vertebra of
    the main curve is genuinely undefined -- any of those five bodies gives
    the same angle. A phantom with that degeneracy reports an arbitrary end
    vertebra and an artificially poor end-vertebra stability, neither of which
    says anything about the algorithm.
    """
    curves = (
        CurveSpec("T1", "T5", proximal_thoracic_deg, "left"),
        CurveSpec("T5", "T12", main_thoracic_deg, "right"),
        CurveSpec("T12", "L5", lumbar_deg, "left"),
        CurveSpec("T1", "T12", 25.0, "kyphosis"),
        CurveSpec("T12", "L5", 45.0, "lordosis"),
    )
    kwargs.setdefault("curves", curves)
    kwargs.setdefault("source", "adolescent_idiopathic_scoliosis")
    return synthetic_spine(axial_rotation_deg=axial_rotation_deg, **kwargs)
