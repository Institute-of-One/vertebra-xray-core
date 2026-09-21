"""Training-free vertebral level assignment from sagittal curvature.

The problem
-----------
A detector hands back N vertebral bodies in cranio-caudal order. Measuring
them is easy; saying which one is T12 is not, and every number in a spine
report -- the curve names, the fusion levels, the kyphosis interval -- depends
on getting it right. Published solutions are almost all learned: a network is
trained to recognise each level's appearance. That needs annotated data, it
inherits that data's population, and when it is wrong it is wrong silently.

The anchor
----------
There is a geometric fact that needs no training. Walking down the spine, the
sagittal curvature reverses sign at the transition from thoracic kyphosis to
lumbar lordosis, and that reversal is at the T12/L1 disc in essentially every
human spine. Curvature is the derivative of the endplate tilt profile, so
"curvature reverses" is exactly "the sagittal tilt profile turns", and the
turning point can be read straight off the landmarks with no model at all.

The same argument gives a second anchor at C7/T1, where cervical lordosis
gives way to thoracic kyphosis. Two anchors are much better than one, because
the count between them is fixed: T1 through T12 is twelve bodies, always. So
the method checks itself. A disagreement is reported rather than absorbed.

Relation to the original implementation
---------------------------------------
This reworks an idea from a 2007 MATLAB prototype that fitted a fifth-order
polynomial to the *frontal* spinal curve and read vertebral levels off the
zero crossings of its second derivative. The mechanism was right and the plane
was wrong: coronal inflections sit wherever the scoliosis happens to put them,
so they identify nothing. Moving the same construction to the sagittal plane
is what turns it from a heuristic into an anatomical anchor.

Why a lateral film alone is not enough
--------------------------------------
A lateral radiograph does not show the sagittal tilt. It shows the tilt of the
chord joining a body's anterior and posterior margins, and that chord is not
the sagittal trace of the endplate once the spine deviates from the midline.
Two separate distortions follow.

Coronal tilt compresses the apparent sagittal tilt by
``tan(beta) = -cos(theta) tan(phi)``, and because ``theta`` varies down the
spine the compression is uneven, which drags the reversal off the true
junction. On phantoms the single-film anchor is right up to a main curve of
about 60 degrees and off by a level beyond that.

Axial rotation swings the chord out of the sagittal plane entirely, mixing the
*coronal* profile into the apparent sagittal one. The coronal profile has a
reversal of its own, so past about 20 degrees of rotation the anchor migrates
to the coronal curve's turning point and is wrong by several levels.

Both distortions live in the projection, not in the spine, so reconstructing
the spine removes them. :func:`label_biplanar` does exactly that: it labels
the lateral film, uses those possibly shifted labels only to pair the two
views, reconstructs, and labels again from the recovered sagittal tilt. On a
grid of main curves from 45 to 90 degrees crossed with axial rotations from 0
to 30 degrees, the single-film route is correct in 5 of 16 cases and the
two-pass route in all 16.

:func:`label_biplanar` is therefore the recommended entry point.
:func:`label_by_sagittal_inflection` remains available for a single film and
is honest about its limits: when it is wrong it is usually also either flagged
as a weak anchor or caught by the count not fitting between C1 and S1.

Other limitations, stated rather than hidden
--------------------------------------------
Transitional anatomy -- a sacralised L5, a lumbarised S1, eleven or thirteen
rib-bearing vertebrae -- shifts the count without shifting the curvature
reversal. No landmark-only method can resolve that, and neither can this one;
what it does instead is report the inter-anchor count, so a spine with the
wrong number of thoracic vertebrae raises a warning rather than a confident
wrong answer. A flat sagittal profile (severe hypokyphosis, or a film cropped
above the lumbar spine) leaves the anchor undetermined and is likewise
reported, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import nomenclature as nom
from ._profile import prune, run_amplitudes, turning_points
from .landmarks import SpineLandmarks

__all__ = [
    "Anchor",
    "LabelingResult",
    "label_biplanar",
    "label_by_sagittal_inflection",
    "label_from_model",
    "label_from_reference",
    "transfer_labels",
    "MIN_REVERSAL_DEG",
]

#: A curvature reversal shallower than this is treated as noise rather than a
#: junction. Physiological kyphosis and lordosis each span tens of degrees, so
#: this is far below any real transition and well above landmark jitter.
MIN_REVERSAL_DEG = 8.0

#: Bodies between the cervicothoracic and thoracolumbar anchors, i.e. between
#: C7 and T12. Fixed by anatomy, and the basis of the consistency check.
_ANCHOR_SEPARATION = nom.index_of(nom.THORACOLUMBAR_JUNCTION) - nom.index_of(
    nom.CERVICOTHORACIC_JUNCTION
)


@dataclass(frozen=True)
class Anchor:
    """One detected curvature reversal, and the level it pins down.

    ``position`` is where the reversal actually falls, in fractional rows; it
    is generally not an integer, because the curvature of a real spine
    reverses at a disc rather than inside a vertebral body. ``index`` is the
    vertebra cranial to that reversal, which is the one the junction is named
    after: the T12/L1 disc makes ``index`` point at T12.
    """

    kind: Literal["thoracolumbar", "cervicothoracic"]
    index: int
    level: str
    reversal_deg: float
    position: float = float("nan")

    #: A reversal weaker than this anchors the labelling poorly, because the
    #: two runs meet at a shallow angle and a degree of landmark noise moves
    #: their intersection by most of a level. Measured on phantoms, anchors
    #: above this hold their level in the high 80s of percent under 1 mm
    #: corner noise; an 11 deg anchor manages about 64%.
    WEAK_REVERSAL_DEG = 15.0

    @property
    def half_level_ambiguous(self) -> bool:
        """Whether the reversal sits about half way between two bodies.

        A reversal inside a vertebral body names that body. One landing on the
        disc between two bodies is a coin toss between them, a genuine
        plus-or-minus one level that no landmark-only method can settle.
        """
        if not np.isfinite(self.position):
            return False
        return abs(self.position - np.floor(self.position) - 0.5) < 0.15

    @property
    def is_weak(self) -> bool:
        """Whether the curvature reverses too gently to pin a level down."""
        return np.isfinite(self.reversal_deg) and self.reversal_deg < self.WEAK_REVERSAL_DEG

    def describe(self) -> str:
        where = "" if not np.isfinite(self.position) else f", reversal at row {self.position:.2f}"
        return (
            f"{self.kind} anchor at row {self.index} -> {self.level} "
            f"({self.reversal_deg:.1f} deg reversal{where})"
        )


@dataclass(frozen=True)
class LabelingResult:
    """Assigned levels plus everything needed to audit the assignment."""

    labels: tuple[str, ...]
    anchors: tuple[Anchor, ...]
    method: str
    anchor_separation: int | None
    expected_separation: int = _ANCHOR_SEPARATION
    warnings: tuple[str, ...] = ()
    count_overflow: bool = False
    """Whether counting outward from the anchor ran past C1 or S1.

    This is the sharpest check the method has. The spine has a fixed number
    of bodies, so an anchor that would put more vertebrae above C1 or below
    S1 than exist is not merely uncertain, it is impossible, and the levels
    are certainly wrong.
    """

    @property
    def is_self_consistent(self) -> bool:
        """Whether two anchors were found and they agree on the count."""
        return self.anchor_separation is not None and (
            self.anchor_separation == self.expected_separation
        )

    @property
    def confidence(self) -> Literal["high", "moderate", "low"]:
        """Coarse trust level, driven by how much evidence was available.

        ``high``
            Two anchors, agreeing on the number of thoracic bodies, neither of
            them weak or sitting on a disc.
        ``moderate``
            One sound anchor, or two that disagree by a single level.
        ``low``
            The anchor is shallow, falls between two bodies, the two anchors
            are irreconcilable, or the count does not fit inside C1 to S1.
            The levels are still the best available estimate; they should be
            checked.
        """
        if not self.anchors or self.count_overflow:
            return "low"
        if any(a.is_weak or a.half_level_ambiguous for a in self.anchors):
            return "low"
        if self.is_self_consistent:
            return "high"
        if self.anchor_separation is None:
            return "moderate"
        return "moderate" if abs(self.anchor_separation - self.expected_separation) <= 1 else "low"

    def apply(self, lm: SpineLandmarks) -> SpineLandmarks:
        """Attach these labels to landmarks with the same number of rows."""
        if len(lm) != len(self.labels):
            raise ValueError(f"{len(self.labels)} labels for {len(lm)} vertebrae")
        return lm.with_labels(self.labels)


# --------------------------------------------------------------------------
# the sagittal anchor
# --------------------------------------------------------------------------


def _sagittal_profile(lateral: SpineLandmarks, anterior_on_image_left: bool) -> np.ndarray:
    """Endplate tilt down the spine, oriented so kyphosis always falls.

    Flipping the sign for the opposite display convention here, once, means
    every downstream test is "is this a minimum" rather than "is this a
    minimum or a maximum depending on how the film was hung".
    """
    tilt = lateral.body_tilt()
    return tilt if anterior_on_image_left else -tilt


def _refine_reversal(profile: np.ndarray, lo: int, k: int, hi: int) -> float:
    """Sub-row position of the curvature reversal at row ``k``.

    Each region of the spine has roughly constant curvature -- thoracic
    kyphosis and lumbar lordosis are close to circular arcs -- so the tilt
    profile is close to piecewise linear, with a corner where the curvature
    reverses. This fits one line to the run from ``lo`` to ``k`` and another
    from ``k`` to ``hi`` and returns where they cross.

    Two alternatives were tried and are worse, both for the same reason: they
    are local, and the tilt profile carries almost no information *at* the
    reversal. A parabola through the three samples around ``k`` is mis-
    specified for a corner and is dragged towards whichever side is
    shallower -- on a normal spine, where lumbar lordosis is far sharper than
    thoracic kyphosis, that is a quarter of a level of pure bias. Fitting a
    line to the differenced profile and solving for its root is worse still,
    because the difference is a step rather than a ramp. The two-line
    intersection is the estimator the piecewise-constant-curvature model
    actually implies, and it uses every vertebra in both runs, which is what
    makes it hold up under landmark noise.
    """
    if k - lo < 2 or hi - k < 2:
        return float(k)
    upper = np.arange(lo, k + 1)
    lower = np.arange(k, hi + 1)
    s_up, b_up = np.polyfit(upper, profile[lo : k + 1], 1)
    s_dn, b_dn = np.polyfit(lower, profile[k : hi + 1], 1)
    if s_up == s_dn:
        return float(k)
    crossing = (b_dn - b_up) / (s_up - s_dn)
    # Beyond a level either side, the two runs are not meeting where the
    # discrete turn says they do and the fit is not describing this junction.
    return float(crossing) if abs(crossing - k) <= 1.0 else float(k)


def _find_anchors(profile: np.ndarray, min_reversal_deg: float) -> list[tuple[str, int, float]]:
    """Interior turning points of the profile, classified by which way they turn.

    With the profile oriented so that kyphosis falls caudally, the
    thoracolumbar junction -- kyphosis giving way to lordosis -- is a minimum,
    and the cervicothoracic junction -- lordosis giving way to kyphosis -- is a
    maximum.
    """
    points = prune(profile, turning_points(profile), min_reversal_deg)
    amplitudes = run_amplitudes(profile, points)
    found: list[tuple[str, int, float, float]] = []
    for k in range(1, len(points) - 1):
        before, after = amplitudes[k - 1], amplitudes[k]
        reversal = min(abs(before), abs(after))
        if reversal < min_reversal_deg:
            continue
        if before < 0 < after:
            kind = "thoracolumbar"
        elif before > 0 > after:
            kind = "cervicothoracic"
        else:
            continue
        position = _refine_reversal(profile, points[k - 1], points[k], points[k + 1])
        # The junction is named after the vertebra the reversal falls in, and
        # a reversal landing exactly on a disc is named after the cranial of
        # the two bodies it separates -- the T12/L1 disc makes T12 the anchor.
        # ``ceil(p - 1/2)`` is that rule: ordinary rounding, with halves going
        # cranial rather than to the nearer even row.
        index = int(np.clip(np.ceil(position - 0.5), 0, len(profile) - 1))
        found.append((kind, index, reversal, position))
    return found


def label_by_sagittal_inflection(
    lateral: SpineLandmarks,
    *,
    anterior_on_image_left: bool = True,
    min_reversal_deg: float = MIN_REVERSAL_DEG,
) -> LabelingResult:
    """Assign vertebral levels from the sagittal curvature reversals.

    Parameters
    ----------
    lateral
        Landmarks from the lateral view, cranial to caudal. The frontal view
        cannot be used: its inflections are wherever the scoliosis puts them.
    anterior_on_image_left
        Display convention of the lateral film.
    min_reversal_deg
        How sharply the curvature must reverse before the turn counts as a
        junction rather than noise.

    Notes
    -----
    When both anchors are found they are cross-checked against the fixed
    twelve-body separation between C7 and T12. The thoracolumbar anchor wins a
    disagreement: it is the sharper of the two reversals in almost every
    spine, and the cervicothoracic one sits at the top of the film where
    landmarks are least reliable.
    """
    if lateral.view != "lateral":
        raise ValueError(
            f"sagittal labelling needs the lateral view, got {lateral.view!r}; "
            "coronal inflections do not identify a level"
        )
    profile = _sagittal_profile(lateral, anterior_on_image_left)
    return _label_from_profile(profile, min_reversal_deg, "sagittal")
def _label_from_profile(
    profile: np.ndarray, min_reversal_deg: float, origin: str
) -> LabelingResult:
    """Anchor and count from an already-oriented sagittal tilt profile.

    ``profile`` must fall caudally through the thoracic spine, whatever it was
    derived from, so that the thoracolumbar junction is a minimum.
    """
    candidates = _find_anchors(profile, min_reversal_deg)
    warnings: list[str] = []

    def strongest(kind: str) -> tuple[str, int, float, float] | None:
        matches = [c for c in candidates if c[0] == kind]
        return max(matches, key=lambda c: c[2]) if matches else None

    tl = strongest("thoracolumbar")
    ct = strongest("cervicothoracic")

    separation = None if (tl is None or ct is None) else tl[1] - ct[1]
    if separation is not None and separation != _ANCHOR_SEPARATION:
        warnings.append(
            f"the two anchors are {separation} bodies apart but C7 to T12 is "
            f"{_ANCHOR_SEPARATION}; the thoracolumbar anchor was used and the "
            "level count may be off, for instance with transitional anatomy "
            "or a missed vertebra"
        )

    if tl is not None:
        origin_index, origin_level, method = tl[1], nom.THORACOLUMBAR_JUNCTION, f"{origin}-tl"
    elif ct is not None:
        origin_index, origin_level, method = ct[1], nom.CERVICOTHORACIC_JUNCTION, f"{origin}-ct"
        warnings.append(
            "no thoracolumbar reversal was found, so levels rest on the "
            "cervicothoracic anchor alone; check that the film reaches the "
            "lumbar spine"
        )
    else:
        raise ValueError(
            "no curvature reversal deep enough to anchor the labelling: the "
            "sagittal tilt profile spans "
            f"{float(profile.max() - profile.min()):.1f} deg but never turns "
            f"by {min_reversal_deg:.1f} deg on both sides of a single "
            "vertebra. A film cropped above the lumbar spine, or a spine with "
            "no lordosis in view, looks like this. Label from a known level "
            "with label_from_reference instead."
        )

    found = []
    if ct is not None:
        found.append(
            Anchor("cervicothoracic", ct[1], nom.CERVICOTHORACIC_JUNCTION, ct[2], ct[3])
        )
    if tl is not None:
        found.append(Anchor("thoracolumbar", tl[1], nom.THORACOLUMBAR_JUNCTION, tl[2], tl[3]))
    anchors = tuple(found)
    for anchor in anchors:
        if anchor.half_level_ambiguous:
            warnings.append(
                f"the {anchor.kind} reversal falls at row {anchor.position:.2f}, "
                "on a disc rather than inside a body, so which of the two "
                "adjacent vertebrae carries the junction is a coin toss and "
                "the whole labelling could be one level out"
            )
        if anchor.is_weak:
            warnings.append(
                f"the {anchor.kind} curvature reverses by only "
                f"{anchor.reversal_deg:.1f} deg, below the "
                f"{Anchor.WEAK_REVERSAL_DEG:.0f} deg this anchor needs to be "
                "reliable; a flat sagittal profile localises the junction poorly"
            )

    labels, overflow, spilled = _count_outward(len(profile), origin_index, origin_level)
    warnings.extend(overflow)
    return LabelingResult(
        labels=labels,
        anchors=anchors,
        method=method,
        anchor_separation=separation,
        warnings=tuple(warnings),
        count_overflow=spilled,
    )


def label_from_model(model, *, min_reversal_deg: float = MIN_REVERSAL_DEG) -> LabelingResult:
    """Assign levels from a reconstructed 3-D spine's true sagittal tilt.

    This is the recommended entry point whenever axial rotation is available,
    because the sagittal tilt of a
    :class:`~vertebra_xray_core.spine3d.SpineModel3D` built with a known
    rotation is the real thing rather than the rotation-contaminated chord a
    lateral film shows. On phantoms the anchor then lands on T12 with its full
    strength at every rotation tested, up to 40 degrees, where the single-film
    route has already degraded to an unusable 8 degree reversal.

    ``model.labels`` is ignored; this function is what produces labels. Pass a
    model built with placeholder levels and replace them with the result.
    """
    # Sagittal tilt is positive when the endplate normal tips posteriorly, so
    # kyphosis *rises* caudally in ``phi``. Negating it matches the convention
    # the anchor finder expects, where the thoracolumbar junction is a minimum.
    profile = -np.asarray(model.phi_deg, dtype=float)
    return _label_from_profile(profile, min_reversal_deg, "model")


def label_biplanar(
    frontal: SpineLandmarks,
    lateral: SpineLandmarks,
    *,
    axial_rotation_deg: np.ndarray | None = None,
    anterior_on_image_left: bool = True,
    min_reversal_deg: float = MIN_REVERSAL_DEG,
) -> LabelingResult:
    """Label a biplanar pair by reconstructing it first. The recommended route.

    There is an apparent circularity here -- reconstruction wants levels, and
    levels are what we are trying to find -- and it dissolves on inspection.
    Reconstruction only needs to know which row of one view is which row of
    the other. So:

    1. label the lateral film on its own, which may be a level or two out;
    2. put those same, possibly shifted, labels on both views, where only
       their agreement matters;
    3. reconstruct, recovering the true sagittal tilt;
    4. label again, from a profile that is no longer distorted.

    The second pass is better than the first for two reasons. The lateral
    film's apparent sagittal tilt is compressed by coronal tilt through
    ``tan(beta) = -cos(theta) tan(phi)``, which shifts the reversal in a
    severe curve; and if ``axial_rotation_deg`` is supplied, the recovered
    tilt is free of rotation contamination as well.

    Both views must have the same number of rows, since step 2 pairs them by
    position. Use :func:`transfer_labels` first if one view is cropped.
    """
    from .spine3d import reconstruct_from_biplanar

    if len(frontal) != len(lateral):
        raise ValueError(
            f"the two views have {len(frontal)} and {len(lateral)} vertebrae; "
            "crop them to the same levels before labelling as a pair"
        )
    first = label_by_sagittal_inflection(
        lateral,
        anterior_on_image_left=anterior_on_image_left,
        min_reversal_deg=min_reversal_deg,
    )
    model = reconstruct_from_biplanar(
        frontal,
        lateral,
        axial_rotation_deg=axial_rotation_deg,
        anterior_on_image_left=anterior_on_image_left,
        match="row",
    )
    second = _label_from_profile(
        -np.asarray(model.phi_deg, dtype=float), min_reversal_deg, "biplanar"
    )
    shifted = second.labels != first.labels
    warnings = tuple(second.warnings)
    if shifted:
        warnings += (
            f"the single-film pass put the first vertebra at {first.labels[0]} and "
            f"the reconstructed pass at {second.labels[0]}; the reconstructed "
            "answer is the one reported, because the lateral film's apparent "
            "sagittal tilt is distorted by coronal tilt",
        )
    from dataclasses import replace as _replace

    return _replace(second, warnings=warnings)


def _count_outward(
    n: int, origin_index: int, origin_level: str
) -> tuple[tuple[str, ...], list[str], bool]:
    """Name ``n`` consecutive bodies given that row ``origin_index`` is ``origin_level``."""
    base = nom.index_of(origin_level)
    first, last = base - origin_index, base - origin_index + n - 1
    warnings: list[str] = []
    if first < 0:
        warnings.append(
            f"counting up from {origin_level} runs {-first} bodies above C1; "
            "the landmark set is longer than the spine, so some rows are "
            "probably duplicates or false detections"
        )
    if last >= len(nom.ALL_LEVELS):
        warnings.append(
            f"counting down from {origin_level} runs "
            f"{last - len(nom.ALL_LEVELS) + 1} bodies below S1; the caudal end "
            "of the landmark set is probably sacral"
        )
    # Out-of-range rows are clamped to the ends of the vocabulary, which
    # repeats a level. That is deliberate: the alternative is inventing names
    # for bodies that do not exist. Callers must not treat a clamped labelling
    # as a level map -- ``count_overflow`` says when that has happened, and
    # everything downstream that needs unique levels refuses them.
    labels = tuple(
        nom.level_at(min(max(base - origin_index + k, 0), len(nom.ALL_LEVELS) - 1))
        for k in range(n)
    )
    return labels, warnings, bool(warnings)


# --------------------------------------------------------------------------
# fallbacks and transfer
# --------------------------------------------------------------------------


def label_from_reference(
    lm: SpineLandmarks, reference_index: int, reference_level: str
) -> LabelingResult:
    """Count outward from one level the operator has identified by hand.

    The usual reference is the caudal-most lumbar body on a film that includes
    the iliac crests. This is the escape hatch for spines the sagittal anchor
    cannot resolve, and it is also how a ground-truth labelling is injected
    when benchmarking.
    """
    if not 0 <= reference_index < len(lm):
        raise ValueError(f"reference_index {reference_index} is outside 0..{len(lm) - 1}")
    labels, warnings, spilled = _count_outward(len(lm), reference_index, reference_level)
    return LabelingResult(
        labels=labels,
        count_overflow=spilled,
        anchors=(
            Anchor(
                kind="thoracolumbar",
                index=reference_index,
                level=reference_level,
                reversal_deg=float("nan"),
            ),
        ),
        method="reference",
        anchor_separation=None,
        warnings=tuple(warnings),
    )


def transfer_labels(
    source: SpineLandmarks, target: SpineLandmarks, *, offset: int = 0
) -> SpineLandmarks:
    """Copy levels from one view to the other, row by row.

    ``offset`` is how many bodies further cranial the *target* view starts, so
    a lateral film cropped one body lower than the frontal one is handled by
    ``offset=-1``. Rows that fall outside the source are an error rather than
    a silent truncation: a biplanar study whose two views cannot be aligned by
    a constant offset needs a human, not a guess.
    """
    if source.labels is None:
        raise ValueError("the source view has no labels to transfer")
    start = -offset
    stop = start + len(target)
    if start < 0 or stop > len(source):
        raise ValueError(
            f"transferring {len(target)} rows at offset {offset} needs source rows "
            f"{start}..{stop - 1}, but the source has {len(source)}"
        )
    return target.with_labels(source.labels[start:stop])
