"""Propagating landmark localisation error into the reported angles.

Automatic Cobb systems report a number. Clinicians know that number moves by
several degrees between readers, and the literature attributes most of that
movement not to drawing the lines badly but to *choosing different end
vertebrae*. Neither part is usually reported per case.

This module resamples the landmarks under a stated localisation error and
reports what happens to the measurement. It separates the two mechanisms,
which is the point:

``fixed``
    The end vertebrae stay where the unperturbed analysis put them, and only
    the endplate lines move. This is the pure geometric sensitivity.
``redetected``
    The whole analysis is rerun on each resample, so the end vertebrae are
    free to move. The extra spread over ``fixed`` is the cost of the
    discrete choice.

The gap between the two is the computational analogue of the clinical
observation, and it is available per patient rather than as a population
average from a reader study.

On phantoms the gap has a threshold in it. With independent corner noise on a
50 degree main thoracic curve, the discrete choice contributes none of the
variance up to about 1.5 mm, 6% at 2 mm, a third at 3 mm and half at 4 mm.
Treat that as a lower bound: independent corner jitter is a poor model of how
two readers come to disagree about which vertebra ends a curve, and it will
understate a mechanism that in a reader study is driven by perception rather
than by noise.

The noise model is deliberately simple and explicit: independent isotropic
Gaussian jitter on each annotated corner, with a standard deviation the caller
states. It is not claimed to be the true distribution of any particular
detector's errors. It is a stated, reproducible perturbation, and a reported
interval derived from it means exactly "this is how much the answer moves if
corners are this uncertain".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cobb as _cobb
from . import labeling as _labeling
from .landmarks import SpineLandmarks

__all__ = [
    "jitter",
    "Interval",
    "CurveUncertainty",
    "bootstrap_cobb",
    "labelling_stability",
    "DEFAULT_RESAMPLES",
]

#: Enough resamples for a stable 95% interval to about a tenth of a degree,
#: while still running in well under a second for a 17-vertebra spine.
DEFAULT_RESAMPLES = 1000


def jitter(lm: SpineLandmarks, sigma: float, rng: np.random.Generator) -> SpineLandmarks:
    """One resample of ``lm`` with independent Gaussian corner noise.

    ``sigma`` is in the units the corners are stored in -- pixels for
    landmarks read off an image, millimetres once they have been scaled.

    The result goes back through the normal constructor, so corners are
    re-canonicalised: if the noise is large enough to reorder a vertebra's
    corners, the resample reflects that rather than hiding it.
    """
    noisy = lm.corners + rng.normal(0.0, sigma, lm.corners.shape)
    return SpineLandmarks(
        corners=_recanonicalise(noisy),
        view=lm.view,
        labels=lm.labels,
        spacing_mm=lm.spacing_mm,
        source=f"{lm.source} + jitter(sigma={sigma})",
    )


def _recanonicalise(corners: np.ndarray) -> np.ndarray:
    from .landmarks import canonicalise_corners

    return canonicalise_corners(corners)


@dataclass(frozen=True)
class Interval:
    """A resampled distribution summarised the way a report needs it."""

    point: float
    mean: float
    sd: float
    low: float
    high: float
    level: float = 0.95

    def describe(self, unit: str = "deg") -> str:
        return (
            f"{self.point:.1f} {unit} "
            f"({self.level:.0%} CI {self.low:.1f} to {self.high:.1f}, SD {self.sd:.1f})"
        )


@dataclass(frozen=True)
class CurveUncertainty:
    """What landmark noise does to one curve."""

    name: str | None
    upper_label: str | None
    lower_label: str | None
    fixed: Interval
    redetected: Interval | None
    end_vertebra_stability: float | None
    sigma: float
    resamples: int

    @property
    def end_vertebra_penalty_deg(self) -> float | None:
        """Extra spread caused by the end vertebrae being free to move.

        Standard deviations are combined in quadrature, so this is the part of
        the total not explained by the endplate lines alone. A negative value
        would mean the redetected spread was the smaller of the two, which
        resampling noise can produce when the penalty is near zero; it is
        clamped to zero.
        """
        if self.redetected is None:
            return None
        extra = self.redetected.sd**2 - self.fixed.sd**2
        return float(np.sqrt(max(extra, 0.0)))

    def describe(self) -> str:
        name = self.name or "curve"
        levels = (
            f" {self.upper_label}-{self.lower_label}"
            if self.upper_label and self.lower_label
            else ""
        )
        line = f"{name}{levels}: {self.fixed.describe()}"
        if self.redetected is not None:
            line += (
                f"; with end vertebrae re-chosen {self.redetected.describe()}"
                f", stable in {self.end_vertebra_stability:.0%} of resamples"
            )
        return line


def _summarise(point: float, samples: np.ndarray, level: float) -> Interval:
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return Interval(
        point=float(point),
        mean=float(np.mean(samples)),
        sd=float(np.std(samples, ddof=1)),
        low=float(low),
        high=float(high),
        level=level,
    )


def bootstrap_cobb(
    lm: SpineLandmarks,
    sigma: float,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int | None = 0,
    definition: _cobb.Definition = "srs",
    min_curve_deg: float = 5.0,
) -> tuple[CurveUncertainty, ...]:
    """Interval for every curve in a frontal view under corner noise ``sigma``.

    ``seed`` defaults to a fixed value so that a report is reproducible; pass
    ``None`` for a fresh draw.

    Curves are matched between resamples by their apex level rather than by
    position in the list, because a resample can gain or lose a marginal
    curve. A resample in which a curve has no counterpart contributes to that
    curve's stability figure but not to its angle distribution.
    """
    rng = np.random.default_rng(seed)
    baseline = _cobb.cobb_angles(lm, definition=definition, min_curve_deg=min_curve_deg)
    if not baseline.curves:
        return ()

    fixed = np.empty((resamples, len(baseline.curves)))
    redetected: list[list[float]] = [[] for _ in baseline.curves]
    matched = np.zeros(len(baseline.curves), dtype=int)

    for r in range(resamples):
        sample = jitter(lm, sigma, rng)
        for c, curve in enumerate(baseline.curves):
            fixed[r, c] = _cobb._pair_angle(sample, curve.upper_index, curve.lower_index, definition)
        found = _cobb.cobb_angles(sample, definition=definition, min_curve_deg=min_curve_deg)
        for c, curve in enumerate(baseline.curves):
            partner = _nearest_curve(found.curves, curve)
            if partner is not None:
                redetected[c].append(partner.angle_deg)
                if (partner.upper_index, partner.lower_index) == (
                    curve.upper_index,
                    curve.lower_index,
                ):
                    matched[c] += 1

    out = []
    for c, curve in enumerate(baseline.curves):
        samples = np.asarray(redetected[c])
        out.append(
            CurveUncertainty(
                name=curve.name,
                upper_label=curve.upper_label,
                lower_label=curve.lower_label,
                fixed=_summarise(curve.angle_deg, fixed[:, c], level),
                redetected=(
                    _summarise(curve.angle_deg, samples, level) if len(samples) > 1 else None
                ),
                end_vertebra_stability=float(matched[c]) / resamples,
                sigma=sigma,
                resamples=resamples,
            )
        )
    return tuple(out)


def _nearest_curve(candidates: tuple[_cobb.Curve, ...], target: _cobb.Curve) -> _cobb.Curve | None:
    """Curve in ``candidates`` whose apex is closest to ``target``'s, if close enough.

    Two levels of slack: a resampled curve whose apex has drifted further than
    that is a different curve, not a noisy version of this one.
    """
    if not candidates:
        return None
    best = min(candidates, key=lambda c: abs(c.apex_index - target.apex_index))
    return best if abs(best.apex_index - target.apex_index) <= 2 else None


def labelling_stability(
    lateral: SpineLandmarks,
    sigma: float,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int | None = 0,
    **labelling_kwargs,
) -> dict[str, float]:
    """How often the sagittal anchor survives corner noise of ``sigma``.

    Returns the fraction of resamples that reproduce the unperturbed
    labelling exactly, the fraction that land within one level of it, and the
    fraction in which no anchor could be found at all. A labelling method that
    is right on clean data and unstable under realistic noise is not usable,
    and this is the measurement that says which it is.
    """
    rng = np.random.default_rng(seed)
    baseline = _labeling.label_by_sagittal_inflection(lateral, **labelling_kwargs)
    from . import nomenclature as nom

    reference = nom.index_of(baseline.labels[0])

    exact = within_one = failed = 0
    for _ in range(resamples):
        sample = jitter(lateral, sigma, rng)
        try:
            result = _labeling.label_by_sagittal_inflection(sample, **labelling_kwargs)
        except ValueError:
            failed += 1
            continue
        offset = nom.index_of(result.labels[0]) - reference
        if offset == 0:
            exact += 1
        if abs(offset) <= 1:
            within_one += 1

    return {
        "exact": exact / resamples,
        "within_one_level": within_one / resamples,
        "no_anchor": failed / resamples,
        "sigma": sigma,
        "resamples": float(resamples),
    }
