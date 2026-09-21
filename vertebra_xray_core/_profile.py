"""Turning-point decomposition of a per-vertebra tilt profile.

Both the coronal curve detector and the sagittal labeller work by cutting the
tilt profile at its reversals, so the decomposition lives here rather than in
either of them.
"""

from __future__ import annotations

import numpy as np

__all__ = ["turning_points", "prune", "run_amplitudes", "FLAT_ATOL_DEG"]


#: Tilts differing by less than this many degrees count as equal. Float noise
#: from a projection and an ``arctan2`` is around 1e-13 deg; the smallest
#: difference any real measurement resolves is many orders of magnitude
#: larger, so this separates the two cleanly.
FLAT_ATOL_DEG = 1e-9


def turning_points(t: np.ndarray, atol: float = FLAT_ATOL_DEG) -> list[int]:
    """Indices bounding every maximal monotone run of ``t``.

    A plateau is collapsed to a single index rather than to its far edge: a
    leading run of equal tilts reports its *last* vertebra, a trailing run its
    *first*, and an interior plateau its midpoint. Reporting the far edge
    instead would name a vertebra several levels away from where the curve
    actually begins, even though the angle would be identical.

    Equality is tested to ``atol`` rather than exactly. Exact comparison looks
    safe -- a genuinely flat stretch of spine is a synthetic construct, and
    two measured vertebrae never tie to the last bit -- but a flat stretch
    that has been through a projection and an ``arctan2`` comes back differing
    in the last few bits, and every one of those differences then becomes its
    own turning point. The run through the flat region fragments into a dozen
    zero-amplitude runs, and which of them survives pruning is decided by
    float noise.
    """
    n = len(t)
    if n < 2:
        return list(range(n))
    changes = [k for k in range(1, n) if abs(t[k] - t[k - 1]) > atol]
    if not changes:
        return [0, n - 1]

    points = [changes[0] - 1]
    direction = 0
    previous = changes[0]
    for k in changes:
        d = np.sign(t[k] - t[k - 1])
        if direction != 0 and d != direction:
            points.append((previous + k - 1) // 2)
        direction = d
        previous = k
    points.append(changes[-1])
    return points


def prune(t: np.ndarray, points: list[int], min_amplitude: float) -> list[int]:
    """Drop turning points bounding runs shallower than ``min_amplitude``.

    Landmark noise puts a wobble of a degree or two on the tilt profile, which
    would otherwise be reported as a swarm of tiny curves. Runs are removed
    smallest-first so that pruning does not depend on the order they are
    visited in.

    Which point goes matters. Absorbing a shallow run at either *end* of the
    profile drops the outermost point and keeps the inner one, because the
    inner one is the extremum and so the real end vertebra: dropping it
    instead would start the surviving curve at a barely tilted vertebra and
    shrink its angle by the whole amplitude of the run that was supposed to
    be negligible. Absorbing an *interior* run drops both of its endpoints,
    which joins the runs on either side into one.
    """
    pts = list(points)
    while len(pts) > 2:
        amps = [abs(t[pts[k + 1]] - t[pts[k]]) for k in range(len(pts) - 1)]
        k = int(np.argmin(amps))
        if amps[k] >= min_amplitude:
            break
        if k == 0:
            pts.pop(0)
        elif k == len(amps) - 1:
            pts.pop(len(pts) - 1)
        else:
            pts.pop(k + 1)
            pts.pop(k)
    return pts


def run_amplitudes(t: np.ndarray, points: list[int]) -> list[float]:
    """Signed change in ``t`` across each run between consecutive ``points``."""
    return [float(t[points[k + 1]] - t[points[k]]) for k in range(len(points) - 1)]
