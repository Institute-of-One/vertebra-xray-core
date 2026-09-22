"""Geometric primitives shared by the 2-D and 3-D measurement paths.

Coordinate conventions
----------------------
Two frames are used and they are never mixed implicitly.

``image``
    Raw pixel/landmark frame as it comes out of a DICOM or an annotation
    file: ``x`` increases to the right, ``y`` increases **downwards**, origin
    at the top-left pixel centre. Landmarks enter the library in this frame.

``math``
    Right-handed frame used by every computation in this package: ``x``
    increases to the right, ``y`` increases **upwards**. Angles measured here
    behave the way the formulae in the literature assume, which removes the
    single most common source of sign errors in Cobb-angle code.

:func:`to_math_frame` performs the one-way conversion at the library
boundary. Nothing downstream flips ``y`` again.

The 3-D patient frame used by :mod:`vertebra_xray_core.spine3d` is documented
there; it is built out of two ``math``-frame projections.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_math_frame",
    "to_image_frame",
    "unit",
    "angle_between",
    "signed_angle",
    "tilt_deg",
    "wrap_to_signed_right_angle",
    "project_onto_plane",
    "rotation_about_z",
    "arc_length",
]


# --------------------------------------------------------------------------
# frame conversion
# --------------------------------------------------------------------------


def to_math_frame(points: np.ndarray, height: float | None = None) -> np.ndarray:
    """Convert ``image``-frame points to the ``math`` frame.

    Parameters
    ----------
    points
        Array whose last axis is of length 2 and holds ``(x, y)`` pairs. Any
        leading shape is preserved, so ``(N, 4, 2)`` corner arrays pass
        through unchanged in structure.
    height
        Image height in pixels. When given, ``y`` becomes ``height - 1 - y``
        so the result stays inside the original pixel extent, which keeps
        overlays on the displayed image correct. When ``None``, ``y`` is
        simply negated; that is enough for every angle in this package
        because angles are invariant to translation.

    Notes
    -----
    The transform is an involution up to the choice of ``height``: applying
    :func:`to_image_frame` with the same ``height`` recovers the input.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[-1] != 2:
        raise ValueError(f"expected trailing axis of length 2, got shape {pts.shape}")
    out = pts.copy()
    if height is None:
        out[..., 1] = -out[..., 1]
    else:
        out[..., 1] = (height - 1.0) - out[..., 1]
    return out


def to_image_frame(points: np.ndarray, height: float | None = None) -> np.ndarray:
    """Inverse of :func:`to_math_frame` (the transform is its own inverse)."""
    return to_math_frame(points, height)


# --------------------------------------------------------------------------
# vectors and angles
# --------------------------------------------------------------------------


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalise ``v`` along ``axis``; zero-length vectors are returned as-is."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Unsigned angle between two vectors, in degrees on ``[0, 180]``.

    Uses the ``atan2`` form rather than ``arccos`` of a dot product: near 0 deg
    and 180 deg the ``arccos`` form loses most of its significant digits, and
    endplate pairs in a mild curve sit exactly in that regime.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape[-1] == 2:
        cross = float(u[0] * v[1] - u[1] * v[0])
    else:
        cross = float(np.linalg.norm(np.cross(u, v)))
    dot = float(np.dot(u, v))
    return float(np.degrees(np.arctan2(abs(cross), dot)))


def signed_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Signed angle from ``u`` to ``v`` in the plane, degrees on ``(-180, 180]``.

    Only defined for 2-D vectors. Counter-clockwise is positive, which in the
    ``math`` frame means "``v`` is rotated towards the patient's left".
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape[-1] != 2 or v.shape[-1] != 2:
        raise ValueError("signed_angle is only defined for 2-D vectors")
    cross = float(u[0] * v[1] - u[1] * v[0])
    dot = float(u[0] * v[0] + u[1] * v[1])
    return float(np.degrees(np.arctan2(cross, dot)))


def wrap_to_signed_right_angle(deg: float | np.ndarray) -> np.ndarray | float:
    """Fold an angle into ``(-90, 90]``.

    An endplate is an undirected line: ``e`` and ``-e`` describe the same
    endplate, so its tilt is only defined modulo 180 deg. Every tilt in this
    package is folded through here so that the two corner orderings of one
    endplate can never produce two different answers.
    """
    x = np.asarray(deg, dtype=float)
    folded = (x + 90.0) % 180.0 - 90.0
    # ``(x + 90) % 180 - 90`` maps exactly +90 to -90; keep the closed end at +90.
    folded = np.where(np.isclose(folded, -90.0), 90.0, folded)
    return float(folded) if np.isscalar(deg) or folded.ndim == 0 else folded


def tilt_deg(vectors: np.ndarray) -> np.ndarray | float:
    """Tilt of one or more endplate vectors against the horizontal, in degrees.

    Returns values on ``(-90, 90]`` in the ``math`` frame, where a positive
    tilt means the vector's right-hand end is the higher one. The result is
    independent of which way round the endplate's two corners were given.
    """
    v = np.atleast_2d(np.asarray(vectors, dtype=float))
    if v.shape[-1] != 2:
        raise ValueError("tilt_deg expects 2-D vectors")
    raw = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    out = wrap_to_signed_right_angle(raw)
    out = np.asarray(out, dtype=float)
    return float(out[0]) if np.ndim(vectors) == 1 else out


# --------------------------------------------------------------------------
# 3-D helpers
# --------------------------------------------------------------------------


def project_onto_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Component of ``v`` lying in the plane through the origin with ``normal``."""
    v = np.asarray(v, dtype=float)
    n = unit(np.asarray(normal, dtype=float))
    return v - np.dot(v, n) * n


def rotation_about_z(deg: float) -> np.ndarray:
    """3x3 rotation matrix about the cranio-caudal axis, ``deg`` degrees."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def arc_length(points: np.ndarray) -> np.ndarray:
    """Cumulative chord length along a polyline, starting at 0."""
    p = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])
