"""The landmark container every measurement in this package consumes.

The measurement engine is deliberately agnostic about where landmarks come
from -- a human clicking corners, the AASCE ground truth, a CNN detector, or
corners derived from a CT segmentation. They all arrive as a
:class:`SpineLandmarks` and are measured by exactly the same code, which is
what makes a detector and a human directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from . import geometry as geo

__all__ = ["SpineLandmarks", "canonicalise_corners", "UL", "UR", "LL", "LR"]

#: Index of each corner along axis 1 of :attr:`SpineLandmarks.corners`.
UL, UR, LL, LR = 0, 1, 2, 3

View = Literal["pa", "ap", "lateral"]


def _quad_with_axis(quad: np.ndarray, cranial: np.ndarray) -> np.ndarray:
    """Order one quad's corners given the local cranial direction ``cranial``."""
    centre = quad.mean(axis=0)
    rel = quad - centre
    up = geo.unit(cranial)
    right = np.array([up[1], -up[0]])  # ``up`` rotated -90 deg, so +y maps to +x
    along = rel @ up
    across = rel @ right
    order = np.argsort(-along, kind="stable")
    superior, inferior = order[:2], order[2:]
    superior = superior[np.argsort(across[superior], kind="stable")]
    inferior = inferior[np.argsort(across[inferior], kind="stable")]
    return quad[[superior[0], superior[1], inferior[0], inferior[1]]]


def canonicalise_corners(quads: np.ndarray) -> np.ndarray:
    """Order every vertebra's four corners as ``UL, UR, LL, LR``.

    Expects ``math``-frame points for a whole spine, ``(N, 4, 2)``, already
    sorted cranial to caudal.

    Corners are split along the **local** cranio-caudal direction -- the chord
    between the neighbouring vertebral centroids -- rather than along the
    image's ``y`` axis. Sorting on global ``y`` looks equivalent and is what
    the obvious implementation does, but it fails whenever a vertebra's tilt
    exceeds ``atan(height / width)``: past roughly 30 deg the inferior-right
    corner of a real vertebral body sits higher in the image than the
    superior-left one, the two pairs swap, and the measured endplate tilt
    jumps by 90 deg. Severely tilted vertebrae are precisely the ones that
    carry the Cobb angle, so this is not a corner case.

    Annotation files also disagree about corner order often enough that
    trusting the order in the file is not safe.
    """
    q = np.asarray(quads, dtype=float)
    if q.ndim != 3 or q.shape[1:] != (4, 2):
        raise ValueError(f"expected (N, 4, 2) quads, got {q.shape}")
    centroids = q.mean(axis=1)
    n = len(q)
    out = np.empty_like(q)
    for i in range(n):
        lo = centroids[max(i - 1, 0)]
        hi = centroids[min(i + 1, n - 1)]
        cranial = lo - hi
        if not np.any(cranial):
            cranial = np.array([0.0, 1.0])
        out[i] = _quad_with_axis(q[i], cranial)
    return out


@dataclass(frozen=True)
class SpineLandmarks:
    """Four corners per vertebral body, in one radiographic projection.

    Attributes
    ----------
    corners
        ``(N, 4, 2)`` array in the ``math`` frame (see
        :mod:`vertebra_xray_core.geometry`), vertebrae ordered cranial to
        caudal, corners ordered ``UL, UR, LL, LR``.
    view
        ``pa``, ``ap`` or ``lateral``. This drives the sign of the
        patient-left axis in 3-D reconstruction, so it is required rather
        than guessed.
    labels
        Vertebral level per row, or ``None`` when the spine has not been
        labelled yet. Labelling is a separate, reportable step.
    spacing_mm
        ``(column, row)`` millimetres per pixel, when known. Angles do not
        need it; distances and the 3-D reconstruction do.
    source
        Free-text provenance, carried into every report.
    """

    corners: np.ndarray
    view: View
    labels: tuple[str, ...] | None = None
    spacing_mm: tuple[float, float] | None = None
    source: str = "unknown"
    meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        c = np.asarray(self.corners, dtype=float)
        if c.ndim != 3 or c.shape[1:] != (4, 2):
            raise ValueError(f"corners must be (N, 4, 2), got {c.shape}")
        if len(c) < 2:
            raise ValueError("at least two vertebrae are needed to measure anything")
        if self.view not in ("pa", "ap", "lateral"):
            raise ValueError(f"unknown view {self.view!r}")
        if self.labels is not None and len(self.labels) != len(c):
            raise ValueError(f"{len(self.labels)} labels for {len(c)} vertebrae")
        object.__setattr__(self, "corners", c)
        if self.labels is not None:
            object.__setattr__(self, "labels", tuple(self.labels))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_image_corners(
        cls,
        corners: np.ndarray,
        view: View,
        *,
        height: float | None = None,
        labels: tuple[str, ...] | None = None,
        spacing_mm: tuple[float, float] | None = None,
        source: str = "unknown",
        meta: dict | None = None,
    ) -> SpineLandmarks:
        """Build from ``image``-frame corners, flipping ``y`` and canonicalising.

        This is the intended entry point for anything read off disk.
        """
        pts = geo.to_math_frame(np.asarray(corners, dtype=float), height)
        if pts.ndim != 3 or pts.shape[1:] != (4, 2):
            raise ValueError(f"corners must be (N, 4, 2), got {pts.shape}")
        if labels is not None and len(labels) != len(pts):
            raise ValueError(f"{len(labels)} labels for {len(pts)} vertebrae")
        # Order the stack cranial-first before canonicalising, because corner
        # assignment needs each vertebra's neighbours.
        order = np.argsort(-pts[:, :, 1].mean(axis=1), kind="stable")
        quads = canonicalise_corners(pts[order])
        if labels is not None:
            labels = tuple(np.asarray(labels, dtype=object)[order])
        return cls(
            corners=quads,
            view=view,
            labels=labels,
            spacing_mm=spacing_mm,
            source=source,
            meta=meta or {},
        )

    def with_labels(self, labels: tuple[str, ...] | None) -> SpineLandmarks:
        """Copy carrying ``labels``."""
        return replace(self, labels=None if labels is None else tuple(labels))

    # -- derived geometry --------------------------------------------------

    def __len__(self) -> int:
        return len(self.corners)

    @property
    def superior_endplate(self) -> np.ndarray:
        """``(N, 2)`` vectors along each superior endplate, left corner to right."""
        return self.corners[:, UR] - self.corners[:, UL]

    @property
    def inferior_endplate(self) -> np.ndarray:
        """``(N, 2)`` vectors along each inferior endplate, left corner to right."""
        return self.corners[:, LR] - self.corners[:, LL]

    @property
    def body_axis(self) -> np.ndarray:
        """``(N, 2)`` vectors from the left mid-height to the right mid-height.

        This is the vector the AASCE reference implementation tilts to obtain
        a Cobb angle. It is the average of the two endplate vectors, so it is
        less sensitive to a single mis-clicked corner but is *not* the SRS
        definition. Both are offered; see :mod:`vertebra_xray_core.cobb`.
        """
        left = 0.5 * (self.corners[:, UL] + self.corners[:, LL])
        right = 0.5 * (self.corners[:, UR] + self.corners[:, LR])
        return right - left

    @property
    def centroids(self) -> np.ndarray:
        """``(N, 2)`` vertebral body centroids, the mean of the four corners."""
        return self.corners.mean(axis=1)

    @property
    def body_height(self) -> np.ndarray:
        """``(N,)`` mean vertical distance between the two endplates."""
        sup = 0.5 * (self.corners[:, UL] + self.corners[:, UR])
        inf = 0.5 * (self.corners[:, LL] + self.corners[:, LR])
        return np.linalg.norm(sup - inf, axis=1)

    def superior_tilt(self) -> np.ndarray:
        """``(N,)`` superior endplate tilts in degrees on ``(-90, 90]``."""
        return np.asarray(geo.tilt_deg(self.superior_endplate), dtype=float)

    def inferior_tilt(self) -> np.ndarray:
        """``(N,)`` inferior endplate tilts in degrees on ``(-90, 90]``."""
        return np.asarray(geo.tilt_deg(self.inferior_endplate), dtype=float)

    def body_tilt(self) -> np.ndarray:
        """``(N,)`` mid-body tilts in degrees on ``(-90, 90]``."""
        return np.asarray(geo.tilt_deg(self.body_axis), dtype=float)

    def index_of_label(self, label: str) -> int:
        """Row holding ``label``; raises when unlabelled or absent."""
        if self.labels is None:
            raise ValueError("these landmarks have not been labelled")
        try:
            return self.labels.index(label)
        except ValueError:
            raise ValueError(f"{label!r} is not among {self.labels}") from None
