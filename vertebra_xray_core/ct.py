"""Vertebral geometry from a CT segmentation.

A labelled CT segmentation is the only widely available source of a spine's
*true* three-dimensional vertebral orientation, including the axial rotation
that two radiographs cannot determine. That makes it the reference standard
this package validates the biplanar path against, and it is also how a real
spine gets into the pipeline without any patient radiograph being published.

What this module does not do is segment anything. It takes a mask that some
other tool produced -- a public benchmark's manual annotation,
TotalSegmentator, or anything else writing the same convention -- and turns it
into a :class:`~vertebra_xray_core.spine3d.SpineModel3D`.

Isolating the vertebral body
----------------------------
Whole-vertebra masks include the posterior elements, and the spinous process
is long enough to dominate a principal-axis fit: orient a thoracic vertebra by
the inertia tensor of its whole mask and the "antero-posterior" axis comes out
pointing down the spinous process. Every angle derived from it is then wrong
by tens of degrees.

So the body is isolated first. Walking posteriorly from the anterior margin,
the mask's cross-section is wide through the body, pinches at the pedicles,
widens again at the laminae and narrows to the spinous process. The pinch is
anatomy, not noise, and cutting there separates the body cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from .spine3d import SpineModel3D, euler_from_matrix

__all__ = [
    "VertebraGeometry",
    "ras_to_patient_frame",
    "isolate_vertebral_body",
    "orientation_from_points",
    "vertebra_geometry",
    "model_from_segmentation",
]

#: Nibabel loads a NIfTI into RAS+ world coordinates: ``+x`` right, ``+y``
#: anterior, ``+z`` superior. This package's patient frame has ``+X`` towards
#: the patient's *left*, so only the first axis flips.
_RAS_TO_PATIENT = np.diag([-1.0, 1.0, 1.0])


def ras_to_patient_frame(points_ras: np.ndarray) -> np.ndarray:
    """Convert RAS+ world coordinates to this package's patient frame."""
    return np.asarray(points_ras, dtype=float) @ _RAS_TO_PATIENT.T


@dataclass(frozen=True)
class VertebraGeometry:
    """One vertebral body's position, orientation and size, from its mask."""

    label: int
    centroid: np.ndarray  # (3,) mm, patient frame
    rotation: np.ndarray  # (3, 3), columns are lateral, anterior, normal
    width_mm: float
    depth_mm: float
    height_mm: float
    voxels: int
    body_voxels: int
    touches_boundary: bool
    iterations: int = 1

    @property
    def angles_deg(self) -> tuple[float, float, float]:
        """``(theta, phi, psi)`` -- coronal tilt, sagittal tilt, axial rotation."""
        return euler_from_matrix(self.rotation)

    @property
    def looks_like_a_vertebral_body(self) -> bool:
        """Crude plausibility check on the isolated body's proportions.

        A vertebral body is wider than it is tall and its dimensions sit in a
        narrow anatomical range. A "body" outside that range means the
        posterior elements were not separated, or the mask is a fragment of a
        vertebra cut off by the field of view. Either way its orientation
        should not be trusted.
        """
        return (
            15.0 <= self.width_mm <= 70.0
            and 12.0 <= self.depth_mm <= 55.0
            and 8.0 <= self.height_mm <= 45.0
            and self.width_mm > self.height_mm
        )


# --------------------------------------------------------------------------
# body isolation
# --------------------------------------------------------------------------


def isolate_vertebral_body(
    points: np.ndarray,
    anterior_axis: np.ndarray | None = None,
    *,
    bin_mm: float = 1.5,
    pinch_fraction: float = 0.55,
) -> np.ndarray:
    """Boolean mask selecting the vertebral body among a vertebra's points.

    Parameters
    ----------
    points
        ``(N, 3)`` patient-frame coordinates of one vertebra's voxels.
    anterior_axis
        Direction to walk along. ``None`` uses the patient's anterior axis,
        which is within about 30 degrees of any vertebra's own and is a good
        enough starting point; :func:`vertebra_geometry` then refines it.
    bin_mm
        Histogram bin width along that axis.
    pinch_fraction
        The cut is placed at the first bin posterior to the widest one whose
        count falls below this fraction of it. Raising it cuts more
        aggressively and risks trimming the body itself.

    Notes
    -----
    If no pinch is found -- a cervical vertebra, where the posterior elements
    are not slender, or a mask that is body-only already -- everything is
    kept. Silently cutting at an arbitrary depth would be worse than not
    cutting.
    """
    pts = np.asarray(points, dtype=float)
    axis = geo.unit(np.array([0.0, 1.0, 0.0]) if anterior_axis is None else anterior_axis)
    along = pts @ axis
    lo, hi = along.min(), along.max()
    if hi - lo < 3 * bin_mm:
        return np.ones(len(pts), dtype=bool)

    edges = np.arange(lo, hi + bin_mm, bin_mm)
    counts, _ = np.histogram(along, bins=edges)
    if len(counts) < 4:
        return np.ones(len(pts), dtype=bool)

    peak = int(np.argmax(counts))
    threshold = pinch_fraction * counts[peak]
    cut = None
    for k in range(peak - 1, 0, -1):  # walk posteriorly, i.e. to lower ``along``
        if counts[k] < threshold:
            cut = edges[k + 1]
            break
    if cut is None:
        return np.ones(len(pts), dtype=bool)
    return along >= cut


# --------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------


def orientation_from_points(points: np.ndarray) -> np.ndarray:
    """Orientation matrix of a vertebral body from its voxel cloud.

    The three principal axes of the point cloud are matched to the patient's
    left-right, antero-posterior and cranio-caudal axes by closest alignment
    rather than by eigenvalue order. Ordering by eigenvalue looks equivalent,
    because a body is widest left-to-right and shortest top-to-bottom, but
    width and depth are nearly equal in the upper thoracic spine and the two
    axes then swap at random.

    The frame is rebuilt from the normal and the lateral axis alone, with the
    antero-posterior axis as their cross product, so the result is exactly
    orthonormal and right-handed even though the principal axes of a real
    mask are only approximately so.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        raise ValueError(f"need at least 4 points to orient a body, got {len(pts)}")
    centred = pts - pts.mean(axis=0)
    # Eigendecomposition of the 3x3 covariance, not an SVD of the whole point
    # matrix. They give the same axes, but a vertebra at fine CT spacing has
    # of the order of a million voxels and this runs a dozen times per
    # vertebra as the body isolation converges; the covariance costs one pass
    # over the points and a 3x3 solve, the SVD does not.
    covariance = centred.T @ centred / len(centred)
    _, vectors = np.linalg.eigh(covariance)
    axes = vectors.T[::-1]  # rows: principal directions, most-spread first

    patient_axes = np.eye(3)  # columns X (left), Y (anterior), Z (cranial)
    alignment = np.abs(axes @ patient_axes)
    assigned: dict[int, np.ndarray] = {}
    remaining = set(range(3))
    targets = set(range(3))
    while remaining:
        best = max(
            ((i, j) for i in remaining for j in targets), key=lambda ij: alignment[ij[0], ij[1]]
        )
        i, j = best
        direction = axes[i]
        if np.dot(direction, patient_axes[:, j]) < 0:
            direction = -direction
        assigned[j] = direction
        remaining.discard(i)
        targets.discard(j)

    normal = geo.unit(assigned[2])
    lateral = geo.unit(assigned[0] - np.dot(assigned[0], normal) * normal)
    anterior = np.cross(normal, lateral)
    return np.stack([lateral, anterior, normal], axis=1)


def vertebra_geometry(
    points: np.ndarray,
    label: int,
    *,
    touches_boundary: bool = False,
    max_iterations: int = 12,
    tolerance_deg: float = 0.05,
) -> VertebraGeometry:
    """Measure one vertebra from its patient-frame voxel coordinates.

    Body isolation and orientation are alternated to convergence. They depend
    on each other -- the cut needs the antero-posterior axis, and the axis
    needs the cut -- and the patient's own anterior axis is only a starting
    point, up to a vertebra's full axial rotation away from the right one.

    A single refinement pass is not enough. The iteration converges steadily
    rather than in one step: starting from the patient axis on a vertebra
    rotated 45 degrees, the error in the recovered rotation falls through
    roughly 24, 13, 7 and 2 degrees over the first five passes. Stopping early
    leaves a bias that looks like a plausible measurement.
    """
    pts = np.asarray(points, dtype=float)
    axis = None
    rotation = None
    body = np.ones(len(pts), dtype=bool)
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        body = isolate_vertebral_body(pts, axis)
        rotation = orientation_from_points(pts[body])
        moved = 180.0 if axis is None else geo.angle_between(axis, rotation[:, 1])
        axis = rotation[:, 1]
        if moved < tolerance_deg:
            break

    local = (pts[body] - pts[body].mean(axis=0)) @ rotation
    extent = local.max(axis=0) - local.min(axis=0)
    return VertebraGeometry(
        label=label,
        centroid=pts[body].mean(axis=0),
        rotation=rotation,
        width_mm=float(extent[0]),
        depth_mm=float(extent[1]),
        height_mm=float(extent[2]),
        voxels=int(len(pts)),
        body_voxels=int(body.sum()),
        touches_boundary=touches_boundary,
        iterations=iterations,
    )


# --------------------------------------------------------------------------
# whole-spine assembly
# --------------------------------------------------------------------------


def model_from_segmentation(
    mask: np.ndarray,
    affine: np.ndarray,
    label_names: dict[int, str],
    *,
    min_voxels: int = 200,
    drop_boundary: bool = True,
    drop_implausible: bool = True,
) -> tuple[SpineModel3D, dict[int, VertebraGeometry]]:
    """Build a :class:`SpineModel3D` from a labelled vertebral segmentation.

    Parameters
    ----------
    mask
        Integer label volume in voxel space.
    affine
        The NIfTI affine mapping voxel indices to RAS+ world millimetres.
    label_names
        Which integer labels to use and what level each one is. Labels absent
        from this mapping are ignored, which is how sacrum, coccyx and any
        dataset-specific extras are excluded.
    min_voxels
        Labels with fewer voxels than this are fragments, not vertebrae.
    drop_boundary
        Discard vertebrae whose mask touches the edge of the volume. Their
        centroid and orientation are biased by whatever the field of view cut
        off, and a long-film study is exactly where partial vertebrae appear.
    drop_implausible
        Discard vertebrae whose isolated body fails
        :attr:`VertebraGeometry.looks_like_a_vertebral_body`.

    Returns
    -------
    The model, and the per-label geometry including the vertebrae that were
    dropped, so the rejection can be audited rather than merely happening.
    """
    mask = np.asarray(mask)
    shape = np.array(mask.shape)
    present = [int(v) for v in np.unique(mask) if v in label_names]
    if not present:
        raise ValueError(
            f"none of the labels in the mask {sorted(np.unique(mask))[:12]} are in label_names"
        )

    # One pass over the volume, then group. A per-label ``mask == label`` scan
    # is the obvious way and costs a full sweep of a quarter-billion voxels
    # for each of two dozen vertebrae.
    occupied = np.argwhere(np.isin(mask, present))
    values = mask[tuple(occupied.T)]
    order = np.argsort(values, kind="stable")
    occupied, values = occupied[order], values[order]
    starts = np.searchsorted(values, present, side="left")
    ends = np.searchsorted(values, present, side="right")

    measured: dict[int, VertebraGeometry] = {}
    for label, start, end in zip(present, starts, ends, strict=True):
        indices = occupied[start:end]
        if len(indices) < min_voxels:
            continue
        touches = bool((indices.min(axis=0) == 0).any() or (indices.max(axis=0) == shape - 1).any())
        world = np.hstack([indices, np.ones((len(indices), 1))]) @ np.asarray(affine).T
        measured[label] = vertebra_geometry(
            ras_to_patient_frame(world[:, :3]), label, touches_boundary=touches
        )

    keep = [
        label
        for label, geometry in measured.items()
        if not (drop_boundary and geometry.touches_boundary)
        and not (drop_implausible and not geometry.looks_like_a_vertebral_body)
    ]
    if len(keep) < 2:
        raise ValueError(
            f"only {len(keep)} usable vertebrae of {len(measured)} measured; "
            "the field of view is probably too small or the mask is not whole vertebrae"
        )

    # Cranial first, which for this frame means decreasing Z.
    keep.sort(key=lambda label: -measured[label].centroid[2])
    angles = np.array([measured[label].angles_deg for label in keep])
    return (
        SpineModel3D(
            labels=tuple(label_names[label] for label in keep),
            centroids=np.stack([measured[label].centroid for label in keep]),
            theta_deg=angles[:, 0],
            phi_deg=angles[:, 1],
            psi_deg=angles[:, 2],
            width_mm=np.array([measured[label].width_mm for label in keep]),
            depth_mm=np.array([measured[label].depth_mm for label in keep]),
            height_mm=np.array([measured[label].height_mm for label in keep]),
            source="ct-segmentation",
            meta={
                "dropped": {
                    label: measured[label] for label in measured if label not in set(keep)
                },
            },
        ),
        measured,
    )
