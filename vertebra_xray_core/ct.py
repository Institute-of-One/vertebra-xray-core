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

So the body is isolated first, by thickness rather than by direction. The body
is the only part of a vertebra that is thick in every direction; pedicles,
laminae and the transverse and spinous processes are all slender. Keeping the
voxels further than a few millimetres from the outside of the mask, and then
the largest connected piece of what survives, leaves the body and nothing
else -- with no reference to which way the vertebra is facing, which is what
makes it work on a spine that is rotated and bent.
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
    "BODY_CORE_RADIUS_MM",
    "orientation_from_points",
    "orientation_from_core_and_posterior",
    "endplate_normal_from_core",
    "vertebra_geometry",
    "vertebra_from_mask",
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


#: Depth of the inward erosion that separates the vertebral body from its
#: posterior elements, in millimetres.
#:
#: Chosen by sweeping it against anatomical plausibility on VerSe. At 4.5 mm
#: the lumbar pedicles survive -- they are 15 mm and more across down there --
#: and L5 comes out 78 mm wide against a true 50. At 7.5 mm the correction
#: added back for the erosion over-inflates the upper thoracic bodies. At
#: 6 mm every vertebra measured was anatomically plausible and the mean
#: dimensions match published morphometry to a few millimetres: L3 at 45 x 39
#: x 28 mm, T12 at 42 x 37 x 26.
BODY_CORE_RADIUS_MM = 6.0

#: How far behind the body core the rest of the vertebra's centroid must sit
#: before that direction is trusted to fix the antero-posterior axis. Real
#: posterior elements displace it by well over a centimetre; anything under
#: this is the eroded shell of a body-only mask and carries no direction.
MIN_POSTERIOR_OFFSET_MM = 4.0


def isolate_vertebral_body(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    radius_mm: float = BODY_CORE_RADIUS_MM,
    min_core_fraction: float = 0.04,
) -> tuple[np.ndarray, float]:
    """Body core within one vertebra's mask, and the erosion depth that produced it.

    Every voxel further than ``radius_mm`` from the outside of the mask is
    kept, and of the pieces that survive, the largest connected one is the
    body. Nothing about the vertebra's orientation enters, which is what makes
    this work on a mask that has been rotated any which way.

    Parameters
    ----------
    volume
        Binary sub-volume containing exactly one vertebra.
    spacing_mm
        Voxel size along each axis. The distance transform is computed in
        millimetres, so anisotropic CT is handled without resampling.
    radius_mm
        How far in to erode. See :data:`BODY_CORE_RADIUS_MM`.
    min_core_fraction
        If the surviving core is smaller than this fraction of the vertebra,
        the erosion is retried shallower. A collapsed or fractured body is
        thin enough to be erased outright by a fixed radius, and VerSe is full
        of them. The depth that finally worked is returned alongside the core,
        because the dimensions have to be corrected by the erosion that
        actually happened rather than the one that was asked for.

    Notes
    -----
    An earlier version walked along the antero-posterior axis looking for the
    narrowing at the pedicles and cut there. On a box phantom with two neat
    pedicles and a spinous process, that works. On real vertebrae it does not:
    the narrowing is shallow, the cut lands in the wrong place, and the
    alternation between "cut using the axis" and "find the axis using the cut"
    then fails to converge. Measured on VerSe, it left the transverse
    processes attached and reported vertebral bodies 60 to 80 mm deep, against
    a true 20 to 40. Shape, not direction, is what separates a body from its
    posterior elements.
    """
    from scipy import ndimage

    binary = np.asarray(volume, dtype=bool)
    if not binary.any():
        return binary, 0.0

    # A shell of background around the mask, because the distance transform
    # measures the distance to the nearest zero voxel and there is no zero
    # beyond the edge of the array. Without the shell, a mask cropped tightly
    # to its own bounding box -- which is how it arrives here -- reports large
    # distances all along its outer surface, so the erosion leaves the outside
    # of the vertebra in place and quietly does nothing.
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    distance = ndimage.distance_transform_edt(padded, sampling=spacing_mm)[
        1:-1, 1:-1, 1:-1
    ]

    for attempt in (radius_mm, 0.75 * radius_mm, 0.5 * radius_mm, 0.25 * radius_mm):
        core = distance >= attempt
        if not core.any():
            continue
        components, count = ndimage.label(core)
        if count == 0:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        largest = components == int(np.argmax(sizes))
        if largest.sum() >= min_core_fraction * binary.sum():
            return largest, attempt
    return binary, 0.0


# --------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------


def orientation_from_core_and_posterior(
    core_points: np.ndarray,
    posterior_points: np.ndarray,
    cranial_hint: np.ndarray | None = None,
) -> np.ndarray:
    """Orientation from the body core plus the direction its posterior elements lie in.

    The endplate normal comes from the core's principal axes, where it is the
    axis of least spread and is unambiguous. The *in-plane* orientation does
    not come from the principal axes at all, and cannot: a thoracic vertebral
    body is as deep as it is wide -- measured on VerSe, width over depth runs
    between 0.8 and 1.3 through the thoracic spine -- so the two in-plane
    eigenvalues are degenerate and their eigenvectors are arbitrary. Taking
    them anyway produces an axial rotation that flips by 90 degrees from one
    vertebra to the next, which is what it did: a median of 27 degrees of
    apparent rotation on supine CT, where the true value is near zero.

    Anatomy settles it. The posterior elements are behind the body and the
    spinous process is long, so the vector from the body's centroid to the
    centroid of everything else points posteriorly whatever the vertebra's
    orientation. Projected into the endplate plane, that fixes the
    antero-posterior axis, and the lateral axis follows from the cross
    product.
    """
    core = np.asarray(core_points, dtype=float)
    normal = None
    if cranial_hint is not None:
        normal = endplate_normal_from_core(core, cranial_hint)
    if normal is None:
        normal = _principal_normal(core, cranial_hint)

    posterior = np.asarray(posterior_points, dtype=float)
    if len(posterior) < 4:
        return orientation_from_points(core)


    backwards = posterior.mean(axis=0) - core.mean(axis=0)
    anterior = -geo.project_onto_plane(backwards, normal)
    # The displacement has to be big enough to mean something. Real posterior
    # elements shift the centroid by a centimetre or more. A mask holding only
    # the vertebral body -- which some segmentation tools produce -- leaves
    # nothing behind the core but the eroded shell, whose centroid sits on top
    # of the core's; the displacement is then a rounding artefact pointing in
    # an arbitrary direction, and trusting it scrambles the orientation
    # silently. Below the threshold there is no information to use and the
    # principal axes, degenerate as they are, are the honest fallback.
    if np.linalg.norm(anterior) < MIN_POSTERIOR_OFFSET_MM:
        return orientation_from_points(core)
    anterior = geo.unit(anterior)
    lateral = np.cross(anterior, normal)
    return np.stack([lateral, anterior, normal], axis=1)


def endplate_normal_from_core(
    core_points: np.ndarray,
    cranial_hint: np.ndarray,
    *,
    cell_mm: float = 2.0,
    central_fraction: float = 0.7,
) -> np.ndarray | None:
    """Endplate normal measured from the core's top and bottom surfaces.

    This is what a Cobb angle is actually defined on, and it is measured here
    the way it is drawn: by finding the two endplate surfaces and fitting a
    plane to each.

    The alternative -- taking the body's axis of least spread -- is what this
    replaces, and it does not work. After erosion a lumbar body core is about
    26 by 18 by 17 mm, so its two smaller principal moments differ by under
    20%: the pair is degenerate and its eigenvectors come out rotated some
    arbitrary angle within the sagittal plane. Measured on VerSe, they sat 40
    and 51 degrees from the cranio-caudal axis on a vertebra whose endplate is
    within 10 degrees of horizontal. No rule for *choosing* between axes can
    fix that, because neither axis is anatomical.

    Erosion leaves a flat surface flat and parallel to where it was, so the
    core's top and bottom faces carry the endplates' orientation even though
    they sit a few millimetres inside them. Only the central
    ``central_fraction`` of each face is used: erosion rounds convex edges, and
    the rim would tilt the fit.

    Returns ``None`` when the core is too small or too flat to fit two planes,
    leaving the caller to fall back.
    """
    points = np.asarray(core_points, dtype=float)
    up = geo.unit(np.asarray(cranial_hint, dtype=float))
    side = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(side, up)) > 0.9:
        side = np.array([0.0, 1.0, 0.0])
    u = geo.unit(np.cross(up, side))
    v = np.cross(up, u)
    local = np.stack([points @ u, points @ v, points @ up], axis=1)

    cells = np.floor(local[:, :2] / cell_mm).astype(np.int64)
    keys = cells[:, 0] * 100003 + cells[:, 1]
    order = np.argsort(keys, kind="stable")
    keys, sorted_local = keys[order], local[order]
    starts = np.flatnonzero(np.concatenate([[True], keys[1:] != keys[:-1]]))
    ends = np.concatenate([starts[1:], [len(keys)]])

    centres, tops, bottoms = [], [], []
    for start, end in zip(starts, ends, strict=True):
        block = sorted_local[start:end]
        centres.append(block[:, :2].mean(axis=0))
        tops.append(block[:, 2].max())
        bottoms.append(block[:, 2].min())
    centres = np.asarray(centres)
    if len(centres) < 12:
        return None

    middle = centres.mean(axis=0)
    radius = central_fraction * np.abs(centres - middle).max(axis=0)
    inside = np.all(np.abs(centres - middle) <= np.maximum(radius, cell_mm), axis=1)
    if inside.sum() < 8:
        return None

    design = np.column_stack([centres[inside], np.ones(inside.sum())])
    normals = []
    for surface in (np.asarray(tops)[inside], np.asarray(bottoms)[inside]):
        fit, *_ = np.linalg.lstsq(design, surface, rcond=None)
        normals.append(geo.unit(np.array([-fit[0], -fit[1], 1.0])))
    local_normal = geo.unit(normals[0] + normals[1])
    return geo.unit(local_normal[0] * u + local_normal[1] * v + local_normal[2] * up)


def _principal_normal(points: np.ndarray, cranial_hint: np.ndarray | None = None) -> np.ndarray:
    """Unit endplate normal of a body core, pointing cranially.

    With ``cranial_hint`` -- the local direction of the spine, from the
    neighbouring vertebrae's centroids -- the normal is the principal axis
    closest to it. Without one it is the axis of least spread, on the grounds
    that a vertebral body is shortest from endplate to endplate.

    The hint matters. After a 6 mm erosion a lumbar body core is about 32 by
    25 by 20 mm, and the last two are close enough that which axis has least
    spread comes down to noise; picking wrong tilts the recovered endplate by
    tens of degrees. On VerSe that showed up as isolated spikes -- minus 50
    degrees at T12 with plus 5 either side -- against a centreline whose own
    tangent was perfectly smooth. A vertebra's endplate normal is never far
    from the line its neighbours sit on, so the hint resolves it.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        raise ValueError(f"need at least 4 points to orient a body, got {len(pts)}")
    centred = pts - pts.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred / len(centred))

    if cranial_hint is None:
        normal = vectors[:, 0]  # smallest eigenvalue
        reference = np.array([0.0, 0.0, 1.0])
    else:
        reference = geo.unit(np.asarray(cranial_hint, dtype=float))
        normal = vectors[:, int(np.argmax(np.abs(reference @ vectors)))]
    return geo.unit(normal if np.dot(normal, reference) >= 0 else -normal)


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
    core_points: np.ndarray,
    label: int,
    *,
    posterior_points: np.ndarray | None = None,
    cranial_hint: np.ndarray | None = None,
    vertebra_voxels: int | None = None,
    eroded_by_mm: float = BODY_CORE_RADIUS_MM,
    touches_boundary: bool = False,
) -> VertebraGeometry:
    """Measure one vertebra from the patient-frame points of its body core.

    ``core_points`` is what :func:`isolate_vertebral_body` selected, already
    converted to millimetres in the patient frame, and ``posterior_points`` is
    everything else in the vertebra's mask. Supplying the second is strongly
    preferred: without it the in-plane orientation falls back to the body's
    principal axes, which are degenerate for a thoracic vertebra and give a
    meaningless axial rotation. See
    :func:`orientation_from_core_and_posterior`.

    Dimensions are the core's extents plus twice ``eroded_by_mm``, because
    erosion removes that much from each side along every direction.
    """
    pts = np.asarray(core_points, dtype=float)
    rotation = (
        orientation_from_points(pts)
        if posterior_points is None
        else orientation_from_core_and_posterior(pts, posterior_points, cranial_hint)
    )
    centroid = pts.mean(axis=0)
    local = (pts - centroid) @ rotation
    extent = local.max(axis=0) - local.min(axis=0) + 2.0 * eroded_by_mm
    return VertebraGeometry(
        label=label,
        centroid=centroid,
        rotation=rotation,
        width_mm=float(extent[0]),
        depth_mm=float(extent[1]),
        height_mm=float(extent[2]),
        voxels=int(vertebra_voxels if vertebra_voxels is not None else len(pts)),
        body_voxels=int(len(pts)),
        touches_boundary=touches_boundary,
    )


def vertebra_from_mask(
    volume: np.ndarray,
    affine: np.ndarray,
    label: int = 1,
    *,
    body_core_radius_mm: float = BODY_CORE_RADIUS_MM,
) -> VertebraGeometry:
    """Measure a single vertebra from a mask holding only that vertebra.

    The same path :func:`model_from_segmentation` takes for each label,
    exposed on its own so one vertebra can be measured or tested without
    assembling a spine around it.
    """
    volume = np.asarray(volume)
    indices = np.argwhere(volume == label)
    if len(indices) < 4:
        raise ValueError(f"label {label} has {len(indices)} voxels, too few to measure")
    return _measure(indices, volume.shape, np.asarray(affine, dtype=float), label,
                    body_core_radius_mm)


def _measure(
    indices: np.ndarray,
    shape: tuple[int, ...],
    affine: np.ndarray,
    label: int,
    radius_mm: float,
    cranial_hint: np.ndarray | None = None,
) -> VertebraGeometry:
    """Shared body of the two entry points above."""
    spacing = tuple(float(v) for v in np.linalg.norm(affine[:3, :3], axis=0))
    extent = np.array(shape)
    touches = bool((indices.min(axis=0) == 0).any() or (indices.max(axis=0) == extent - 1).any())

    # Crop to this vertebra before the distance transform: a full CT volume is
    # a quarter of a billion voxels and one vertebra occupies a thousandth.
    low = indices.min(axis=0)
    box = np.zeros(indices.max(axis=0) + 1 - low, dtype=bool)
    box[tuple((indices - low).T)] = True
    core, eroded_by = isolate_vertebral_body(box, spacing, radius_mm=radius_mm)
    core_indices = np.argwhere(core) + low
    rest_indices = np.argwhere(box & ~core) + low

    def to_patient(idx):
        world = np.hstack([idx, np.ones((len(idx), 1))]) @ affine.T
        return ras_to_patient_frame(world[:, :3])

    return vertebra_geometry(
        to_patient(core_indices),
        label,
        posterior_points=to_patient(rest_indices) if len(rest_indices) else None,
        cranial_hint=cranial_hint,
        vertebra_voxels=len(indices),
        eroded_by_mm=eroded_by,
        touches_boundary=touches,
    )


def _local_spine_axes(centroids: list[np.ndarray]) -> list[np.ndarray]:
    """Cranial direction at each vertebra, from the chord between its neighbours."""
    points = np.asarray(centroids, dtype=float)
    n = len(points)
    axes = []
    for i in range(n):
        above = points[max(i - 1, 0)]
        below = points[min(i + 1, n - 1)]
        chord = above - below
        axes.append(geo.unit(chord) if np.any(chord) else np.array([0.0, 0.0, 1.0]))
    return axes


# --------------------------------------------------------------------------
# whole-spine assembly
# --------------------------------------------------------------------------


def model_from_segmentation(
    mask: np.ndarray,
    affine: np.ndarray,
    label_names: dict[int, str],
    *,
    min_voxels: int = 200,
    body_core_radius_mm: float = BODY_CORE_RADIUS_MM,
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
    body_core_radius_mm
        Erosion depth used to separate the vertebral body from its posterior
        elements; see :func:`isolate_vertebral_body`.
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

    affine = np.asarray(affine, dtype=float)
    usable = {
        label: occupied[start:end]
        for label, start, end in zip(present, starts, ends, strict=True)
        if end - start >= min_voxels
    }

    # First pass for the centroids, which do not depend on orientation, then a
    # second that orients each vertebra using the line its neighbours sit on.
    # Getting the endplate normal from the body's own principal axes alone is
    # ambiguous whenever the eroded core is nearly as tall as it is deep, which
    # in the lumbar spine it is.
    first = {
        label: _measure(indices, mask.shape, affine, label, body_core_radius_mm)
        for label, indices in usable.items()
    }
    order = sorted(first, key=lambda label: -first[label].centroid[2])
    hints = _local_spine_axes([first[label].centroid for label in order])

    measured: dict[int, VertebraGeometry] = {
        label: _measure(
            usable[label], mask.shape, affine, label, body_core_radius_mm, hints[k]
        )
        for k, label in enumerate(order)
    }

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
