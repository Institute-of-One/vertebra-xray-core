"""Synthetic radiographs of a phantom spine.

Why this exists
---------------
Two reasons, one scientific and one practical.

A figure showing which landmarks a detector should place needs a radiograph to
place them on, and no public radiograph can be used: the datasets that carry
vertebral landmarks are gone or frontal-only, and the CT benchmark this work
validates against is ShareAlike-licensed, which cannot be reconciled with a
journal that takes copyright assignment. Rendering the phantom removes the
question -- there is no patient, no licence and no provenance to trace.

It is also the only way to exercise a viewer end to end without patient data.

What is modelled
----------------
Vertebral bodies, pedicles, laminae, spinous and transverse processes, ribs,
and a soft-tissue envelope, each as a region in the vertebra's own frame, then
line-integrated along the beam. The pedicles matter most: seen head-on they
are the paired oval shadows a reader uses to grade rotation, and they are what
this package asks a detector to localise, so a figure in which they are not
visible would be arguing against itself.

What is not modelled: scatter, beam hardening, the heterogeneity of real
trabecular bone, and the thorax beyond a smooth envelope. This produces a
plausible radiograph, not a simulation of one, and nothing in this package
measures anything from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pedicles import normative_pedicles
from .spine3d import Projection, SpineModel3D

__all__ = ["Radiograph", "simulate_radiograph", "ATTENUATION"]

#: Relative linear attenuation, cortical bone taken as 1. Trabecular bone is
#: roughly a third of cortical at radiographic energies and soft tissue an
#: order of magnitude below that; the values only have to order the greys.
ATTENUATION = {
    "cortical": 1.0,
    "trabecular": 0.34,
    "rib": 0.30,
    "pedicle": 1.25,
    "cortex_side": 0.75,
    "soft_tissue": 0.05,
}


@dataclass(frozen=True)
class Radiograph:
    """A rendered projection and where it sits in millimetres.

    ``extent`` is ``(left, right, bottom, top)`` in the projected ``math``
    frame, so landmark coordinates from
    :meth:`~vertebra_xray_core.spine3d.SpineModel3D.project` overlay directly
    without any further transformation.
    """

    image: np.ndarray
    extent: tuple[float, float, float, float]
    view: str

    @property
    def aspect(self) -> float:
        left, right, bottom, top = self.extent
        return abs((right - left) / (top - bottom))


def _local_grid(centre, half, rotation, spacing):
    """Sample points of an axis-aligned box around ``centre``, in world mm."""
    axes = [np.arange(-h, h + spacing, spacing) for h in half]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid @ rotation.T + centre


def _rounded_box(local, half, radius):
    """Membership in a box with rounded edges, for points already in its frame."""
    inner = np.maximum(np.abs(local) - (np.asarray(half) - radius), 0.0)
    return np.linalg.norm(inner, axis=1) <= radius


def _vertebra_parts(model: SpineModel3D, index: int):
    """Regions of one vertebra, as ``(half_extent, offset, attenuation)`` triples.

    Offsets are in the vertebra's own frame: ``+x`` patient left, ``+y``
    anterior, ``+z`` cranial.
    """
    width = float(model.width_mm[index])
    depth = float(model.depth_mm[index])
    height = float(model.height_mm[index])
    # Place the pedicles where the normative table says they are, so that the
    # shadows in the rendering and the landmarks overlaid on it are the same
    # points. A figure whose markers sit beside the anatomy they are meant to
    # mark argues against itself.
    pedicle = normative_pedicles(model.labels[index])

    shell = 2.0
    parts = [
        ((width / 2, depth / 2, height / 2), (0.0, 0.0, 0.0), ATTENUATION["trabecular"]),
        # Cortical endplates. Without them the body is a flat grey block; with
        # them it has the dense upper and lower margins a reader draws the
        # Cobb line along, which is the whole subject of the figure.
        ((width / 2, depth / 2, shell / 2), (0.0, 0.0, height / 2 - shell / 2), ATTENUATION["cortical"]),
        ((width / 2, depth / 2, shell / 2), (0.0, 0.0, -height / 2 + shell / 2), ATTENUATION["cortical"]),
        # Lateral cortices, fainter, so the body has edges.
        ((shell / 2, depth / 2, height / 2), (-width / 2 + shell / 2, 0.0, 0.0), ATTENUATION["cortex_side"]),
        ((shell / 2, depth / 2, height / 2), (width / 2 - shell / 2, 0.0, 0.0), ATTENUATION["cortex_side"]),
    ]
    for side in (-1.0, 1.0):
        parts.append(  # pedicle
            (
                (5.4, 8.5, 7.0),
                (side * pedicle.half_separation, -pedicle.posterior_offset, 0.0),
                ATTENUATION["pedicle"],
            )
        )
        parts.append(  # transverse process
            (
                (14.0, 3.5, 4.0),
                (side * (pedicle.half_separation + 11.0), -pedicle.posterior_offset - 5.0, 0.0),
                ATTENUATION["cortical"],
            )
        )
    parts.append(  # lamina
        ((16.0, 5.0, 4.0), (0.0, -pedicle.posterior_offset - 11.0, 0.0), ATTENUATION["cortical"])
    )
    parts.append(  # spinous process
        ((3.0, 10.0, 7.0), (0.0, -pedicle.posterior_offset - 22.0, -3.0), ATTENUATION["cortical"])
    )
    return parts


def _accumulate_spine(model, volume, origin, spacing):
    rotations = model.rotations
    for index in range(len(model)):
        rotation = rotations[index]
        centre = model.centroids[index]
        for half, offset, mu in _vertebra_parts(model, index):
            part_centre = centre + rotation @ np.asarray(offset, dtype=float)
            margin = max(half) + spacing
            points = _local_grid(part_centre, (margin,) * 3, np.eye(3), spacing)
            local = (points - part_centre) @ rotation
            inside = _rounded_box(local, half, min(2.0, min(half) * 0.45))
            _deposit(volume, origin, spacing, points[inside], mu)


#: Half-width of the rib cage at each thoracic level, in millimetres. The cage
#: widens from the apex down to about T8 and narrows again towards the
#: thoracolumbar junction; a constant radius makes the ribs read as a spring.
_RIB_HALF_WIDTH = {
    "T1": 42.0, "T2": 58.0, "T3": 72.0, "T4": 84.0, "T5": 94.0, "T6": 102.0,
    "T7": 108.0, "T8": 110.0, "T9": 107.0, "T10": 99.0, "T11": 84.0, "T12": 66.0,
}


def _ribs(model, volume, origin, spacing):
    """Rib pairs sweeping laterally from each thoracic vertebra and descending.

    Each rib is traced as an arc that leaves the costovertebral joint running
    posterolaterally, turns at the lateral margin of the cage and comes
    forward towards the sternum, dropping steadily as it goes. On a frontal
    projection that traces the familiar downward hook. Two details do most of
    the work: the cage has to widen and narrow along its length, and the rib
    has to descend far enough, or the arcs close into ovals and the thorax
    looks like a spring.
    """
    for index, label in enumerate(model.labels):
        half_width = _RIB_HALF_WIDTH.get(label)
        if half_width is None:
            continue
        rotation = model.rotations[index]
        centre = model.centroids[index]
        t = np.linspace(0.0, 1.0, 130)
        # Stop before the costal cartilage, which is not radiopaque; carrying
        # the arc round to the midline is what closes it into an oval.
        azimuth = np.radians(10.0 + t * 128.0)
        depth = 0.72 * half_width
        for side in (-1.0, 1.0):
            arc = np.stack(
                [
                    side * half_width * np.sin(azimuth),
                    -24.0 + depth * (1.0 - np.cos(azimuth)),
                    # Most of the descent is in the anterior half: the
                    # posterior rib runs nearly level out to the lateral
                    # margin, the anterior rib then falls steeply.
                    -4.0 - 118.0 * t**2.3,
                ],
                axis=1,
            )
            points = arc @ rotation.T + centre
            _deposit(volume, origin, spacing, points, ATTENUATION["rib"], thickness=3.0)


def _soft_tissue(model, volume, origin, spacing):
    """An elliptical torso that narrows at the shoulders and flares at the pelvis.

    Without it the skeleton floats on white and the picture does not read as a
    radiograph; with a plain rectangle it reads as a phantom in a box.
    """
    z = origin[2] + spacing * np.arange(volume.shape[2])
    top, bottom = model.centroids[:, 2].max(), model.centroids[:, 2].min()
    span = max(top - bottom, 1.0)

    x = origin[0] + spacing * np.arange(volume.shape[0])
    y = origin[1] + spacing * np.arange(volume.shape[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    # Follow the spine rather than the coordinate origin: a torso centred on
    # zero around a spine that is not leaves the patient looking off-centre in
    # their own film.
    trunk_x = float(np.median(model.centroids[:, 0]))

    for k, height in enumerate(z):
        if not (bottom - 70.0) < height < (top + 55.0):
            continue
        # 0 at the pelvis, 1 at the shoulders
        u = np.clip((height - (bottom - 70.0)) / (span + 125.0), 0.0, 1.0)
        half_x = 132.0 + 34.0 * np.sin(np.pi * np.clip(u * 1.15, 0.0, 1.0)) - 26.0 * u**3
        half_y = 96.0 + 18.0 * np.sin(np.pi * u)
        ellipse = ((xx - trunk_x) / half_x) ** 2 + ((yy + 6.0) / half_y) ** 2 <= 1.0
        volume[:, :, k] = np.maximum(volume[:, :, k], ellipse * ATTENUATION["soft_tissue"])


def _deposit(volume, origin, spacing, points, mu, thickness: float = 0.0):
    """Add attenuation at ``points``, optionally smeared into a tube."""
    if thickness:
        steps = np.arange(-thickness / 2, thickness / 2 + spacing, spacing)
        offsets = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
        points = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    index = np.round((points - origin) / spacing).astype(int)
    shape = np.array(volume.shape)
    keep = np.all((index >= 0) & (index < shape), axis=1)
    index = index[keep]
    np.maximum.at(volume, tuple(index.T), mu)


def simulate_radiograph(
    model: SpineModel3D,
    projection: Projection | None = None,
    *,
    spacing_mm: float = 1.6,
    with_ribs: bool = True,
    with_soft_tissue: bool = True,
    blur_mm: float = 1.6,
    noise: float = 0.012,
    seed: int | None = 0,
) -> Radiograph:
    """Render a phantom spine as a radiograph.

    The volume is built once in the patient frame and line-integrated along
    the beam, so the result is a genuine projection of the same geometry every
    other measurement in this package uses: landmarks projected by
    :meth:`SpineModel3D.project` land exactly where the anatomy is drawn.
    """
    from scipy import ndimage

    projection = projection or Projection(view="pa")
    corners = model.body_corners().reshape(-1, 3)
    low = corners.min(axis=0) - np.array([150.0, 130.0, 70.0])
    high = corners.max(axis=0) + np.array([150.0, 130.0, 60.0])
    shape = np.ceil((high - low) / spacing_mm).astype(int) + 1
    volume = np.zeros(tuple(shape), dtype=np.float32)

    if with_soft_tissue:
        _soft_tissue(model, volume, low, spacing_mm)
    _accumulate_spine(model, volume, low, spacing_mm)
    if with_ribs:
        _ribs(model, volume, low, spacing_mm)

    axis = {"pa": 1, "ap": 1, "lateral": 0}[projection.view]
    thickness = volume.sum(axis=axis) * spacing_mm

    if axis == 1:  # frontal: image axes are X and Z
        image = thickness
        horizontal = (low[0], low[0] + spacing_mm * (shape[0] - 1))
    else:  # lateral: image axes are Y and Z, anterior to the left by convention
        image = thickness
        horizontal = (low[1], low[1] + spacing_mm * (shape[1] - 1))
        if projection.anterior_on_image_left:
            image = image[::-1, :]
            horizontal = (-horizontal[1], -horizontal[0])
    vertical = (low[2], low[2] + spacing_mm * (shape[2] - 1))

    if blur_mm:
        image = ndimage.gaussian_filter(image, blur_mm / spacing_mm)
    if noise:
        rng = np.random.default_rng(seed)
        image = image + rng.normal(0.0, noise * max(image.max(), 1e-6), image.shape)

    # Radiographic convention: dense structures dark. The image is a path
    # integral of attenuation, so exponentiating turns it into transmitted
    # intensity, which is what a film records.
    transmitted = np.exp(-image / max(np.percentile(image, 99.5), 1e-6) * 2.4)

    return Radiograph(
        image=transmitted.T,  # (rows = z, cols = horizontal), origin bottom-left
        extent=(horizontal[0], horizontal[1], vertical[0], vertical[1]),
        view=projection.view,
    )
