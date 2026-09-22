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
the shoulder girdle, the iliac wings, sacrum and femoral heads, and a torso
containing aerated lungs, the mediastinum, the abdomen and the arms, then
line-integrated along the beam.

The lungs are not decoration. Vertebral density on a film is not uniform down
the spine: the thoracic bodies are seen through air and stand out sharply,
the lumbar ones through the abdomen and are washed out, and the hardest
levels to read are the ones at the diaphragm where that changes. A rendering
of one uniform density hides the part of the problem that matters.

The pedicles matter most: seen head-on they
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
    "soft_tissue": 0.050,
    "abdomen": 0.080,
    "mediastinum": 0.105,
    "lung": 0.006,
    "iliac_wing": 0.33,
    "pelvic_rim": 0.56,
    "sacrum": 0.50,
    "femoral_head": 0.78,
    "scapula": 0.29,
    "humerus": 0.80,
    "clavicle": 0.72,
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
        #
        # The antero-posterior semi-axis is set from the half-width rather
        # than chosen freely, but it has to be set low enough that the arc
        # ends behind the sternum. At 0.72 the anterior end of a mid-thoracic
        # rib reached 114 mm in front of the vertebral body, past the front of
        # the chest wall, and on a lateral projection the ribs came out
        # sticking through the skin.
        azimuth = np.radians(10.0 + t * 118.0)
        depth = 0.50 * half_width
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


def _height_of(model, label: str, fallback: float) -> float:
    """Cranio-caudal position of one named vertebra, or ``fallback`` if absent."""
    labels = list(model.labels)
    if label not in labels:
        return fallback
    return float(model.centroids[labels.index(label), 2])


def _trunk_axis(model):
    """Where the middle of the trunk is at each height, following the spine.

    A scoliotic trunk is not a vertical barrel with a snake inside it. The
    chest wall and the abdomen are carried by the spine, so the body outline
    shifts with the curve; holding it straight and then centring the pelvis
    on the mean put L5 and the sacrum 56 mm apart in the phantom used for
    Figure 1, which is a dislocation, not a deformity.

    Smoothed over five levels, because soft tissue does not reproduce
    segmental detail, and flat outside the spine's own range so the pelvis
    and the neck sit under and over their nearest vertebra.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import uniform_filter1d

    order = np.argsort(model.centroids[:, 2])
    heights = model.centroids[order, 2]
    heights = heights + np.arange(len(heights)) * 1e-6  # strictly increasing
    lateral = uniform_filter1d(model.centroids[order, 0], size=5, mode="nearest")
    curve = PchipInterpolator(heights, lateral)
    low, high = float(heights[0]), float(heights[-1])
    return lambda h: float(curve(min(max(h, low), high)))


def _torso(model, volume, origin, spacing):
    """Torso, lungs, mediastinum, abdomen and arms, assigned slice by slice.

    Assigned rather than accumulated, because the lungs have to *remove*
    attenuation from the soft tissue they sit inside; the skeleton is added
    afterwards and overrides whatever it passes through.

    The thoracic and abdominal densities are separated by a domed boundary
    rather than a plane. A flat one draws a ruled line straight across the
    film, which is the single detail that gives a rendering away.
    """
    from scipy.interpolate import PchipInterpolator

    z = origin[2] + spacing * np.arange(volume.shape[2])
    top, bottom = model.centroids[:, 2].max(), model.centroids[:, 2].min()
    span = max(top - bottom, 1.0)

    axis_x = _trunk_axis(model)
    lung_apex = _height_of(model, "T2", top - 0.08 * span)
    diaphragm = _height_of(model, "T10", bottom + 0.34 * span)
    hip = _height_of(model, "L5", bottom)
    shoulder_z = _height_of(model, "T1", top)

    # Half-widths at the landmarks that give a trunk its outline: thighs,
    # hips, iliac crest, waist, chest, shoulders, neck. Each number is one an
    # anatomist can argue with, which is the point of listing them.
    #
    # Interpolated smoothly rather than linearly. Straight segments between
    # knots give the silhouette visible corners at the waist and shoulder,
    # and a body outline with corners in it is not a body outline. Pchip is
    # used because it passes through the knots without overshooting them into
    # a flare the anatomy does not have.
    knots = np.array(
        [hip - 210.0, hip - 110.0, hip + 30.0, diaphragm - 20.0,
         lung_apex - 46.0, lung_apex + 30.0, lung_apex + 130.0]
    )
    knots = np.maximum.accumulate(knots) + np.arange(7) * 1e-3
    width_of = PchipInterpolator(knots, np.array([132.0, 174.0, 166.0, 118.0, 148.0, 156.0, 56.0]))
    depth_of = PchipInterpolator(knots, np.array([60.0, 100.0, 112.0, 92.0, 106.0, 96.0, 58.0]))

    x = origin[0] + spacing * np.arange(volume.shape[0])
    y = origin[1] + spacing * np.arange(volume.shape[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    soft, dense = ATTENUATION["soft_tissue"], ATTENUATION["abdomen"]

    for k, height in enumerate(z):
        if not (knots[0] - 4.0) < height < (shoulder_z + 66.0):
            continue
        centre = axis_x(height)
        half_x = max(float(width_of(height)), 12.0)
        half_y = max(float(depth_of(height)), 12.0)
        inside = ((xx - centre) / half_x) ** 2 + ((yy + 6.0) / half_y) ** 2 <= 1.0

        # Distance from each lung's axis, in units of its own half-width. The
        # hemidiaphragm and the lung apex are domes, so the lung is shortest
        # over the dome and reaches lowest at the costophrenic recess.
        radial = np.minimum(
            ((xx - (centre - 62.0)) / 58.0) ** 2 + ((yy + 4.0) / 74.0) ** 2,
            ((xx - (centre + 62.0)) / 58.0) ** 2 + ((yy + 4.0) / 74.0) ** 2,
        )
        reach = np.clip(radial, 0.0, 1.8)
        floor = diaphragm + 12.0 - 70.0 * reach
        ceiling = lung_apex - 36.0 * reach

        below = 1.0 / (1.0 + np.exp(np.clip((height - floor) / 7.0, -30.0, 30.0)))
        slab = np.where(inside, soft + (dense - soft) * below, 0.0)

        thorax = inside & (height > floor) & (height < ceiling)
        slab = np.where(thorax & (radial <= 1.0), ATTENUATION["lung"], slab)
        heart = (
            ((xx - (centre + 30.0)) / 56.0) ** 2
            + ((yy - 22.0) / 48.0) ** 2
            + ((height - (diaphragm + 54.0)) / 74.0) ** 2
        ) <= 1.0
        column = np.abs(xx - centre) < 26.0
        slab = np.where(thorax & (column | heart), ATTENUATION["mediastinum"], slab)

        # Shoulders and the upper arms. On a frontal film they sit clear of
        # the spine; on a lateral one they lie directly in the beam, and the
        # upper thoracic bodies are read through them or not at all.
        shoulder_centre = axis_x(shoulder_z)
        arm = (
            np.minimum(
                ((xx - (shoulder_centre - 132.0)) / 56.0) ** 2,
                ((xx - (shoulder_centre + 132.0)) / 56.0) ** 2,
            )
            + ((yy + 6.0) / 70.0) ** 2
            + ((height - (shoulder_z - 62.0)) / 118.0) ** 2
        ) <= 1.0
        slab = np.where(arm, np.maximum(slab, soft), slab)
        volume[:, :, k] = slab


def _shoulder_girdle(model, volume, origin, spacing):
    """Humeral heads, shafts, scapulae and clavicles over the upper thorax.

    These are the reason the upper thoracic endplates are the hardest ones to
    place on a lateral film: T1 to about T4 are seen through the humeral head
    and the overlapping scapulae, and a reader who cannot see an endplate
    cannot choose it as an end vertebra. Leaving them out would make the
    lateral view look easier than it is, in a figure whose whole argument is
    about what each view can and cannot supply.
    """
    labels = list(model.labels)
    top_index = int(np.argmax(model.centroids[:, 2]))
    if "T1" in labels:
        top_index = labels.index("T1")
    origin_x = float(model.centroids[top_index, 0])
    origin_y = float(model.centroids[top_index, 1])
    origin_z = float(model.centroids[top_index, 2])
    root = np.array([origin_x, origin_y, origin_z])

    for side in (-1.0, 1.0):
        centre = root + np.array([side * 130.0, -8.0, -46.0])
        points = _local_grid(centre, (28.0, 28.0, 28.0), np.eye(3), spacing)
        offset = np.linalg.norm(points - centre, axis=1)
        _deposit(volume, origin, spacing, points[offset <= 25.0], ATTENUATION["trabecular"])
        _deposit(
            volume, origin, spacing,
            points[(offset <= 25.0) & (offset >= 21.5)],
            ATTENUATION["humerus"],
        )

        shaft = np.linspace(0.0, 1.0, 70)
        path = root + np.stack(
            [side * (130.0 + 7.0 * shaft), -8.0 + 5.0 * shaft, -66.0 - 122.0 * shaft],
            axis=1,
        )
        _tube(volume, origin, spacing, path, 12.0, ATTENUATION["trabecular"])
        _tube(volume, origin, spacing, path, 12.0, ATTENUATION["humerus"], inner=8.0)

        # Scapula: a thin plate behind the ribs, its medial border nearly
        # vertical and its lateral border running out to the acromion.
        t = np.linspace(0.0, 1.0, 44)
        for step in np.linspace(0.0, 1.0, 34):
            _deposit(
                volume, origin, spacing,
                root + np.stack(
                    [np.full_like(t, side * (44.0 + 88.0 * step)),
                     -58.0 - 16.0 * step + 6.0 * t,
                     -6.0 - 118.0 * t + 46.0 * step * (1.0 - t)],
                    axis=1,
                ),
                ATTENUATION["scapula"], thickness=3.0,
            )

        # Clavicle, an S from the sternal end out to the acromion.
        c = np.linspace(0.0, 1.0, 50)
        _tube(
            volume, origin, spacing,
            root + np.stack(
                [side * 126.0 * c, 54.0 - 62.0 * c**1.4, 14.0 - 44.0 * c**1.6],
                axis=1,
            ),
            5.0, ATTENUATION["clavicle"],
        )


def _pelvis(model, volume, origin, spacing):
    """Sacrum, iliac wings, pubic rami and the proximal femora.

    A scoliosis film is taken to include the pelvis -- the iliac crests carry
    the Risser grade and pelvic obliquity is read from them -- so a rendering
    that stops at L5 is not the picture a reader is looking at. It also gives
    the lower lumbar spine the dense background it actually has.

    The ilium is filled between its crest and the pelvic brim, and only as far
    forward as the anterior superior spine; carried round to the pubis it
    fills the obturator ring and the inlet, and the result is a paddle. Beyond
    the spine there is a pubic ramus and nothing else.
    """
    from scipy.interpolate import PchipInterpolator

    labels = list(model.labels)
    if "L5" not in labels:
        return
    index = labels.index("L5")

    # Under L5, because L5 and the sacrum articulate. Centring the pelvis on
    # the mean instead leaves them apart by however far the lumbar spine has
    # drifted -- 56 mm in the phantom used for Figure 1, which reads as a
    # dislocation. The chest wall follows the spine as well (see
    # :func:`_trunk_axis`), so the pelvis stays inside it.
    #
    # Level, not tilted with L5: pelvic obliquity is not modelled, and a
    # pelvis carrying the lumbar tilt would claim something this phantom does
    # not represent.
    base = model.centroids[index].copy()
    half_w = 0.5 * float(model.width_mm[index])
    body_height = float(model.height_mm[index])

    def curve(values, at):
        return PchipInterpolator([0.0, 0.4, 0.75, 1.0], values)(at)

    # Sacrum: a wedge continuing the column, narrowing and tipping backwards.
    for step in np.linspace(0.0, 1.0, 24):
        half = (half_w * (1.0 - 0.46 * step) + spacing, 10.0 + spacing, 3.0 + spacing)
        centre = base + np.array([0.0, -9.0 * step, -body_height / 2 - 6.0 - 66.0 * step])
        points = _local_grid(centre, half, np.eye(3), spacing)
        inner = (half[0] - spacing, half[1] - spacing, half[2])
        _deposit(
            volume, origin, spacing,
            points[_rounded_box(points - centre, inner, 3.0)],
            ATTENUATION["sacrum"],
        )

    # The ilium, from the sacroiliac joint to the anterior superior spine.
    u = np.linspace(0.0, 1.0, 72)
    lateral = half_w + curve([10.0, 98.0, 130.0, 112.0], u)
    forward = curve([-30.0, -4.0, 30.0, 68.0], u)
    crest = curve([14.0, 46.0, 30.0, -8.0], u)
    brim = curve([-26.0, -48.0, -62.0, -72.0], u)

    for side in (-1.0, 1.0):
        rail = np.column_stack([side * lateral, forward])
        # The fossa is thin bone and the margins are not; drawn at one density
        # the wing is a solid paddle, and drawn too thin it is a wire.
        for step in np.linspace(0.0, 1.0, 40):
            _deposit(
                volume, origin, spacing,
                base + np.column_stack([rail, brim + (crest - brim) * step]),
                ATTENUATION["iliac_wing"], thickness=3.5,
            )
        for margin, radius in ((crest, 5.0), (brim, 4.0)):
            _tube(
                volume, origin, spacing, base + np.column_stack([rail, margin]),
                radius, ATTENUATION["pelvic_rim"],
            )

    symphysis = np.array([0.0, 74.0, -86.0])
    for side in (-1.0, 1.0):
        spine = np.array([side * (half_w + 112.0), 68.0, -72.0])
        tuberosity = np.array([side * (half_w + 88.0), -6.0, -114.0])
        r = np.linspace(0.0, 1.0, 44)[:, None]

        # Superior pubic ramus, from the anterior spine in to the symphysis,
        # and the ischiopubic ramus closing the obturator ring beneath it.
        _tube(volume, origin, spacing,
              base + spine + (symphysis + np.array([side * 20.0, 0.0, 0.0]) - spine) * r,
              6.0, ATTENUATION["pelvic_rim"])
        _tube(volume, origin, spacing,
              base + symphysis + np.array([side * 20.0, 0.0, 0.0])
              + (tuberosity - symphysis - np.array([side * 20.0, 0.0, 0.0])) * r
              + np.array([0.0, 0.0, -18.0]) * np.sin(np.pi * r),
              6.0, ATTENUATION["sacrum"])

        # Acetabular body, carrying the wing down to the hip joint.
        drop = np.linspace(0.0, 1.0, 26)
        for offset in np.linspace(-24.0, 24.0, 16):
            _deposit(
                volume, origin, spacing,
                base + np.stack(
                    [side * (half_w + 92.0 + 6.0 * drop),
                     np.full_like(drop, 12.0 + offset),
                     -50.0 - 38.0 * drop],
                    axis=1,
                ),
                ATTENUATION["iliac_wing"], thickness=4.0,
            )

        head = base + np.array([side * (half_w + 94.0), 12.0, -84.0])
        points = _local_grid(head, (26.0, 26.0, 26.0), np.eye(3), spacing)
        offset = np.linalg.norm(points - head, axis=1)
        _deposit(volume, origin, spacing, points[offset <= 23.0], ATTENUATION["trabecular"])
        _deposit(
            volume, origin, spacing,
            points[(offset <= 23.0) & (offset >= 20.0)],
            ATTENUATION["femoral_head"],
        )

        # Neck and greater trochanter. Without them the head is a ball
        # floating below the pelvis rather than the top of a femur.
        neck = np.linspace(0.0, 1.0, 40)
        _tube(
            volume, origin, spacing,
            base + np.stack(
                [side * (half_w + 94.0 + 26.0 * neck), np.full_like(neck, 12.0),
                 -84.0 - 24.0 * neck],
                axis=1,
            ),
            11.0, ATTENUATION["trabecular"],
        )
        trochanter = base + np.array([side * (half_w + 120.0), 10.0, -110.0])
        points = _local_grid(trochanter, (18.0, 18.0, 18.0), np.eye(3), spacing)
        offset = np.linalg.norm(points - trochanter, axis=1)
        _deposit(volume, origin, spacing, points[offset <= 15.0], ATTENUATION["trabecular"])
        _deposit(
            volume, origin, spacing,
            points[(offset <= 15.0) & (offset >= 12.5)],
            ATTENUATION["pelvic_rim"],
        )


def _tube(volume, origin, spacing, path, radius, mu, inner: float = 0.0):
    """Add attenuation in a round tube along ``path``.

    A square kernel is fine for something a few millimetres across, but a
    humerus smeared with one comes out as a bar of metal down the side of the
    film, which is the sort of detail that makes a reader stop trusting the
    picture.

    ``inner`` leaves the middle of the tube alone, which is how a long bone is
    drawn: a dense cortex around a lucent medullary canal. Filled solid, a
    humerus comes out as black as a vertebral body and reads as hardware.
    """
    steps = np.arange(-radius, radius + spacing, spacing)
    offsets = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
    distance = np.linalg.norm(offsets, axis=1)
    offsets = offsets[(distance <= radius) & (distance >= inner)]
    points = (np.asarray(path)[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    _deposit(volume, origin, spacing, points, mu)


def _deposit(volume, origin, spacing, points, mu, thickness: float = 0.0):
    """Add attenuation at ``points``, optionally smeared into a box."""
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
    with_pelvis: bool = True,
    with_shoulders: bool = True,
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
    low = corners.min(axis=0) - np.array([200.0, 150.0, 200.0])
    high = corners.max(axis=0) + np.array([200.0, 150.0, 80.0])
    shape = np.ceil((high - low) / spacing_mm).astype(int) + 1
    volume = np.zeros(tuple(shape), dtype=np.float32)

    if with_soft_tissue:
        _torso(model, volume, low, spacing_mm)
    if with_pelvis:
        _pelvis(model, volume, low, spacing_mm)
    if with_shoulders:
        _shoulder_girdle(model, volume, low, spacing_mm)
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
