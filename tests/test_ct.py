"""Recovering vertebral orientation from a voxel mask.

The fixtures are voxelised vertebrae with posterior elements attached and a
prescribed orientation, so the recovered angles can be compared against the
number that was put in. That is what makes this testable without any CT: the
question is not whether a segmentation is right, it is whether the geometry
extracted from one is.

Everything goes through the same voxel path real data takes. An earlier
version measured point clouds directly, which let the tests pass against an
isolation method that then failed on every real vertebra.
"""

from __future__ import annotations

import numpy as np
import pytest

from vertebra_xray_core import ct
from vertebra_xray_core.spine3d import orientation_matrix

#: Fine enough that the fixture is not the limiting factor. At 0.9 mm the
#: phantom's 7 mm processes are only eight voxels across, they bridge to the
#: body where an oblique rasterisation stair-steps, and the recovered
#: orientation of a steeply tilted vertebra is out by tens of degrees -- an
#: artefact of the fixture, not of the method: the same case recovers to 1.2
#: degrees at 0.4 mm.
SPACING = 0.5


def _vertebra_points(
    theta: float,
    phi: float,
    psi: float,
    *,
    width: float = 45.0,
    depth: float = 32.0,
    height: float = 24.0,
    with_posterior_elements: bool = True,
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Patient-frame points of one vertebra at a prescribed orientation.

    The body is a box. The posterior elements are a pair of pedicles, a pair
    of transverse processes and a long spinous process, all of them slender --
    which is what the thickness-based isolation is built to remove, and what
    ruins a principal-axis fit that skips it.
    """
    rotation = orientation_matrix(theta, phi, psi)
    step = SPACING * 0.6

    def slab(x_range, y_range, z_range):
        # Sort each range: a mirrored part is written with its bounds the other
        # way round, and np.arange on a reversed range returns nothing at all,
        # which silently drops one side of the vertebra and biases every
        # measurement that depends on its symmetry.
        axes = [np.arange(min(a, b), max(a, b), step) for a, b in (x_range, y_range, z_range)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    half_w, half_d, half_h = width / 2, depth / 2, height / 2
    parts = [slab((-half_w, half_w), (-half_d, half_d), (-half_h, half_h))]
    if with_posterior_elements:
        # Absolute overlaps, not fractions of the body. Sizing the posterior
        # elements as fractions of the body left the transverse processes
        # barely grazing it, so whether they joined the body at all came down
        # to how the rasterisation happened to round -- and the isolated core
        # swung between 9% and 52% of the vertebra from one orientation to the
        # next. Each piece here overlaps its neighbour by several millimetres,
        # and every one of them is 7 to 9 mm thick: solidly attached, and
        # comfortably thinner than the erosion depth.
        back = -half_d
        for side in (-1.0, 1.0):
            parts.append(  # pedicle, overlapping the body by 5 mm
                slab((side * 14 - 4.5, side * 14 + 4.5), (back - 12, back + 5), (-6, 6))
            )
            parts.append(  # transverse process, springing from the pedicle
                slab((side * 10, side * 40), (back - 11, back - 4), (-4, 4))
            )
        parts.append(slab((-16, 16), (back - 20, back - 10), (-4, 4)))  # lamina
        parts.append(slab((-3.5, 3.5), (back - 36, back - 18), (-7, 7)))  # spinous process

    return np.concatenate(parts) @ rotation.T + np.asarray(centre, dtype=float)


def _rasterise(clouds: dict[int, np.ndarray], spacing: float = SPACING):
    """Turn patient-frame point clouds into a labelled volume and its RAS affine."""
    everything = np.concatenate(list(clouds.values()))
    low = everything.min(axis=0) - 3.0 * spacing
    shape = np.ceil((everything.max(axis=0) + 3.0 * spacing - low) / spacing).astype(int) + 1
    volume = np.zeros(shape, dtype=np.int16)
    for label, cloud in clouds.items():
        index = np.round((cloud - low) / spacing).astype(int)
        volume[index[:, 0], index[:, 1], index[:, 2]] = label

    # Patient frame to RAS+ flips the first axis; the affine maps voxel
    # indices to RAS millimetres.
    affine = np.eye(4)
    affine[:3, :3] = np.diag([-spacing, spacing, spacing])
    affine[:3, 3] = [-low[0], low[1], low[2]]
    return volume, affine


def _single(theta, phi, psi, **kwargs):
    volume, affine = _rasterise({1: _vertebra_points(theta, phi, psi, **kwargs)})
    return ct.vertebra_from_mask(volume, affine, 1)


#: Orientations spanning what a human spine actually does: coronal tilt up to
#: about 35 deg in a severe curve, sagittal tilt up to about 25 deg, and axial
#: rotation up to about 30 deg.
ORIENTATIONS = [
    (0.0, 0.0, 0.0),
    (15.0, -10.0, 0.0),
    (-25.0, 18.0, 0.0),
    (22.0, -12.0, 20.0),
    (-35.0, 25.0, -30.0),
]


@pytest.mark.parametrize("theta,phi,psi", ORIENTATIONS)
def test_orientation_is_recovered_from_a_voxelised_vertebra(theta, phi, psi):
    """Accurate to a fraction of a degree typically, a few degrees at the extremes.

    The residual at the corners of the range is voxelisation rather than the
    method: a vertebra rasterised at an oblique angle has stair-stepped
    surfaces, and the pedicles bridge to the body across them in places where
    an axis-aligned one would leave them separate.
    """
    got_theta, got_phi, got_psi = _single(theta, phi, psi).angles_deg
    assert got_theta == pytest.approx(theta, abs=2.0)
    assert got_phi == pytest.approx(phi, abs=2.0)
    assert got_psi == pytest.approx(psi, abs=4.0)


def test_typical_orientations_are_recovered_to_under_a_degree():
    for theta, phi, psi in [(0.0, 0.0, 0.0), (15.0, -10.0, 0.0), (22.0, -12.0, 20.0)]:
        got = _single(theta, phi, psi).angles_deg
        assert max(abs(a - b) for a, b in zip(got, (theta, phi, psi), strict=True)) < 1.0


def test_the_body_alone_cannot_give_an_axial_rotation():
    """Why the posterior elements are needed, stated as a failing alternative.

    A vertebral body is nearly as deep as it is wide -- on VerSe, width over
    depth runs from 0.8 to 1.3 through the thoracic spine -- so its two
    in-plane principal axes are degenerate and their eigenvectors are
    arbitrary. Orienting from the body alone therefore produces an axial
    rotation that can be 90 degrees out, and on real supine CT it was: a
    median of 27 degrees of apparent rotation where the truth is near zero.
    """
    theta, phi, psi = 6.0, -4.0, 0.0
    square = _vertebra_points(theta, phi, psi, width=36.0, depth=35.0,
                              with_posterior_elements=False)
    centred = square - square.mean(axis=0)
    spread = np.sort(np.linalg.eigvalsh(centred.T @ centred / len(centred)))
    # The two in-plane eigenvalues, largest and middle, are within a few per
    # cent of each other: their eigenvectors carry no information about which
    # way the vertebra faces.
    assert spread[2] / spread[1] < 1.12

    with_elements = _single(theta, phi, psi).angles_deg
    assert with_elements[2] == pytest.approx(psi, abs=3.0)


def test_axial_rotation_is_recovered_beyond_the_anatomical_range():
    """The posterior elements remove the degeneracy, not merely soften it."""
    for psi in (30.0, 45.0, 60.0):
        got = _single(10.0, -8.0, psi).angles_deg
        assert got[2] == pytest.approx(psi, abs=8.0)
        assert got[0] == pytest.approx(10.0, abs=2.0)


@pytest.mark.parametrize("theta,phi,psi", ORIENTATIONS)
def test_the_isolated_body_has_the_dimensions_it_was_built_with(theta, phi, psi):
    """Dimensions are approximate and only feed the projected corner model.

    The erosion correction assumes the body shrinks by the erosion depth on
    every side, which a rasterised box at an oblique angle does not do
    exactly. Angles do not depend on these numbers.
    """
    geometry = _single(theta, phi, psi)
    assert geometry.width_mm == pytest.approx(45.0, abs=8.0)
    assert geometry.depth_mm == pytest.approx(32.0, abs=17.0)
    assert geometry.height_mm == pytest.approx(24.0, abs=8.0)
    assert geometry.looks_like_a_vertebral_body


def test_the_posterior_elements_are_actually_removed():
    """The isolation must cut the body out, not merely trim its edges."""
    geometry = _single(0.0, 0.0, 0.0)
    assert geometry.body_voxels < 0.75 * geometry.voxels


def test_skipping_body_isolation_is_what_goes_wrong():
    """Regression, and the reason the isolation step exists.

    Measuring the whole vertebra rather than its body reports a vertebral body
    as deep as the spinous process is long. On VerSe this produced bodies 60
    to 80 mm deep against a true 20 to 40, which is not a tolerance question:
    every dimension downstream, and the projected corners built from them, is
    then describing a shape no vertebra has.
    """
    points = _vertebra_points(20.0, -15.0, 0.0)
    rotation = ct.orientation_from_points(points)
    local = (points - points.mean(axis=0)) @ rotation
    naive_depth = float(local[:, 1].max() - local[:, 1].min())

    isolated = _single(20.0, -15.0, 0.0)
    assert naive_depth > 60.0
    assert isolated.depth_mm < 0.75 * naive_depth


def test_a_body_only_mask_survives_isolation():
    """Nothing slender to remove means the body is kept, not eroded away."""
    geometry = _single(10.0, -8.0, 0.0, with_posterior_elements=False)
    assert geometry.looks_like_a_vertebral_body
    assert geometry.width_mm == pytest.approx(45.0, abs=6.0)


def test_a_thin_body_is_not_erased_by_the_erosion():
    """A collapsed vertebra must still be measurable; VerSe is full of them."""
    geometry = _single(0.0, 0.0, 0.0, height=11.0, with_posterior_elements=False)
    assert geometry.body_voxels > 0
    assert geometry.width_mm == pytest.approx(45.0, abs=6.0)


def test_the_centroid_lands_where_the_body_was_put():
    centre = (12.0, -5.0, 300.0)
    geometry = _single(18.0, -9.0, 12.0, centre=centre)
    assert np.allclose(geometry.centroid, centre, atol=3.0)


def test_implausible_proportions_are_rejected():
    geometry = _single(0.0, 0.0, 0.0, width=20.0, depth=18.0, height=62.0,
                       with_posterior_elements=False)
    assert not geometry.looks_like_a_vertebral_body


def test_the_ras_flip_only_touches_the_left_right_axis():
    ras = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert np.allclose(ct.ras_to_patient_frame(ras), [[-1.0, 2.0, 3.0], [4.0, 5.0, -6.0]])
    assert np.allclose(ct.ras_to_patient_frame(ct.ras_to_patient_frame(ras)), ras)


def test_isolation_handles_anisotropic_voxels():
    """Distance is measured in millimetres, so thick slices do not change the cut."""
    body = np.zeros((40, 40, 20), dtype=bool)
    body[8:32, 8:32, 4:16] = True
    body[18:22, 0:8, 8:12] = True  # a slender process sticking out
    core, used = ct.isolate_vertebral_body(body, (1.0, 1.0, 2.0), radius_mm=3.0)
    assert core.sum() > 0
    assert used == pytest.approx(3.0)
    assert not core[18:22, 0:8, 8:12].any(), "the process should have been eroded away"


# -- a whole spine ---------------------------------------------------------


def test_a_whole_segmentation_becomes_a_spine_model():
    angles = [(-20.0, -12.0, 0.0), (-8.0, -6.0, 5.0), (6.0, 2.0, 10.0), (18.0, 9.0, 5.0)]
    clouds = {
        8 + k: _vertebra_points(*a, centre=(0.0, 0.0, -34.0 * k)) for k, a in enumerate(angles)
    }
    volume, affine = _rasterise(clouds)
    names = {8: "T1", 9: "T2", 10: "T3", 11: "T4"}
    model, measured = ct.model_from_segmentation(volume, affine, names, drop_boundary=False)

    assert model.labels == ("T1", "T2", "T3", "T4")
    assert np.allclose(model.theta_deg, [a[0] for a in angles], atol=3.0)
    assert np.allclose(model.phi_deg, [a[1] for a in angles], atol=3.0)
    assert np.allclose(model.psi_deg, [a[2] for a in angles], atol=3.0)
    assert np.all(np.diff(model.centroids[:, 2]) < 0), "cranial first"
    assert set(measured) == set(names)


def test_labels_outside_the_mapping_are_ignored():
    clouds = {
        8 + k: _vertebra_points(0.0, 0.0, 0.0, centre=(0.0, 0.0, -34.0 * k)) for k in range(3)
    }
    volume, affine = _rasterise(clouds)
    model, _ = ct.model_from_segmentation(volume, affine, {8: "T1", 9: "T2"}, drop_boundary=False)
    assert model.labels == ("T1", "T2")


def test_too_few_usable_vertebrae_is_an_error_with_a_reason():
    clouds = {
        8 + k: _vertebra_points(0.0, 0.0, 0.0, centre=(0.0, 0.0, -34.0 * k)) for k in range(2)
    }
    volume, affine = _rasterise(clouds)
    with pytest.raises(ValueError, match="usable vertebrae"):
        ct.model_from_segmentation(volume, affine, {8: "T1"}, drop_boundary=False)
