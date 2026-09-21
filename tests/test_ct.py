"""Recovering vertebral orientation from a voxel mask.

The fixtures here are voxelised vertebrae with posterior elements attached and
a prescribed orientation, so the recovered angles can be compared against the
number that was put in. That is what makes this testable without any CT: the
question is not whether a segmentation is right, it is whether the geometry
extracted from one is.
"""

from __future__ import annotations

import numpy as np
import pytest

from vertebra_xray_core import ct
from vertebra_xray_core.spine3d import euler_from_matrix, orientation_matrix

VOXEL_MM = 0.8


def _voxelise(
    theta: float,
    phi: float,
    psi: float,
    *,
    width: float = 45.0,
    depth: float = 30.0,
    height: float = 22.0,
    with_posterior_elements: bool = True,
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Patient-frame points of one vertebra at a prescribed orientation.

    The body is a box. The posterior elements are a pair of pedicles and a
    long spinous process, all of them behind the body and slender -- which is
    what creates the cross-sectional pinch the isolation step looks for, and
    what wrecks a principal-axis fit that skips it.
    """
    rotation = orientation_matrix(theta, phi, psi)

    def box(cx, cy, cz, sx, sy, sz):
        xs = np.arange(-sx / 2, sx / 2, VOXEL_MM)
        ys = np.arange(-sy / 2, sy / 2, VOXEL_MM)
        zs = np.arange(-sz / 2, sz / 2, VOXEL_MM)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        return grid + np.array([cx, cy, cz])

    parts = [box(0.0, 0.0, 0.0, width, depth, height)]
    if with_posterior_elements:
        for side in (-1.0, 1.0):
            parts.append(box(side * 0.30 * width, -0.62 * depth, 0.0, 0.16 * width, 0.30 * depth, 0.55 * height))
        parts.append(box(0.0, -1.15 * depth, -0.15 * height, 0.10 * width, 0.75 * depth, 0.45 * height))
        parts.append(box(0.0, -0.80 * depth, 0.0, 0.85 * width, 0.14 * depth, 0.40 * height))

    local = np.concatenate(parts)
    return local @ rotation.T + np.asarray(centre, dtype=float)


ORIENTATIONS = [
    (0.0, 0.0, 0.0),
    (15.0, -10.0, 0.0),
    (-25.0, 18.0, 0.0),
    (22.0, -12.0, 20.0),
    (-35.0, 25.0, -30.0),
]


@pytest.mark.parametrize("theta,phi,psi", ORIENTATIONS)
def test_orientation_is_recovered_from_a_voxelised_vertebra(theta, phi, psi):
    points = _voxelise(theta, phi, psi)
    geometry = ct.vertebra_geometry(points, label=1)
    got_theta, got_phi, got_psi = geometry.angles_deg
    assert got_theta == pytest.approx(theta, abs=2.0)
    assert got_phi == pytest.approx(phi, abs=2.0)
    assert got_psi == pytest.approx(psi, abs=2.0)


@pytest.mark.parametrize("theta,phi,psi", ORIENTATIONS)
def test_the_isolated_body_has_the_dimensions_it_was_built_with(theta, phi, psi):
    geometry = ct.vertebra_geometry(_voxelise(theta, phi, psi), label=1)
    assert geometry.width_mm == pytest.approx(45.0, abs=3.0)
    assert geometry.depth_mm == pytest.approx(30.0, abs=4.0)
    assert geometry.height_mm == pytest.approx(22.0, abs=3.0)
    assert geometry.looks_like_a_vertebral_body


def test_skipping_body_isolation_is_what_goes_wrong():
    """Regression, and the reason the isolation step exists.

    Orienting the whole vertebra by its principal axes puts the
    antero-posterior axis down the spinous process, and the recovered angles
    are wrong by far more than any measurement tolerance.
    """
    theta, phi, psi = 20.0, -15.0, 0.0
    points = _voxelise(theta, phi, psi)
    naive = euler_from_matrix(ct.orientation_from_points(points))
    careful = ct.vertebra_geometry(points, label=1).angles_deg
    naive_error = max(abs(a - b) for a, b in zip(naive, (theta, phi, psi), strict=True))
    careful_error = max(abs(a - b) for a, b in zip(careful, (theta, phi, psi), strict=True))
    assert careful_error < 2.0
    assert naive_error > 3 * careful_error


def test_a_body_only_mask_is_left_alone():
    """Nothing to pinch means nothing is cut, rather than an arbitrary cut."""
    points = _voxelise(10.0, -8.0, 0.0, with_posterior_elements=False)
    keep = ct.isolate_vertebral_body(points)
    assert keep.all()


def test_the_centroid_lands_where_the_body_was_put():
    centre = (12.0, -5.0, 300.0)
    geometry = ct.vertebra_geometry(_voxelise(18.0, -9.0, 12.0, centre=centre), label=1)
    assert np.allclose(geometry.centroid, centre, atol=2.0)


def test_implausible_proportions_are_rejected():
    flat = _voxelise(0.0, 0.0, 0.0, width=20.0, depth=18.0, height=60.0, with_posterior_elements=False)
    assert not ct.vertebra_geometry(flat, label=1).looks_like_a_vertebral_body


def test_the_ras_flip_only_touches_the_left_right_axis():
    ras = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert np.allclose(ct.ras_to_patient_frame(ras), [[-1.0, 2.0, 3.0], [4.0, 5.0, -6.0]])
    assert np.allclose(ct.ras_to_patient_frame(ct.ras_to_patient_frame(ras)), ras)


# -- a whole spine ---------------------------------------------------------


def _voxel_volume(model_angles, spacing=1.0):
    """Rasterise a short stack of vertebrae into a labelled volume plus affine."""
    clouds = {}
    for k, (theta, phi, psi) in enumerate(model_angles):
        clouds[k + 8] = _voxelise(theta, phi, psi, centre=(0.0, 0.0, -32.0 * k))

    everything = np.concatenate(list(clouds.values()))
    lo = everything.min(axis=0) - 4.0
    shape = np.ceil((everything.max(axis=0) + 4.0 - lo) / spacing).astype(int) + 1
    volume = np.zeros(shape, dtype=np.int16)
    for label, cloud in clouds.items():
        idx = np.round((cloud - lo) / spacing).astype(int)
        volume[idx[:, 0], idx[:, 1], idx[:, 2]] = label

    # Patient frame back to RAS+ is the same flip, and the affine maps voxel
    # indices to RAS millimetres.
    affine = np.eye(4)
    affine[:3, :3] = np.diag([-spacing, spacing, spacing])
    affine[:3, 3] = [-lo[0], lo[1], lo[2]]
    return volume, affine


def test_a_whole_segmentation_becomes_a_spine_model():
    angles = [(-20.0, -12.0, 0.0), (-8.0, -6.0, 5.0), (6.0, 2.0, 10.0), (18.0, 9.0, 5.0)]
    volume, affine = _voxel_volume(angles)
    names = {8: "T1", 9: "T2", 10: "T3", 11: "T4"}
    model, measured = ct.model_from_segmentation(volume, affine, names, drop_boundary=False)

    assert model.labels == ("T1", "T2", "T3", "T4")
    assert np.allclose(model.theta_deg, [a[0] for a in angles], atol=2.5)
    assert np.allclose(model.phi_deg, [a[1] for a in angles], atol=2.5)
    assert np.allclose(model.psi_deg, [a[2] for a in angles], atol=2.5)
    assert np.all(np.diff(model.centroids[:, 2]) < 0), "cranial first"
    assert set(measured) == set(names)


def test_labels_outside_the_mapping_are_ignored():
    volume, affine = _voxel_volume([(0.0, 0.0, 0.0), (5.0, -5.0, 0.0), (10.0, -8.0, 0.0)])
    model, _ = ct.model_from_segmentation(
        volume, affine, {8: "T1", 9: "T2"}, drop_boundary=False
    )
    assert model.labels == ("T1", "T2")


def test_too_few_usable_vertebrae_is_an_error_with_a_reason():
    volume, affine = _voxel_volume([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="usable vertebrae"):
        ct.model_from_segmentation(volume, affine, {8: "T1"}, drop_boundary=False)
