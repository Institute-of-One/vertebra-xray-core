"""The 3-D model, its projections, and the biplanar inverse problem."""

from __future__ import annotations

import numpy as np
import pytest

from vertebra_xray_core import cobb, cobb3d, phantom
from vertebra_xray_core import geometry as geo
from vertebra_xray_core.spine3d import (
    Projection,
    axial_rotation_bias,
    endplate_normal,
    orientation_matrix,
    reconstruct_from_biplanar,
    rot_x,
    rot_y,
    rot_z,
    solve_orientation,
)

PA = Projection(view="pa")
LAT = Projection(view="lateral")


@pytest.mark.parametrize("rot", [rot_x, rot_y, rot_z])
@pytest.mark.parametrize("deg", [0.0, 17.0, -43.0, 90.0])
def test_rotations_are_proper_and_orthonormal(rot, deg):
    r = rot(deg)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


@pytest.mark.parametrize("psi", [0.0, 10.0, 35.0, -60.0])
def test_the_endplate_normal_does_not_depend_on_axial_rotation(psi):
    """The claim the whole 3-D argument rests on: R_z acts first, about z."""
    theta, phi = 22.0, -14.0
    full = orientation_matrix(theta, phi, psi)[:, 2]
    assert np.allclose(full, endplate_normal(theta, phi), atol=1e-12)


def test_a_positive_coronal_tilt_raises_the_patients_left_side():
    e = orientation_matrix(20.0, 0.0, 0.0)[:, 0]  # left-right body direction
    assert e[2] > 0, "the +X (patient left) end of the body should be higher"


def test_projecting_a_model_recovers_the_coronal_tilt_exactly():
    model = phantom.adolescent_idiopathic_scoliosis()
    assert np.allclose(model.project(PA).body_tilt(), model.theta_deg, atol=1e-9)


def test_the_lateral_view_couples_sagittal_tilt_to_coronal_tilt():
    """``tan(beta) = -cos(theta) tan(phi)``, derived in the module docstring.

    A pipeline that reads the lateral tilt straight off as the sagittal tilt
    is wrong by this factor, and the error grows with the scoliosis.
    """
    model = phantom.adolescent_idiopathic_scoliosis()
    beta = model.project(LAT).body_tilt()
    expected = -np.degrees(
        np.arctan(np.cos(np.radians(model.theta_deg)) * np.tan(np.radians(model.phi_deg)))
    )
    assert np.allclose(beta, expected, atol=1e-9)


@pytest.mark.parametrize("psi", [0.0, 15.0])
def test_biplanar_reconstruction_is_exact_when_axial_rotation_is_known(psi):
    model = phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=psi)
    rebuilt = reconstruct_from_biplanar(
        model.project(PA), model.project(LAT), axial_rotation_deg=np.full(len(model), psi)
    )
    assert np.allclose(rebuilt.theta_deg, model.theta_deg, atol=1e-8)
    assert np.allclose(rebuilt.phi_deg, model.phi_deg, atol=1e-8)
    assert np.allclose(rebuilt.endplate_normals, model.endplate_normals, atol=1e-8)


def test_solve_orientation_inverts_the_forward_projection():
    for theta, phi, psi in [(0.0, 0.0, 0.0), (25.0, -12.0, 0.0), (-33.0, 20.0, 18.0)]:
        model = phantom.synthetic_spine(
            curves=(), axial_rotation_deg=psi
        )  # geometry only; angles injected below
        del model
        from vertebra_xray_core.spine3d import _measured_tilts

        alpha, beta = _measured_tilts(theta, phi, psi, -1.0)
        got_theta, got_phi = solve_orientation(alpha, beta, psi)
        assert got_theta == pytest.approx(theta, abs=1e-6)
        assert got_phi == pytest.approx(phi, abs=1e-6)


def test_ignoring_axial_rotation_biases_the_reconstruction_monotonically():
    previous = 0.0
    for psi in (0.0, 5.0, 10.0, 20.0, 30.0):
        error = axial_rotation_bias(20.0, -15.0, psi)["normal_error_deg"]
        assert error >= previous - 1e-9
        previous = error
    assert axial_rotation_bias(20.0, -15.0, 0.0)["normal_error_deg"] == pytest.approx(0.0, abs=1e-9)
    assert axial_rotation_bias(20.0, -15.0, 20.0)["normal_error_deg"] > 5.0


def test_reconstruction_needs_both_views_labelled():
    model = phantom.adolescent_idiopathic_scoliosis()
    frontal = model.project(PA).with_labels(None)
    with pytest.raises(ValueError, match="labelled"):
        reconstruct_from_biplanar(frontal, model.project(LAT))


def test_reconstruction_matches_views_by_level_not_by_row():
    """A lateral view cropped at the top must still line up with the frontal one."""
    model = phantom.adolescent_idiopathic_scoliosis()
    frontal = model.project(PA)
    lateral = model.project(LAT)
    from vertebra_xray_core.landmarks import SpineLandmarks

    cropped = SpineLandmarks(
        corners=lateral.corners[3:],
        view="lateral",
        labels=lateral.labels[3:],
        source="cropped",
    )
    rebuilt = reconstruct_from_biplanar(frontal, cropped)
    assert rebuilt.labels == model.labels[3:]
    assert np.allclose(rebuilt.theta_deg, model.theta_deg[3:], atol=1e-6)


# -- three-dimensional angles ----------------------------------------------


def test_the_plane_of_maximum_curvature_is_coronal_for_a_purely_coronal_curve():
    model = phantom.synthetic_spine(curves=(phantom.CurveSpec("T5", "T12", 40.0, "right"),))
    result = cobb3d.measure_between_levels(model, "T5", "T12")
    assert result.pmc_from_coronal_deg == pytest.approx(0.0, abs=0.05)
    assert result.pmc_deg == pytest.approx(40.0, abs=1e-3)
    assert result.sagittal_deg == pytest.approx(0.0, abs=1e-6)


def test_the_plane_of_maximum_curvature_is_sagittal_for_a_purely_sagittal_curve():
    model = phantom.synthetic_spine(curves=(phantom.CurveSpec("T1", "T12", 35.0, "kyphosis"),))
    result = cobb3d.measure_between_levels(model, "T1", "T12")
    assert abs(result.pmc_from_sagittal_deg) == pytest.approx(0.0, abs=0.05)
    assert result.coronal_deg == pytest.approx(0.0, abs=1e-6)


def test_the_three_dimensional_angle_is_never_below_the_coronal_one():
    model = phantom.adolescent_idiopathic_scoliosis()
    frontal = model.project(PA)
    for result in cobb3d.cobb3d_for_curves(model, cobb.cobb_angles(frontal).curves):
        assert result.pmc_deg >= result.coronal_deg - 1e-9
        assert result.pmc_deg >= result.sagittal_deg - 1e-9
        assert result.coronal_underestimate_deg >= 0.0


def test_the_projected_angle_has_a_half_turn_period():
    model = phantom.adolescent_idiopathic_scoliosis()
    normals = model.endplate_normals
    for omega in (0.0, 31.0, 77.0):
        a = cobb3d.projected_angle_deg(normals[2], normals[-3], omega)
        b = cobb3d.projected_angle_deg(normals[2], normals[-3], omega + 180.0)
        assert float(a) == pytest.approx(float(b), abs=1e-9)


def test_a_lateral_film_under_reads_the_sagittal_angle_when_the_spine_is_scoliotic():
    """The coronal tilt swings the body's AP axis out of the sagittal plane."""
    model = phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=50.0)
    true_sagittal = cobb3d.measure_between_levels(model, "T1", "T12").sagittal_deg
    on_film = cobb3d.as_seen_on_film(model, "T1", "T12", LAT)
    assert on_film < true_sagittal
    straight = phantom.normal_adult_spine()
    assert cobb3d.as_seen_on_film(straight, "T1", "T12", LAT) == pytest.approx(
        cobb3d.measure_between_levels(straight, "T1", "T12").sagittal_deg, abs=1e-9
    )


def test_a_divergent_beam_leaves_the_coronal_tilt_alone_without_axial_rotation():
    """Cone-beam divergence cannot change a frontal Cobb angle when psi is zero.

    The chord a reader marks on a frontal film joins the left and right ends
    of an endplate. With no axial rotation that chord lies in a plane of
    constant depth, so both ends are magnified identically and the tilt is
    exactly invariant. Treating a long film as a parallel projection is
    therefore not an approximation here -- it is exact.
    """
    model = phantom.adolescent_idiopathic_scoliosis()
    parallel = model.project(PA).body_tilt()
    divergent = model.project(Projection(view="pa", sod_mm=1000.0, sdd_mm=1200.0)).body_tilt()
    assert np.allclose(parallel, divergent, atol=1e-9)


@pytest.mark.parametrize("psi,floor_deg", [(5.0, 0.5), (15.0, 2.0), (30.0, 4.0)])
def test_axial_rotation_lets_beam_divergence_into_the_coronal_measurement(psi, floor_deg):
    """Rotation swings the chord out of the constant-depth plane, and then it does.

    This is a second, independent route by which axial rotation corrupts a
    biplanar measurement: the first is that it biases the orientation solve,
    this one is that it makes the projection model itself wrong.
    """
    model = phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=psi)
    parallel = model.project(PA).body_tilt()
    divergent = model.project(Projection(view="pa", sod_mm=1000.0, sdd_mm=1200.0)).body_tilt()
    assert np.abs(parallel - divergent).max() > floor_deg


def test_projection_rejects_a_half_specified_beam_geometry():
    with pytest.raises(ValueError, match="both"):
        Projection(view="pa", sod_mm=1000.0)


def test_body_corners_form_a_box_of_the_right_size():
    model = phantom.normal_adult_spine()
    corners = model.body_corners()
    for i in range(len(model)):
        spans = corners[i].max(axis=0) - corners[i].min(axis=0)
        expected = np.array([model.width_mm[i], model.depth_mm[i], model.height_mm[i]])
        assert np.linalg.norm(spans) >= np.linalg.norm(expected) - 1e-9
        assert geo.angle_between(
            corners[i][7] - corners[i][6], model.rotations[i][:, 2]
        ) == pytest.approx(0.0, abs=1e-6)


# -- which plane the angle is measured in ----------------------------------


def test_the_unrestricted_maximum_over_all_planes_is_degenerate():
    """Why the family of measurement planes has to be restricted.

    For any two endplates that are not parallel there is a viewing direction
    that makes their traces perpendicular, so the unrestricted maximum is 90
    degrees for every curve and says nothing about the spine. "The largest
    Cobb angle over all planes" is therefore not a quantity; restricting to
    planes containing the cranio-caudal axis is what makes it one, and that
    family is exactly what a radiograph of a rotating standing patient can
    realise.
    """
    model = phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=50.0)
    normals = model.endplate_normals
    upper, lower = normals[4], normals[11]

    rng = np.random.default_rng(0)
    directions = rng.normal(size=(4000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    unrestricted = max(cobb3d.angle_in_plane(upper, lower, m) for m in directions)
    restricted, _ = cobb3d.plane_of_maximum_curvature(upper, lower)

    assert unrestricted > 89.0, "the unrestricted maximum saturates at a right angle"
    assert restricted < 60.0, "the restricted one stays a Cobb angle"


def test_the_classical_centroid_plane_agrees_with_the_dihedral_angle():
    """Peloux and Stagnara's plan d'election, against the angle between normals.

    The plane through the two end vertebrae's centroids and the apex is a
    different construction from maximising a projection, and it is worth
    knowing they measure nearly the same thing: on phantoms they agree to a
    tenth of a degree, and on 60 real curves to a median of 0.39.
    """
    model = phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=50.0, lumbar_deg=32.0)
    labels = list(model.labels)
    normals = model.endplate_normals
    for curve in cobb.cobb_angles(model.project(PA)).curves:
        if curve.apex_label in (curve.upper_label, curve.lower_label):
            continue
        plane = cobb3d.centroid_plane_normal(
            model, curve.upper_label, curve.apex_label, curve.lower_label
        )
        upper = normals[labels.index(curve.upper_label)]
        lower = normals[labels.index(curve.lower_label)]
        assert cobb3d.angle_in_plane(upper, lower, plane) == pytest.approx(
            geo.angle_between(upper, lower), abs=0.5
        )


def test_collinear_centroids_are_refused_rather_than_producing_a_plane():
    """A perfectly straight spine has no plane through three of its centroids."""
    from dataclasses import replace

    straight = phantom.synthetic_spine(curves=())
    straight = replace(
        straight,
        centroids=np.stack([np.array([0.0, 0.0, -30.0 * k]) for k in range(len(straight))]),
    )
    with pytest.raises(ValueError, match="collinear"):
        cobb3d.centroid_plane_normal(straight, "T2", "T6", "T10")
