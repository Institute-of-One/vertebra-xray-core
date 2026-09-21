"""Frame conventions and the angle primitives everything else rests on."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vertebra_xray_core import geometry as geo


def test_math_frame_conversion_is_its_own_inverse():
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [120.5, 998.25]])
    for height in (None, 1000.0):
        assert np.allclose(geo.to_image_frame(geo.to_math_frame(pts, height), height), pts)


def test_math_frame_keeps_points_inside_the_image_when_height_is_given():
    pts = np.array([[10.0, 0.0], [10.0, 999.0]])
    flipped = geo.to_math_frame(pts, height=1000.0)
    assert flipped[0, 1] == 999.0
    assert flipped[1, 1] == 0.0


def test_endplate_tilt_ignores_which_way_round_the_corners_are():
    for degrees in (-89.0, -45.0, -1.0, 0.0, 1.0, 30.0, 60.0, 89.0):
        v = np.array([math.cos(math.radians(degrees)), math.sin(math.radians(degrees))])
        assert geo.tilt_deg(v) == pytest.approx(geo.tilt_deg(-v))
        assert geo.tilt_deg(v) == pytest.approx(degrees)


def test_tilt_folds_into_the_half_open_right_angle():
    for raw in np.arange(-720.0, 720.0, 7.5):
        folded = geo.wrap_to_signed_right_angle(raw)
        assert -90.0 < folded <= 90.0
        assert math.isclose((raw - folded) % 180.0, 0.0, abs_tol=1e-9) or math.isclose(
            (raw - folded) % 180.0, 180.0, abs_tol=1e-9
        )


def test_angle_between_stays_accurate_for_nearly_parallel_vectors():
    # The arccos-of-a-dot-product form loses most of its digits here, which is
    # exactly the regime a mild curve lives in.
    tiny = 1e-6
    u = np.array([1.0, 0.0])
    v = np.array([math.cos(tiny), math.sin(tiny)])
    assert geo.angle_between(u, v) == pytest.approx(math.degrees(tiny), rel=1e-9)


def test_angle_between_matches_in_three_dimensions():
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    assert geo.angle_between(u, v) == pytest.approx(90.0)
    assert geo.angle_between(u, -u) == pytest.approx(180.0)


def test_signed_angle_is_antisymmetric():
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    assert geo.signed_angle(u, v) == pytest.approx(90.0)
    assert geo.signed_angle(v, u) == pytest.approx(-90.0)


def test_signed_angle_rejects_three_dimensional_input():
    with pytest.raises(ValueError, match="2-D"):
        geo.signed_angle(np.zeros(3), np.zeros(3))


def test_projection_onto_a_plane_removes_only_the_normal_component():
    v = np.array([1.0, 2.0, 3.0])
    n = np.array([0.0, 0.0, 1.0])
    assert np.allclose(geo.project_onto_plane(v, n), [1.0, 2.0, 0.0])


def test_unit_leaves_a_zero_vector_alone_rather_than_dividing_by_zero():
    assert np.allclose(geo.unit(np.zeros(3)), np.zeros(3))
