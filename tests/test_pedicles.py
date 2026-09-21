"""Pedicles as the two landmarks that make a biplanar measurement determined."""

from __future__ import annotations

import numpy as np
import pytest

from vertebra_xray_core import geometry as geo
from vertebra_xray_core.pedicles import (
    NORMATIVE_PEDICLES_BY_LEVEL,
    PedicleGeometry,
    estimate_axial_rotation,
    normative_pedicles,
    project_pedicle_offsets,
    solve_orientation_with_pedicles,
)
from vertebra_xray_core.spine3d import _measured_tilts, endplate_normal, solve_orientation

LATERAL_SIGN = -1.0
PEDICLES = normative_pedicles("T8")

ORIENTATIONS = [
    (0.0, 0.0, 0.0),
    (20.0, -15.0, 0.0),
    (20.0, -15.0, 10.0),
    (-30.0, 22.0, -25.0),
    (35.0, -28.0, 30.0),
]


def _round_trip(theta, phi, psi, pedicles=PEDICLES):
    alpha, beta = _measured_tilts(theta, phi, psi, LATERAL_SIGN)
    offsets = project_pedicle_offsets(theta, phi, psi, pedicles)
    return solve_orientation_with_pedicles(alpha, beta, offsets, pedicles)


@pytest.mark.parametrize("theta,phi,psi", ORIENTATIONS)
def test_two_pedicles_make_the_inverse_problem_determined(theta, phi, psi):
    """The whole point: with them, nothing has to be assumed about rotation."""
    got_theta, got_phi, got_psi = _round_trip(theta, phi, psi)
    assert got_theta == pytest.approx(theta, abs=1e-6)
    assert got_phi == pytest.approx(phi, abs=1e-6)
    assert got_psi == pytest.approx(psi, abs=1e-6)


def test_the_solution_is_exact_across_the_whole_anatomical_range():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        theta, phi, psi = rng.uniform(-35, 35), rng.uniform(-30, 30), rng.uniform(-35, 35)
        got = _round_trip(theta, phi, psi)
        worst = max(
            worst,
            geo.angle_between(endplate_normal(got[0], got[1]), endplate_normal(theta, phi)),
            abs(got[2] - psi),
        )
    assert worst < 1e-5


@pytest.mark.parametrize("psi", [5.0, 10.0, 20.0, 30.0])
def test_without_pedicles_the_same_case_is_biased(psi):
    """The baseline the pedicles remove, stated as a test so it cannot drift."""
    theta, phi = 20.0, -15.0
    alpha, beta = _measured_tilts(theta, phi, psi, LATERAL_SIGN)
    naive = solve_orientation(alpha, beta, 0.0)
    naive_error = geo.angle_between(
        endplate_normal(*naive), endplate_normal(theta, phi)
    )
    with_pedicles = _round_trip(theta, phi, psi)
    assert naive_error > 0.35 * psi, "the bias grows with the rotation"
    assert (
        geo.angle_between(
            endplate_normal(with_pedicles[0], with_pedicles[1]),
            endplate_normal(theta, phi),
        )
        < 1e-6
    )


def test_rotation_moves_the_pedicle_midpoint_and_narrows_the_pair():
    """The two signals the inverse uses, in the direction the anatomy implies."""
    flat = project_pedicle_offsets(0.0, 0.0, 0.0, PEDICLES)
    assert 0.5 * (flat[0] + flat[1]) == pytest.approx(0.0, abs=1e-9)
    assert flat[1] - flat[0] == pytest.approx(2 * PEDICLES.half_separation, abs=1e-9)

    turned = project_pedicle_offsets(0.0, 0.0, 20.0, PEDICLES)
    midpoint = 0.5 * (turned[0] + turned[1])
    assert midpoint == pytest.approx(
        PEDICLES.posterior_offset * np.sin(np.radians(20.0)), abs=1e-9
    )
    assert turned[1] - turned[0] < flat[1] - flat[0]


def test_the_midpoint_is_the_well_conditioned_signal():
    """Near zero rotation the separation is flat and the midpoint is not.

    This is why the inverse leans on the midpoint: the separation varies as
    cos(psi), whose derivative vanishes at zero, so it carries almost no
    information about a small rotation.
    """
    small = 2.0
    flat = project_pedicle_offsets(0.0, 0.0, 0.0, PEDICLES)
    turned = project_pedicle_offsets(0.0, 0.0, small, PEDICLES)
    midpoint_change = abs(0.5 * (turned[0] + turned[1]) - 0.5 * (flat[0] + flat[1]))
    separation_change = abs((turned[1] - turned[0]) - (flat[1] - flat[0]))
    assert midpoint_change > 20 * separation_change


@pytest.mark.parametrize("psi", [-25.0, -8.0, 0.0, 12.0, 28.0])
def test_axial_rotation_alone_is_recoverable_from_the_pedicles(psi):
    offsets = project_pedicle_offsets(0.0, 0.0, psi, PEDICLES)
    assert estimate_axial_rotation(offsets, PEDICLES) == pytest.approx(psi, abs=1e-6)


def test_a_normative_table_is_good_enough():
    """Patient-specific pedicle geometry buys very little.

    Measured on phantoms across the anatomical range: taking the geometry from
    a table that is wrong by 2 mm in both parameters leaves a median residual
    of about a quarter of a degree, against the 5.5 degrees that ignoring
    rotation costs.
    """
    rng = np.random.default_rng(3)
    table = normative_pedicles("T8")
    residuals = []
    for _ in range(150):
        theta, phi, psi = rng.uniform(-35, 35), rng.uniform(-30, 30), rng.uniform(-30, 30)
        truth = PedicleGeometry(
            table.half_separation + rng.normal(0.0, 2.0),
            table.posterior_offset + rng.normal(0.0, 2.0),
        )
        alpha, beta = _measured_tilts(theta, phi, psi, LATERAL_SIGN)
        offsets = project_pedicle_offsets(theta, phi, psi, truth)
        got = solve_orientation_with_pedicles(alpha, beta, offsets, table)
        residuals.append(
            geo.angle_between(endplate_normal(got[0], got[1]), endplate_normal(theta, phi))
        )
    assert np.median(residuals) < 1.0
    assert np.quantile(residuals, 0.9) < 3.0


def test_the_normative_table_covers_the_thoracolumbar_spine_and_is_anatomical():
    assert set(NORMATIVE_PEDICLES_BY_LEVEL) >= {f"T{i}" for i in range(1, 13)}
    assert set(NORMATIVE_PEDICLES_BY_LEVEL) >= {f"L{i}" for i in range(1, 6)}
    for level, geometry in NORMATIVE_PEDICLES_BY_LEVEL.items():
        assert 8.0 < geometry.half_separation < 25.0, level
        assert 15.0 < geometry.posterior_offset < 35.0, level
    # The posterior offset, which carries the rotation signal, grows caudally.
    assert (
        NORMATIVE_PEDICLES_BY_LEVEL["L3"].posterior_offset
        > NORMATIVE_PEDICLES_BY_LEVEL["T1"].posterior_offset
    )


def test_an_unknown_level_falls_back_rather_than_failing():
    assert normative_pedicles("C4") == normative_pedicles(None)
