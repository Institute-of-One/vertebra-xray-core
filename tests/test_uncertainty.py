"""Propagating landmark error into the reported angles."""

from __future__ import annotations

import numpy as np
import pytest

from vertebra_xray_core import cobb, phantom, uncertainty
from vertebra_xray_core.spine3d import Projection

PA = Projection(view="pa")


def _frontal():
    return phantom.adolescent_idiopathic_scoliosis(
        main_thoracic_deg=45.0, lumbar_deg=30.0
    ).project(PA)


def test_zero_noise_leaves_the_measurement_untouched():
    frontal = _frontal()
    results = uncertainty.bootstrap_cobb(frontal, sigma=0.0, resamples=20)
    for result in results:
        assert result.fixed.sd == pytest.approx(0.0, abs=1e-9)
        assert result.fixed.low == pytest.approx(result.fixed.point, abs=1e-9)
        assert result.fixed.high == pytest.approx(result.fixed.point, abs=1e-9)
        assert result.end_vertebra_stability == 1.0


def test_the_interval_brackets_the_point_estimate():
    results = uncertainty.bootstrap_cobb(_frontal(), sigma=1.0, resamples=200)
    for result in results:
        assert result.fixed.low <= result.fixed.point <= result.fixed.high
        assert result.fixed.mean == pytest.approx(result.fixed.point, abs=1.0)


def test_the_interval_widens_with_the_stated_localisation_error():
    frontal = _frontal()
    widths = []
    for sigma in (0.5, 1.0, 2.0, 4.0):
        result = uncertainty.bootstrap_cobb(frontal, sigma=sigma, resamples=200)[0]
        widths.append(result.fixed.high - result.fixed.low)
    assert widths == sorted(widths)


def test_the_bootstrap_is_reproducible_from_its_seed():
    frontal = _frontal()
    a = uncertainty.bootstrap_cobb(frontal, sigma=1.5, resamples=100, seed=7)
    b = uncertainty.bootstrap_cobb(frontal, sigma=1.5, resamples=100, seed=7)
    assert [x.fixed.sd for x in a] == [x.fixed.sd for x in b]


def test_the_sensitivity_matches_the_geometry_it_comes_from():
    """One millimetre of corner error is worth roughly three degrees of Cobb.

    An endplate line spans the body width, so a corner error of ``sigma``
    tilts it by about ``sqrt(2) sigma / width`` radians, and a Cobb angle
    combines two such lines. For a 45 mm thoracic body at 1 mm that is close
    to 3.6 deg, which is also the order of the inter-observer spread reported
    in reader studies -- so the noise model is at least the right size.
    """
    frontal = _frontal()
    result = uncertainty.bootstrap_cobb(frontal, sigma=1.0, resamples=400)[0]
    assert 1.5 < result.fixed.sd < 6.0


def test_letting_the_end_vertebrae_move_is_reported_separately():
    results = uncertainty.bootstrap_cobb(_frontal(), sigma=2.0, resamples=300)
    for result in results:
        assert result.redetected is not None
        assert 0.0 <= result.end_vertebra_stability <= 1.0
        assert result.end_vertebra_penalty_deg >= 0.0


def test_end_vertebra_choice_becomes_the_dominant_problem_as_noise_grows():
    frontal = _frontal()
    gentle = uncertainty.bootstrap_cobb(frontal, sigma=0.5, resamples=300)[0]
    harsh = uncertainty.bootstrap_cobb(frontal, sigma=4.0, resamples=300)[0]
    assert gentle.end_vertebra_stability > harsh.end_vertebra_stability


def test_jitter_returns_usable_landmarks_and_keeps_the_labels():
    frontal = _frontal()
    rng = np.random.default_rng(0)
    noisy = uncertainty.jitter(frontal, 1.0, rng)
    assert noisy.labels == frontal.labels
    assert noisy.view == frontal.view
    assert noisy.corners.shape == frontal.corners.shape
    assert not np.allclose(noisy.corners, frontal.corners)
    # still measurable, not a degenerate object
    assert cobb.cobb_angles(noisy).curves


def test_a_straight_spine_yields_no_intervals_rather_than_an_error():
    straight = phantom.normal_adult_spine().project(PA)
    assert uncertainty.bootstrap_cobb(straight, sigma=1.0, resamples=10) == ()
