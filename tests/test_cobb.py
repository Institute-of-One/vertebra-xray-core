"""Coronal measurement, checked against phantoms whose angles are exact."""

from __future__ import annotations

import numpy as np
import pytest

from vertebra_xray_core import cobb, phantom
from vertebra_xray_core import nomenclature as nom
from vertebra_xray_core.phantom import CurveSpec, synthetic_spine
from vertebra_xray_core.spine3d import Projection

PA = Projection(view="pa")


def _frontal(model):
    return model.project(PA)


def test_a_straight_spine_has_no_curves():
    model = synthetic_spine(curves=(CurveSpec("T1", "T12", 30.0, "kyphosis"),))
    assert cobb.cobb_angles(_frontal(model)).curves == ()


@pytest.mark.parametrize("prescribed", [10.0, 25.0, 45.0, 70.0])
def test_a_single_curve_measures_exactly_what_was_prescribed(prescribed):
    model = synthetic_spine(curves=(CurveSpec("T5", "T12", prescribed, "right"),))
    result = cobb.cobb_angles(_frontal(model))
    assert len(result.curves) == 1
    assert result.curves[0].angle_deg == pytest.approx(prescribed, abs=1e-6)


def test_a_double_curve_does_not_inflate_at_the_shared_end_vertebra():
    """Regression: superposing the two curves adds their tilts at T12.

    A phantom built that way reports 60 deg and 45 deg for a 45/30 pair, and
    every downstream validation silently inherits the error.
    """
    model = phantom.adolescent_idiopathic_scoliosis(
        main_thoracic_deg=45.0, lumbar_deg=30.0, proximal_thoracic_deg=20.0
    )
    angles = sorted(c.angle_deg for c in cobb.cobb_angles(_frontal(model)).curves)
    assert angles == pytest.approx([20.0, 30.0, 45.0], abs=1e-6)


def test_convexity_and_apex_come_out_on_the_right_side():
    model = phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=45.0, lumbar_deg=30.0)
    result = cobb.cobb_angles(_frontal(model))
    thoracic = result.by_name("MT")
    lumbar = result.by_name("L")
    assert thoracic.convexity == "right"
    assert lumbar.convexity == "left"
    assert nom.region_of(thoracic.apex_label) == "thoracic"
    assert nom.region_of(lumbar.apex_label) == "lumbar"
    assert thoracic.is_major


def test_a_non_standard_display_convention_mirrors_the_convexity():
    model = phantom.adolescent_idiopathic_scoliosis()
    flipped = cobb.cobb_angles(_frontal(model), right_on_image_left=False)
    assert flipped.by_name("MT").convexity == "left"


def test_the_two_definitions_agree_on_a_phantom_and_differ_in_general():
    model = phantom.adolescent_idiopathic_scoliosis()
    frontal = _frontal(model)
    srs = cobb.cobb_angles(frontal, definition="srs")
    axis = cobb.cobb_angles(frontal, definition="body_axis")
    # Both endplates of a phantom body share its orientation, so the two
    # definitions coincide exactly here. On real annotations they do not, and
    # that difference is a measurement choice rather than an algorithmic one.
    assert srs.max_angle_deg == pytest.approx(axis.max_angle_deg, abs=1e-6)


def test_small_curves_are_absorbed_rather_than_reported():
    model = synthetic_spine(
        curves=(CurveSpec("T2", "T6", 3.0, "right"), CurveSpec("T6", "T12", 40.0, "left"))
    )
    result = cobb.cobb_angles(_frontal(model), min_curve_deg=5.0)
    assert [c.angle_deg for c in result.curves] == pytest.approx([40.0], abs=0.5)


def test_all_pairs_max_never_undercuts_the_structural_maximum():
    model = phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=50.0, lumbar_deg=48.0)
    frontal = _frontal(model)
    structural = cobb.cobb_angles(frontal, definition="body_axis").max_angle_deg
    pairwise, _, _ = cobb.all_pairs_max(frontal)
    assert pairwise >= structural - 1e-9


def test_the_aasce_triplet_puts_the_maximum_in_the_middle_slot():
    model = phantom.adolescent_idiopathic_scoliosis()
    pt, mt, tl = cobb.aasce_triplet(_frontal(model))
    assert mt >= pt and mt >= tl


def test_a_lateral_view_is_refused_rather_than_measured_as_coronal():
    model = phantom.normal_adult_spine()
    with pytest.raises(ValueError, match="frontal view"):
        cobb.detect_curves(model.project(Projection(view="lateral")))


def test_the_tilt_profile_is_reported_for_auditing():
    model = phantom.adolescent_idiopathic_scoliosis()
    frontal = _frontal(model)
    result = cobb.cobb_angles(frontal)
    assert np.allclose(result.tilt_profile_deg, frontal.body_tilt())


def test_scoliosis_threshold_is_applied_as_documented():
    mild = synthetic_spine(curves=(CurveSpec("T5", "T12", 8.0, "right"),))
    marked = synthetic_spine(curves=(CurveSpec("T5", "T12", 22.0, "right"),))
    assert not cobb.cobb_angles(_frontal(mild)).curves[0].is_scoliotic
    assert cobb.cobb_angles(_frontal(marked)).curves[0].is_scoliotic
