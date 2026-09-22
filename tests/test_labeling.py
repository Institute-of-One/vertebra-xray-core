"""Training-free vertebral labelling from the sagittal curvature reversal."""

from __future__ import annotations

import numpy as np
import pytest

from vertebra_xray_core import labeling, phantom, uncertainty
from vertebra_xray_core import nomenclature as nom
from vertebra_xray_core.phantom import CurveSpec, synthetic_spine
from vertebra_xray_core.spine3d import Projection

LAT = Projection(view="lateral")


def _unlabelled_lateral(model):
    return model.project(LAT).with_labels(None)


def _ais_with_cervical(**kwargs):
    return synthetic_spine(
        levels=nom.span("C5", "L5"),
        curves=(
            CurveSpec("C5", "C7", 25.0, "lordosis"),
            CurveSpec("C7", "T12", 30.0, "kyphosis"),
            CurveSpec("T12", "L5", 45.0, "lordosis"),
            CurveSpec("T5", "T12", 45.0, "right"),
            CurveSpec("T12", "L4", 30.0, "left"),
        ),
        **kwargs,
    )


CASES = {
    "normal T1-L5": phantom.normal_adult_spine(),
    "normal C7-L5": phantom.normal_adult_spine(levels=nom.span("C7", "L5")),
    "AIS T1-L5": phantom.adolescent_idiopathic_scoliosis(),
    "AIS with cervical C5-L5": _ais_with_cervical(),
    "moderate AIS": phantom.adolescent_idiopathic_scoliosis(
        main_thoracic_deg=60.0, lumbar_deg=40.0
    ),
    "mildly rotated": phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=10.0),
}


@pytest.mark.parametrize("name", list(CASES))
def test_every_level_is_named_correctly_on_a_clean_spine(name):
    model = CASES[name]
    result = labeling.label_by_sagittal_inflection(_unlabelled_lateral(model))
    assert result.labels == model.labels


def test_the_anchor_lands_on_the_thoracolumbar_junction():
    model = phantom.normal_adult_spine()
    result = labeling.label_by_sagittal_inflection(_unlabelled_lateral(model))
    anchor = next(a for a in result.anchors if a.kind == "thoracolumbar")
    assert model.labels[anchor.index] == nom.THORACOLUMBAR_JUNCTION
    assert anchor.position == pytest.approx(anchor.index, abs=0.1)


def test_two_anchors_cross_check_the_thoracic_count():
    result = labeling.label_by_sagittal_inflection(_unlabelled_lateral(_ais_with_cervical()))
    assert {a.kind for a in result.anchors} == {"cervicothoracic", "thoracolumbar"}
    assert result.anchor_separation == 12
    assert result.is_self_consistent
    assert result.confidence == "high"
    assert result.warnings == ()


def test_a_missing_vertebra_is_reported_rather_than_absorbed():
    """Drop one thoracic body: the two anchors then disagree about the count."""
    model = _ais_with_cervical()
    lateral = _unlabelled_lateral(model)
    from vertebra_xray_core.landmarks import SpineLandmarks

    keep = [i for i in range(len(lateral)) if i != 8]
    gapped = SpineLandmarks(corners=lateral.corners[keep], view="lateral", source="gapped")
    result = labeling.label_by_sagittal_inflection(gapped)
    assert result.anchor_separation == 11
    assert not result.is_self_consistent
    assert any("apart" in w for w in result.warnings)


def test_a_flat_sagittal_profile_is_flagged_as_a_weak_anchor():
    model = synthetic_spine(
        curves=(
            CurveSpec("T1", "T12", 12.0, "kyphosis"),
            CurveSpec("T12", "L5", 50.0, "lordosis"),
            CurveSpec("T6", "T12", 55.0, "right"),
            CurveSpec("T12", "L4", 35.0, "left"),
        )
    )
    result = labeling.label_by_sagittal_inflection(_unlabelled_lateral(model))
    assert result.labels == model.labels, "still correct, just less trustworthy"
    assert result.confidence == "low"
    assert any("reverses by only" in w for w in result.warnings)


def test_a_spine_with_no_reversal_refuses_to_guess():
    model = synthetic_spine(curves=(CurveSpec("T5", "T12", 40.0, "right"),))
    with pytest.raises(ValueError, match="no curvature reversal"):
        labeling.label_by_sagittal_inflection(_unlabelled_lateral(model))


def test_the_frontal_view_is_refused_because_its_inflections_mean_nothing():
    model = phantom.adolescent_idiopathic_scoliosis()
    with pytest.raises(ValueError, match="lateral view"):
        labeling.label_by_sagittal_inflection(model.project(Projection(view="pa")))


def test_the_opposite_display_convention_is_handled():
    model = phantom.normal_adult_spine()
    mirrored = model.project(Projection(view="lateral", anterior_on_image_left=False)).with_labels(
        None
    )
    result = labeling.label_by_sagittal_inflection(mirrored, anterior_on_image_left=False)
    assert result.labels == model.labels


def test_labelling_from_a_known_reference_level_counts_outward():
    model = phantom.adolescent_idiopathic_scoliosis()
    lateral = _unlabelled_lateral(model)
    result = labeling.label_from_reference(lateral, len(lateral) - 1, "L5")
    assert result.labels == model.labels


def test_counting_off_the_end_of_the_spine_is_reported():
    model = phantom.adolescent_idiopathic_scoliosis()
    lateral = _unlabelled_lateral(model)
    below = labeling.label_from_reference(lateral, 0, "L3")
    assert any("below S1" in w for w in below.warnings)
    above = labeling.label_from_reference(lateral, len(lateral) - 1, "C3")
    assert any("above C1" in w for w in above.warnings)


# -- axial rotation --------------------------------------------------------


@pytest.mark.parametrize("psi", [0.0, 5.0, 10.0])
def test_a_single_lateral_film_tolerates_mild_axial_rotation(psi):
    model = phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=psi)
    result = labeling.label_by_sagittal_inflection(_unlabelled_lateral(model))
    assert result.labels == model.labels


@pytest.mark.parametrize("psi", [20.0, 30.0])
def test_heavy_axial_rotation_is_declared_unreliable_rather_than_answered_wrongly(psi):
    """The failure mode that matters is a confident wrong answer, not a wrong one.

    Rotation mixes the coronal profile into the apparent sagittal one, and
    past about 20 degrees the anchor migrates to the coronal curve's turning
    point. What the method must do is notice, by whichever of its checks
    happens to fire.
    """
    model = phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=psi)
    result = labeling.label_by_sagittal_inflection(_unlabelled_lateral(model))
    assert result.labels != model.labels, "the premise of this test"
    assert result.confidence == "low"
    assert result.warnings


def test_a_severe_coronal_curve_defeats_a_single_lateral_film():
    """Coronal tilt compresses the apparent sagittal profile and shifts the anchor.

    This is a limitation of the film, not of the estimator, which is why
    reconstructing first fixes it.
    """
    model = phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=90.0)
    single = labeling.label_by_sagittal_inflection(_unlabelled_lateral(model))
    assert single.labels != model.labels
    both = labeling.label_biplanar(
        model.project(Projection(view="pa")).with_labels(None), _unlabelled_lateral(model)
    )
    assert both.labels == model.labels
    assert any("distorted by coronal tilt" in w for w in both.warnings)


@pytest.mark.parametrize("main", [45.0, 60.0, 75.0, 90.0])
@pytest.mark.parametrize("psi", [0.0, 10.0, 20.0, 30.0])
def test_the_two_pass_biplanar_route_holds_across_the_whole_grid(main, psi):
    model = phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=main, axial_rotation_deg=psi)
    result = labeling.label_biplanar(
        model.project(Projection(view="pa")).with_labels(None),
        _unlabelled_lateral(model),
        axial_rotation_deg=np.full(len(model), psi),
    )
    assert result.labels == model.labels


def test_biplanar_labelling_needs_the_two_views_cropped_alike():
    model = phantom.adolescent_idiopathic_scoliosis()
    from vertebra_xray_core.landmarks import SpineLandmarks

    lateral = _unlabelled_lateral(model)
    short = SpineLandmarks(corners=lateral.corners[2:], view="lateral", source="short")
    with pytest.raises(ValueError, match="same levels"):
        labeling.label_biplanar(model.project(Projection(view="pa")).with_labels(None), short)


def test_an_impossible_level_count_is_reported_as_such():
    """T12 placed near the top of a long film would put S1 six bodies too low."""
    model = phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=20.0)
    result = labeling.label_by_sagittal_inflection(_unlabelled_lateral(model))
    assert result.count_overflow
    assert result.confidence == "low"


@pytest.mark.parametrize("psi", [0.0, 10.0, 20.0, 30.0, 40.0])
def test_knowing_the_axial_rotation_restores_the_anchor_completely(psi):
    from vertebra_xray_core.spine3d import reconstruct_from_biplanar

    model = phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=psi)
    rebuilt = reconstruct_from_biplanar(
        model.project(Projection(view="pa")),
        model.project(LAT),
        axial_rotation_deg=np.full(len(model), psi),
    )
    result = labeling.label_from_model(rebuilt)
    assert result.labels == model.labels
    anchor = next(a for a in result.anchors if a.kind == "thoracolumbar")
    assert not anchor.is_weak


def test_labels_transfer_between_the_two_views():
    model = phantom.adolescent_idiopathic_scoliosis()
    lateral = _unlabelled_lateral(model)
    labelled = labeling.label_by_sagittal_inflection(lateral).apply(lateral)
    frontal = labeling.transfer_labels(labelled, model.project(Projection(view="pa")))
    assert frontal.labels == model.labels


def test_transferring_across_a_mismatched_crop_is_an_error_not_a_guess():
    model = phantom.adolescent_idiopathic_scoliosis()
    lateral = _unlabelled_lateral(model)
    labelled = labeling.label_by_sagittal_inflection(lateral).apply(lateral)
    with pytest.raises(ValueError, match="source has"):
        labeling.transfer_labels(labelled, model.project(Projection(view="pa")), offset=3)


@pytest.mark.slow
def test_the_anchor_survives_realistic_landmark_noise():
    """A method that is right on clean data and unstable under noise is not usable."""
    model = phantom.adolescent_idiopathic_scoliosis()
    lateral = _unlabelled_lateral(model)
    stability = uncertainty.labelling_stability(lateral, sigma=1.0, resamples=300)
    # Measured at about 0.75 exact and 0.95 within a level on this phantom.
    assert stability["exact"] > 0.65
    assert stability["within_one_level"] > 0.90
    assert stability["no_anchor"] == 0.0


@pytest.mark.slow
def test_a_weak_anchor_really_is_less_stable_than_a_strong_one():
    """The confidence field has to track something measurable, or it is decoration."""
    strong = _unlabelled_lateral(phantom.normal_adult_spine())
    weak = _unlabelled_lateral(
        synthetic_spine(
            curves=(
                CurveSpec("T1", "T12", 12.0, "kyphosis"),
                CurveSpec("T12", "L5", 50.0, "lordosis"),
            )
        )
    )
    strong_rate = uncertainty.labelling_stability(strong, 2.0, resamples=300)["exact"]
    weak_rate = uncertainty.labelling_stability(weak, 2.0, resamples=300)["exact"]
    assert strong_rate > weak_rate
    assert labeling.label_by_sagittal_inflection(weak).confidence == "low"


def test_the_reported_profile_is_the_one_that_was_analysed():
    model = phantom.normal_adult_spine()
    lateral = _unlabelled_lateral(model)
    profile = labeling._sagittal_profile(lateral, anterior_on_image_left=True)
    assert np.allclose(profile, lateral.body_tilt())
