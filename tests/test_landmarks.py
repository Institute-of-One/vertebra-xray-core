"""Corner canonicalisation, which is where a whole class of 90 deg bugs lives."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from vertebra_xray_core.landmarks import LL, LR, UL, UR, SpineLandmarks


def _stack(tilt_deg: float, *, width: float = 45.0, height: float = 27.0, n: int = 5):
    """A column of ``n`` identical vertebrae, each tilted by ``tilt_deg``.

    Built in ``image`` coordinates, where ``y`` grows downwards, so that the
    fixtures go through the same entry point real annotations do.

    Successive bodies are offset along their own axis rather than straight
    down the image, because that is how a spine stacks: a vertebra tilted 40
    deg sits in a column that is itself running at 40 deg. Offsetting straight
    down instead would build a shape no spine has, and the corner assignment
    for it is genuinely ambiguous.
    """
    t = math.radians(tilt_deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    box = np.array(
        [
            [-width / 2, -height / 2],
            [+width / 2, -height / 2],
            [-width / 2, +height / 2],
            [+width / 2, +height / 2],
        ]
    )
    quad = box @ rot.T
    step = np.array([0.0, height * 1.4]) @ rot.T
    return np.stack([quad + [200.0, 100.0] + k * step for k in range(n)])


def test_vertebrae_come_back_ordered_cranial_first():
    lm = SpineLandmarks.from_image_corners(_stack(0.0), "pa", height=1000.0)
    ys = lm.centroids[:, 1]
    assert np.all(np.diff(ys) < 0), "cranial-first means decreasing y in the math frame"


@pytest.mark.parametrize("tilt", [0.0, 10.0, 25.0, 31.0, 40.0, 55.0, -40.0])
def test_tilt_survives_past_the_angle_where_a_global_y_sort_breaks(tilt):
    """Regression: sorting corners on image ``y`` fails past atan(height/width).

    A 45 x 27 body flips its corner pairs at about 31 deg, and a naive
    canonicaliser then reports a tilt 90 deg away from the truth. Severely
    tilted vertebrae are the ones carrying the Cobb angle, so this must hold
    well past that point.
    """
    lm = SpineLandmarks.from_image_corners(_stack(tilt), "pa", height=1000.0)
    # image-frame tilt is positive clockwise, math-frame positive anticlockwise
    assert np.allclose(lm.superior_tilt(), -tilt)
    assert np.allclose(lm.inferior_tilt(), -tilt)
    assert np.allclose(lm.body_tilt(), -tilt)


def test_corner_order_in_the_file_does_not_matter():
    base = _stack(20.0, n=3)
    reference = SpineLandmarks.from_image_corners(base, "pa", height=1000.0)
    for permutation in itertools.permutations(range(4)):
        shuffled = base[:, permutation, :]
        got = SpineLandmarks.from_image_corners(shuffled, "pa", height=1000.0)
        assert np.allclose(got.corners, reference.corners)


def test_vertebra_order_in_the_file_does_not_matter_and_labels_follow():
    base = _stack(0.0, n=4)
    labels = ("T9", "T10", "T11", "T12")
    order = [2, 0, 3, 1]
    got = SpineLandmarks.from_image_corners(
        base[order], "pa", height=1000.0, labels=tuple(labels[i] for i in order)
    )
    assert got.labels == labels


def test_derived_geometry_matches_the_construction():
    width, height = 45.0, 27.0
    lm = SpineLandmarks.from_image_corners(_stack(0.0, width=width, height=height), "pa")
    assert np.allclose(lm.body_height, height)
    assert np.allclose(np.linalg.norm(lm.body_axis, axis=1), width)
    assert np.allclose(lm.corners[:, UR] - lm.corners[:, UL], lm.superior_endplate)
    assert np.allclose(lm.corners[:, LR] - lm.corners[:, LL], lm.inferior_endplate)


def test_labels_must_match_the_number_of_vertebrae():
    with pytest.raises(ValueError, match="labels for"):
        SpineLandmarks.from_image_corners(_stack(0.0, n=3), "pa", labels=("T1", "T2"))


def test_an_unknown_view_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown view"):
        SpineLandmarks.from_image_corners(_stack(0.0), "frontal")


def test_looking_up_a_label_on_unlabelled_landmarks_explains_itself():
    lm = SpineLandmarks.from_image_corners(_stack(0.0), "pa")
    with pytest.raises(ValueError, match="not been labelled"):
        lm.index_of_label("T12")
