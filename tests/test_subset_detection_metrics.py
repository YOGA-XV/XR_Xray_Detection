import pytest

from scripts.subset_detection_metrics import (
    Box,
    average_precision,
    box_iou,
    evaluate_subset,
    is_large_box,
    is_medium_box,
    is_overlap_box,
    is_small_box,
    is_thin_box,
    match_predictions,
)


def test_box_iou_uses_xywh_normalized_boxes():
    box_a = Box(cls=0, xc=0.5, yc=0.5, w=0.4, h=0.4, conf=1.0, image_id="a")
    box_b = Box(cls=0, xc=0.5, yc=0.5, w=0.2, h=0.2, conf=1.0, image_id="a")

    assert box_iou(box_a, box_b) == pytest.approx(0.25)


def test_subset_definitions_match_protocol():
    assert is_small_box(Box(cls=0, xc=0.5, yc=0.5, w=0.09, h=0.10, conf=1.0, image_id="a"))
    assert not is_small_box(Box(cls=0, xc=0.5, yc=0.5, w=0.10, h=0.10, conf=1.0, image_id="a"))
    assert is_medium_box(Box(cls=0, xc=0.5, yc=0.5, w=0.10, h=0.10, conf=1.0, image_id="a"))
    assert is_medium_box(Box(cls=0, xc=0.5, yc=0.5, w=0.20, h=0.20, conf=1.0, image_id="a"))
    assert not is_medium_box(Box(cls=0, xc=0.5, yc=0.5, w=0.25, h=0.20, conf=1.0, image_id="a"))
    assert is_large_box(Box(cls=0, xc=0.5, yc=0.5, w=0.25, h=0.20, conf=1.0, image_id="a"))
    assert is_thin_box(Box(cls=0, xc=0.5, yc=0.5, w=0.31, h=0.10, conf=1.0, image_id="a"))
    assert not is_thin_box(Box(cls=0, xc=0.5, yc=0.5, w=0.30, h=0.10, conf=1.0, image_id="a"))


def test_overlap_definition_uses_max_intersection_over_target_area():
    target = Box(cls=0, xc=0.5, yc=0.5, w=0.4, h=0.4, conf=1.0, image_id="a")
    overlapping = Box(cls=1, xc=0.6, yc=0.5, w=0.4, h=0.4, conf=1.0, image_id="a")
    separate = Box(cls=1, xc=0.1, yc=0.1, w=0.1, h=0.1, conf=1.0, image_id="a")

    assert is_overlap_box(target, [target, overlapping, separate], threshold=0.3)
    assert not is_overlap_box(target, [target, separate], threshold=0.3)


def test_evaluate_subset_supports_overlap_threshold():
    gt = [
        Box(cls=0, xc=0.5, yc=0.5, w=0.4, h=0.4, conf=1.0, image_id="a"),
        Box(cls=1, xc=0.6, yc=0.5, w=0.4, h=0.4, conf=1.0, image_id="a"),
        Box(cls=0, xc=0.1, yc=0.1, w=0.1, h=0.1, conf=1.0, image_id="b"),
    ]
    predictions = [
        Box(cls=0, xc=0.5, yc=0.5, w=0.4, h=0.4, conf=0.9, image_id="a"),
        Box(cls=1, xc=0.6, yc=0.5, w=0.4, h=0.4, conf=0.8, image_id="a"),
        Box(cls=0, xc=0.1, yc=0.1, w=0.1, h=0.1, conf=0.7, image_id="b"),
    ]

    result = evaluate_subset(predictions, gt, "overlap", iou_threshold=0.5, overlap_threshold=0.3)

    assert result["gt_count"] == 2
    assert result["prediction_count"] == 2
    assert result["ap"] == 1.0
    assert result["recall"] == 1.0


def test_match_predictions_sorts_by_confidence_and_matches_once():
    gt = [Box(cls=0, xc=0.5, yc=0.5, w=0.2, h=0.2, conf=1.0, image_id="a")]
    predictions = [
        Box(cls=0, xc=0.1, yc=0.1, w=0.2, h=0.2, conf=0.9, image_id="a"),
        Box(cls=0, xc=0.5, yc=0.5, w=0.2, h=0.2, conf=0.8, image_id="a"),
        Box(cls=0, xc=0.5, yc=0.5, w=0.2, h=0.2, conf=0.7, image_id="a"),
    ]

    matches = match_predictions(predictions, gt, iou_threshold=0.5)

    assert matches.true_positives == [0, 1, 0]
    assert matches.false_positives == [1, 0, 1]
    assert matches.confidences == [0.9, 0.8, 0.7]


def test_average_precision_is_one_for_single_perfect_detection():
    gt = [Box(cls=0, xc=0.5, yc=0.5, w=0.2, h=0.2, conf=1.0, image_id="a")]
    predictions = [Box(cls=0, xc=0.5, yc=0.5, w=0.2, h=0.2, conf=0.9, image_id="a")]

    matches = match_predictions(predictions, gt, iou_threshold=0.5)

    assert average_precision(matches, gt_count=1) == 1.0
