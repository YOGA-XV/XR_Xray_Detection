from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    cls: int
    xc: float
    yc: float
    w: float
    h: float
    conf: float
    image_id: str


@dataclass(frozen=True)
class MatchResult:
    true_positives: list[int]
    false_positives: list[int]
    confidences: list[float]


def is_small_box(box: Box) -> bool:
    return box.w * box.h < 0.01


def is_medium_box(box: Box) -> bool:
    area = box.w * box.h
    return 0.01 <= area < 0.05


def is_large_box(box: Box) -> bool:
    return box.w * box.h >= 0.05


def is_thin_box(box: Box) -> bool:
    if box.w <= 0 or box.h <= 0:
        return False
    return max(box.w, box.h) / min(box.w, box.h) > 3.0


def xyxy(box: Box) -> tuple[float, float, float, float]:
    return (
        box.xc - box.w / 2,
        box.yc - box.h / 2,
        box.xc + box.w / 2,
        box.yc + box.h / 2,
    )


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = a.w * a.h + b.w * b.h - inter
    if union <= 0:
        return 0.0
    return inter / union


def intersection_over_target_area(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    target_area = a.w * a.h
    if target_area <= 0:
        return 0.0
    return (inter_w * inter_h) / target_area


def is_overlap_box(box: Box, boxes: list[Box], threshold: float) -> bool:
    for other in boxes:
        if other is box:
            continue
        if other.image_id != box.image_id:
            continue
        if intersection_over_target_area(box, other) >= threshold:
            return True
    return False


def match_predictions(predictions: list[Box], ground_truth: list[Box], iou_threshold: float) -> MatchResult:
    sorted_predictions = sorted(predictions, key=lambda box: box.conf, reverse=True)
    matched_gt: set[int] = set()
    true_positives: list[int] = []
    false_positives: list[int] = []
    confidences: list[float] = []

    for pred in sorted_predictions:
        best_index = None
        best_iou = 0.0
        for index, gt in enumerate(ground_truth):
            if index in matched_gt:
                continue
            if pred.image_id != gt.image_id or pred.cls != gt.cls:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        confidences.append(pred.conf)
        if best_index is not None and best_iou >= iou_threshold:
            matched_gt.add(best_index)
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    return MatchResult(true_positives=true_positives, false_positives=false_positives, confidences=confidences)


def average_precision(matches: MatchResult, gt_count: int) -> float:
    if gt_count == 0:
        return 0.0
    cumulative_tp = 0
    cumulative_fp = 0
    precisions = [1.0]
    recalls = [0.0]

    for tp, fp in zip(matches.true_positives, matches.false_positives):
        cumulative_tp += tp
        cumulative_fp += fp
        precision = cumulative_tp / max(cumulative_tp + cumulative_fp, 1)
        recall = cumulative_tp / gt_count
        precisions.append(precision)
        recalls.append(recall)

    precisions.append(0.0)
    recalls.append(1.0)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def read_yolo_labels(label_dir: Path, with_confidence: bool) -> list[Box]:
    boxes: list[Box] = []
    for path in sorted(label_dir.glob("*.txt")):
        image_id = path.stem
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            expected = 6 if with_confidence else 5
            if len(parts) != expected:
                raise ValueError(f"{path}:{line_number}: expected {expected} columns, got {len(parts)}")
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if with_confidence else 1.0
            boxes.append(Box(cls=cls, xc=xc, yc=yc, w=w, h=h, conf=conf, image_id=image_id))
    return boxes


def evaluate_subset(
    predictions: list[Box],
    ground_truth: list[Box],
    subset: str,
    iou_threshold: float,
    overlap_threshold: float = 0.3,
) -> dict[str, float]:
    if subset == "small":
        predicate = is_small_box
    elif subset == "medium":
        predicate = is_medium_box
    elif subset == "large":
        predicate = is_large_box
    elif subset == "thin":
        predicate = is_thin_box
    elif subset == "overlap":
        predicate = lambda box: is_overlap_box(box, ground_truth, threshold=overlap_threshold)
    else:
        raise ValueError("subset must be one of: small, medium, large, thin, overlap")

    subset_gt = [box for box in ground_truth if predicate(box)]
    subset_predictions = [box for box in predictions if any(box.image_id == gt.image_id and box.cls == gt.cls for gt in subset_gt)]
    matches = match_predictions(subset_predictions, subset_gt, iou_threshold=iou_threshold)
    ap = average_precision(matches, gt_count=len(subset_gt))
    recall = sum(matches.true_positives) / len(subset_gt) if subset_gt else 0.0
    return {
        "gt_count": len(subset_gt),
        "prediction_count": len(subset_predictions),
        "ap": ap,
        "recall": recall,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AP/Recall for protocol-defined target subsets from YOLO txt predictions.")
    parser.add_argument("--labels", type=Path, required=True, help="Ground-truth YOLO label directory.")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction txt directory with confidence column.")
    parser.add_argument("--subset", choices=("small", "medium", "large", "thin", "overlap"), required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--overlap-threshold", type=float, default=0.3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truth = read_yolo_labels(args.labels, with_confidence=False)
    predictions = read_yolo_labels(args.predictions, with_confidence=True)
    result = evaluate_subset(predictions, ground_truth, args.subset, args.iou, overlap_threshold=args.overlap_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
