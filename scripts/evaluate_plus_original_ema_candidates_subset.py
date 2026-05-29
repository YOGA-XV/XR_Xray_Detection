from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.subset_detection_metrics import evaluate_subset, read_yolo_labels
from ultralytics import YOLO


JOBS = [
    (
        "XR-Plus-Original-EMA",
        Path("runs/train_plus_original_ema_candidates/xr_plus_original_ema_img768_lr0005_cm20/weights/best.pt"),
    ),
    (
        "XR-Plus-Original-EMA-AIFI-Lite",
        Path(
            "runs/train_plus_original_ema_candidates/"
            "xr_plus_original_ema_aifi_lite_img768_lr0005_cm20/weights/best.pt"
        ),
    ),
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = ROOT / "runs/summary/plus_original_ema_candidates_subset"
    pred_project = output_dir / "predictions"
    labels_dir = ROOT / "datasets/SPXray/labels/val"
    images_dir = ROOT / "datasets/SPXray/images/val"
    ground_truth = read_yolo_labels(labels_dir, with_confidence=False)

    rows: list[dict[str, Any]] = []
    for name, rel_weights in JOBS:
        weights = ROOT / rel_weights
        if not weights.exists():
            raise FileNotFoundError(weights)

        run_dir = pred_project / name
        if run_dir.exists():
            shutil.rmtree(run_dir)

        model = YOLO(str(weights))
        model.predict(
            source=str(images_dir),
            imgsz=768,
            conf=0.001,
            iou=0.7,
            device="0",
            save_txt=True,
            save_conf=True,
            xr_pcn=True,
            xr_egi=True,
            verbose=False,
            project=str(pred_project),
            name=name,
            exist_ok=True,
        )
        predictions = read_yolo_labels(run_dir / "labels", with_confidence=True)
        row: dict[str, Any] = {"model": name, "weights": str(rel_weights)}
        for subset in ("small", "thin", "overlap"):
            metrics = evaluate_subset(predictions, ground_truth, subset, iou_threshold=0.5, overlap_threshold=0.3)
            row[f"ap_{subset}"] = metrics["ap"]
            row[f"recall_{subset}"] = metrics["recall"]
            row[f"{subset}_gt_count"] = metrics["gt_count"]
        rows.append(row)
        (output_dir / f"{name}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv(output_dir / "subset_metrics.csv", rows)
    (output_dir / "subset_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
