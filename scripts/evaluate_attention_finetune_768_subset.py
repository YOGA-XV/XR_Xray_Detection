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
        "nano_original_ema_768",
        Path("runs/train_attention_ablation_finetune_768/nano_original_ema_lr0005_img768_cm20/weights/best.pt"),
    ),
    (
        "nano_original_ema_aifi_lite_768",
        Path(
            "runs/train_attention_ablation_finetune_768/"
            "nano_original_ema_aifi_lite_lr0005_img768_cm20/weights/best.pt"
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
    output_dir = ROOT / "runs/summary/attention_finetune_768_subset"
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
        small = evaluate_subset(predictions, ground_truth, "small", iou_threshold=0.5)
        thin = evaluate_subset(predictions, ground_truth, "thin", iou_threshold=0.5)
        row = {
            "model": name,
            "weights": str(rel_weights),
            "ap_small": small["ap"],
            "recall_small": small["recall"],
            "small_gt_count": small["gt_count"],
            "ap_thin": thin["ap"],
            "recall_thin": thin["recall"],
            "thin_gt_count": thin["gt_count"],
        }
        rows.append(row)
        (output_dir / f"{name}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv(output_dir / "subset_metrics.csv", rows)
    print(json.dumps({"output_dir": str(output_dir), "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
