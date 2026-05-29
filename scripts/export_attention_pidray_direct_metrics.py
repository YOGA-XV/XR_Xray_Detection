from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


MODELS = [
    (
        "nano+原版 EMA",
        Path("runs/train_attention_ablation/nano_original_ema/weights/best.pt"),
    ),
    (
        "nano+原版 EMA + AIFI-Lite",
        Path("runs/train_attention_ablation/nano_original_ema_aifi_lite/weights/best.pt"),
    ),
]

SPLITS = [
    ("overall", Path("configs/datasets/PIDray_pcn_egi_test.yaml")),
    ("easy", Path("configs/datasets/PIDray_pcn_egi_easy.yaml")),
    ("hard", Path("configs/datasets/PIDray_pcn_egi_hard.yaml")),
    ("hidden", Path("configs/datasets/PIDray_pcn_egi_hidden.yaml")),
]


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def rounded(value: Any) -> float:
    return round(float(scalar(value)), 6)


def safe_int(value: Any) -> int | str:
    if value == "":
        return ""
    return int(scalar(value))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return scalar(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SPXray-trained original-attention models directly on PIDray splits."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/summary/attention_pidray_direct"))
    parser.add_argument("--project", type=Path, default=Path("runs/val_attention_pidray_direct"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    project = args.project if args.project.is_absolute() else ROOT / args.project
    output_dir.mkdir(parents=True, exist_ok=True)
    project.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    for model_name, rel_weights in MODELS:
        weights = ROOT / rel_weights
        if not weights.exists():
            raise FileNotFoundError(f"{model_name}: missing weights: {weights}")

        model = YOLO(str(weights))
        for split_name, rel_data in SPLITS:
            data_yaml = ROOT / rel_data
            if not data_yaml.exists():
                raise FileNotFoundError(f"{split_name}: missing data yaml: {data_yaml}")

            run_name = (
                model_name.replace("+", "plus")
                .replace(" ", "_")
                .replace("原版", "original")
                .replace("，", "_")
                .replace("-", "_")
                .lower()
                + f"_{split_name}"
            )
            print(f"Evaluating {model_name} on PIDray {split_name}...")
            metrics = model.val(
                data=str(data_yaml),
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                plots=False,
                save_json=False,
                project=str(project),
                name=run_name,
                exist_ok=True,
            )
            results = metrics.results_dict
            speed = metrics.speed
            preprocess_ms = float(speed.get("preprocess", 0.0))
            inference_ms = float(speed.get("inference", 0.0))
            postprocess_ms = float(speed.get("postprocess", 0.0))
            total_ms = preprocess_ms + inference_ms + postprocess_ms
            precision = float(results["metrics/precision(B)"])
            recall = float(results["metrics/recall(B)"])
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

            summary_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "weights": str(weights.relative_to(ROOT)),
                    "data": str(data_yaml.relative_to(ROOT)),
                    "precision": round(precision, 6),
                    "recall": round(recall, 6),
                    "f1": round(f1, 6),
                    "map50": rounded(results["metrics/mAP50(B)"]),
                    "map50_95": rounded(results["metrics/mAP50-95(B)"]),
                    "fitness": rounded(results["fitness"]),
                    "preprocess_ms": round(preprocess_ms, 6),
                    "inference_ms": round(inference_ms, 6),
                    "postprocess_ms": round(postprocess_ms, 6),
                    "total_ms": round(total_ms, 6),
                    "fps_end_to_end": round(1000.0 / total_ms, 6) if total_ms > 0 else "",
                }
            )

            counts = {
                str(row["Class"]): {"images": safe_int(row["Images"]), "instances": safe_int(row["Instances"])}
                for row in metrics.summary(decimals=6)
            }
            box = metrics.box
            names = metrics.names
            for metric_i, class_i in enumerate(box.ap_class_index):
                class_name = names[int(class_i)]
                class_counts = counts.get(class_name, {})
                per_class_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "class": class_name,
                        "images": class_counts.get("images", ""),
                        "instances": class_counts.get("instances", ""),
                        "precision": rounded(box.p[metric_i]),
                        "recall": rounded(box.r[metric_i]),
                        "f1": rounded(box.f1[metric_i]),
                        "ap50": rounded(box.ap50[metric_i]),
                        "ap75": rounded(box.all_ap[metric_i][5]),
                        "ap50_95": rounded(box.ap[metric_i]),
                    }
                )

    summary_csv = output_dir / "attention_pidray_direct_metrics.csv"
    per_class_csv = output_dir / "attention_pidray_direct_per_class_ap.csv"
    summary_json = output_dir / "attention_pidray_direct_metrics.json"
    write_csv(summary_csv, summary_rows)
    write_csv(per_class_csv, per_class_rows)
    summary_json.write_text(
        json.dumps(json_ready({"summary": summary_rows, "per_class": per_class_rows}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {summary_csv}")
    print(f"Wrote {per_class_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
