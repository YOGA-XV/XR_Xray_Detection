from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


MODELS = [
    ("YOLOv8n", "runs/train_pidray/yolov8n_pidray/weights/best.pt", False),
    ("YOLOv8s", "runs/train_pidray/yolov8s_pidray/weights/best.pt", False),
    ("XR-Nano", "runs/train_pidray/xr_nano_pidray/weights/best.pt", True),
    ("XR-Lite", "runs/train_pidray/xr_lite_pidray/weights/best.pt", True),
    ("XR-Plus", "runs/train_pidray/xr_plus_pidray/weights/best.pt", True),
]

SPLITS = [
    ("overall", "PIDray_test.yaml", "PIDray_pcn_egi_test.yaml"),
    ("easy", "PIDray_easy.yaml", "PIDray_pcn_egi_easy.yaml"),
    ("hard", "PIDray_hard.yaml", "PIDray_pcn_egi_hard.yaml"),
    ("hidden", "PIDray_hidden.yaml", "PIDray_pcn_egi_hidden.yaml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export structured PIDray stage-4 validation metrics.")
    parser.add_argument("--root", type=Path, default=Path("G:/XR_Xray_Detection"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/summary"))
    parser.add_argument("--project", type=Path, default=Path("runs/val_pidray_structured"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--json-from-csv",
        action="store_true",
        help="Only rebuild JSON from existing summary CSV files without rerunning validation.",
    )
    return parser.parse_args()


def safe_float(value: object) -> float:
    return round(float(value), 6)


def safe_int(value: object) -> int | str:
    if value == "":
        return ""
    return int(value)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    root = args.root
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    project = args.project if args.project.is_absolute() else root / args.project
    output_dir.mkdir(parents=True, exist_ok=True)
    project.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "pidray_stage4_val_metrics.csv"
    per_class_csv = output_dir / "pidray_stage4_val_per_class_ap.csv"
    summary_json = output_dir / "pidray_stage4_val_metrics.json"

    if args.json_from_csv:
        summary_rows = read_csv_rows(summary_csv)
        per_class_rows = read_csv_rows(per_class_csv)
        summary_json.write_text(
            json.dumps({"summary": summary_rows, "per_class": per_class_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {summary_json}")
        return

    summary_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []

    for model_name, weight_rel, uses_xr_data in MODELS:
        weights = root / weight_rel
        if not weights.exists():
            raise FileNotFoundError(f"Missing weights for {model_name}: {weights}")

        for split_name, rgb_yaml, xr_yaml in SPLITS:
            data_yaml = root / "configs/datasets" / (xr_yaml if uses_xr_data else rgb_yaml)
            if not data_yaml.exists():
                raise FileNotFoundError(f"Missing data yaml for {model_name} {split_name}: {data_yaml}")

            run_name = f"{model_name.lower().replace('-', '_')}_{split_name}"
            print(f"Evaluating {model_name} on PIDray {split_name}...")
            metrics = YOLO(str(weights)).val(
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
            summary_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "weights": str(weights),
                    "data": str(data_yaml),
                    "precision": safe_float(results["metrics/precision(B)"]),
                    "recall": safe_float(results["metrics/recall(B)"]),
                    "map50": safe_float(results["metrics/mAP50(B)"]),
                    "map50_95": safe_float(results["metrics/mAP50-95(B)"]),
                    "fitness": safe_float(results["fitness"]),
                    "preprocess_ms": safe_float(speed.get("preprocess", 0.0)),
                    "inference_ms": safe_float(speed.get("inference", 0.0)),
                    "postprocess_ms": safe_float(speed.get("postprocess", 0.0)),
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
                        "precision": safe_float(box.p[metric_i]),
                        "recall": safe_float(box.r[metric_i]),
                        "f1": safe_float(box.f1[metric_i]),
                        "ap50": safe_float(box.ap50[metric_i]),
                        "ap75": safe_float(box.all_ap[metric_i][5]),
                        "ap50_95": safe_float(box.ap[metric_i]),
                    }
                )

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    with per_class_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)

    summary_json.write_text(
        json.dumps(json_ready({"summary": summary_rows, "per_class": per_class_rows}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {summary_csv}")
    print(f"Wrote {per_class_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
