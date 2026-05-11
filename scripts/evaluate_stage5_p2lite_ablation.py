from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_yolo_results import load_best_result, numeric_subset
from scripts.subset_detection_metrics import evaluate_subset, read_yolo_labels


XR_NANO_REFERENCE_EXPERIMENT = "A" + "3_pcn_egi"

METRIC_COLUMNS = [
    "experiment",
    "best_epoch",
    "precision",
    "recall",
    "f1",
    "map50",
    "map75",
    "map50_95",
    "ap_small",
    "recall_small",
    "ap_thin",
    "recall_thin",
    "delta_map50_95_vs_XR_Nano",
    "delta_ap_small_vs_XR_Nano",
    "delta_ap_thin_vs_XR_Nano",
]


def load_reference_row(summary_csv: Path, experiments: tuple[str, ...]) -> dict[str, float | str]:
    with summary_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "experiment" not in fieldnames:
            raise ValueError(
                f"{summary_csv} must be a stage summary CSV with an 'experiment' column; "
                f"found columns: {fieldnames}. Use runs/stage2/stage2_summary.csv for the XR-Nano reference."
            )
        for row in reader:
            if row["experiment"] in experiments:
                parsed: dict[str, float | str] = {"experiment": "XR-Nano"}
                for key, value in row.items():
                    if key == "experiment":
                        continue
                    parsed[key] = float(value)
                return parsed
    raise ValueError(f"{summary_csv} does not contain an XR-Nano reference experiment")


def build_summary_row(
    *,
    experiment: str,
    best_metrics: dict[str, float | int],
    small_metrics: dict[str, float],
    thin_metrics: dict[str, float],
    xr_nano_reference: dict[str, float | str],
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "experiment": experiment,
        "best_epoch": best_metrics["epoch"],
        "precision": best_metrics["metrics/precision(B)"],
        "recall": best_metrics["metrics/recall(B)"],
        "f1": best_metrics["metrics/F1(B)"],
        "map50": best_metrics["metrics/mAP50(B)"],
        "map75": best_metrics["metrics/mAP75(B)"],
        "map50_95": best_metrics["metrics/mAP50-95(B)"],
        "ap_small": small_metrics["ap"],
        "recall_small": small_metrics["recall"],
        "ap_thin": thin_metrics["ap"],
        "recall_thin": thin_metrics["recall"],
    }
    row["delta_map50_95_vs_XR_Nano"] = float(row["map50_95"]) - float(xr_nano_reference["map50_95"])
    row["delta_ap_small_vs_XR_Nano"] = float(row["ap_small"]) - float(xr_nano_reference["ap_small"])
    row["delta_ap_thin_vs_XR_Nano"] = float(row["ap_thin"]) - float(xr_nano_reference["ap_thin"])
    return row


def write_csv(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow({key: row[key] for key in METRIC_COLUMNS})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_existing_outputs(
    *,
    results_csv: Path,
    labels_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    baseline_summary: Path,
    experiment: str,
    iou: float = 0.5,
) -> dict[str, float | int | str]:
    best_metrics = numeric_subset(load_best_result(results_csv, "metrics/mAP50-95(B)"))
    final_metrics = {
        "source": str(results_csv),
        "best_metric": "metrics/mAP50-95(B)",
        "best": best_metrics,
    }
    write_json(output_dir / "yolov8n_xr_p2lite_final_metrics.json", final_metrics)

    ground_truth = read_yolo_labels(labels_dir, with_confidence=False)
    predictions = read_yolo_labels(predictions_dir, with_confidence=True)
    small_metrics = evaluate_subset(predictions, ground_truth, "small", iou_threshold=iou)
    thin_metrics = evaluate_subset(predictions, ground_truth, "thin", iou_threshold=iou)
    write_json(output_dir / "yolov8n_xr_p2lite_ap_small.json", small_metrics)
    write_json(output_dir / "yolov8n_xr_p2lite_ap_thin.json", thin_metrics)

    xr_nano_reference = load_reference_row(baseline_summary, ("XR-Nano", XR_NANO_REFERENCE_EXPERIMENT))
    summary_row = build_summary_row(
        experiment=experiment,
        best_metrics=best_metrics,
        small_metrics=small_metrics,
        thin_metrics=thin_metrics,
        xr_nano_reference=xr_nano_reference,
    )
    analysis = {
        "experiment": experiment,
        "role": "XR-Nano + P2-Lite auxiliary ablation",
        "summary": summary_row,
        "xr_nano_reference": xr_nano_reference,
        "interpretation": {
            "p2_small_gain": summary_row["delta_ap_small_vs_XR_Nano"],
            "p2_thin_gain": summary_row["delta_ap_thin_vs_XR_Nano"],
            "overall_map_change": summary_row["delta_map50_95_vs_XR_Nano"],
            "use": "Use this result to isolate the contribution of P2-Lite before judging full XR-Plus.",
        },
    }
    write_json(output_dir / "yolov8n_xr_p2lite_analysis.json", analysis)
    write_csv(output_dir / "stage5_p2lite_ablation_summary.csv", summary_row)
    return summary_row


def run_prediction(
    *,
    weights: Path,
    source: Path,
    project: Path,
    name: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save_txt=True,
        save_conf=True,
        xr_pcn=True,
        xr_egi=True,
        project=str(project),
        name=name,
    )
    return project / name / "labels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and summarize the Stage5 XR-Nano+P2-Lite auxiliary ablation.")
    parser.add_argument("--train-dir", type=Path, required=True, help="Training run directory containing results.csv and weights/best.pt.")
    parser.add_argument("--labels", type=Path, default=Path("datasets/SPXray/labels/val"))
    parser.add_argument("--images", type=Path, default=Path("datasets/SPXray/images/val"))
    parser.add_argument("--predictions", type=Path, help="Existing prediction labels directory. Required when --skip-predict is set.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/stage5/p2lite_ablation"))
    parser.add_argument("--baseline-summary", type=Path, default=Path("runs/stage2/stage2_summary.csv"))
    parser.add_argument("--predict-project", type=Path, default=Path("runs/predict"))
    parser.add_argument("--predict-name", default="yolov8n_xr_p2lite_val")
    parser.add_argument("--experiment", default="S5_p2lite")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7, help="Prediction NMS IoU. Subset AP still uses --metric-iou.")
    parser.add_argument("--metric-iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-predict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_csv = args.train_dir / "results.csv"

    if args.skip_predict:
        if args.predictions is None:
            raise ValueError("--predictions is required when --skip-predict is set")
        predictions_dir = args.predictions
    else:
        predictions_dir = run_prediction(
            weights=args.train_dir / "weights" / "best.pt",
            source=args.images,
            project=args.predict_project,
            name=args.predict_name,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )

    summary = evaluate_existing_outputs(
        results_csv=results_csv,
        labels_dir=args.labels,
        predictions_dir=predictions_dir,
        output_dir=args.output_dir,
        baseline_summary=args.baseline_summary,
        experiment=args.experiment,
        iou=args.metric_iou,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
