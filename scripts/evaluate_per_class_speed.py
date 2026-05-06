from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


DEFAULT_MODELS = {
    "A3_pcn_egi": Path("runs/train_stage2/yolov8n_pcn_egi/weights/best.pt"),
    "XR-Lite": Path("runs/train_stage4/yolov8n_xr_lite/weights/best.pt"),
    "P2-Lite": Path("runs/train_stage5/yolov8n_xr_p2lite/weights/best.pt"),
    "XR-Plus": Path("runs/train_stage5/yolov8n_xr_plus/weights/best.pt"),
}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    *,
    name: str,
    weights: Path,
    data: Path,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = YOLO(str(weights))
    layers, params, gradients, flops = model.info(imgsz=imgsz)
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    metrics = model.val(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        plots=False,
        verbose=False,
        split="val",
        project=str(output_dir / "_val_runs"),
        name=run_name,
        exist_ok=True,
        save_json=False,
    )

    speed = metrics.speed
    preprocess_ms = float(speed.get("preprocess", 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))
    total_ms = preprocess_ms + inference_ms + postprocess_ms

    summary = {
        "model": name,
        "weights": str(weights),
        "layers": layers,
        "params": params,
        "params_m": round(params / 1e6, 5),
        "gradients": gradients,
        "gflops": round(float(flops), 5),
        "preprocess_ms": round(preprocess_ms, 5),
        "inference_ms": round(inference_ms, 5),
        "postprocess_ms": round(postprocess_ms, 5),
        "total_ms": round(total_ms, 5),
        "fps_inference_only": round(1000.0 / inference_ms, 5) if inference_ms > 0 else "",
        "fps_end_to_end": round(1000.0 / total_ms, 5) if total_ms > 0 else "",
        "precision": metrics.results_dict["metrics/precision(B)"],
        "recall": metrics.results_dict["metrics/recall(B)"],
        "map50": metrics.results_dict["metrics/mAP50(B)"],
        "map50_95": metrics.results_dict["metrics/mAP50-95(B)"],
    }

    count_rows = {
        str(row["Class"]): {"images": row["Images"], "instances": row["Instances"]}
        for row in metrics.summary(decimals=6)
    }
    box = metrics.box
    names = metrics.names
    per_class_rows = []
    for metric_i, class_i in enumerate(box.ap_class_index):
        class_name = names[int(class_i)]
        counts = count_rows.get(class_name, {})
        per_class_rows.append(
            {
                "model": name,
                "class": class_name,
                "images": counts.get("images", ""),
                "instances": counts.get("instances", ""),
                "precision": round(float(box.p[metric_i]), 6),
                "recall": round(float(box.r[metric_i]), 6),
                "f1": round(float(box.f1[metric_i]), 6),
                "ap50": round(float(box.ap50[metric_i]), 6),
                "ap75": round(float(box.all_ap[metric_i][5]), 6),
                "ap50_95": round(float(box.ap[metric_i]), 6),
            }
        )

    return summary, per_class_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate per-class AP and speed/complexity for completed XR models.")
    parser.add_argument("--data", type=Path, default=Path("configs/datasets/SPXray_pcn_egi.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/stage5/per_class_speed"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "WEIGHTS"),
        help="Optional model override. Can be passed multiple times. Defaults to A3, XR-Lite, P2-Lite, XR-Plus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data = args.data if args.data.is_absolute() else ROOT / args.data
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    models = {name: Path(path) for name, path in args.model} if args.model else DEFAULT_MODELS

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for name, weights in models.items():
        weights = weights if weights.is_absolute() else ROOT / weights
        if not weights.exists():
            raise FileNotFoundError(f"{name}: weights not found: {weights}")
        summary, class_rows = evaluate_model(
            name=name,
            weights=weights,
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            output_dir=args.output_dir,
        )
        summary_rows.append(summary)
        per_class_rows.extend(class_rows)

    summary_fields = [
        "model",
        "weights",
        "layers",
        "params",
        "params_m",
        "gradients",
        "gflops",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "total_ms",
        "fps_inference_only",
        "fps_end_to_end",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]
    class_fields = ["model", "class", "images", "instances", "precision", "recall", "f1", "ap50", "ap75", "ap50_95"]

    write_csv(args.output_dir / "model_speed_complexity.csv", summary_rows, summary_fields)
    write_csv(args.output_dir / "per_class_ap.csv", per_class_rows, class_fields)
    (args.output_dir / "model_speed_complexity.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "models": list(models)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
