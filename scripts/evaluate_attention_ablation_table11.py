from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_yolo_results import load_best_result, numeric_subset
from scripts.subset_detection_metrics import evaluate_subset, read_yolo_labels
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops_with_torch_profiler


@dataclass(frozen=True)
class ModelSpec:
    name: str
    train_dir: Path


DEFAULT_MODELS = [
    ModelSpec("SE", Path("runs/train_attention_ablation/nano_se")),
    ModelSpec("CBAM", Path("runs/train_attention_ablation/nano_cbam")),
    ModelSpec("nano+原版 EMA", Path("runs/train_attention_ablation/nano_original_ema")),
    ModelSpec("nano+原版 AIFI", Path("runs/train_attention_ablation/nano_original_aifi")),
    ModelSpec("nano+原版 EMA + AIFI-Lite", Path("runs/train_attention_ablation/nano_original_ema_aifi_lite")),
    ModelSpec("nano+EMA-Lite + 原版 AIFI", Path("runs/train_attention_ablation/nano_ema_lite_original_aifi")),
    ModelSpec("nano+原版 EMA + 原版 AIFI", Path("runs/train_attention_ablation/nano_original_ema_original_aifi")),
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_gflops(*, model: Any, imgsz: int, info_flops: float) -> float:
    if info_flops > 0:
        return round(float(info_flops), 5)
    return round(float(get_flops_with_torch_profiler(model, imgsz=imgsz)), 5)


def safe_run_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def clean_generated_dir(path: Path, required_parent: Path) -> None:
    resolved = path.resolve()
    parent = required_parent.resolve()
    if parent not in resolved.parents and resolved != parent:
        raise ValueError(f"refuse to remove path outside {parent}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def run_prediction(
    *,
    model: YOLO,
    source: Path,
    project: Path,
    name: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> Path:
    run_dir = project / name
    clean_generated_dir(run_dir, project)
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
        verbose=False,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    return run_dir / "labels"


def evaluate_model(
    *,
    spec: ModelSpec,
    data: Path,
    images: Path,
    labels: Path,
    output_dir: Path,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    warmup_val_runs: int,
    conf: float,
    nms_iou: float,
    metric_iou: float,
) -> dict[str, Any]:
    train_dir = spec.train_dir if spec.train_dir.is_absolute() else ROOT / spec.train_dir
    weights = train_dir / "weights" / "best.pt"
    results_csv = train_dir / "results.csv"
    if not weights.exists():
        raise FileNotFoundError(f"{spec.name}: missing weights: {weights}")
    if not results_csv.exists():
        raise FileNotFoundError(f"{spec.name}: missing results.csv: {results_csv}")

    best = numeric_subset(load_best_result(results_csv, "metrics/mAP50-95(B)"))
    model = YOLO(str(weights))
    layers, params, gradients, flops = model.info(imgsz=imgsz)
    gflops = resolve_gflops(model=model.model, imgsz=imgsz, info_flops=float(flops))

    run_name = safe_run_name(spec.name)
    val_kwargs = {
        "data": str(data),
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "plots": False,
        "verbose": False,
        "split": "val",
        "project": str(output_dir / "_val_runs"),
        "exist_ok": True,
        "save_json": False,
    }
    for warmup_i in range(warmup_val_runs):
        model.val(name=f"{run_name}_warmup{warmup_i + 1}", **val_kwargs)
    metrics = model.val(name=run_name, **val_kwargs)
    speed = metrics.speed
    preprocess_ms = float(speed.get("preprocess", 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))
    total_ms = preprocess_ms + inference_ms + postprocess_ms

    predictions_dir = run_prediction(
        model=model,
        source=images,
        project=output_dir / "predictions",
        name=run_name,
        imgsz=imgsz,
        conf=conf,
        iou=nms_iou,
        device=device,
    )
    ground_truth = read_yolo_labels(labels, with_confidence=False)
    predictions = read_yolo_labels(predictions_dir, with_confidence=True)
    small = evaluate_subset(predictions, ground_truth, "small", iou_threshold=metric_iou)
    thin = evaluate_subset(predictions, ground_truth, "thin", iou_threshold=metric_iou)

    row = {
        "model": spec.name,
        "train_dir": str(train_dir.relative_to(ROOT)),
        "weights": str(weights.relative_to(ROOT)),
        "best_epoch": best["epoch"],
        "params": params,
        "params_m": round(params / 1e6, 5),
        "gflops": gflops,
        "preprocess_ms": round(preprocess_ms, 5),
        "inference_ms": round(inference_ms, 5),
        "postprocess_ms": round(postprocess_ms, 5),
        "total_ms": round(total_ms, 5),
        "fps_inference_only": round(1000.0 / inference_ms, 5) if inference_ms > 0 else "",
        "fps_end_to_end": round(1000.0 / total_ms, 5) if total_ms > 0 else "",
        "precision": best["metrics/precision(B)"],
        "recall": best["metrics/recall(B)"],
        "f1": best["metrics/F1(B)"],
        "map50": best["metrics/mAP50(B)"],
        "map75": best["metrics/mAP75(B)"],
        "map50_95": best["metrics/mAP50-95(B)"],
        "ap_small": small["ap"],
        "recall_small": small["recall"],
        "small_gt_count": small["gt_count"],
        "ap_thin": thin["ap"],
        "recall_thin": thin["recall"],
        "thin_gt_count": thin["gt_count"],
    }
    write_json(output_dir / f"{run_name}.json", row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Table 11 metrics for original-attention ablation models.")
    parser.add_argument("--data", type=Path, default=Path("configs/datasets/SPXray_pcn_egi.yaml"))
    parser.add_argument("--images", type=Path, default=Path("datasets/SPXray/images/val"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/SPXray/labels/val"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/summary/attention_ablation_table11"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-val-runs", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--metric-iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data = args.data if args.data.is_absolute() else ROOT / args.data
    args.images = args.images if args.images.is_absolute() else ROOT / args.images
    args.labels = args.labels if args.labels.is_absolute() else ROOT / args.labels
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows = [
        evaluate_model(
            spec=spec,
            data=args.data,
            images=args.images,
            labels=args.labels,
            output_dir=args.output_dir,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            warmup_val_runs=args.warmup_val_runs,
            conf=args.conf,
            nms_iou=args.nms_iou,
            metric_iou=args.metric_iou,
        )
        for spec in DEFAULT_MODELS
    ]
    fields = list(rows[0])
    write_csv(args.output_dir / "attention_ablation_table11_metrics.csv", rows, fields)
    write_json(args.output_dir / "attention_ablation_table11_metrics.json", rows)
    print(json.dumps({"output_dir": str(args.output_dir), "models": [row["model"] for row in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
