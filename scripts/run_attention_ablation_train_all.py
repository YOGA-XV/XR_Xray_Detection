from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class TrainJob:
    index: int
    name: str
    model_yaml: Path


JOBS = [
    TrainJob(1, "nano_se", ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-nano-se.yaml"),
    TrainJob(2, "nano_cbam", ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-nano-cbam.yaml"),
    TrainJob(3, "nano_original_ema", ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-nano-original-ema.yaml"),
    TrainJob(4, "nano_original_aifi", ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-nano-original-aifi.yaml"),
    TrainJob(
        5,
        "nano_original_ema_aifi_lite",
        ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-nano-original-ema-aifi-lite.yaml",
    ),
    TrainJob(
        6,
        "nano_ema_lite_original_aifi",
        ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-nano-ema-lite-original-aifi.yaml",
    ),
    TrainJob(
        7,
        "nano_original_ema_original_aifi",
        ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-nano-original-ema-original-aifi.yaml",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the XR-Nano attention ablation trainings in sequence.")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/datasets/SPXray_pcn_egi.yaml")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train_attention_ablation")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rerun-completed", action="store_true", help="Train even if a completed run directory exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned jobs.")
    return parser.parse_args()


def is_completed(project: Path, name: str, epochs: int) -> bool:
    run_dir = project / name
    best = run_dir / "weights/best.pt"
    results = run_dir / "results.csv"
    if not best.exists() or not results.exists():
        return False
    try:
        last = [line for line in results.read_text(encoding="utf-8").splitlines() if line.strip()][-1]
        epoch = int(float(last.split(",", 1)[0].strip()))
    except (IndexError, ValueError):
        return False
    return epoch >= epochs


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Running attention ablation training jobs sequentially", flush=True)
    print(f"Root: {ROOT}", flush=True)
    print(f"Data: {args.data}", flush=True)
    print(f"Output project: {args.project}", flush=True)
    print(f"epochs={args.epochs}, batch={args.batch}, workers={args.workers}, deterministic=False", flush=True)
    print("=" * 60, flush=True)

    if args.dry_run:
        for job in JOBS:
            status = "completed" if is_completed(args.project, job.name, args.epochs) else "pending"
            print(f"[{job.index}/{len(JOBS)}] {job.name}: {status} ({job.model_yaml})", flush=True)
        return

    from ultralytics import YOLO

    for job in JOBS:
        if not args.rerun_completed and is_completed(args.project, job.name, args.epochs):
            print(f"[{job.index}/{len(JOBS)}] Skipping {job.name}: completed run already exists.", flush=True)
            continue

        print(f"[{job.index}/{len(JOBS)}] Training {job.name}", flush=True)
        model = YOLO(str(job.model_yaml))
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            optimizer="SGD",
            cos_lr=True,
            close_mosaic=10,
            seed=args.seed,
            deterministic=False,
            pretrained=str(args.pretrained),
            patience=0,
            project=str(args.project),
            name=job.name,
            exist_ok=True,
        )
        del model
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    print("=" * 60, flush=True)
    print("All attention ablation training jobs completed successfully.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
