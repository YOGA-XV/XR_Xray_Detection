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
    pretrained: Path


JOBS = [
    TrainJob(
        1,
        "xr_plus_original_ema_img768_lr0005_cm20",
        ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-plus-original-ema.yaml",
        ROOT / "runs/train_attention_ablation/nano_original_ema/weights/best.pt",
    ),
    TrainJob(
        2,
        "xr_plus_original_ema_aifi_lite_img768_lr0005_cm20",
        ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-plus-original-ema-aifi-lite.yaml",
        ROOT / "runs/train_attention_ablation/nano_original_ema_aifi_lite/weights/best.pt",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two XR-Plus candidates based on original EMA variants.")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/datasets/SPXray_pcn_egi.yaml")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train_plus_original_ema_candidates")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
    print("Running XR-Plus original EMA candidate training jobs sequentially", flush=True)
    print(f"Root: {ROOT}", flush=True)
    print(f"Data: {args.data}", flush=True)
    print(f"Output project: {args.project}", flush=True)
    print(
        f"epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, workers={args.workers}, "
        f"lr0={args.lr0}, close_mosaic={args.close_mosaic}, deterministic=False",
        flush=True,
    )
    print("=" * 60, flush=True)

    for job in JOBS:
        if not job.model_yaml.exists():
            raise FileNotFoundError(f"model yaml not found: {job.model_yaml}")
        if not job.pretrained.exists():
            raise FileNotFoundError(f"pretrained weights not found: {job.pretrained}")

    if args.dry_run:
        for job in JOBS:
            status = "completed" if is_completed(args.project, job.name, args.epochs) else "pending"
            print(f"[{job.index}/2] {job.name}: {status}", flush=True)
            print(f"  model: {job.model_yaml}", flush=True)
            print(f"  pretrained: {job.pretrained}", flush=True)
        return

    from ultralytics import YOLO

    for job in JOBS:
        if not args.rerun_completed and is_completed(args.project, job.name, args.epochs):
            print(f"[{job.index}/2] Skipping {job.name}: completed run already exists.", flush=True)
            continue

        print(f"[{job.index}/2] Training {job.name}", flush=True)
        print(f"  pretrained: {job.pretrained}", flush=True)
        model = YOLO(str(job.model_yaml))
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            optimizer="SGD",
            lr0=args.lr0,
            lrf=args.lrf,
            cos_lr=True,
            close_mosaic=args.close_mosaic,
            seed=args.seed,
            deterministic=False,
            pretrained=str(job.pretrained),
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
    print("All XR-Plus original EMA candidate training jobs completed successfully.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
