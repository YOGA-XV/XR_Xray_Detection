from __future__ import annotations

import argparse
import csv
import json
import os
import site
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops_with_torch_profiler


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: Path


MODELS = [
    ModelSpec("YOLOv8n", Path("runs/train_spxray/yolov8n_spxray/weights/best.pt")),
    ModelSpec("YOLOv8s", Path("runs/train_spxray/yolov8s_spxray/weights/best.pt")),
    ModelSpec("A1-PCN", Path("runs/train_stage2/yolov8n_pcn/weights/best.pt")),
    ModelSpec("A2-EGI", Path("runs/train_stage2/yolov8n_egi_raw/weights/best.pt")),
    ModelSpec("XR-Nano", Path("runs/train_stage2/yolov8n_pcn_egi/weights/best.pt")),
    ModelSpec("XR-Lite", Path("runs/train_stage4/yolov8n_xr_lite/weights/best.pt")),
    ModelSpec("P2-Lite", Path("runs/train_stage5/yolov8n_xr_p2lite/weights/best.pt")),
    ModelSpec("XR-Plus", Path("runs/train_stage5/yolov8n_xr_plus/weights/best.pt")),
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_optional_dll_dirs() -> None:
    if os.name != "nt":
        return
    extra_paths = []
    for site_dir in site.getsitepackages():
        tensorrt_libs = Path(site_dir) / "tensorrt_libs"
        if tensorrt_libs.exists():
            extra_paths.append(str(tensorrt_libs))
            os.add_dll_directory(str(tensorrt_libs))
    if extra_paths:
        os.environ["PATH"] = ";".join(extra_paths) + ";" + os.environ.get("PATH", "")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def input_channels(model: torch.nn.Module) -> int:
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return int(module.in_channels)
    raise RuntimeError("could not infer input channels")


def resolve_gflops(model: YOLO, imgsz: int, info_flops: float) -> float:
    if info_flops > 0:
        return float(info_flops)
    return float(get_flops_with_torch_profiler(model.model, imgsz=imgsz))


def summarize_ms(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    if len(samples) < 5:
        return float(statistics.mean(samples)), float(statistics.pstdev(samples))
    sorted_samples = sorted(samples)
    trim = max(1, int(len(sorted_samples) * 0.05))
    clipped = sorted_samples[trim:-trim] or sorted_samples
    return float(statistics.mean(clipped)), float(statistics.pstdev(clipped))


def torch_device_name(device: str) -> str:
    if device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if device.isdigit():
        return f"cuda:{device}"
    return device


def benchmark_pytorch(
    *,
    yolo: YOLO,
    channels: int,
    imgsz: int,
    warmup: int,
    repeat: int,
    device: str,
) -> tuple[float, float, float]:
    torch_device = torch_device_name(device)
    model = yolo.model
    model.to(torch_device)
    model.float()
    model.eval()
    x = torch.rand(1, channels, imgsz, imgsz, device=torch_device, dtype=torch.float32)
    if torch_device != "cpu":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if torch_device != "cpu":
            torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(repeat):
            if torch_device != "cpu":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if torch_device != "cpu":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    mean_ms, std_ms = summarize_ms(samples)
    peak_gb = 0.0
    if torch_device != "cpu":
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return mean_ms, std_ms, peak_gb


def export_onnx(weights: Path, imgsz: int, device: str) -> Path:
    yolo = YOLO(str(weights))
    exported = yolo.export(format="onnx", imgsz=imgsz, batch=1, device=device, half=False, simplify=True, verbose=False)
    return Path(exported)


def benchmark_onnx(
    *,
    onnx_path: Path,
    provider: str,
    provider_options: dict[str, Any] | None,
    warmup: int,
    repeat: int,
) -> tuple[float | None, float | None, str, str]:
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        if provider not in available:
            return None, None, "", f"{provider} unavailable; available={sorted(available)}"
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if provider == "CPUExecutionProvider":
            providers = [provider]
        elif provider_options:
            providers: list[Any] = [(provider, provider_options), "CPUExecutionProvider"]
        else:
            providers = [provider, "CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        active_provider = session.get_providers()[0] if session.get_providers() else ""
        if provider != "CPUExecutionProvider" and active_provider != provider:
            return None, None, active_provider, f"{provider} fell back to {session.get_providers()}"
        inputs = {}
        for item in session.get_inputs():
            shape = [1 if not isinstance(dim, int) or dim <= 0 else dim for dim in item.shape]
            dtype = np.float16 if "float16" in item.type else np.float32
            inputs[item.name] = np.random.rand(*shape).astype(dtype)
        for _ in range(warmup):
            session.run(None, inputs)
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            session.run(None, inputs)
            samples.append((time.perf_counter() - start) * 1000)
        mean_ms, std_ms = summarize_ms(samples)
        return mean_ms, std_ms, active_provider, ""
    except Exception as exc:  # noqa: BLE001
        return None, None, "", f"{type(exc).__name__}: {exc}"


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fps_from_ms(ms: float | None) -> float | None:
    if ms is None or ms <= 0:
        return None
    return 1000.0 / ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Table 13 complexity and speed fields.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/summary/table13_speed_20260529"))
    parser.add_argument("--skip-tensorrt", action="store_true")
    parser.add_argument("--models", default="", help="Comma-separated model names to benchmark. Empty means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    add_optional_dll_dirs()
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1")
    os.environ.setdefault("ORT_TENSORRT_CACHE_PATH", str(output_dir / "trt_cache"))

    requested_models = {name.strip() for name in args.models.split(",") if name.strip()}
    rows: list[dict[str, Any]] = []
    for spec in MODELS:
        if requested_models and spec.name not in requested_models:
            continue
        weights = ROOT / spec.weights
        if not weights.exists():
            raise FileNotFoundError(f"{spec.name}: {weights}")

        print(f"=== {spec.name} ===", flush=True)
        yolo = YOLO(str(weights))
        layers, params, _gradients, info_flops = yolo.info(imgsz=args.imgsz)
        gflops = resolve_gflops(yolo, args.imgsz, float(info_flops))
        channels = input_channels(yolo.model)
        pt_ms, pt_std, peak_gb = benchmark_pytorch(
            yolo=yolo,
            channels=channels,
            imgsz=args.imgsz,
            warmup=args.warmup,
            repeat=args.repeat,
            device=args.device,
        )

        onnx_path = export_onnx(weights, args.imgsz, args.device)
        onnx_ms, onnx_std, onnx_provider, onnx_error = benchmark_onnx(
            onnx_path=onnx_path,
            provider="CUDAExecutionProvider",
            provider_options=None,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        trt_ms = trt_std = None
        trt_provider = ""
        trt_error = "skipped"
        if not args.skip_tensorrt:
            trt_cache_path = output_dir / "trt_cache" / spec.name.replace("/", "_").replace(" ", "_")
            trt_cache_path.mkdir(parents=True, exist_ok=True)
            trt_ms, trt_std, trt_provider, trt_error = benchmark_onnx(
                onnx_path=onnx_path,
                provider="TensorrtExecutionProvider",
                provider_options={
                    "trt_fp16_enable": "True",
                    "trt_engine_cache_enable": "True",
                    "trt_engine_cache_path": str(trt_cache_path),
                },
                warmup=max(5, args.warmup // 3),
                repeat=max(50, args.repeat // 2),
            )

        row = {
            "model": spec.name,
            "weights": str(spec.weights),
            "onnx": str(onnx_path.relative_to(ROOT)) if onnx_path.is_relative_to(ROOT) else str(onnx_path),
            "input_size": args.imgsz,
            "input_channels": channels,
            "layers": layers,
            "params": params,
            "params_m": round(params / 1e6, 5),
            "gflops": round(gflops, 5),
            "pytorch_fp32_ms": round(pt_ms, 5),
            "pytorch_fp32_ms_std": round(pt_std, 5),
            "pytorch_fp32_fps": round(fps_from_ms(pt_ms) or 0.0, 5),
            "onnx_ms": "" if onnx_ms is None else round(onnx_ms, 5),
            "onnx_ms_std": "" if onnx_std is None else round(onnx_std, 5),
            "onnx_fps": "" if onnx_ms is None else round(fps_from_ms(onnx_ms) or 0.0, 5),
            "tensorrt_fp16_ms": "" if trt_ms is None else round(trt_ms, 5),
            "tensorrt_fp16_ms_std": "" if trt_std is None else round(trt_std, 5),
            "tensorrt_fp16_fps": "" if trt_ms is None else round(fps_from_ms(trt_ms) or 0.0, 5),
            "latency_ms": round(pt_ms, 5),
            "gpu_memory_gb": round(peak_gb, 5),
            "model_size_mb": round(file_size_mb(weights), 5),
            "onnx_size_mb": round(file_size_mb(onnx_path), 5) if onnx_path.exists() else "",
            "onnx_provider": onnx_provider,
            "onnx_error": onnx_error,
            "tensorrt_provider": trt_provider,
            "tensorrt_error": trt_error,
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
        write_csv(output_dir / "table13_speed_metrics.csv", rows)
        (output_dir / "table13_speed_metrics.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
