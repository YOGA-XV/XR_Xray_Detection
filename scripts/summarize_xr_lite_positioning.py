from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


METRIC_ALIASES = {
    "map50_95": ("map50_95", "metrics/mAP50-95(B)"),
    "map50": ("map50", "metrics/mAP50(B)"),
    "precision": ("precision", "metrics/precision(B)"),
    "recall": ("recall", "metrics/recall(B)"),
    "fps_end_to_end": ("fps_end_to_end",),
    "params_m": ("params_m",),
    "gflops": ("gflops",),
}


XR_FAMILY = {"XR-Lite", "P2-Lite", "XR-Plus"}
N_FAMILY = {"yolov8n", "XR-Nano", "XR-Lite", "P2-Lite", "XR-Plus"}
LEGACY_XR_NANO_ID = "A" + "3_pcn_egi"


def normalize_model_name(name: str) -> str:
    return "XR-Nano" if name == LEGACY_XR_NANO_ID else name


def parse_float(row: dict[str, str], metric: str) -> float | None:
    for key in METRIC_ALIASES[metric]:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def load_baseline_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = normalize_model_name(row["model"])
            rows.append(
                {
                    "model": name,
                    "source": str(path),
                    "family": "n_family" if name == "yolov8n" else "baseline",
                    "map50_95": parse_float(row, "map50_95"),
                    "map50": parse_float(row, "map50"),
                    "precision": parse_float(row, "precision"),
                    "recall": parse_float(row, "recall"),
                    "fps_end_to_end": None,
                    "params_m": None,
                    "gflops": None,
                }
            )
    return rows


def load_speed_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = normalize_model_name(row["model"])
            rows.append(
                {
                    "model": name,
                    "source": str(path),
                    "family": "xr_family" if name in XR_FAMILY else "n_family",
                    "map50_95": parse_float(row, "map50_95"),
                    "map50": parse_float(row, "map50"),
                    "precision": parse_float(row, "precision"),
                    "recall": parse_float(row, "recall"),
                    "fps_end_to_end": parse_float(row, "fps_end_to_end"),
                    "params_m": parse_float(row, "params_m"),
                    "gflops": parse_float(row, "gflops"),
                }
            )
    return rows


def leader(rows: list[dict[str, Any]], metric: str, candidates: set[str] | None = None) -> dict[str, Any] | None:
    filtered = [
        row
        for row in rows
        if row.get(metric) is not None and (candidates is None or row["model"] in candidates)
    ]
    if not filtered:
        return None
    return max(filtered, key=lambda row: float(row[metric]))


def model_name(row: dict[str, Any] | None) -> str | None:
    return None if row is None else str(row["model"])


def build_positioning(*, baseline_csv: Path, speed_csv: Path) -> dict[str, Any]:
    baseline_rows = load_baseline_rows(baseline_csv)
    speed_rows = load_speed_rows(speed_csv)
    rows = baseline_rows + speed_rows

    claims = {
        "overall_map50_95_leader": model_name(leader(rows, "map50_95")),
        "n_family_map50_95_leader": model_name(leader(rows, "map50_95", N_FAMILY)),
        "n_family_precision_leader": model_name(leader(rows, "precision", N_FAMILY)),
        "xr_family_map50_95_leader": model_name(leader(rows, "map50_95", XR_FAMILY)),
        "xr_family_fps_leader": model_name(leader(rows, "fps_end_to_end", XR_FAMILY)),
        "xr_family_precision_leader": model_name(leader(rows, "precision", XR_FAMILY)),
    }

    warnings: list[str] = []
    if claims["overall_map50_95_leader"] != "XR-Lite":
        warnings.append("not the overall mAP50:95 leader")
    if claims["n_family_map50_95_leader"] != "XR-Lite":
        warnings.append("not the YOLOv8n-family mAP50:95 leader")

    xr_lite = next((row for row in rows if row["model"] == "XR-Lite"), None)
    xr_nano = next((row for row in rows if row["model"] == "XR-Nano"), None)
    deltas = {}
    if xr_lite and xr_nano:
        for metric in ("map50_95", "precision", "recall", "fps_end_to_end", "params_m", "gflops"):
            if xr_lite.get(metric) is not None and xr_nano.get(metric) is not None:
                deltas[f"xr_lite_minus_xr_nano_{metric}"] = float(xr_lite[metric]) - float(xr_nano[metric])

    recommended_claim = (
        "XR-Lite is the strongest XR structural variant in the current Stage 5 comparison "
        "when judged by mAP50:95, end-to-end FPS, and precision within the XR family. "
        "It should not be reported as the global accuracy leader."
    )

    return {
        "claims": claims,
        "warnings": warnings,
        "deltas": deltas,
        "recommended_claim": recommended_claim,
        "rows": rows,
    }


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# XR-Lite Honest Positioning",
        "",
        "## Defensible Claim",
        "",
        report["recommended_claim"],
        "",
        "## Leaders",
        "",
    ]
    for key, value in report["claims"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    for warning in report["warnings"]:
        lines.append(f"- {warning}")
    lines.extend(["", "## Model Rows", "", "| Model | mAP50:95 | Precision | Recall | FPS | Params(M) | GFLOPs |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in report["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    format_value(row["map50_95"]),
                    format_value(row["precision"]),
                    format_value(row["recall"]),
                    format_value(row["fps_end_to_end"]),
                    format_value(row["params_m"]),
                    format_value(row["gflops"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize defensible XR-Lite positioning from existing results.")
    parser.add_argument("--baseline-csv", type=Path, default=Path("runs/baseline/yolov8_baseline_summary.csv"))
    parser.add_argument("--speed-csv", type=Path, default=Path("runs/stage5/per_class_speed_warm/model_speed_complexity.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/positioning/xr_lite"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_csv = args.baseline_csv if args.baseline_csv.is_absolute() else ROOT / args.baseline_csv
    speed_csv = args.speed_csv if args.speed_csv.is_absolute() else ROOT / args.speed_csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    report = build_positioning(baseline_csv=baseline_csv, speed_csv=speed_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "xr_lite_positioning.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "xr_lite_positioning.md", report)
    print(json.dumps({"output_dir": str(output_dir), "claims": report["claims"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
