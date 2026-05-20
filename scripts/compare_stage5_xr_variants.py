from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


COLUMNS = [
    "variant",
    "params",
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
    "delta_map50_95_vs_ref",
    "delta_ap_small_vs_ref",
    "delta_ap_thin_vs_ref",
]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    yaml_path: Path | None
    final_metrics: Path
    ap_small: Path
    ap_thin: Path
    params: int | None = None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_params(yaml_path: Path) -> int:
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(str(yaml_path), nc=12, ch=4, verbose=False)
    return sum(p.numel() for p in model.parameters())


def empty_row(spec: VariantSpec) -> dict[str, Any]:
    row = {column: "" for column in COLUMNS}
    row["variant"] = spec.name
    row["params"] = spec.params if spec.params is not None else (count_params(spec.yaml_path) if spec.yaml_path else "")
    return row


def load_variant_row(spec: VariantSpec, allow_missing: bool) -> dict[str, Any]:
    paths = [spec.final_metrics, spec.ap_small, spec.ap_thin]
    if any(not path.exists() for path in paths):
        if allow_missing:
            return empty_row(spec)
        missing = [str(path) for path in paths if not path.exists()]
        raise FileNotFoundError(f"Missing metric files for {spec.name}: {missing}")

    final_metrics = read_json(spec.final_metrics)
    small = read_json(spec.ap_small)
    thin = read_json(spec.ap_thin)
    best = final_metrics["best"]

    return {
        "variant": spec.name,
        "params": spec.params if spec.params is not None else (count_params(spec.yaml_path) if spec.yaml_path else ""),
        "best_epoch": best["epoch"],
        "precision": best["metrics/precision(B)"],
        "recall": best["metrics/recall(B)"],
        "f1": best["metrics/F1(B)"],
        "map50": best["metrics/mAP50(B)"],
        "map75": best["metrics/mAP75(B)"],
        "map50_95": best["metrics/mAP50-95(B)"],
        "ap_small": small["ap"],
        "recall_small": small["recall"],
        "ap_thin": thin["ap"],
        "recall_thin": thin["recall"],
        "delta_map50_95_vs_ref": "",
        "delta_ap_small_vs_ref": "",
        "delta_ap_thin_vs_ref": "",
    }


def apply_reference_deltas(rows: list[dict[str, Any]], reference_name: str | None) -> None:
    if reference_name is None:
        return
    reference = next((row for row in rows if row["variant"] == reference_name), None)
    if reference is None:
        raise ValueError(f"Reference variant {reference_name!r} is not present in rows")
    if reference["map50_95"] == "":
        return
    for row in rows:
        if row["map50_95"] == "":
            continue
        row["delta_map50_95_vs_ref"] = float(row["map50_95"]) - float(reference["map50_95"])
        row["delta_ap_small_vs_ref"] = float(row["ap_small"]) - float(reference["ap_small"])
        row["delta_ap_thin_vs_ref"] = float(row["ap_thin"]) - float(reference["ap_thin"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def compare_variants(
    *,
    specs: list[VariantSpec],
    output_csv: Path,
    output_json: Path,
    reference_name: str | None,
    allow_missing: bool,
) -> list[dict[str, Any]]:
    rows = [load_variant_row(spec, allow_missing=allow_missing) for spec in specs]
    apply_reference_deltas(rows, reference_name)
    write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def default_specs() -> list[VariantSpec]:
    model_dir = ROOT / "ultralytics/cfg/models/v8"
    return [
        VariantSpec(
            name="XR-Lite",
            yaml_path=model_dir / "yolov8n-xr-lite.yaml",
            final_metrics=ROOT / "runs/stage4/yolov8n_xr_lite_final_metrics.json",
            ap_small=ROOT / "runs/stage4/yolov8n_xr_lite_ap_small.json",
            ap_thin=ROOT / "runs/stage4/yolov8n_xr_lite_ap_thin.json",
        ),
        VariantSpec(
            name="XR-Plus",
            yaml_path=model_dir / "yolov8n-xr-plus.yaml",
            final_metrics=ROOT / "runs/stage5/yolov8n_xr_plus_final_metrics.json",
            ap_small=ROOT / "runs/stage5/yolov8n_xr_plus_ap_small.json",
            ap_thin=ROOT / "runs/stage5/yolov8n_xr_plus_ap_thin.json",
        ),
        VariantSpec(
            name="XR-Plus-v2",
            yaml_path=model_dir / "yolov8n-xr-plus-v2.yaml",
            final_metrics=ROOT / "runs/stage5/yolov8n_xr_plus_v2_final_metrics.json",
            ap_small=ROOT / "runs/stage5/yolov8n_xr_plus_v2_ap_small.json",
            ap_thin=ROOT / "runs/stage5/yolov8n_xr_plus_v2_ap_thin.json",
        ),
        VariantSpec(
            name="XR-Plus-v3",
            yaml_path=model_dir / "yolov8n-xr-plus-v3.yaml",
            final_metrics=ROOT / "runs/stage5/yolov8n_xr_plus_v3_final_metrics.json",
            ap_small=ROOT / "runs/stage5/yolov8n_xr_plus_v3_ap_small.json",
            ap_thin=ROOT / "runs/stage5/yolov8n_xr_plus_v3_ap_thin.json",
        ),
        VariantSpec(
            name="XR-Plus-v4",
            yaml_path=model_dir / "yolov8n-xr-plus-v4.yaml",
            final_metrics=ROOT / "runs/stage5/yolov8n_xr_plus_v4_final_metrics.json",
            ap_small=ROOT / "runs/stage5/yolov8n_xr_plus_v4_ap_small.json",
            ap_thin=ROOT / "runs/stage5/yolov8n_xr_plus_v4_ap_thin.json",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare XR-Lite, XR-Plus, and XR-Plus v2/v3/v4 stage5 results.")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "runs/stage5/xr_plus_variants_summary.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "runs/stage5/xr_plus_variants_summary.json")
    parser.add_argument("--reference", default="XR-Lite")
    parser.add_argument("--allow-missing", action="store_true", help="Keep rows with params even when metric files are not ready.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compare_variants(
        specs=default_specs(),
        output_csv=args.output_csv,
        output_json=args.output_json,
        reference_name=args.reference,
        allow_missing=args.allow_missing,
    )
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
