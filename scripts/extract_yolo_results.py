from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_last_result(results_csv: Path) -> dict[str, str]:
    with results_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{results_csv} has no metric rows")
    return {key.strip(): value.strip() for key, value in rows[-1].items()}


def load_best_result(results_csv: Path, metric_name: str) -> dict[str, str]:
    with results_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"{results_csv} has no metric rows")
    return max(rows, key=lambda row: float(row[metric_name]))


def numeric_subset(row: dict[str, str]) -> dict[str, float | int]:
    fields = [
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/F1(B)",
        "metrics/mAP50(B)",
        "metrics/mAP75(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]
    output: dict[str, float | int] = {}
    for field in fields:
        value = row[field]
        output[field] = int(float(value)) if field == "epoch" else float(value)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract last and best metric rows from an Ultralytics results.csv file.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-metric", default="metrics/mAP50-95(B)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    last = load_last_result(args.results)
    best = load_best_result(args.results, args.best_metric)
    summary = {
        "source": str(args.results),
        "best_metric": args.best_metric,
        "last": numeric_subset(last),
        "best": numeric_subset(best),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
