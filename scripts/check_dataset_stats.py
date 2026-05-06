from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SCALE_BINS = ("small", "medium", "large")


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    names: list[str]
    splits: tuple[str, ...] = ("train", "val", "test")


def classify_scale(area: float) -> str:
    """Classify objects using the protocol in docs/完整版试验设计方案.md section 5.3."""
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def is_thin_box(width: float, height: float) -> bool:
    """Return true for elongated boxes using section 5.4: max(w,h) / min(w,h) > 3."""
    if width <= 0 or height <= 0:
        return False
    return max(width, height) / min(width, height) > 3.0


def parse_yolo_label_line(line: str) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 columns, got {len(parts)}")
    cls_raw, xc_raw, yc_raw, width_raw, height_raw = parts
    cls = int(float(cls_raw))
    xc = float(xc_raw)
    yc = float(yc_raw)
    width = float(width_raw)
    height = float(height_raw)
    if width < 0 or height < 0:
        raise ValueError("width and height must be non-negative")
    return cls, xc, yc, width, height


def iter_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def read_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    labels = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            labels.append(parse_yolo_label_line(stripped))
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: {exc}") from exc
    return labels


def empty_split_stats() -> dict:
    return {
        "image_count": 0,
        "label_count": 0,
        "missing_label_count": 0,
        "instance_count": 0,
        "class_instances": {},
        "class_images": {},
        "scale_counts": {name: 0 for name in SCALE_BINS},
        "thin_count": 0,
        "area_values": [],
        "aspect_ratio_values": [],
    }


def collect_dataset_stats(config: DatasetConfig) -> dict:
    split_stats = {}
    overall_class_instances: Counter[int] = Counter()
    overall_class_images: Counter[int] = Counter()
    overall_scale_counts: Counter[str] = Counter({name: 0 for name in SCALE_BINS})
    overall_area_values: list[float] = []
    overall_aspect_ratio_values: list[float] = []
    overall_images = 0
    overall_labels = 0
    overall_missing_labels = 0
    overall_instances = 0
    overall_thin = 0

    for split in config.splits:
        image_dir = config.root / "images" / split
        label_dir = config.root / "labels" / split
        images = iter_images(image_dir)
        label_files_seen = 0
        missing_labels = 0
        class_instances: Counter[int] = Counter()
        class_images: Counter[int] = Counter()
        scale_counts: Counter[str] = Counter({name: 0 for name in SCALE_BINS})
        area_values: list[float] = []
        aspect_ratio_values: list[float] = []
        thin_count = 0
        instance_count = 0

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                label_files_seen += 1
            else:
                missing_labels += 1
            labels = read_label_file(label_path)
            image_classes = set()

            for cls, _xc, _yc, width, height in labels:
                area = width * height
                ratio = max(width, height) / min(width, height) if width > 0 and height > 0 else 0.0
                scale = classify_scale(area)
                thin = is_thin_box(width, height)

                class_instances[cls] += 1
                image_classes.add(cls)
                scale_counts[scale] += 1
                area_values.append(area)
                aspect_ratio_values.append(ratio)
                instance_count += 1
                if thin:
                    thin_count += 1

            for cls in image_classes:
                class_images[cls] += 1

        split_stats[split] = {
            "image_count": len(images),
            "label_count": label_files_seen,
            "missing_label_count": missing_labels,
            "instance_count": instance_count,
            "class_instances": counter_to_string_dict(class_instances, len(config.names)),
            "class_images": counter_to_string_dict(class_images, len(config.names)),
            "scale_counts": {name: scale_counts[name] for name in SCALE_BINS},
            "thin_count": thin_count,
            "area_values": area_values,
            "aspect_ratio_values": aspect_ratio_values,
        }

        overall_images += len(images)
        overall_labels += label_files_seen
        overall_missing_labels += missing_labels
        overall_instances += instance_count
        overall_thin += thin_count
        overall_class_instances.update(class_instances)
        overall_class_images.update(class_images)
        overall_scale_counts.update(scale_counts)
        overall_area_values.extend(area_values)
        overall_aspect_ratio_values.extend(aspect_ratio_values)

    return {
        "dataset_root": str(config.root),
        "class_names": config.names,
        "small_definition": "area = w*h/(W*H) < 0.01",
        "medium_definition": "0.01 <= area < 0.05",
        "large_definition": "area >= 0.05",
        "thin_definition": "max(w,h)/min(w,h) > 3",
        "splits": split_stats,
        "overall": {
            "image_count": overall_images,
            "label_count": overall_labels,
            "missing_label_count": overall_missing_labels,
            "instance_count": overall_instances,
            "class_instances": counter_to_string_dict(overall_class_instances, len(config.names)),
            "class_images": counter_to_string_dict(overall_class_images, len(config.names)),
            "scale_counts": {name: overall_scale_counts[name] for name in SCALE_BINS},
            "thin_count": overall_thin,
            "area_values": overall_area_values,
            "aspect_ratio_values": overall_aspect_ratio_values,
        },
    }


def counter_to_string_dict(counter: Counter[int], class_count: int) -> dict[str, int]:
    return {str(class_id): counter.get(class_id, 0) for class_id in range(class_count)}


def histogram(values: Iterable[float], bins: list[float]) -> list[dict[str, float | int | str]]:
    values_list = list(values)
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        count = sum(1 for value in values_list if left <= value < right)
        rows.append({"bin": f"[{left},{right})", "left": left, "right": right, "count": count})
    if bins:
        last = bins[-1]
        rows.append({"bin": f">={last}", "left": last, "right": "", "count": sum(1 for value in values_list if value >= last)})
    return rows


def load_dataset_config(path: Path) -> DatasetConfig:
    data = parse_simple_yaml(path)
    root_raw = data.get("path")
    if not root_raw:
        raise ValueError(f"{path} must define 'path'")
    root = Path(root_raw)
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    names_raw = data.get("names")
    if not names_raw:
        raise ValueError(f"{path} must define 'names'")
    names = ast.literal_eval(names_raw) if isinstance(names_raw, str) else names_raw
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("'names' must be a list of class names")
    return DatasetConfig(root=root, names=names)


def parse_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(stats: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_stats = strip_raw_values(stats)
    (output_dir / "summary.json").write_text(
        json.dumps(compact_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    split_rows = []
    for split, split_stats in stats["splits"].items():
        split_rows.append({
            "split": split,
            "image_count": split_stats["image_count"],
            "label_count": split_stats["label_count"],
            "missing_label_count": split_stats["missing_label_count"],
            "instance_count": split_stats["instance_count"],
            "small": split_stats["scale_counts"]["small"],
            "medium": split_stats["scale_counts"]["medium"],
            "large": split_stats["scale_counts"]["large"],
            "thin_count": split_stats["thin_count"],
        })
    write_csv(output_dir / "split_summary.csv", split_rows, list(split_rows[0].keys()) if split_rows else [])

    class_rows = []
    class_names = stats["class_names"]
    for class_id, class_name in enumerate(class_names):
        row = {"class_id": class_id, "class_name": class_name}
        for split, split_stats in stats["splits"].items():
            row[f"{split}_instances"] = split_stats["class_instances"][str(class_id)]
            row[f"{split}_images"] = split_stats["class_images"][str(class_id)]
        row["overall_instances"] = stats["overall"]["class_instances"][str(class_id)]
        row["overall_images"] = stats["overall"]["class_images"][str(class_id)]
        class_rows.append(row)
    write_csv(output_dir / "class_distribution.csv", class_rows, list(class_rows[0].keys()) if class_rows else [])

    area_bins = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]
    aspect_bins = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    write_csv(output_dir / "bbox_area_histogram.csv", histogram(stats["overall"]["area_values"], area_bins), ["bin", "left", "right", "count"])
    write_csv(output_dir / "aspect_ratio_histogram.csv", histogram(stats["overall"]["aspect_ratio_values"], aspect_bins), ["bin", "left", "right", "count"])


def strip_raw_values(stats: dict) -> dict:
    def strip_node(node: dict) -> dict:
        return {key: value for key, value in node.items() if key not in {"area_values", "aspect_ratio_values"}}

    stripped = {key: value for key, value in stats.items() if key != "splits"}
    stripped["overall"] = strip_node(stats["overall"])
    stripped["splits"] = {split: strip_node(split_stats) for split, split_stats in stats["splits"].items()}
    return stripped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check YOLO-format X-ray dataset statistics for the frozen experiment protocol.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML path.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for summary JSON and CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(args.data)
    stats = collect_dataset_stats(config)
    write_outputs(stats, args.output)
    print(f"Dataset root: {stats['dataset_root']}")
    print(f"Images: {stats['overall']['image_count']}")
    print(f"Instances: {stats['overall']['instance_count']}")
    print(f"Outputs: {args.output}")


if __name__ == "__main__":
    main()
