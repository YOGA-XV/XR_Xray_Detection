from pathlib import Path
import shutil
import uuid

from scripts.check_dataset_stats import (
    DatasetConfig,
    classify_scale,
    collect_dataset_stats,
    count_overlaps,
    is_thin_box,
)


def write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_scale_thresholds_match_experiment_protocol():
    assert classify_scale(0.0099) == "small"
    assert classify_scale(0.01) == "medium"
    assert classify_scale(0.0499) == "medium"
    assert classify_scale(0.05) == "large"


def test_thin_threshold_matches_experiment_protocol():
    assert is_thin_box(width=0.30, height=0.10) is False
    assert is_thin_box(width=0.31, height=0.10) is True
    assert is_thin_box(width=0.05, height=0.30) is True


def test_count_overlaps_uses_target_area_threshold():
    labels = [
        (0, 0.5, 0.5, 0.4, 0.4),
        (1, 0.6, 0.5, 0.4, 0.4),
        (1, 0.1, 0.1, 0.1, 0.1),
    ]

    assert count_overlaps(labels, threshold=0.3) == 2
    assert count_overlaps(labels, threshold=0.8) == 0


def test_collect_dataset_stats_counts_instances_images_scale_and_thin():
    workspace_tmp = Path("test_artifacts") / f"tiny_xray_{uuid.uuid4().hex}"
    root = workspace_tmp / "TinyXray"
    touch_image(root / "images" / "train" / "a.jpg")
    touch_image(root / "images" / "train" / "b.jpg")
    touch_image(root / "images" / "val" / "c.jpg")

    write_label(
        root / "labels" / "train" / "a.txt",
        [
            "0 0.5 0.5 0.10 0.05",
            "1 0.5 0.5 0.20 0.05",
        ],
    )
    write_label(root / "labels" / "train" / "b.txt", ["1 0.5 0.5 0.30 0.30"])
    write_label(root / "labels" / "val" / "c.txt", ["0 0.5 0.5 0.10 0.20"])

    config = DatasetConfig(
        root=root,
        names=["Baton", "Knife"],
        splits=("train", "val"),
    )

    try:
        stats = collect_dataset_stats(config)

        assert stats["dataset_root"] == str(root)
        assert stats["splits"]["train"]["image_count"] == 2
        assert stats["splits"]["train"]["label_count"] == 2
        assert stats["splits"]["train"]["instance_count"] == 3
        assert stats["splits"]["train"]["class_instances"]["0"] == 1
        assert stats["splits"]["train"]["class_instances"]["1"] == 2
        assert stats["splits"]["train"]["class_images"]["1"] == 2
        assert stats["splits"]["train"]["scale_counts"] == {
            "small": 1,
            "medium": 1,
            "large": 1,
        }
        assert stats["splits"]["train"]["thin_count"] == 1
        assert stats["splits"]["train"]["overlap_counts"]["overlap_0.3"] == 2
        assert stats["splits"]["train"]["overlap_counts"]["overlap_0.5"] == 2
        assert stats["overall"]["image_count"] == 3
        assert stats["overall"]["instance_count"] == 4
        assert stats["overall"]["overlap_counts"]["overlap_0.3"] == 2
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)
