import numpy as np
import torch
from pathlib import Path
import shutil
import uuid
from types import SimpleNamespace

from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.augment import Format
from ultralytics.data.utils import get_hash
from ultralytics.data.xr_preprocess import (
    apply_xr_preprocess,
    apply_xr_preprocess_tensor,
    pseudo_color_normalize,
    pseudo_color_normalize_tensor,
    scharr_edge_map,
    scharr_edge_map_tensor,
)
from ultralytics.engine.predictor import BasePredictor
from ultralytics.nn.tasks import adapt_input_conv_weight


def test_pseudo_color_normalize_preserves_shape_and_clips_outliers():
    image = np.array(
        [
            [[0, 10, 20], [50, 60, 70], [100, 110, 120]],
            [[150, 160, 170], [200, 210, 220], [250, 250, 250]],
        ],
        dtype=np.uint8,
    )

    normalized = pseudo_color_normalize(image, low_percentile=1, high_percentile=99)

    assert normalized.shape == image.shape
    assert normalized.dtype == np.uint8
    assert normalized.min() == 0
    assert normalized.max() == 255


def test_scharr_edge_map_returns_single_channel_uint8_edges():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, 4:, :] = 255

    edge = scharr_edge_map(image)

    assert edge.shape == (8, 8, 1)
    assert edge.dtype == np.uint8
    assert edge.max() == 255
    assert edge[:, 3:5].sum() > 0


def test_apply_xr_preprocess_supports_pcn_and_egi_modes():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, 4:, :] = 255

    pcn_only = apply_xr_preprocess(image, use_pcn=True, use_egi=False)
    egi_only = apply_xr_preprocess(image, use_pcn=False, use_egi=True)
    pcn_egi = apply_xr_preprocess(image, use_pcn=True, use_egi=True)

    assert pcn_only.shape == (8, 8, 3)
    assert egi_only.shape == (8, 8, 4)
    assert pcn_egi.shape == (8, 8, 4)
    assert egi_only[..., 3].max() == 255


def test_tensor_xr_preprocess_runs_after_normalization_and_appends_edge_channel():
    image = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    image[:, :, :, 4:] = 1.0

    pcn_only = apply_xr_preprocess_tensor(image.clone(), use_pcn=True, use_egi=False)
    egi_only = apply_xr_preprocess_tensor(image.clone(), use_pcn=False, use_egi=True)
    pcn_egi = apply_xr_preprocess_tensor(image.clone(), use_pcn=True, use_egi=True)

    assert pcn_only.shape == (1, 3, 8, 8)
    assert egi_only.shape == (1, 4, 8, 8)
    assert pcn_egi.shape == (1, 4, 8, 8)
    assert egi_only[:, 3].max() > 0
    assert pcn_egi.min() >= 0
    assert pcn_egi.max() <= 1


def test_tensor_pcn_matches_per_channel_p1_p99_definition():
    image = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4) / 100

    normalized = pseudo_color_normalize_tensor(image, low_percentile=1, high_percentile=99)

    flat = image.flatten(2)
    count = flat.shape[-1]
    low_idx = max(int(count * 0.01), 1)
    high_idx = min(max(int(np.ceil(count * 0.99)), 1), count)
    low = flat.kthvalue(low_idx, dim=2).values[..., None, None]
    high = flat.kthvalue(high_idx, dim=2).values[..., None, None]
    expected = ((image - low) / (high - low + 1e-6)).clamp(0, 1)

    assert torch.allclose(normalized, expected)


def test_tensor_scharr_edge_map_uses_rgb_channel_order():
    image = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    image[:, 0, :, 4:] = 1.0

    edge = scharr_edge_map_tensor(image)

    assert edge.shape == (1, 1, 8, 8)
    assert edge.max() > 0


def test_format_keeps_edge_channel_when_converting_bgr_edge_to_tensor():
    # HWC order is B, G, R, Edge before Format. Expected tensor order is R, G, B, Edge.
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    image[..., 3] = 40

    tensor = Format()._format_img(image)

    assert tensor.shape == (4, 2, 2)
    assert torch.equal(tensor[0], torch.full((2, 2), 30, dtype=torch.uint8))
    assert torch.equal(tensor[1], torch.full((2, 2), 20, dtype=torch.uint8))
    assert torch.equal(tensor[2], torch.full((2, 2), 10, dtype=torch.uint8))
    assert torch.equal(tensor[3], torch.full((2, 2), 40, dtype=torch.uint8))


def test_adapt_input_conv_weight_initializes_extra_channel_from_rgb_mean():
    source = torch.arange(2 * 3 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3, 3)
    target = torch.zeros((2, 4, 3, 3), dtype=torch.float32)

    adapted = adapt_input_conv_weight(target, source)

    assert torch.equal(adapted[:, :3], source)
    assert torch.equal(adapted[:, 3], source.mean(dim=1))


def test_predictor_keeps_edge_channel_when_converting_bgr_edge_to_tensor():
    predictor = BasePredictor.__new__(BasePredictor)
    predictor.args = SimpleNamespace(rect=False, xr_pcn=False, xr_egi=True)
    predictor.imgsz = (2, 2)
    predictor.device = torch.device("cpu")
    predictor.model = SimpleNamespace(format="pt", dynamic=False, stride=32, fp16=False)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30

    tensor = predictor.preprocess([image])

    assert tensor.shape == (1, 4, 2, 2)
    assert torch.allclose(tensor[0, 0], torch.full((2, 2), 30 / 255))
    assert torch.allclose(tensor[0, 1], torch.full((2, 2), 20 / 255))
    assert torch.allclose(tensor[0, 2], torch.full((2, 2), 10 / 255))
    assert torch.allclose(tensor[0, 3], torch.zeros((2, 2)))


def test_yolo_dataset_leaves_stage2_preprocess_for_gpu_batch_step():
    workspace_tmp = Path("test_artifacts") / f"stage2_xr_{uuid.uuid4().hex}"
    image_dir = workspace_tmp / "images" / "val"
    label_dir = workspace_tmp / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, 8:, :] = 255
    import cv2

    cv2.imwrite(str(image_dir / "sample.jpg"), image)
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    im_file = str(image_dir / "sample.jpg")
    label_file = str(label_dir / "sample.txt")
    cache = {
        "labels": [
            {
                "im_file": im_file,
                "shape": (16, 16),
                "cls": np.array([[0]], dtype=np.float32),
                "bboxes": np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
            }
        ],
        "hash": get_hash([label_file, im_file]),
        "results": (1, 0, 0, 0, 1),
        "msgs": [],
        "version": DATASET_CACHE_VERSION,
    }
    with open(label_dir.with_suffix(".cache"), "wb") as f:
        np.save(f, cache)

    try:
        base_data = {"names": {0: "item"}, "nc": 1}
        pcn = YOLODataset(
            img_path=str(image_dir),
            imgsz=32,
            batch_size=1,
            augment=False,
            data={**base_data, "channels": 3, "xr_pcn": True, "xr_egi": False},
            task="detect",
        )
        egi = YOLODataset(
            img_path=str(image_dir),
            imgsz=32,
            batch_size=1,
            augment=False,
            data={**base_data, "channels": 4, "xr_pcn": False, "xr_egi": True},
            task="detect",
        )
        pcn_egi = YOLODataset(
            img_path=str(image_dir),
            imgsz=32,
            batch_size=1,
            augment=False,
            data={**base_data, "channels": 4, "xr_pcn": True, "xr_egi": True},
            task="detect",
        )

        assert pcn[0]["img"].shape[0] == 3
        assert egi[0]["img"].shape[0] == 3
        assert pcn_egi[0]["img"].shape[0] == 3
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)
