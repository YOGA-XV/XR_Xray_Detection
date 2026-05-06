# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def pseudo_color_normalize(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply per-image, per-channel P1/P99 pseudo-color normalization."""
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"PCN expects an HWC image with at least 3 channels, got shape {image.shape}")

    color = image[..., :3].astype(np.float32)
    out = np.empty_like(color, dtype=np.float32)
    for c in range(3):
        channel = color[..., c]
        low = np.percentile(channel, low_percentile)
        high = np.percentile(channel, high_percentile)
        out[..., c] = np.clip((channel - low) / (high - low + eps), 0.0, 1.0)

    normalized = (out * 255.0).round().astype(np.uint8)
    if image.shape[2] == 3:
        return normalized
    return np.concatenate([normalized, image[..., 3:]], axis=2)


def scharr_edge_map(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return a normalized single-channel Scharr edge map for an HWC OpenCV BGR image."""
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Scharr edge extraction expects an HWC image with at least 3 channels, got shape {image.shape}")

    bgr = image[..., :3].astype(np.float32)
    gray = 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]
    edge_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    edge_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(edge_x * edge_x + edge_y * edge_y)
    mag_min = float(mag.min())
    mag_max = float(mag.max())
    edge = np.clip((mag - mag_min) / (mag_max - mag_min + eps), 0.0, 1.0)
    return (edge[..., None] * 255.0).round().astype(np.uint8)


def apply_xr_preprocess(image: np.ndarray, use_pcn: bool = False, use_egi: bool = False) -> np.ndarray:
    """Apply the Stage 2 XR input pipeline and return a 3-channel or 4-channel HWC image."""
    if not use_pcn and not use_egi:
        return image
    processed = pseudo_color_normalize(image) if use_pcn else image
    if not use_egi:
        return processed
    return np.concatenate([processed[..., :3], scharr_edge_map(processed)], axis=2)


def pseudo_color_normalize_tensor(
    image: torch.Tensor,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply per-image, per-channel P1/P99 normalization to a BCHW tensor in [0, 1]."""
    if image.ndim != 4 or image.shape[1] < 3:
        raise ValueError(f"Tensor PCN expects a BCHW tensor with at least 3 channels, got shape {tuple(image.shape)}")

    dtype = image.dtype
    color = image[:, :3].float()
    flat = color.flatten(2)
    count = flat.shape[-1]
    low_idx = max(int(count * low_percentile / 100.0), 1)
    high_idx = min(max(int(np.ceil(count * high_percentile / 100.0)), 1), count)
    low = flat.kthvalue(low_idx, dim=2).values[:, :, None, None]
    high = flat.kthvalue(high_idx, dim=2).values[:, :, None, None]
    normalized = ((color - low) / (high - low + eps)).clamp_(0.0, 1.0).to(dtype)
    if image.shape[1] == 3:
        return normalized
    return torch.cat((normalized, image[:, 3:]), dim=1)


def scharr_edge_map_tensor(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return a normalized Scharr edge map for a BCHW RGB tensor in [0, 1]."""
    if image.ndim != 4 or image.shape[1] < 3:
        raise ValueError(
            f"Tensor Scharr edge extraction expects a BCHW tensor with at least 3 channels, got shape {tuple(image.shape)}"
        )

    dtype = image.dtype
    rgb = image[:, :3].float()
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    kernel_x = torch.tensor(
        [[[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]],
        device=image.device,
        dtype=gray.dtype,
    ).unsqueeze(0)
    kernel_y = torch.tensor(
        [[[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]]],
        device=image.device,
        dtype=gray.dtype,
    ).unsqueeze(0)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    edge_x = F.conv2d(padded, kernel_x)
    edge_y = F.conv2d(padded, kernel_y)
    mag = torch.sqrt(edge_x * edge_x + edge_y * edge_y)
    flat = mag.flatten(2)
    mag_min = flat.min(dim=2).values[:, :, None, None]
    mag_max = flat.max(dim=2).values[:, :, None, None]
    return ((mag - mag_min) / (mag_max - mag_min + eps)).clamp_(0.0, 1.0).to(dtype)


def apply_xr_preprocess_tensor(image: torch.Tensor, use_pcn: bool = False, use_egi: bool = False) -> torch.Tensor:
    """Apply the Stage 2 XR input pipeline to a normalized BCHW tensor on its current device."""
    if not use_pcn and not use_egi:
        return image
    processed = pseudo_color_normalize_tensor(image) if use_pcn else image
    if not use_egi:
        return processed
    return torch.cat((processed[:, :3], scharr_edge_map_tensor(processed)), dim=1)
