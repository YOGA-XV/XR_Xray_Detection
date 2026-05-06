# Ultralytics YOLOv8-only build

from .model import YOLO

from ultralytics.models.yolo import classify, detect, obb, pose, segment

__all__ = "YOLO", "classify", "detect", "obb", "pose", "segment"
