# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def auto_annotate(
    data: str | Path,
    det_model: str = "yolov8x.pt",
    sam_model: str = "sam_b.pt",
    device: str = "",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 300,
    classes: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """Automatically annotate images using a YOLO object detection model and a SAM segmentation model.

    This function processes images in a specified directory, detects objects using a YOLO model, and then generates
    segmentation masks using a SAM model. The resulting annotations are saved as text files in YOLO format.

    Args:
        data (str | Path): Path to a folder containing images to be annotated.
        det_model (str): Path or name of the pre-trained YOLO detection model.
        sam_model (str): Path or name of the pre-trained SAM segmentation model.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda', '0'). Empty string for auto-selection.
        conf (float): Confidence threshold for detection model.
        iou (float): IoU threshold for filtering overlapping boxes in detection results.
        imgsz (int): Input image resize dimension.
        max_det (int): Maximum number of detections per image.
        classes (list[int], optional): Filter predictions to specified class IDs, returning only relevant detections.
        output_dir (str | Path, optional): Directory to save the annotated results. If None, creates a default directory
            based on the input data path.

    Examples:
        >>> from ultralytics.data.annotator import auto_annotate
        >>> auto_annotate(data="ultralytics/assets", det_model="yolov8n.pt", sam_model="mobile_sam.pt")
    """
    raise NotImplementedError(
        "auto_annotate requires SAM which is not available in this YOLOv8-only build. "
        "Use the full ultralytics package for SAM support."
    )
