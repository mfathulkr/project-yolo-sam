from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import SAM, YOLO

from pool_segmentation_compare.io_utils import ensure_dir, list_images, load_image_shape, save_binary_mask


def run_yolo_sam_pipeline(
    images_dir: Path,
    output_dir: Path,
    yolo_weights: str,
    sam_checkpoint: str,
    conf_threshold: float,
    image_size: int,
    yolo_device: str,
    sam_device: str,
) -> None:
    masks_dir = ensure_dir(output_dir / "masks")
    detector = YOLO(yolo_weights)
    segmenter = SAM(sam_checkpoint)

    for image_path in tqdm(list_images(images_dir), desc="Pipeline A"):
        height, width = load_image_shape(image_path)
        det_results = detector.predict(
            source=str(image_path),
            conf=conf_threshold,
            imgsz=image_size,
            device=yolo_device,
            verbose=False,
        )

        boxes = det_results[0].boxes.xyxy
        if boxes is None or len(boxes) == 0:
            merged_mask = np.zeros((height, width), dtype=bool)
        else:
            sam_results = segmenter(
                str(image_path),
                bboxes=boxes.cpu().numpy().tolist(),
                device=sam_device,
                verbose=False,
            )
            mask_tensor = sam_results[0].masks.data if sam_results and sam_results[0].masks is not None else None
            if mask_tensor is None or len(mask_tensor) == 0:
                merged_mask = np.zeros((height, width), dtype=bool)
            else:
                merged_mask = mask_tensor.cpu().numpy().astype(bool).any(axis=0)

        save_binary_mask(merged_mask, masks_dir / f"{image_path.stem}.png")
