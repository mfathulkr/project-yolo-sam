from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from yolo_sam.io_utils import ensure_dir, list_images, save_binary_mask
from isaid_vehicle.models.grounding_dino_local import LocalGroundingDinoDetector
from yolo_sam.models.sam2_local import LocalSam2ImageSegmenter


def run_grounded_sam2_pipeline(
    images_dir: Path,
    output_dir: Path,
    detector_model_id: str,
    detector_device: str | int,
    detector_torch_dtype: str,
    text_prompt: str,
    threshold: float,
    text_threshold: float,
    nms_iou: float,
    max_boxes: int,
    sam2_model_id: str,
    sam2_device: str | int,
    sam2_torch_dtype: str,
    mask_threshold: float,
    box_batch_size: int,
    hf_token: str | None = None,
) -> None:
    masks_dir = ensure_dir(output_dir / "masks")
    raw_dir = ensure_dir(output_dir / "raw")

    detector = LocalGroundingDinoDetector(
        model_id=detector_model_id,
        device=detector_device,
        torch_dtype=detector_torch_dtype,
        token=hf_token,
    )
    segmenter = LocalSam2ImageSegmenter(
        model_id=sam2_model_id,
        device=sam2_device,
        torch_dtype=sam2_torch_dtype,
        token=hf_token,
    )

    for index, image_path in enumerate(tqdm(list_images(images_dir), desc="GroundingDINO + SAM2"), start=1):
        mask_path = masks_dir / f"{image_path.stem}.png"
        raw_path = raw_dir / f"{image_path.stem}.json"
        if mask_path.exists() and raw_path.exists():
            continue

        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
            height, width = image.size[1], image.size[0]

        detections = detector.detect(
            image=image,
            text_prompt=text_prompt,
            threshold=threshold,
            text_threshold=text_threshold,
            nms_iou=nms_iou,
            max_boxes=max_boxes,
        )
        if not detections.boxes:
            merged_mask = np.zeros((height, width), dtype=bool)
            scores: list[float] = []
        else:
            result = segmenter.segment_boxes(
                image=image,
                boxes=detections.boxes,
                mask_threshold=mask_threshold,
                box_batch_size=box_batch_size,
            )
            merged_mask = result.merged_mask
            scores = result.scores

        save_binary_mask(merged_mask, mask_path)
        raw_payload = {
            "prompt_type": "grounding_dino_text_to_bbox",
            "text_prompt": text_prompt,
            "num_boxes": len(detections.boxes),
            "num_masks": len(detections.boxes),
            "detection_scores": detections.scores,
            "sam2_scores": scores,
            "labels": detections.labels,
            "input_boxes": detections.boxes,
        }
        with raw_path.open("w", encoding="utf-8") as handle:
            json.dump(raw_payload, handle, indent=2)

        detector.maybe_clear_cuda_cache(index)
        segmenter.maybe_clear_cuda_cache(index)
