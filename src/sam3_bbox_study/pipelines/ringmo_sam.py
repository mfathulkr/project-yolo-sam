from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sam3_bbox_study.data.coco_boxes import load_ground_truth_boxes
from sam3_bbox_study.io_utils import ensure_dir, list_images, save_binary_mask
from sam3_bbox_study.models.ringmo_sam_local import LocalRingMoSamSegmenter


def load_yolo_boxes_from_raw(raw_dir: Path, image_name: str) -> list[list[float]]:
    raw_path = raw_dir / f"{Path(image_name).stem}.json"
    if not raw_path.exists():
        return []
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    return [[float(value) for value in box] for box in payload.get("input_boxes", [])]


def run_ringmo_sam_pipeline(
    images_dir: Path,
    coco_dir: Path,
    gt_output_dir: Path,
    yolo_output_dir: Path,
    yolo_raw_dir: Path,
    model_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: str,
    normalize: bool,
    class_ids: list[int],
) -> None:
    gt_masks_dir = ensure_dir(gt_output_dir / "masks")
    gt_raw_dir = ensure_dir(gt_output_dir / "raw")
    yolo_masks_dir = ensure_dir(yolo_output_dir / "masks")
    yolo_raw_out_dir = ensure_dir(yolo_output_dir / "raw")

    gt_boxes_by_name = load_ground_truth_boxes(coco_dir)
    segmenter = LocalRingMoSamSegmenter(
        model_root=model_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        normalize=normalize,
        class_ids=class_ids,
    )

    for image_path in tqdm(list_images(images_dir), desc="RingMo-SAM"):
        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
            height, width = image.size[1], image.size[0]

        gt_boxes = gt_boxes_by_name.get(image_path.name, [])
        yolo_boxes = load_yolo_boxes_from_raw(yolo_raw_dir, image_path.name)
        class_map = segmenter.predict_class_map(image)
        if class_map.shape != (height, width):
            class_map_image = Image.fromarray(class_map, mode="L").resize((width, height), Image.Resampling.NEAREST)
            class_map = np.asarray(class_map_image, dtype=np.uint8)

        class_mask = np.isin(class_map, class_ids)
        unique_ids, unique_counts = np.unique(class_map, return_counts=True)
        class_histogram = {str(int(key)): int(value) for key, value in zip(unique_ids, unique_counts, strict=True)}

        for boxes, masks_dir, raw_dir, prompt_type in [
            (gt_boxes, gt_masks_dir, gt_raw_dir, "gt_bbox"),
            (yolo_boxes, yolo_masks_dir, yolo_raw_out_dir, "yolo_bbox"),
        ]:
            box_mask = segmenter.boxes_to_mask(boxes, class_map.shape)
            merged_mask = class_mask & box_mask
            save_binary_mask(merged_mask, masks_dir / f"{image_path.stem}.png")
            payload = {
                "prompt_type": prompt_type,
                "num_boxes": len(boxes),
                "input_boxes": boxes,
                "class_ids": class_ids,
                "class_histogram": class_histogram,
                "normalize_input": normalize,
                "model": "AI-Cyber/RingMo-SAM",
            }
            with (raw_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
