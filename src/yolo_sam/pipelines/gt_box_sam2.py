from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from yolo_sam.data.coco_boxes import load_ground_truth_boxes
from yolo_sam.io_utils import ensure_dir, list_images, save_binary_mask
from yolo_sam.models.sam2_local import LocalSam2ImageSegmenter


def run_gt_box_sam2_pipeline(
    images_dir: Path,
    coco_dir: Path,
    output_dir: Path,
    model_id: str,
    sam_device: str | int,
    torch_dtype: str,
    mask_threshold: float,
    box_batch_size: int,
    hf_token: str | None = None,
) -> None:
    masks_dir = ensure_dir(output_dir / "masks")
    raw_dir = ensure_dir(output_dir / "raw")

    boxes_by_name = load_ground_truth_boxes(coco_dir)
    segmenter = LocalSam2ImageSegmenter(
        model_id=model_id,
        device=sam_device,
        torch_dtype=torch_dtype,
        token=hf_token,
    )

    for index, image_path in enumerate(tqdm(list_images(images_dir), desc="GT bbox + SAM2"), start=1):
        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
            height, width = image.size[1], image.size[0]

        boxes_list = boxes_by_name.get(image_path.name, [])
        if not boxes_list:
            merged_mask = np.zeros((height, width), dtype=bool)
            raw_payload = {
                "num_boxes": 0,
                "num_masks": 0,
                "scores": [],
                "input_boxes": [],
                "prompt_type": "gt_bbox",
            }
        else:
            result = segmenter.segment_boxes(
                image=image,
                boxes=boxes_list,
                mask_threshold=mask_threshold,
                box_batch_size=box_batch_size,
            )
            merged_mask = result.merged_mask
            raw_payload = {
                "num_boxes": len(boxes_list),
                "num_masks": result.num_masks,
                "scores": result.scores,
                "input_boxes": boxes_list,
                "prompt_type": "gt_bbox",
            }

        save_binary_mask(merged_mask, masks_dir / f"{image_path.stem}.png")
        with (raw_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(raw_payload, handle, indent=2)

        segmenter.maybe_clear_cuda_cache(index)
