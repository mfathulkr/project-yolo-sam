from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from yolo_sam.data.coco_boxes import load_ground_truth_boxes
from yolo_sam.io_utils import ensure_dir, list_images, save_binary_mask
from yolo_sam.models.sam3_local import LocalSam3ImageSegmenter


def run_gt_box_sam3_pipeline(
    images_dir: Path,
    coco_dir: Path,
    output_dir: Path,
    model_dir: Path,
    prompt: str,
    device: str | int,
    torch_dtype: str,
    output_prob_thresh: float,
    mask_threshold: float,
    hf_token: str | None = None,
) -> None:
    masks_dir = ensure_dir(output_dir / "masks")
    raw_dir = ensure_dir(output_dir / "raw")

    boxes_by_name = load_ground_truth_boxes(coco_dir)
    segmenter = LocalSam3ImageSegmenter(
        model_dir=model_dir,
        device=device,
        torch_dtype=torch_dtype,
        hf_token=hf_token,
    )

    for index, image_path in enumerate(tqdm(list_images(images_dir), desc="GT bbox + SAM3"), start=1):
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
                "sam3_boxes": [],
            }
        else:
            result = segmenter.segment(
                image=image,
                prompt=None,
                boxes=boxes_list,
                box_labels=[1] * len(boxes_list),
                output_prob_thresh=output_prob_thresh,
                mask_threshold=mask_threshold,
            )
            merged_mask = result.merged_mask
            raw_payload = {
                "num_boxes": len(boxes_list),
                "num_masks": result.num_masks,
                "scores": result.scores,
                "input_boxes": boxes_list,
                "prompt_type": "gt_bbox",
                "sam3_boxes": result.boxes,
            }

        save_binary_mask(merged_mask, masks_dir / f"{image_path.stem}.png")
        with (raw_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(raw_payload, handle, indent=2)

        segmenter.maybe_clear_cuda_cache(index)
