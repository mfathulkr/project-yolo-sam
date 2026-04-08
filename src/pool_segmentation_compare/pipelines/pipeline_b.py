from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from pool_segmentation_compare.io_utils import ensure_dir, list_images, save_binary_mask
from pool_segmentation_compare.models.sam3_local import LocalSam3ImageSegmenter


def encode_image_base64(image_path: Path) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError(f"Failed to encode image: {image_path}")
    return base64.b64encode(buffer).decode("utf-8")


def polygons_to_mask(polygons: list[list[list[float]]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.int32)
        if len(points) < 3:
            continue
        cv2.fillPoly(mask, [points], color=1)
    return mask.astype(bool)


def response_to_mask(response_json: dict, shape: tuple[int, int]) -> np.ndarray:
    merged = np.zeros(shape, dtype=bool)
    for prompt_result in response_json.get("prompt_results", []):
        for prediction in prompt_result.get("predictions", []):
            polygons = prediction.get("masks", [])
            if polygons:
                merged |= polygons_to_mask(polygons, shape)
    return merged


def run_sam3_hosted_pipeline(
    images_dir: Path,
    output_dir: Path,
    api_key: str,
    endpoint: str,
    prompt: str,
    model_id: str,
    output_prob_thresh: float,
    timeout_seconds: int = 180,
) -> None:
    masks_dir = ensure_dir(output_dir / "masks")
    raw_dir = ensure_dir(output_dir / "raw")

    for image_path in tqdm(list_images(images_dir), desc="Pipeline B (hosted)"):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]

        payload = {
            "image": {"type": "base64", "value": encode_image_base64(image_path)},
            "prompts": [{"type": "text", "text": prompt}],
            "output_prob_thresh": output_prob_thresh,
            "format": "polygon",
            "model_id": model_id,
        }

        response = requests.post(
            endpoint,
            params={"api_key": api_key},
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        response_json = response.json()

        with (raw_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(response_json, handle, indent=2)

        merged_mask = response_to_mask(response_json, (height, width))
        save_binary_mask(merged_mask, masks_dir / f"{image_path.stem}.png")


def run_sam3_local_pipeline(
    images_dir: Path,
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

    segmenter = LocalSam3ImageSegmenter(
        model_dir=model_dir,
        device=device,
        torch_dtype=torch_dtype,
        hf_token=hf_token,
    )

    for index, image_path in enumerate(tqdm(list_images(images_dir), desc="Pipeline B (local)"), start=1):
        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
        result = segmenter.segment(
            image=image,
            prompt=prompt,
            output_prob_thresh=output_prob_thresh,
            mask_threshold=mask_threshold,
        )
        save_binary_mask(result.merged_mask, masks_dir / f"{image_path.stem}.png")

        raw_payload = {
            "num_masks": result.num_masks,
            "scores": result.scores,
            "boxes": result.boxes,
        }
        with (raw_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(raw_payload, handle, indent=2)

        segmenter.maybe_clear_cuda_cache(index)
