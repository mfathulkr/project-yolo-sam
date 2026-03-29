from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

from pool_segmentation_compare.io_utils import ensure_dir, list_images, save_binary_mask
from pool_segmentation_compare.models.download import ensure_sam3_model_dir


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


def masks_to_merged_mask(masks: object, shape: tuple[int, int]) -> np.ndarray:
    if masks is None:
        return np.zeros(shape, dtype=bool)

    if torch.is_tensor(masks):
        masks_array = masks.detach().cpu().numpy()
    else:
        masks_array = np.asarray(masks)

    if masks_array.size == 0:
        return np.zeros(shape, dtype=bool)
    if masks_array.ndim == 2:
        masks_array = masks_array[None, ...]
    return masks_array.astype(bool).any(axis=0)


def resolve_torch_device(device_value: str | int) -> str:
    if isinstance(device_value, int):
        return f"cuda:{device_value}"

    normalized = str(device_value).strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        return normalized
    if normalized.isdigit():
        return f"cuda:{normalized}"
    return normalized


def resolve_torch_dtype(dtype_value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    normalized = str(dtype_value).strip().lower()
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_value}")
    return mapping[normalized]


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

    local_model_dir = ensure_sam3_model_dir(model_dir, token=hf_token)
    resolved_device = resolve_torch_device(device)
    resolved_dtype = resolve_torch_dtype(torch_dtype)

    processor = Sam3Processor.from_pretrained(str(local_model_dir), local_files_only=True)
    model = Sam3Model.from_pretrained(
        str(local_model_dir),
        local_files_only=True,
        torch_dtype=resolved_dtype,
    ).to(resolved_device)
    model.eval()

    for index, image_path in enumerate(tqdm(list_images(images_dir), desc="Pipeline B (local)"), start=1):
        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(resolved_device)

        with torch.inference_mode():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=output_prob_thresh,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        merged_mask = masks_to_merged_mask(results.get("masks"), tuple(image.size[::-1]))
        save_binary_mask(merged_mask, masks_dir / f"{image_path.stem}.png")

        raw_payload = {
            "num_masks": int(merged_mask.any()) if results.get("masks") is None else int(len(results.get("masks"))),
            "scores": results.get("scores", []).tolist() if torch.is_tensor(results.get("scores")) else results.get("scores", []),
        }
        with (raw_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(raw_payload, handle, indent=2)

        if resolved_device.startswith("cuda") and index % 10 == 0:
            torch.cuda.empty_cache()
