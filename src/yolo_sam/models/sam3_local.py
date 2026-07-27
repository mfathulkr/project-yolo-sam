from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from yolo_sam.models.download import ensure_sam3_model_dir


def masks_to_merged_mask(masks: object, shape: tuple[int, int]) -> np.ndarray:
    instance_masks = masks_to_instance_masks(masks, shape)
    if not instance_masks:
        return np.zeros(shape, dtype=bool)
    return np.any(np.stack(instance_masks, axis=0), axis=0)


def masks_to_instance_masks(masks: object, shape: tuple[int, int]) -> list[np.ndarray]:
    if masks is None:
        return []
    if torch.is_tensor(masks):
        masks_array = masks.detach().cpu().numpy()
    else:
        masks_array = np.asarray(masks)
    if masks_array.size == 0:
        return []
    if masks_array.shape[-2:] != shape:
        raise ValueError(
            f"Processed mask shape {masks_array.shape[-2:]} does not match image shape {shape}"
        )
    flattened = masks_array.reshape(-1, shape[0], shape[1]).astype(bool)
    return [mask for mask in flattened]


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


def move_inputs_to_device(
    inputs: dict[str, object],
    device: str,
    dtype: torch.dtype,
) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


@dataclass
class Sam3SegmentationResult:
    merged_mask: np.ndarray
    instance_masks: list[np.ndarray]
    num_masks: int
    scores: list[float]
    boxes: list[list[float]]


class LocalSam3ImageSegmenter:
    def __init__(
        self,
        model_dir: Path,
        device: str | int,
        torch_dtype: str,
        hf_token: str | None = None,
    ) -> None:
        local_model_dir = ensure_sam3_model_dir(model_dir, token=hf_token)
        self.device = resolve_torch_device(device)
        self.dtype = resolve_torch_dtype(torch_dtype)
        self.processor = Sam3Processor.from_pretrained(str(local_model_dir), local_files_only=True)
        self.model = Sam3Model.from_pretrained(
            str(local_model_dir),
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def segment(
        self,
        image: Image.Image,
        prompt: str | None,
        output_prob_thresh: float,
        mask_threshold: float,
        boxes: list[list[float]] | None = None,
        box_labels: list[int] | None = None,
    ) -> Sam3SegmentationResult:
        processor_kwargs: dict[str, object] = {
            "images": image,
            "return_tensors": "pt",
        }
        if prompt is not None:
            processor_kwargs["text"] = prompt
        if boxes is not None:
            processor_kwargs["input_boxes"] = [boxes]
            processor_kwargs["input_boxes_labels"] = [box_labels or [1] * len(boxes)]

        inputs = self.processor(**processor_kwargs)
        inputs = move_inputs_to_device(inputs, self.device, self.dtype)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=output_prob_thresh,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        target_shape = tuple(image.size[::-1])
        merged_mask = masks_to_merged_mask(results.get("masks"), target_shape)
        instance_masks = masks_to_instance_masks(results.get("masks"), target_shape)
        scores = results.get("scores", [])
        raw_boxes = results.get("boxes", [])

        if torch.is_tensor(scores):
            scores_list = scores.detach().cpu().tolist()
        else:
            scores_list = list(scores)

        if torch.is_tensor(raw_boxes):
            boxes_list = raw_boxes.detach().cpu().tolist()
        else:
            boxes_list = list(raw_boxes)

        if results.get("masks") is None:
            num_masks = int(merged_mask.any())
        else:
            num_masks = int(len(results.get("masks")))

        return Sam3SegmentationResult(
            merged_mask=merged_mask,
            instance_masks=instance_masks,
            num_masks=num_masks,
            scores=scores_list,
            boxes=boxes_list,
        )

    def maybe_clear_cuda_cache(self, step_index: int, interval: int = 10) -> None:
        if self.device.startswith("cuda") and step_index % interval == 0:
            torch.cuda.empty_cache()
