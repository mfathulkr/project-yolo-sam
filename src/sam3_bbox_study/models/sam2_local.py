from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

from sam3_bbox_study.models.sam3_local import move_inputs_to_device, resolve_torch_device, resolve_torch_dtype


def collapse_masks_to_merged_mask(masks: object, shape: tuple[int, int]) -> np.ndarray:
    if masks is None:
        return np.zeros(shape, dtype=bool)
    if torch.is_tensor(masks):
        masks_array = masks.detach().cpu().numpy()
    else:
        masks_array = np.asarray(masks)
    if masks_array.size == 0:
        return np.zeros(shape, dtype=bool)
    if masks_array.ndim == 2:
        return masks_array.astype(bool)
    if masks_array.shape[-2:] != shape:
        masks_array = np.asarray(masks_array)
    reduce_axes = tuple(range(masks_array.ndim - 2))
    return masks_array.astype(bool).any(axis=reduce_axes)


@dataclass
class Sam2SegmentationResult:
    merged_mask: np.ndarray
    num_masks: int
    scores: list[float]


class LocalSam2ImageSegmenter:
    def __init__(
        self,
        model_id: str,
        device: str | int,
        torch_dtype: str,
        token: str | None = None,
    ) -> None:
        self.device = resolve_torch_device(device)
        self.dtype = resolve_torch_dtype(torch_dtype)
        self.processor = Sam2Processor.from_pretrained(model_id, token=token)
        self.model = Sam2Model.from_pretrained(model_id, torch_dtype=self.dtype, token=token).to(self.device)
        self.model.eval()

    def segment_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        mask_threshold: float,
        box_batch_size: int = 16,
    ) -> Sam2SegmentationResult:
        height, width = image.size[1], image.size[0]
        if not boxes:
            return Sam2SegmentationResult(
                merged_mask=np.zeros((height, width), dtype=bool),
                num_masks=0,
                scores=[],
            )

        merged = np.zeros((height, width), dtype=bool)
        scores: list[float] = []
        box_batch_size = max(1, int(box_batch_size))

        for start in range(0, len(boxes), box_batch_size):
            batch_boxes = boxes[start : start + box_batch_size]
            inputs = self.processor(
                images=image,
                input_boxes=[batch_boxes],
                return_tensors="pt",
            )
            inputs = move_inputs_to_device(inputs, self.device, self.dtype)
            with torch.inference_mode():
                outputs = self.model(**inputs, multimask_output=False)

            processed_masks = self.processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                mask_threshold=mask_threshold,
                binarize=True,
            )
            merged |= collapse_masks_to_merged_mask(processed_masks, (height, width))

            raw_scores = getattr(outputs, "iou_scores", None)
            if raw_scores is not None:
                if torch.is_tensor(raw_scores):
                    scores.extend(float(value) for value in raw_scores.detach().cpu().reshape(-1).tolist())
                else:
                    scores.extend(float(value) for value in np.asarray(raw_scores).reshape(-1).tolist())

        return Sam2SegmentationResult(
            merged_mask=merged,
            num_masks=len(boxes),
            scores=scores,
        )

    def maybe_clear_cuda_cache(self, step_index: int, interval: int = 10) -> None:
        if self.device.startswith("cuda") and step_index % interval == 0:
            torch.cuda.empty_cache()
