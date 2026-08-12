from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

from yolo_sam.models.download import ensure_sam3_model_dir
from yolo_sam.models.sam2_local import masks_to_instance_masks
from yolo_sam.models.sam3_local import (
    move_inputs_to_device,
    resolve_torch_device,
    resolve_torch_dtype,
)


@dataclass
class Sam3TrackerSegmentationResult:
    instance_masks: list[np.ndarray]
    scores: list[float]


class LocalSam3TrackerImageSegmenter:
    """SAM3 PVS image interface for one instance mask per bbox prompt."""

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
        self.processor = Sam3TrackerProcessor.from_pretrained(
            str(local_model_dir),
            local_files_only=True,
        )
        self.model = Sam3TrackerModel.from_pretrained(
            str(local_model_dir),
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def segment_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        mask_threshold: float,
        box_batch_size: int = 16,
    ) -> Sam3TrackerSegmentationResult:
        height, width = image.size[1], image.size[0]
        if not boxes:
            return Sam3TrackerSegmentationResult(instance_masks=[], scores=[])

        instance_masks: list[np.ndarray] = []
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
            batch_instance_masks = masks_to_instance_masks(
                processed_masks,
                (height, width),
            )
            if len(batch_instance_masks) != len(batch_boxes):
                raise ValueError(
                    "SAM3 PVS returned "
                    f"{len(batch_instance_masks)} masks for {len(batch_boxes)} boxes"
                )
            instance_masks.extend(batch_instance_masks)

            raw_scores = getattr(outputs, "iou_scores", None)
            if raw_scores is not None:
                if torch.is_tensor(raw_scores):
                    scores.extend(
                        float(value)
                        for value in raw_scores.detach().cpu().reshape(-1).tolist()
                    )
                else:
                    scores.extend(
                        float(value)
                        for value in np.asarray(raw_scores).reshape(-1).tolist()
                    )

        return Sam3TrackerSegmentationResult(
            instance_masks=instance_masks,
            scores=scores,
        )

    def maybe_clear_cuda_cache(self, step_index: int, interval: int = 10) -> None:
        if self.device.startswith("cuda") and step_index % interval == 0:
            torch.cuda.empty_cache()
