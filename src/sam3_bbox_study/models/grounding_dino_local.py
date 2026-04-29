from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam3_bbox_study.models.sam3_local import resolve_torch_device, resolve_torch_dtype


@dataclass
class GroundingDinoDetectionResult:
    boxes: list[list[float]]
    scores: list[float]
    labels: list[str]


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = order[1:][iou <= iou_threshold]

    return keep


class LocalGroundingDinoDetector:
    def __init__(
        self,
        model_id: str,
        device: str | int,
        torch_dtype: str,
        token: str | None = None,
    ) -> None:
        self.device = resolve_torch_device(device)
        self.dtype = resolve_torch_dtype(torch_dtype)
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            token=token,
        ).to(self.device)
        self.model.eval()

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float,
        text_threshold: float,
        nms_iou: float,
        max_boxes: int,
    ) -> GroundingDinoDetectionResult:
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        moved_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if torch.is_floating_point(value):
                    moved_inputs[key] = value.to(device=self.device, dtype=self.dtype)
                else:
                    moved_inputs[key] = value.to(device=self.device)
            else:
                moved_inputs[key] = value

        with torch.inference_mode():
            outputs = self.model(**moved_inputs)

        input_ids = moved_inputs.get("input_ids")
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        raw_boxes = results["boxes"].detach().cpu().numpy() if torch.is_tensor(results["boxes"]) else np.asarray(results["boxes"])
        raw_scores = (
            results["scores"].detach().cpu().numpy() if torch.is_tensor(results["scores"]) else np.asarray(results["scores"])
        )
        labels = [str(label) for label in results.get("text_labels", results.get("labels", []))]

        if len(raw_boxes) == 0:
            return GroundingDinoDetectionResult(boxes=[], scores=[], labels=[])

        keep = nms_numpy(raw_boxes, raw_scores, nms_iou)
        keep = keep[: max(0, int(max_boxes))]
        return GroundingDinoDetectionResult(
            boxes=raw_boxes[keep].astype(float).tolist(),
            scores=raw_scores[keep].astype(float).tolist(),
            labels=[labels[index] if index < len(labels) else "" for index in keep],
        )

    def maybe_clear_cuda_cache(self, step_index: int, interval: int = 10) -> None:
        if self.device.startswith("cuda") and step_index % interval == 0:
            torch.cuda.empty_cache()
