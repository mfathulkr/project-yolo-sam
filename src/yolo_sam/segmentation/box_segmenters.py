from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from yolo_sam.models.sam1_local import LocalSam1ImageSegmenter
from yolo_sam.models.sam2_local import LocalSam2ImageSegmenter
from yolo_sam.models.sam3_local import LocalSam3ImageSegmenter


DEFAULT_BOX_BATCH_SIZE = 16


@dataclass(frozen=True)
class SingleBoxSegmentation:
    mask: np.ndarray
    score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got {self.mask.shape}")


class BoxSegmenter(Protocol):
    model_id: str
    model_version: str

    def segment_box(
        self,
        image: Image.Image,
        box_xyxy: list[float],
    ) -> SingleBoxSegmentation: ...

    def segment_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: list[list[float]],
    ) -> list[SingleBoxSegmentation]: ...


def _single_mask_or_empty(
    masks: list[np.ndarray],
    image: Image.Image,
) -> np.ndarray:
    if not masks:
        return np.zeros((image.height, image.width), dtype=bool)
    if len(masks) != 1:
        raise ValueError(f"Single-bbox inference returned {len(masks)} masks")
    return masks[0].astype(bool)


class Sam1BoxSegmenter:
    model_version = "sam1"

    def __init__(
        self,
        segmenter: LocalSam1ImageSegmenter,
        model_id: str,
        mask_threshold: float,
        model_version: str = "sam1",
    ) -> None:
        self.segmenter = segmenter
        self.model_id = model_id
        self.model_version = model_version
        self.mask_threshold = mask_threshold

    def segment_box(self, image: Image.Image, box_xyxy: list[float]) -> SingleBoxSegmentation:
        return self.segment_boxes(image, [box_xyxy])[0]

    def segment_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: list[list[float]],
    ) -> list[SingleBoxSegmentation]:
        result = self.segmenter.segment_boxes(
            image=image,
            boxes=boxes_xyxy,
            mask_threshold=self.mask_threshold,
            box_batch_size=min(DEFAULT_BOX_BATCH_SIZE, max(1, len(boxes_xyxy))),
        )
        if len(result.instance_masks) != len(boxes_xyxy):
            raise ValueError(
                f"SAM1 returned {len(result.instance_masks)} masks for {len(boxes_xyxy)} boxes"
            )
        return [
            SingleBoxSegmentation(
                mask=mask.astype(bool),
                score=result.scores[index] if index < len(result.scores) else None,
            )
            for index, mask in enumerate(result.instance_masks)
        ]


class Sam2BoxSegmenter:
    model_version = "sam2"

    def __init__(
        self,
        segmenter: LocalSam2ImageSegmenter,
        model_id: str,
        mask_threshold: float,
        model_version: str = "sam2",
    ) -> None:
        self.segmenter = segmenter
        self.model_id = model_id
        self.model_version = model_version
        self.mask_threshold = mask_threshold

    def segment_box(self, image: Image.Image, box_xyxy: list[float]) -> SingleBoxSegmentation:
        return self.segment_boxes(image, [box_xyxy])[0]

    def segment_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: list[list[float]],
    ) -> list[SingleBoxSegmentation]:
        result = self.segmenter.segment_boxes(
            image=image,
            boxes=boxes_xyxy,
            mask_threshold=self.mask_threshold,
            box_batch_size=min(DEFAULT_BOX_BATCH_SIZE, max(1, len(boxes_xyxy))),
        )
        if len(result.instance_masks) != len(boxes_xyxy):
            raise ValueError(
                f"SAM2 returned {len(result.instance_masks)} masks for {len(boxes_xyxy)} boxes"
            )
        return [
            SingleBoxSegmentation(
                mask=mask.astype(bool),
                score=result.scores[index] if index < len(result.scores) else None,
            )
            for index, mask in enumerate(result.instance_masks)
        ]


def bbox_iou_xyxy(left: list[float], right: list[float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    inter_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    inter_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = inter_width * inter_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def match_masks_to_input_boxes(
    input_boxes: list[list[float]],
    output_boxes: list[list[float]],
    output_masks: list[np.ndarray],
    output_scores: list[float],
    image: Image.Image,
    *,
    min_match_iou: float = 0.0,
) -> list[SingleBoxSegmentation]:
    if len(output_boxes) != len(output_masks):
        raise ValueError(
            f"SAM3 returned {len(output_boxes)} boxes and {len(output_masks)} masks"
        )
    if min_match_iou < 0.0 or min_match_iou > 1.0:
        raise ValueError("min_match_iou must be in [0, 1]")

    def empty_result() -> SingleBoxSegmentation:
        return SingleBoxSegmentation(
            mask=np.zeros((image.height, image.width), dtype=bool),
            score=None,
            metadata={"matched_output_index": None, "match_iou": 0.0},
        )

    matched = [empty_result() for _ in input_boxes]
    if not input_boxes or not output_boxes:
        return matched

    iou_matrix = np.asarray(
        [
            [
                bbox_iou_xyxy(input_box, output_box)
                for output_box in output_boxes
            ]
            for input_box in input_boxes
        ],
        dtype=np.float64,
    )
    input_indices, output_indices = linear_sum_assignment(
        iou_matrix,
        maximize=True,
    )
    for input_index, output_index in zip(
        input_indices.tolist(),
        output_indices.tolist(),
        strict=True,
    ):
        match_iou = float(iou_matrix[input_index, output_index])
        if match_iou <= min_match_iou:
            continue
        matched[input_index] = SingleBoxSegmentation(
            mask=output_masks[output_index].astype(bool),
            score=(
                float(output_scores[output_index])
                if output_index < len(output_scores)
                else None
            ),
            metadata={
                "matched_output_index": output_index,
                "match_iou": match_iou,
            },
        )
    return matched


class Sam3BoxSegmenter:
    model_version = "sam3"

    def __init__(
        self,
        segmenter: LocalSam3ImageSegmenter,
        model_id: str,
        output_prob_threshold: float,
        mask_threshold: float,
        model_version: str = "sam3",
    ) -> None:
        self.segmenter = segmenter
        self.model_id = model_id
        self.model_version = model_version
        self.output_prob_threshold = output_prob_threshold
        self.mask_threshold = mask_threshold

    def segment_box(self, image: Image.Image, box_xyxy: list[float]) -> SingleBoxSegmentation:
        return self.segment_boxes(image, [box_xyxy])[0]

    def segment_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: list[list[float]],
    ) -> list[SingleBoxSegmentation]:
        result = self.segmenter.segment(
            image=image,
            prompt=None,
            output_prob_thresh=self.output_prob_threshold,
            mask_threshold=self.mask_threshold,
            boxes=boxes_xyxy,
            box_labels=[1] * len(boxes_xyxy),
        )
        return match_masks_to_input_boxes(
            input_boxes=boxes_xyxy,
            output_boxes=result.boxes,
            output_masks=result.instance_masks,
            output_scores=result.scores,
            image=image,
        )
