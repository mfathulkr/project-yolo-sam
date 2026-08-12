from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from PIL import Image

from yolo_sam.models.sam1_local import LocalSam1ImageSegmenter
from yolo_sam.models.sam2_local import LocalSam2ImageSegmenter
from yolo_sam.models.sam3_tracker_local import LocalSam3TrackerImageSegmenter


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


class Sam3BoxSegmenter:
    model_version = "sam3"

    def __init__(
        self,
        segmenter: LocalSam3TrackerImageSegmenter,
        model_id: str,
        mask_threshold: float,
        box_batch_size: int = DEFAULT_BOX_BATCH_SIZE,
        model_version: str = "sam3",
    ) -> None:
        self.segmenter = segmenter
        self.model_id = model_id
        self.model_version = model_version
        self.mask_threshold = mask_threshold
        self.box_batch_size = max(1, int(box_batch_size))

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
            box_batch_size=min(self.box_batch_size, max(1, len(boxes_xyxy))),
        )
        if len(result.instance_masks) != len(boxes_xyxy):
            raise ValueError(
                f"SAM3 PVS returned {len(result.instance_masks)} masks for "
                f"{len(boxes_xyxy)} boxes"
            )
        return [
            SingleBoxSegmentation(
                mask=mask.astype(bool),
                score=result.scores[index] if index < len(result.scores) else None,
                metadata={"inference_interface": "sam3_tracker_pvs"},
            )
            for index, mask in enumerate(result.instance_masks)
        ]
