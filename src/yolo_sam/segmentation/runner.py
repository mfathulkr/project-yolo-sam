from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from yolo_sam.data.contracts import (
    BBoxSource,
    BBoxXYWH,
    PredictionRecord,
    PredictionStatus,
    PromptType,
    validate_primary_bbox_source,
)
from yolo_sam.segmentation.box_segmenters import BoxSegmenter


@dataclass(frozen=True)
class SegmentationTask:
    image_id: str
    instance_id: str
    bbox: BBoxXYWH
    bbox_source: BBoxSource
    prompt_type: PromptType

    def __post_init__(self) -> None:
        expected_source = {
            PromptType.GT_BBOX: {
                BBoxSource.HUMAN_ANNOTATION,
                BBoxSource.ORIGINAL_DETECTION_ANNOTATION,
            },
            PromptType.YOLO_BBOX: {BBoxSource.YOLO_PREDICTION},
        }[self.prompt_type]
        if self.bbox_source not in expected_source:
            raise ValueError(
                f"{self.prompt_type} is incompatible with bbox source {self.bbox_source}"
            )
        validate_primary_bbox_source(self.bbox_source)


@dataclass(frozen=True)
class CompletedPrediction:
    record: PredictionRecord
    mask: np.ndarray


def encode_binary_mask(mask: np.ndarray) -> dict[str, object]:
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {mask.shape}")
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": counts,
    }


def decode_binary_mask(rle: dict[str, object]) -> np.ndarray:
    return mask_utils.decode(rle).astype(bool)


def bbox_xywh_to_xyxy(bbox: BBoxXYWH) -> list[float]:
    return [
        bbox.x,
        bbox.y,
        bbox.x + bbox.width,
        bbox.y + bbox.height,
    ]


def run_segmentation_tasks(
    run_id: str,
    image: Image.Image,
    tasks: list[SegmentationTask],
    segmenter: BoxSegmenter,
    continue_on_error: bool = False,
) -> list[CompletedPrediction]:
    completed: list[CompletedPrediction] = []
    empty = np.zeros((image.height, image.width), dtype=bool)
    batch_started = time.perf_counter()
    try:
        batch_results = segmenter.segment_boxes(
            image=image,
            boxes_xyxy=[bbox_xywh_to_xyxy(task.bbox) for task in tasks],
        )
        if len(batch_results) != len(tasks):
            raise ValueError(
                f"Segmenter returned {len(batch_results)} results for {len(tasks)} tasks"
            )
    except (AttributeError, NotImplementedError):
        batch_results = [
            segmenter.segment_box(
                image=image,
                box_xyxy=bbox_xywh_to_xyxy(task.bbox),
            )
            for task in tasks
        ]
    batch_elapsed_ms = (time.perf_counter() - batch_started) * 1000.0

    for task, result in zip(tasks, batch_results, strict=True):
        try:
            mask = result.mask.astype(bool)
            if mask.shape != empty.shape:
                raise ValueError(
                    f"Prediction mask shape {mask.shape} does not match image shape {empty.shape}"
                )
            status = (
                PredictionStatus.OK
                if mask.any()
                else PredictionStatus.EMPTY_MASK
            )
            predicted_mask_rle = encode_binary_mask(mask)
            confidence = result.score
        except Exception:
            if not continue_on_error:
                raise
            mask = empty.copy()
            status = PredictionStatus.INFERENCE_ERROR
            predicted_mask_rle = encode_binary_mask(mask)
            confidence = None

        elapsed_ms = batch_elapsed_ms / max(len(tasks), 1)
        record = PredictionRecord(
            run_id=run_id,
            model_id=segmenter.model_id,
            model_version=segmenter.model_version,
            image_id=task.image_id,
            instance_id=task.instance_id,
            prompt_type=task.prompt_type,
            prompt_source=task.bbox_source,
            input_bbox=task.bbox,
            predicted_mask_rle=predicted_mask_rle,
            status=status,
            runtime_ms=elapsed_ms,
            confidence=confidence,
        )
        completed.append(CompletedPrediction(record=record, mask=mask))
    return completed


def write_predictions_jsonl(
    predictions: list[CompletedPrediction],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(
                json.dumps(
                    prediction.record.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
