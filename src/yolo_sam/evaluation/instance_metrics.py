from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class BinaryMaskMetrics:
    iou: float
    dice: float
    precision: float
    recall: float
    boundary_iou: float
    true_positive_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int


@dataclass(frozen=True)
class InstanceMetricRow:
    run_id: str
    model_id: str
    model_version: str
    prompt_type: str
    image_id: str
    instance_id: str
    source_scene_id: str
    reference_type: str
    stratum: str
    iou: float
    dice: float
    precision: float
    recall: float
    boundary_iou: float
    true_positive_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AggregateMetrics:
    count: int
    mean_iou: float
    mean_dice: float
    mean_precision: float
    mean_recall: float
    mean_boundary_iou: float
    success_at_iou_50: float
    success_at_iou_75: float
    success_at_iou_90: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_ratio(numerator: int, denominator: int, both_empty_value: float) -> float:
    if denominator == 0:
        return both_empty_value
    return float(numerator / denominator)


def mask_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {mask.shape}")
    binary = mask.astype(np.uint8)
    if not binary.any():
        return binary.astype(bool)
    diagonal = float(np.hypot(mask.shape[0], mask.shape[1]))
    iterations = max(1, int(round(dilation_ratio * diagonal)))
    eroded = cv2.erode(
        binary,
        np.ones((3, 3), dtype=np.uint8),
        iterations=iterations,
    )
    return (binary.astype(bool) & ~eroded.astype(bool))


def binary_mask_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    boundary_dilation_ratio: float = 0.02,
) -> BinaryMaskMetrics:
    if prediction.shape != reference.shape:
        raise ValueError(
            f"Prediction and reference shapes differ: {prediction.shape} vs {reference.shape}"
        )
    if prediction.ndim != 2:
        raise ValueError(f"Expected 2D masks, got {prediction.shape}")

    pred = prediction.astype(bool)
    ref = reference.astype(bool)
    true_positive = int(np.logical_and(pred, ref).sum())
    false_positive = int(np.logical_and(pred, ~ref).sum())
    false_negative = int(np.logical_and(~pred, ref).sum())
    union = true_positive + false_positive + false_negative
    pred_pixels = true_positive + false_positive
    ref_pixels = true_positive + false_negative

    pred_boundary = mask_boundary(pred, dilation_ratio=boundary_dilation_ratio)
    ref_boundary = mask_boundary(ref, dilation_ratio=boundary_dilation_ratio)
    boundary_intersection = int(np.logical_and(pred_boundary, ref_boundary).sum())
    boundary_union = int(np.logical_or(pred_boundary, ref_boundary).sum())

    return BinaryMaskMetrics(
        iou=_safe_ratio(true_positive, union, both_empty_value=1.0),
        dice=_safe_ratio(2 * true_positive, pred_pixels + ref_pixels, both_empty_value=1.0),
        precision=_safe_ratio(
            true_positive,
            pred_pixels,
            both_empty_value=1.0 if ref_pixels == 0 else 0.0,
        ),
        recall=_safe_ratio(
            true_positive,
            ref_pixels,
            both_empty_value=1.0 if pred_pixels == 0 else 0.0,
        ),
        boundary_iou=_safe_ratio(
            boundary_intersection,
            boundary_union,
            both_empty_value=1.0,
        ),
        true_positive_pixels=true_positive,
        false_positive_pixels=false_positive,
        false_negative_pixels=false_negative,
    )


def aggregate_instance_metrics(rows: Iterable[InstanceMetricRow]) -> AggregateMetrics:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot aggregate an empty metric row collection")
    ious = np.asarray([row.iou for row in materialized], dtype=np.float64)
    return AggregateMetrics(
        count=len(materialized),
        mean_iou=float(ious.mean()),
        mean_dice=float(np.mean([row.dice for row in materialized])),
        mean_precision=float(np.mean([row.precision for row in materialized])),
        mean_recall=float(np.mean([row.recall for row in materialized])),
        mean_boundary_iou=float(np.mean([row.boundary_iou for row in materialized])),
        success_at_iou_50=float(np.mean(ious >= 0.50)),
        success_at_iou_75=float(np.mean(ious >= 0.75)),
        success_at_iou_90=float(np.mean(ious >= 0.90)),
    )


def evaluate_prediction_references(
    *,
    run_id: str,
    model_id: str,
    model_version: str,
    prompt_type: str,
    image_id: str,
    instance_id: str,
    source_scene_id: str,
    stratum: str,
    prediction: np.ndarray,
    references: dict[str, np.ndarray],
    boundary_dilation_ratio: float = 0.02,
) -> list[InstanceMetricRow]:
    if not references:
        raise ValueError("At least one reference mask is required")
    rows: list[InstanceMetricRow] = []
    for reference_type, reference in sorted(references.items()):
        metrics = binary_mask_metrics(
            prediction,
            reference,
            boundary_dilation_ratio=boundary_dilation_ratio,
        )
        rows.append(
            InstanceMetricRow(
                run_id=run_id,
                model_id=model_id,
                model_version=model_version,
                prompt_type=prompt_type,
                image_id=image_id,
                instance_id=instance_id,
                source_scene_id=source_scene_id,
                reference_type=reference_type,
                stratum=stratum,
                iou=metrics.iou,
                dice=metrics.dice,
                precision=metrics.precision,
                recall=metrics.recall,
                boundary_iou=metrics.boundary_iou,
                true_positive_pixels=metrics.true_positive_pixels,
                false_positive_pixels=metrics.false_positive_pixels,
                false_negative_pixels=metrics.false_negative_pixels,
            )
        )
    return rows


def reference_inflation_rows(
    rows: Iterable[InstanceMetricRow],
    human_reference_type: str = "human",
    pseudo_reference_type: str = "pseudo_sam1",
) -> list[dict[str, object]]:
    paired: dict[tuple[str, str, str, str], dict[str, InstanceMetricRow]] = {}
    for row in rows:
        key = (row.run_id, row.model_id, row.prompt_type, row.instance_id)
        paired.setdefault(key, {})[row.reference_type] = row

    output: list[dict[str, object]] = []
    for key, references in sorted(paired.items()):
        if human_reference_type not in references or pseudo_reference_type not in references:
            continue
        human = references[human_reference_type]
        pseudo = references[pseudo_reference_type]
        output.append(
            {
                "run_id": key[0],
                "model_id": key[1],
                "prompt_type": key[2],
                "instance_id": key[3],
                "source_scene_id": human.source_scene_id,
                "stratum": human.stratum,
                "human_iou": human.iou,
                "pseudo_iou": pseudo.iou,
                "iou_inflation": pseudo.iou - human.iou,
                "dice_inflation": pseudo.dice - human.dice,
                "precision_inflation": pseudo.precision - human.precision,
                "recall_inflation": pseudo.recall - human.recall,
                "boundary_iou_inflation": pseudo.boundary_iou - human.boundary_iou,
            }
        )
    return output
