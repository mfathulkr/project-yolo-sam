from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def normalized_rle(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle["counts"]
    return {
        "size": [int(value) for value in rle["size"]],
        "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
    }


def binary_metrics_from_rle(
    prediction_rle: dict[str, Any],
    reference_rle: dict[str, Any],
    *,
    known_positive_instance: bool = True,
) -> dict[str, float | int]:
    prediction = normalized_rle(prediction_rle)
    reference = normalized_rle(reference_rle)
    if prediction["size"] != reference["size"]:
        raise ValueError(
            f"Maske boyutları farklı: {prediction['size']} != {reference['size']}"
        )

    prediction_pixels = int(mask_utils.area(prediction))
    reference_pixels = int(mask_utils.area(reference))
    intersection = int(
        mask_utils.area(mask_utils.merge([prediction, reference], intersect=True))
    )
    false_positive = prediction_pixels - intersection
    false_negative = reference_pixels - intersection
    union = intersection + false_positive + false_negative

    if known_positive_instance and reference_pixels == 0:
        return {
            "iou": 0.0,
            "dice": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "true_positive_pixels": intersection,
            "false_positive_pixels": false_positive,
            "false_negative_pixels": false_negative,
        }

    def ratio(numerator: int, denominator: int, both_empty: float) -> float:
        return both_empty if denominator == 0 else float(numerator / denominator)

    return {
        "iou": ratio(intersection, union, 1.0),
        "dice": ratio(
            2 * intersection,
            prediction_pixels + reference_pixels,
            1.0,
        ),
        "precision": ratio(
            intersection,
            prediction_pixels,
            1.0 if reference_pixels == 0 else 0.0,
        ),
        "recall": ratio(
            intersection,
            reference_pixels,
            1.0 if prediction_pixels == 0 else 0.0,
        ),
        "true_positive_pixels": intersection,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
    }


def compare_dense_and_rle(
    prediction_rle: dict[str, Any],
    reference_rle: dict[str, Any],
) -> dict[str, float]:
    from yolo_sam.evaluation.instance_metrics import binary_mask_metrics

    prediction = mask_utils.decode(normalized_rle(prediction_rle)).astype(bool)
    reference = mask_utils.decode(normalized_rle(reference_rle)).astype(bool)
    dense = binary_mask_metrics(
        prediction,
        reference,
        known_positive_instance=True,
    )
    sparse = binary_metrics_from_rle(prediction_rle, reference_rle)
    return {
        key: abs(float(sparse[key]) - float(getattr(dense, key)))
        for key in ("iou", "dice", "precision", "recall")
    }
