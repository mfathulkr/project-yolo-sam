from __future__ import annotations

import numpy as np


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    true_positive = float(np.logical_and(pred, gt).sum())
    false_positive = float(np.logical_and(pred, ~gt).sum())
    false_negative = float(np.logical_and(~pred, gt).sum())

    union = true_positive + false_positive + false_negative
    pred_area = true_positive + false_positive
    gt_area = true_positive + false_negative

    precision = true_positive / pred_area if pred_area > 0 else 0.0
    recall = true_positive / gt_area if gt_area > 0 else 0.0
    dice_denominator = (2.0 * true_positive) + false_positive + false_negative
    dice = (2.0 * true_positive / dice_denominator) if dice_denominator > 0 else 0.0

    return {
        "iou": true_positive / union if union > 0 else 0.0,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "pred_area_ratio": pred_area / gt_area if gt_area > 0 else 0.0,
    }
