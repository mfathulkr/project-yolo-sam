from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def bbox_iou_xywh(left: list[float], right: list[float]) -> float:
    left_x, left_y, left_width, left_height = left
    right_x, right_y, right_width, right_height = right
    intersection_x1 = max(left_x, right_x)
    intersection_y1 = max(left_y, right_y)
    intersection_x2 = min(left_x + left_width, right_x + right_width)
    intersection_y2 = min(left_y + left_height, right_y + right_height)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    union = (
        left_width * left_height
        + right_width * right_height
        - intersection
    )
    return float(intersection / union) if union > 0 else 0.0


def coco_ap_at(coco_eval: COCOeval, threshold: float) -> float:
    thresholds = np.asarray(coco_eval.params.iouThrs)
    threshold_index = int(np.argmin(np.abs(thresholds - threshold)))
    precision = coco_eval.eval["precision"][threshold_index, :, :, 0, -1]
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else 0.0


def coco_ap50_95(coco_eval: COCOeval) -> float:
    precision = coco_eval.eval["precision"][:, :, :, 0, -1]
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else 0.0


def fixed_precision_recall(
    coco: COCO,
    detections: list[dict[str, Any]],
    *,
    iou_threshold: float,
    confidence_threshold: float,
) -> dict[str, float | int]:
    ground_truth_by_image = {
        int(image_id): [
            [float(value) for value in coco.anns[annotation_id]["bbox"]]
            for annotation_id in coco.getAnnIds(imgIds=[image_id])
        ]
        for image_id in coco.getImgIds()
    }
    detections_by_image: dict[int, list[dict[str, Any]]] = {}
    for detection in detections:
        if float(detection["score"]) < confidence_threshold:
            continue
        detections_by_image.setdefault(int(detection["image_id"]), []).append(
            detection
        )

    true_positive = 0
    false_positive = 0
    total_ground_truth = sum(
        len(boxes) for boxes in ground_truth_by_image.values()
    )
    for image_id, ground_truth_boxes in ground_truth_by_image.items():
        matched_ground_truth: set[int] = set()
        image_detections = sorted(
            detections_by_image.get(image_id, []),
            key=lambda row: float(row["score"]),
            reverse=True,
        )
        for detection in image_detections:
            best_iou = 0.0
            best_index = -1
            detection_bbox = [float(value) for value in detection["bbox"]]
            for ground_truth_index, ground_truth_bbox in enumerate(
                ground_truth_boxes
            ):
                if ground_truth_index in matched_ground_truth:
                    continue
                iou = bbox_iou_xywh(detection_bbox, ground_truth_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_index = ground_truth_index
            if best_index >= 0 and best_iou >= iou_threshold:
                true_positive += 1
                matched_ground_truth.add(best_index)
            else:
                false_positive += 1

    false_negative = total_ground_truth - true_positive
    precision_denominator = true_positive + false_positive
    return {
        "precision": (
            true_positive / float(precision_denominator)
            if precision_denominator
            else 0.0
        ),
        "recall": (
            true_positive / float(total_ground_truth)
            if total_ground_truth
            else 0.0
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def select_f1_confidence_threshold(
    coco: COCO,
    detections: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.50,
) -> dict[str, float | int | str]:
    """Select a confidence threshold on validation detections only."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")

    ground_truth_by_image = {
        int(image_id): [
            [float(value) for value in coco.anns[annotation_id]["bbox"]]
            for annotation_id in coco.getAnnIds(imgIds=[image_id])
        ]
        for image_id in coco.getImgIds()
    }
    total_ground_truth = sum(
        len(boxes) for boxes in ground_truth_by_image.values()
    )
    matched_ground_truth = {
        image_id: set() for image_id in ground_truth_by_image
    }
    ordered = sorted(
        detections,
        key=lambda row: float(row["score"]),
        reverse=True,
    )
    if not ordered:
        return {
            "selection_method": "max_f1",
            "selection_iou_threshold": iou_threshold,
            "selected_confidence_threshold": 1.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": total_ground_truth,
            "candidate_thresholds": 0,
        }

    true_positive = 0
    false_positive = 0
    best: dict[str, float | int | str] | None = None
    best_key = (-1.0, -1.0, -1.0, -1.0)
    candidate_count = 0
    index = 0
    while index < len(ordered):
        threshold = float(ordered[index]["score"])
        end = index
        while (
            end < len(ordered)
            and float(ordered[end]["score"]) == threshold
        ):
            detection = ordered[end]
            image_id = int(detection["image_id"])
            available = ground_truth_by_image.get(image_id)
            if available is None:
                raise ValueError(
                    f"Detection references unknown image ID {image_id}"
                )
            best_iou = 0.0
            best_index = -1
            detection_bbox = [
                float(value) for value in detection["bbox"]
            ]
            for ground_truth_index, ground_truth_bbox in enumerate(available):
                if ground_truth_index in matched_ground_truth[image_id]:
                    continue
                iou = bbox_iou_xywh(detection_bbox, ground_truth_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_index = ground_truth_index
            if best_index >= 0 and best_iou >= iou_threshold:
                matched_ground_truth[image_id].add(best_index)
                true_positive += 1
            else:
                false_positive += 1
            end += 1

        false_negative = total_ground_truth - true_positive
        precision = true_positive / (true_positive + false_positive)
        recall = (
            true_positive / total_ground_truth
            if total_ground_truth
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        candidate_count += 1
        candidate = {
            "selection_method": "max_f1",
            "selection_iou_threshold": iou_threshold,
            "selected_confidence_threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "candidate_thresholds": candidate_count,
        }
        candidate_key = (f1, recall, precision, threshold)
        if candidate_key > best_key:
            best = candidate
            best_key = candidate_key
        index = end

    if best is None:
        raise RuntimeError("Confidence threshold selection produced no candidate")
    best["candidate_thresholds"] = candidate_count
    return best


def evaluate_coco_bbox_detections(
    coco: COCO,
    detections: list[dict[str, Any]],
    *,
    fixed_confidence_threshold: float,
    max_detections: int,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "images": len(coco.getImgIds()),
        "ground_truth_instances": len(coco.getAnnIds()),
        "detections_for_ap": len(detections),
        "fixed_confidence_threshold": fixed_confidence_threshold,
    }
    if detections:
        coco_detections = coco.loadRes(detections)
        coco_eval = COCOeval(coco, coco_detections, "bbox")
        coco_eval.params.maxDets = [1, 10, max_detections]
        coco_eval.evaluate()
        coco_eval.accumulate()
        metrics.update(
            {
                "bbox_AP50": coco_ap_at(coco_eval, 0.50),
                "bbox_AP75": coco_ap_at(coco_eval, 0.75),
                "bbox_AP90": coco_ap_at(coco_eval, 0.90),
                "bbox_AP50_95": coco_ap50_95(coco_eval),
            }
        )
    else:
        metrics.update(
            {
                "bbox_AP50": 0.0,
                "bbox_AP75": 0.0,
                "bbox_AP90": 0.0,
                "bbox_AP50_95": 0.0,
            }
        )

    for threshold in (0.50, 0.75, 0.90):
        threshold_metrics = fixed_precision_recall(
            coco,
            detections,
            iou_threshold=threshold,
            confidence_threshold=fixed_confidence_threshold,
        )
        suffix = int(threshold * 100)
        for name, value in threshold_metrics.items():
            metrics[f"{name}_at_bbox_iou{suffix}"] = value
    return metrics
