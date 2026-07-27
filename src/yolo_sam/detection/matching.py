from __future__ import annotations

from dataclasses import dataclass

from yolo_sam.data.contracts import BBoxXYWH


@dataclass(frozen=True)
class Detection:
    bbox: BBoxXYWH
    confidence: float
    class_id: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be in [0, 1]")


@dataclass(frozen=True)
class DetectionMatch:
    ground_truth_index: int
    detection_index: int
    bbox_iou: float


@dataclass(frozen=True)
class DetectionMatchingResult:
    matches: tuple[DetectionMatch, ...]
    unmatched_ground_truth_indices: tuple[int, ...]
    unmatched_detection_indices: tuple[int, ...]


def bbox_iou_xywh(left: BBoxXYWH, right: BBoxXYWH) -> float:
    left_x2 = left.x + left.width
    left_y2 = left.y + left.height
    right_x2 = right.x + right.width
    right_y2 = right.y + right.height
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left.x, right.x))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left.y, right.y))
    intersection = intersection_width * intersection_height
    union = left.area + right.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def match_detections_to_ground_truth(
    ground_truth_boxes: list[BBoxXYWH],
    detections: list[Detection],
    iou_threshold: float,
) -> DetectionMatchingResult:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    unmatched_ground_truth = set(range(len(ground_truth_boxes)))
    unmatched_detections = set(range(len(detections)))
    matches: list[DetectionMatch] = []

    detection_order = sorted(
        range(len(detections)),
        key=lambda index: (-detections[index].confidence, index),
    )
    for detection_index in detection_order:
        if not unmatched_ground_truth:
            break
        candidates = [
            (
                bbox_iou_xywh(
                    ground_truth_boxes[ground_truth_index],
                    detections[detection_index].bbox,
                ),
                ground_truth_index,
            )
            for ground_truth_index in unmatched_ground_truth
        ]
        best_iou, best_ground_truth_index = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
        if best_iou < iou_threshold:
            continue
        matches.append(
            DetectionMatch(
                ground_truth_index=best_ground_truth_index,
                detection_index=detection_index,
                bbox_iou=best_iou,
            )
        )
        unmatched_ground_truth.remove(best_ground_truth_index)
        unmatched_detections.remove(detection_index)

    return DetectionMatchingResult(
        matches=tuple(sorted(matches, key=lambda match: match.ground_truth_index)),
        unmatched_ground_truth_indices=tuple(sorted(unmatched_ground_truth)),
        unmatched_detection_indices=tuple(sorted(unmatched_detections)),
    )
