from __future__ import annotations

import unittest

from yolo_sam.data.contracts import BBoxXYWH
from yolo_sam.detection.matching import (
    Detection,
    match_detections_to_ground_truth,
)


class DetectionMatchingTest(unittest.TestCase):
    def test_matching_is_one_to_one_and_confidence_ordered(self) -> None:
        ground_truth = [
            BBoxXYWH(0, 0, 10, 10),
            BBoxXYWH(20, 0, 10, 10),
        ]
        detections = [
            Detection(BBoxXYWH(0, 0, 10, 10), confidence=0.9),
            Detection(BBoxXYWH(1, 0, 10, 10), confidence=0.8),
            Detection(BBoxXYWH(20, 0, 10, 10), confidence=0.7),
        ]
        result = match_detections_to_ground_truth(
            ground_truth,
            detections,
            iou_threshold=0.5,
        )
        self.assertEqual(
            [(match.ground_truth_index, match.detection_index) for match in result.matches],
            [(0, 0), (1, 2)],
        )
        self.assertEqual(result.unmatched_detection_indices, (1,))

    def test_low_iou_detection_leaves_ground_truth_unmatched(self) -> None:
        result = match_detections_to_ground_truth(
            [BBoxXYWH(0, 0, 2, 2)],
            [Detection(BBoxXYWH(10, 10, 2, 2), confidence=0.9)],
            iou_threshold=0.5,
        )
        self.assertEqual(result.unmatched_ground_truth_indices, (0,))
        self.assertEqual(result.unmatched_detection_indices, (0,))


if __name__ == "__main__":
    unittest.main()
