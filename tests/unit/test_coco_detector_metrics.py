from __future__ import annotations

import unittest

from pycocotools.coco import COCO

from yolo_sam.detection.coco_metrics import (
    bbox_iou_xywh,
    fixed_precision_recall,
    select_f1_confidence_threshold,
)


class CocoDetectorMetricsTest(unittest.TestCase):
    def test_bbox_iou_xywh(self) -> None:
        self.assertEqual(
            bbox_iou_xywh([0, 0, 10, 10], [20, 20, 5, 5]),
            0.0,
        )
        self.assertAlmostEqual(
            bbox_iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]),
            1.0 / 3.0,
        )

    def test_validation_threshold_maximizes_f1_without_test_data(self) -> None:
        coco = COCO()
        coco.dataset = {
            "images": [{"id": 1}, {"id": 2}],
            "categories": [{"id": 1, "name": "plane"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [20, 20, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                },
            ],
        }
        coco.createIndex()
        detections = [
            {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.90},
            {"image_id": 2, "bbox": [20, 20, 10, 10], "score": 0.70},
            {"image_id": 1, "bbox": [40, 40, 5, 5], "score": 0.60},
        ]

        selected = select_f1_confidence_threshold(coco, detections)
        self.assertAlmostEqual(
            float(selected["selected_confidence_threshold"]),
            0.70,
        )
        self.assertAlmostEqual(float(selected["f1"]), 1.0)
        fixed = fixed_precision_recall(
            coco,
            detections,
            iou_threshold=0.50,
            confidence_threshold=float(
                selected["selected_confidence_threshold"]
            ),
        )
        self.assertEqual(fixed["true_positive"], 2)
        self.assertEqual(fixed["false_positive"], 0)


if __name__ == "__main__":
    unittest.main()
