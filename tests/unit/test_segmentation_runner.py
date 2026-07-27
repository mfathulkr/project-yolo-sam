from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from yolo_sam.data.contracts import (
    BBoxSource,
    BBoxXYWH,
    PredictionStatus,
    PromptType,
)
from yolo_sam.segmentation.box_segmenters import SingleBoxSegmentation
from yolo_sam.segmentation.runner import (
    SegmentationTask,
    decode_binary_mask,
    run_segmentation_tasks,
)


class _FixtureSegmenter:
    model_id = "fixture"
    model_version = "fixture-v1"

    def segment_box(
        self,
        image: Image.Image,
        box_xyxy: list[float],
    ) -> SingleBoxSegmentation:
        mask = np.zeros((image.height, image.width), dtype=bool)
        x1, y1, x2, y2 = (int(value) for value in box_xyxy)
        mask[y1:y2, x1:x2] = True
        return SingleBoxSegmentation(mask=mask, score=0.8)

    def segment_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: list[list[float]],
    ) -> list[SingleBoxSegmentation]:
        return [self.segment_box(image, box) for box in boxes_xyxy]


class SegmentationRunnerTest(unittest.TestCase):
    def test_runner_emits_one_prediction_per_instance(self) -> None:
        image = Image.new("RGB", (8, 8))
        tasks = [
            SegmentationTask(
                image_id="image",
                instance_id="instance",
                bbox=BBoxXYWH(1, 2, 3, 4),
                bbox_source=BBoxSource.HUMAN_ANNOTATION,
                prompt_type=PromptType.GT_BBOX,
            )
        ]
        completed = run_segmentation_tasks("run", image, tasks, _FixtureSegmenter())
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].record.status, PredictionStatus.OK)
        decoded = decode_binary_mask(completed[0].record.predicted_mask_rle or {})
        self.assertEqual(int(decoded.sum()), 12)

    def test_prompt_and_bbox_source_must_match(self) -> None:
        with self.assertRaises(ValueError):
            SegmentationTask(
                image_id="image",
                instance_id="instance",
                bbox=BBoxXYWH(1, 1, 2, 2),
                bbox_source=BBoxSource.YOLO_PREDICTION,
                prompt_type=PromptType.GT_BBOX,
            )


if __name__ == "__main__":
    unittest.main()
