from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from yolo_sam.segmentation.box_segmenters import (
    DEFAULT_BOX_BATCH_SIZE,
    Sam1BoxSegmenter,
    Sam2BoxSegmenter,
    Sam3BoxSegmenter,
)


class _RecordingSegmenter:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def segment_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        mask_threshold: float,
        box_batch_size: int,
    ) -> SimpleNamespace:
        self.batch_sizes.append(box_batch_size)
        masks = [
            np.zeros((image.height, image.width), dtype=bool)
            for _ in boxes
        ]
        return SimpleNamespace(instance_masks=masks, scores=[0.5] * len(boxes))


class BoxSegmenterTest(unittest.TestCase):
    def test_sam1_and_sam2_cap_dense_scene_box_batches(self) -> None:
        image = Image.new("RGB", (8, 8))
        boxes = [[0.0, 0.0, 2.0, 2.0]] * 40

        for wrapper_class in (Sam1BoxSegmenter, Sam2BoxSegmenter):
            with self.subTest(wrapper=wrapper_class.__name__):
                underlying = _RecordingSegmenter()
                wrapper = wrapper_class(
                    segmenter=underlying,
                    model_id="fixture",
                    mask_threshold=0.0,
                )
                results = wrapper.segment_boxes(image, boxes)

                self.assertEqual(len(results), len(boxes))
                self.assertEqual(
                    underlying.batch_sizes,
                    [DEFAULT_BOX_BATCH_SIZE],
                )

    def test_sam3_pvs_returns_one_mask_per_box_without_score_filtering(self) -> None:
        image = Image.new("RGB", (8, 8))
        boxes = [[0.0, 0.0, 2.0, 2.0]] * 40
        underlying = _RecordingSegmenter()
        wrapper = Sam3BoxSegmenter(
            segmenter=underlying,
            model_id="fixture",
            mask_threshold=0.0,
            box_batch_size=32,
        )

        results = wrapper.segment_boxes(image, boxes)

        self.assertEqual(len(results), len(boxes))
        self.assertEqual(underlying.batch_sizes, [32])
        self.assertTrue(
            all(
                result.metadata["inference_interface"] == "sam3_tracker_pvs"
                for result in results
            )
        )

if __name__ == "__main__":
    unittest.main()
