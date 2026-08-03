from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from yolo_sam.segmentation.box_segmenters import (
    DEFAULT_BOX_BATCH_SIZE,
    Sam1BoxSegmenter,
    Sam2BoxSegmenter,
    bbox_iou_xyxy,
    match_masks_to_input_boxes,
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

    def test_sam3_outputs_are_matched_one_to_one_by_box_iou(self) -> None:
        image = Image.new("RGB", (8, 8))
        left_mask = np.zeros((8, 8), dtype=bool)
        right_mask = np.zeros((8, 8), dtype=bool)
        left_mask[:, :4] = True
        right_mask[:, 4:] = True
        matched = match_masks_to_input_boxes(
            input_boxes=[[0, 0, 4, 8], [4, 0, 8, 8]],
            output_boxes=[[4, 0, 8, 8], [0, 0, 4, 8]],
            output_masks=[right_mask, left_mask],
            output_scores=[0.7, 0.8],
            image=image,
        )
        self.assertTrue(np.array_equal(matched[0].mask, left_mask))
        self.assertTrue(np.array_equal(matched[1].mask, right_mask))

    def test_missing_early_output_does_not_shift_later_sam3_masks(self) -> None:
        image = Image.new("RGB", (8, 8))
        right_mask = np.zeros((8, 8), dtype=bool)
        right_mask[:2, 4:6] = True

        matched = match_masks_to_input_boxes(
            input_boxes=[[0, 0, 2, 2], [4, 0, 6, 2]],
            output_boxes=[[4, 0, 6, 2]],
            output_masks=[right_mask],
            output_scores=[0.9],
            image=image,
        )

        self.assertFalse(matched[0].mask.any())
        self.assertIsNone(matched[0].metadata["matched_output_index"])
        self.assertTrue(np.array_equal(matched[1].mask, right_mask))
        self.assertEqual(matched[1].metadata["matched_output_index"], 0)

    def test_zero_iou_sam3_output_is_not_assigned(self) -> None:
        image = Image.new("RGB", (8, 8))
        unrelated_mask = np.zeros((8, 8), dtype=bool)
        unrelated_mask[6:, 6:] = True

        matched = match_masks_to_input_boxes(
            input_boxes=[[0, 0, 2, 2]],
            output_boxes=[[6, 6, 8, 8]],
            output_masks=[unrelated_mask],
            output_scores=[0.9],
            image=image,
        )

        self.assertFalse(matched[0].mask.any())
        self.assertIsNone(matched[0].metadata["matched_output_index"])

    def test_bbox_iou_xyxy(self) -> None:
        self.assertEqual(bbox_iou_xyxy([0, 0, 2, 2], [3, 3, 4, 4]), 0.0)
        self.assertAlmostEqual(
            bbox_iou_xyxy([0, 0, 2, 2], [1, 0, 3, 2]),
            1 / 3,
        )


if __name__ == "__main__":
    unittest.main()
