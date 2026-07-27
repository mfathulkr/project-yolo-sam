from __future__ import annotations

import unittest

from yolo_sam.data.contracts import (
    BBoxSource,
    BBoxXYWH,
    ReferenceType,
    validate_primary_bbox_source,
)
from yolo_sam.data.samrs import clip_bbox_to_image, horizontal_bbox_from_rhbox
from yolo_sam.data.isaid import clip_bbox_to_tile


class ContractsTest(unittest.TestCase):
    def test_bbox_clipping(self) -> None:
        bbox = BBoxXYWH(x=-2, y=4, width=10, height=12)
        clipped = bbox.clipped(image_width=6, image_height=10)
        self.assertEqual(clipped.to_list(), [0.0, 4, 6.0, 6.0])

    def test_invalid_bbox_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BBoxXYWH(x=0, y=0, width=0, height=1)

    def test_mask_derived_bbox_is_rejected_for_primary_study(self) -> None:
        with self.assertRaises(ValueError):
            validate_primary_bbox_source(BBoxSource.MASK_DERIVED)
        validate_primary_bbox_source(BBoxSource.ORIGINAL_DETECTION_ANNOTATION)

    def test_reference_type_values_are_stable(self) -> None:
        self.assertEqual(ReferenceType.HUMAN.value, "human")
        self.assertEqual(ReferenceType.PSEUDO_SAM1.value, "pseudo_sam1")

    def test_rhbox_conversion_uses_original_box_geometry(self) -> None:
        self.assertEqual(
            horizontal_bbox_from_rhbox([10, 20, 35, 55]),
            [10.0, 20.0, 25.0, 35.0],
        )
        with self.assertRaises(ValueError):
            horizontal_bbox_from_rhbox([10, 20, 5, 55])

    def test_original_detection_bbox_is_clipped_to_image_bounds(self) -> None:
        self.assertEqual(
            clip_bbox_to_image(
                [-10.0, 900.0, 50.0, 200.0],
                image_width=1024,
                image_height=1024,
            ),
            [0.0, 900.0, 40.0, 124.0],
        )

    def test_isaid_bbox_is_clipped_geometrically(self) -> None:
        self.assertEqual(
            clip_bbox_to_tile([90, 80, 30, 50], tile_x=100, tile_y=100, tile_size=64),
            [0.0, 0.0, 20.0, 30.0],
        )
        self.assertIsNone(
            clip_bbox_to_tile([0, 0, 10, 10], tile_x=100, tile_y=100, tile_size=64)
        )


if __name__ == "__main__":
    unittest.main()
