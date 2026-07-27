from __future__ import annotations

import unittest

import numpy as np

from yolo_sam.models.sam2_local import masks_to_instance_masks
from yolo_sam.models.sam3_local import masks_to_instance_masks as sam3_masks_to_instances


class ModelMaskConversionTest(unittest.TestCase):
    def test_sam1_sam2_batch_masks_remain_per_instance(self) -> None:
        masks = np.zeros((1, 2, 1, 4, 5), dtype=np.uint8)
        masks[0, 0, 0, 0, 0] = 1
        masks[0, 1, 0, 3, 4] = 1
        converted = masks_to_instance_masks(masks, (4, 5))
        self.assertEqual(len(converted), 2)
        self.assertTrue(converted[0][0, 0])
        self.assertTrue(converted[1][3, 4])

    def test_sam3_masks_remain_per_instance(self) -> None:
        masks = np.zeros((3, 4, 5), dtype=np.uint8)
        masks[:, 1, 1] = 1
        converted = sam3_masks_to_instances(masks, (4, 5))
        self.assertEqual(len(converted), 3)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            masks_to_instance_masks(np.zeros((1, 3, 3)), (4, 4))


if __name__ == "__main__":
    unittest.main()
