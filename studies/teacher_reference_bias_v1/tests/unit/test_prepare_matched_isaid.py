from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from yolo_sam.data.isaid import rasterize_clipped_annotation


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "prepare_matched_isaid_plane.py"
SPEC = importlib.util.spec_from_file_location("prepare_matched_isaid_plane", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load prepare_matched_isaid_plane.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareMatchedIsaidTest(unittest.TestCase):
    def test_boundary_sliver_is_encoded_as_lossless_rle(self) -> None:
        result = rasterize_clipped_annotation(
            annotation={
                "bbox": [0.0, -1.0, 11.0, 3.0],
                "segmentation": [
                    [0.0, -1.0, 10.0, -1.0, 10.0, 1.0, 0.0, 1.0]
                ],
            },
            tile_x=0,
            tile_y=0,
            tile_size=16,
            min_instance_area=8,
        )
        self.assertIsNotNone(result)
        segmentation, bbox, area = result
        decoded = mask_utils.decode(segmentation).astype(bool)

        self.assertEqual(decoded.shape, (16, 16))
        self.assertEqual(int(decoded.sum()), area)
        self.assertGreater(area, 0)
        self.assertEqual(bbox, [0.0, 0.0, 11.0, 2.0])
        self.assertTrue(np.all(decoded[:2, :11]))

    def test_train_validation_scene_split_is_disjoint_and_deterministic(self) -> None:
        rows = [
            {
                "image_id": index,
                "file_name": f"scene_{index}.png",
                "source_scene_id": f"scene_{index}",
                "plane_instances": 1 if index % 3 == 0 else 0,
            }
            for index in range(30)
        ]
        first = MODULE.split_train_validation_scenes(rows, 0.2, seed=42)
        second = MODULE.split_train_validation_scenes(rows, 0.2, seed=42)
        self.assertEqual(first, second)
        train_names, validation_names, _ = first
        self.assertFalse(train_names & validation_names)
        self.assertEqual(len(train_names | validation_names), len(rows))


if __name__ == "__main__":
    unittest.main()
