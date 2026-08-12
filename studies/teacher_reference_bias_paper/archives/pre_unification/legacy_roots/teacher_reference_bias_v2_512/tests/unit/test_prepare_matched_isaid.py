from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from yolo_sam.data.isaid import rasterize_clipped_annotation


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "prepare_matched_512_from_master.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_matched_512_from_master",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load prepare_matched_512_from_master.py")
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

    def test_balanced_test_selection_is_exact_and_deterministic(self) -> None:
        rows = []
        for scene_index in range(8):
            for stratum in MODULE.STRATA:
                for item_index in range(2):
                    rows.append(
                        {
                            "source_scene_id": f"scene_{scene_index}",
                            "stratum": stratum,
                            "file_name": (
                                f"scene_{scene_index}_{stratum}_{item_index}.png"
                            ),
                        }
                    )
        positive = pd.DataFrame(rows)
        first_scenes, first_coverage = MODULE.select_test_scene_pool(
            positive,
            target_per_stratum=4,
            seed=42,
            trials=16,
        )
        second_scenes, second_coverage = MODULE.select_test_scene_pool(
            positive,
            target_per_stratum=4,
            seed=42,
            trials=16,
        )
        self.assertEqual(first_scenes, second_scenes)
        self.assertEqual(first_coverage, second_coverage)

        selected = MODULE.select_exact_test_rows(
            positive,
            scene_pool=first_scenes,
            target_per_stratum=4,
            seed=42,
        )
        self.assertEqual(len(selected), 16)
        self.assertEqual(
            selected["stratum"].value_counts().to_dict(),
            {stratum: 4 for stratum in MODULE.STRATA},
        )


if __name__ == "__main__":
    unittest.main()
