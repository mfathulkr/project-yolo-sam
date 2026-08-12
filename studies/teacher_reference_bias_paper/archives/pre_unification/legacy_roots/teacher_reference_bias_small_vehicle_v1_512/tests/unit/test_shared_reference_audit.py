from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_shared_isaid_samrs_references.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_shared_isaid_samrs_references",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SharedReferenceAuditTest(unittest.TestCase):
    def test_unique_object_sensitivity_weights_repeated_views_once(self) -> None:
        matches = pd.DataFrame(
            {
                "instance_id": ["view-a", "view-b", "view-c"],
                "human_object_key": ["object-1", "object-1", "object-2"],
                "source_scene_id": ["scene-1", "scene-1", "scene-2"],
            }
        )
        rows = []
        for model in MODULE.MODELS:
            for reference_type, values in {
                "human": (0.0, 1.0, 1.0),
                "pseudo_sam1": (1.0, 1.0, 1.0),
            }.items():
                for instance_id, source_scene_id, value in zip(
                    ("view-a", "view-b", "view-c"),
                    ("scene-1", "scene-1", "scene-2"),
                    values,
                    strict=True,
                ):
                    rows.append(
                        {
                            "model_version": model,
                            "reference_type": reference_type,
                            "instance_id": instance_id,
                            "source_scene_id": source_scene_id,
                            "iou": value,
                            "dice": value,
                            "precision": value,
                            "recall": value,
                            "boundary_iou": value,
                        }
                    )

        result = MODULE.unique_object_sensitivity(
            pd.DataFrame(rows),
            matches,
            bootstrap_samples=100,
            bootstrap_seed=42,
        )

        self.assertTrue((result["unique_human_objects"] == 2).all())
        self.assertTrue(
            np.allclose(result["human_mean_iou"], 0.75)
        )
        self.assertTrue(
            np.allclose(result["mean_iou_inflation"], 0.25)
        )

    def test_mapping_requires_both_score_and_exact_pixels(self) -> None:
        self.assertEqual(
            MODULE.classify_tile_mapping(
                0.999,
                exact_pixels=True,
                min_template_score=0.995,
            ),
            "matched",
        )
        self.assertEqual(
            MODULE.classify_tile_mapping(
                0.999,
                exact_pixels=False,
                min_template_score=0.995,
            ),
            "pixel_mismatch",
        )
        self.assertEqual(
            MODULE.classify_tile_mapping(
                0.990,
                exact_pixels=True,
                min_template_score=0.995,
            ),
            "low_template_score",
        )

    def test_locates_regular_tile_inside_source(self) -> None:
        rng = np.random.default_rng(42)
        source = rng.integers(0, 256, size=(80, 90, 3), dtype=np.uint8)
        tile = source[17:57, 23:63].copy()

        x, y, score, exact = MODULE.locate_tile_in_source(
            source,
            tile,
            scale=0.5,
        )

        self.assertEqual((x, y), (23, 17))
        self.assertGreater(score, 0.999)
        self.assertTrue(exact)

    def test_locates_right_padded_boundary_tile(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.integers(0, 256, size=(70, 35, 3), dtype=np.uint8)
        tile = np.full((50, 50, 3), 117, dtype=np.uint8)
        tile[:, :35] = source[11:61]

        x, y, score, exact = MODULE.locate_tile_in_source(
            source,
            tile,
            scale=0.5,
        )

        self.assertEqual((x, y), (0, 11))
        self.assertGreater(score, 0.999)
        self.assertTrue(exact)

    def test_locates_bottom_padded_boundary_tile(self) -> None:
        rng = np.random.default_rng(99)
        source = rng.integers(0, 256, size=(35, 75, 3), dtype=np.uint8)
        tile = np.full((50, 50, 3), 116, dtype=np.uint8)
        tile[:35] = source[:, 19:69]

        x, y, score, exact = MODULE.locate_tile_in_source(
            source,
            tile,
            scale=0.5,
        )

        self.assertEqual((x, y), (19, 0))
        self.assertGreater(score, 0.999)
        self.assertTrue(exact)


if __name__ == "__main__":
    unittest.main()
