from __future__ import annotations

import unittest

from yolo_sam.data.split import (
    SplitCandidate,
    SplitManifestRow,
    grouped_stratified_split,
    validate_split_manifest,
)


class GroupedSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            SplitCandidate(
                image_id=f"scene_{scene}_{tile}",
                source_scene_id=f"scene_{scene}",
                stratum="overlap_high" if scene % 2 else "no_overlap_low",
                instance_count=scene + 1,
            )
            for scene in range(12)
            for tile in range(2)
        ]

    def test_grouped_split_is_deterministic_and_scene_safe(self) -> None:
        fractions = {"train": 0.7, "validation": 0.15, "test": 0.15}
        first = grouped_stratified_split(self.candidates, fractions, seed=42)
        second = grouped_stratified_split(self.candidates, fractions, seed=42)
        self.assertEqual(first, second)
        validate_split_manifest(first)
        self.assertEqual({row.split for row in first}, set(fractions))

    def test_manifest_rejects_scene_leakage(self) -> None:
        rows = [
            SplitManifestRow("a", "scene", "overall", 1, "train"),
            SplitManifestRow("b", "scene", "overall", 1, "test"),
        ]
        with self.assertRaises(ValueError):
            validate_split_manifest(rows)


if __name__ == "__main__":
    unittest.main()
