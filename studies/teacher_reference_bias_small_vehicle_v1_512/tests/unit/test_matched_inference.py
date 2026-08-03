from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from yolo_sam.data.contracts import BBoxSource, PromptType


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_matched_gt_bbox_segmentation.py"
)
SPEC = importlib.util.spec_from_file_location("run_matched_gt_bbox_segmentation", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load matched inference script")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MatchedInferenceTest(unittest.TestCase):
    def test_coco_annotations_become_stable_instance_tasks(self) -> None:
        tasks = MODULE.coco_tasks_for_image(
            "isaid_small_vehicle",
            {"id": 7, "file_name": "tile.png"},
            [
                {
                    "id": 11,
                    "bbox": [1, 2, 3, 4],
                    "bbox_source": "human_annotation",
                }
            ],
        )
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].instance_id, "isaid_small_vehicle:7:11")
        self.assertEqual(tasks[0].bbox_source, BBoxSource.HUMAN_ANNOTATION)
        self.assertEqual(tasks[0].prompt_type, PromptType.GT_BBOX)


if __name__ == "__main__":
    unittest.main()
