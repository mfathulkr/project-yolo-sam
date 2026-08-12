from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_matched_yolo_bbox_segmentation.py"
)
SPEC = importlib.util.spec_from_file_location("run_matched_yolo_bbox_segmentation", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load YOLO bbox pipeline script")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class YoloBboxPipelineTest(unittest.TestCase):
    def test_xyxy_to_xywh(self) -> None:
        bbox = MODULE.xyxy_to_xywh([10, 20, 35, 55])
        self.assertEqual(bbox.to_list(), [10, 20, 25, 35])


if __name__ == "__main__":
    unittest.main()
