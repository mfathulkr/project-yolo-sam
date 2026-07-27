from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "evaluate_matched_predictions.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_matched_predictions", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load matched evaluator script")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MatchedEvaluatorTest(unittest.TestCase):
    def test_instance_set_validation_rejects_missing_predictions(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.validate_instance_sets(
                predictions=[{"instance_id": "a"}],
                coco_reference={
                    "a": np.ones((1, 1), dtype=bool),
                    "b": np.ones((1, 1), dtype=bool),
                },
                pseudo={},
            )

    def test_instance_set_validation_accepts_exact_dual_reference_sets(self) -> None:
        mask = np.ones((1, 1), dtype=bool)
        MODULE.validate_instance_sets(
            predictions=[{"instance_id": "a"}],
            coco_reference={"a": mask},
            pseudo={"a": mask},
        )


if __name__ == "__main__":
    unittest.main()
