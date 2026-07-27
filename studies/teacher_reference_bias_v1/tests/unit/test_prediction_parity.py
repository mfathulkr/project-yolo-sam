from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from teacher_reference_bias.reporting.prediction_parity import (
    summarize_prediction_masks,
)


class PredictionParityTest(unittest.TestCase):
    def test_runtime_and_model_metadata_do_not_change_mask_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.jsonl"
            row = {
                "instance_id": "instance-1",
                "status": "ok",
                "model_version": "unpinned",
                "runtime_ms": 10.0,
                "predicted_mask_rle": {"size": [2, 2], "counts": "13"},
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            before = summarize_prediction_masks(path)

            row["model_version"] = "explicit-revision"
            row["runtime_ms"] = 20.0
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            after = summarize_prediction_masks(path)

        self.assertEqual(
            before["canonical_mask_sha256"],
            after["canonical_mask_sha256"],
        )
        self.assertNotEqual(before["model_versions"], after["model_versions"])

    def test_mask_change_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.jsonl"
            row = {
                "instance_id": "instance-1",
                "status": "ok",
                "predicted_mask_rle": {"size": [2, 2], "counts": "13"},
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            before = summarize_prediction_masks(path)

            row["predicted_mask_rle"]["counts"] = "22"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            after = summarize_prediction_masks(path)

        self.assertNotEqual(
            before["canonical_mask_sha256"],
            after["canonical_mask_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
