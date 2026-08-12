from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_matched_detector.py"
SPEC = importlib.util.spec_from_file_location("run_matched_detector", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load detector evaluation script")
DETECTOR_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DETECTOR_SCRIPT)


class FakeDetector:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return iter(kwargs["source"])


class DetectorStreamingTest(unittest.TestCase):
    def test_detector_evaluation_uses_bounded_streaming_batches(self) -> None:
        detector = FakeDetector()

        results = DETECTOR_SCRIPT.stream_detector_results(
            detector,
            image_paths=["one.png", "two.png", "three.png"],
            image_size=1024,
            confidence_threshold=0.001,
            nms_iou_threshold=0.7,
            max_detections=500,
            device="0",
            batch_size=2,
        )

        self.assertEqual(list(results), ["one.png", "two.png", "three.png"])
        self.assertEqual(len(detector.calls), 2)
        self.assertEqual(
            [call["source"] for call in detector.calls],
            [["one.png", "two.png"], ["three.png"]],
        )
        self.assertTrue(all(call["stream"] for call in detector.calls))
        self.assertTrue(all(call["batch"] == 2 for call in detector.calls))
        self.assertTrue(all(len(call["source"]) <= 2 for call in detector.calls))

    def test_non_positive_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            list(
                DETECTOR_SCRIPT.stream_detector_results(
                    FakeDetector(),
                    image_paths=["one.png"],
                    image_size=1024,
                    confidence_threshold=0.001,
                    nms_iou_threshold=0.7,
                    max_detections=500,
                    device="0",
                    batch_size=0,
                )
            )


if __name__ == "__main__":
    unittest.main()
