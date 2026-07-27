from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from yolo_sam.data.contracts import (
    BBoxSource,
    BBoxXYWH,
    PromptType,
)
from yolo_sam.evaluation.instance_metrics import binary_mask_metrics
from yolo_sam.evaluation.metrics import compute_mask_metrics
from yolo_sam.segmentation.box_segmenters import SingleBoxSegmentation
from yolo_sam.segmentation.runner import (
    SegmentationTask,
    run_segmentation_tasks,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = ROOT / "scripts" / "evaluate_matched_predictions.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_matched_predictions_integration",
    EVALUATOR_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load matched evaluator script")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class _ExactBoxSegmenter:
    model_id = "fixture-sam"
    model_version = "fixture-v1"

    def segment_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: list[list[float]],
    ) -> list[SingleBoxSegmentation]:
        results = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = (int(value) for value in box)
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            results.append(SingleBoxSegmentation(mask=mask, score=0.99))
        return results


class TinyEndToEndIntegrationTest(unittest.TestCase):
    def test_legacy_and_canonical_nonempty_metric_kernels_match(self) -> None:
        prediction = np.zeros((8, 8), dtype=bool)
        reference = np.zeros((8, 8), dtype=bool)
        prediction[1:5, 2:6] = True
        reference[2:7, 3:6] = True

        legacy = compute_mask_metrics(prediction, reference)
        canonical = binary_mask_metrics(prediction, reference)
        for metric in ("iou", "dice", "precision", "recall"):
            self.assertAlmostEqual(legacy[metric], getattr(canonical, metric))

    def test_prediction_jsonl_to_dual_reference_metrics(self) -> None:
        image = Image.new("RGB", (8, 8))
        task = SegmentationTask(
            image_id="fixture:1",
            instance_id="fixture:1:1",
            bbox=BBoxXYWH(2, 3, 2, 2),
            bbox_source=BBoxSource.HUMAN_ANNOTATION,
            prompt_type=PromptType.GT_BBOX,
        )
        completed = run_segmentation_tasks(
            "fixture-run",
            image,
            [task],
            _ExactBoxSegmenter(),
        )
        payload = completed[0].record.to_dict()
        payload.update(
            {
                "source_scene_id": "scene-1",
                "stratum": "no_overlap_low_mask_area",
            }
        )

        with tempfile.TemporaryDirectory() as temporary:
            prediction_path = Path(temporary) / "predictions.jsonl"
            prediction_path.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            predictions = EVALUATOR.read_jsonl(prediction_path)

        human = completed[0].mask.copy()
        pseudo = human.copy()
        pseudo[0, 0] = True
        human_references = {task.instance_id: human}
        pseudo_references = {task.instance_id: pseudo}

        EVALUATOR.validate_instance_sets(
            predictions,
            human_references,
            pseudo_references,
        )
        metric_rows = EVALUATOR.evaluate_rows(
            predictions,
            human_references,
            "human",
            pseudo_references,
        )
        by_reference = {row.reference_type: row for row in metric_rows}
        self.assertEqual(set(by_reference), {"human", "pseudo_sam1"})
        self.assertAlmostEqual(by_reference["human"].iou, 1.0)
        self.assertAlmostEqual(by_reference["pseudo_sam1"].iou, 0.8)

        summaries = EVALUATOR.summary_rows(metric_rows)
        overall = {
            row["reference_type"]: row
            for row in summaries
            if row["stratum"] == "overall"
        }
        self.assertEqual(overall["human"]["count"], 1)
        self.assertAlmostEqual(overall["human"]["mean_iou"], 1.0)

        union_rows = EVALUATOR.image_union_rows(
            predictions,
            human_references,
            "human",
            pseudo_references,
        )
        union_by_reference = {
            row["reference_type"]: row for row in union_rows
        }
        self.assertAlmostEqual(union_by_reference["human"]["iou"], 1.0)
        self.assertAlmostEqual(union_by_reference["pseudo_sam1"]["iou"], 0.8)


if __name__ == "__main__":
    unittest.main()
