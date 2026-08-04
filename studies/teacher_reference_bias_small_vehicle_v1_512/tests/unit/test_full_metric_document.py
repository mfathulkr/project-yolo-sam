from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from teacher_reference_bias.reporting.full_metric_document import (
    build_detector_table,
    build_qualitative_examples,
    build_segmentation_table,
    numeric_value,
    ranking_comparison_sentence,
)
from teacher_reference_bias.validation import validate_detector_base_provenance


class FullMetricDocumentTest(unittest.TestCase):
    def test_ranking_comparison_does_not_claim_change_when_order_is_same(
        self,
    ) -> None:
        sentence = ranking_comparison_sentence(
            "SAM1 > SAM2 > SAM3",
            "SAM1 > SAM2 > SAM3",
        )

        self.assertIn("her iki referansta da", sentence)
        self.assertIn("skor düzeylerindeki değişim", sentence)
        self.assertNotIn("sıralamadaki değişim", sentence)

    def test_ranking_comparison_flags_real_order_change(self) -> None:
        sentence = ranking_comparison_sentence(
            "SAM1 > SAM2 > SAM3",
            "SAM2 > SAM1 > SAM3",
        )

        self.assertIn("sıralamadaki değişim", sentence)
        self.assertIn("SAM2 > SAM1 > SAM3", sentence)

    def test_detector_provenance_accepts_fresh_base_model(self) -> None:
        validate_detector_base_provenance(
            training_args={"model": "/repo/models/yolo/yolo26x.pt"},
            training_manifest={
                "inputs": {"base_weights": "/repo/models/yolo/yolo26x.pt"},
                "parameters": {
                    "base_weights": "models/yolo/yolo26x.pt",
                    "resume": False,
                },
            },
            expected_base_model="models/yolo/yolo26x.pt",
        )

    def test_detector_provenance_accepts_matching_resume_checkpoint(self) -> None:
        checkpoint = "/repo/results/detector/seed_42/train/weights/last.pt"
        validate_detector_base_provenance(
            training_args={"model": checkpoint, "resume": checkpoint},
            training_manifest={
                "inputs": {
                    "base_weights": "/repo/models/yolo/yolo26x.pt",
                    "resume_checkpoint": checkpoint,
                },
                "parameters": {
                    "base_weights": "models/yolo/yolo26x.pt",
                    "resume": True,
                },
            },
            expected_base_model="models/yolo/yolo26x.pt",
        )

    def test_detector_provenance_rejects_wrong_declared_base(self) -> None:
        checkpoint = "/repo/results/detector/seed_42/train/weights/last.pt"
        with self.assertRaisesRegex(ValueError, "does not declare yolo26x.pt"):
            validate_detector_base_provenance(
                training_args={"model": checkpoint, "resume": checkpoint},
                training_manifest={
                    "inputs": {
                        "base_weights": "/repo/models/yolo/yolo26n.pt",
                        "resume_checkpoint": checkpoint,
                    },
                    "parameters": {
                        "base_weights": "models/yolo/yolo26n.pt",
                        "resume": True,
                    },
                },
                expected_base_model="models/yolo/yolo26x.pt",
            )

    def test_detector_provenance_rejects_unrelated_resume_checkpoint(self) -> None:
        checkpoint = "/repo/results/detector/seed_42/train/weights/last.pt"
        with self.assertRaisesRegex(ValueError, "paths disagree"):
            validate_detector_base_provenance(
                training_args={
                    "model": checkpoint,
                    "resume": "/repo/other/weights/last.pt",
                },
                training_manifest={
                    "inputs": {
                        "base_weights": "/repo/models/yolo/yolo26x.pt",
                        "resume_checkpoint": checkpoint,
                    },
                    "parameters": {
                        "base_weights": "models/yolo/yolo26x.pt",
                        "resume": True,
                    },
                },
                expected_base_model="models/yolo/yolo26x.pt",
            )

    def test_numeric_value_accepts_mean_std_display(self) -> None:
        self.assertEqual(numeric_value("0.812 ± 0.017"), 0.812)
        self.assertIsNone(numeric_value("+0.350"))
        self.assertIsNone(numeric_value("3 seed"))

    def test_build_segmentation_table_keeps_gt_and_summarizes_yolo_seeds(
        self,
    ) -> None:
        rows = []
        for model in ("sam1", "sam2", "sam3"):
            rows.append(
                {
                    "dataset_id": "sample",
                    "reference_type": "human",
                    "stratum": "overall",
                    "model": model,
                    "bbox_source": "gt_bbox",
                    "detector_seed": None,
                    "instance_count": 10,
                    "source_scene_count": 4,
                    "mean_iou": 0.8,
                    "mean_dice": 0.9,
                    "mean_precision": 0.8,
                    "mean_recall": 0.9,
                    "mean_boundary_iou": 0.7,
                    "success_at_iou_50": 1.0,
                    "success_at_iou_75": 0.8,
                    "success_at_iou_90": 0.2,
                }
            )
            for seed, iou in ((42, 0.6), (123, 0.7), (2026, 0.8)):
                row = rows[-1].copy()
                row.update(
                    {
                        "bbox_source": "yolo_bbox",
                        "detector_seed": seed,
                        "mean_iou": iou,
                    }
                )
                rows.append(row)

        table = build_segmentation_table(
            pd.DataFrame(rows),
            dataset_id="sample",
            reference_type="human",
            stratum="overall",
        )

        self.assertEqual(len(table), 6)
        self.assertEqual(
            list(table.columns),
            [
                "Pipeline",
                "Images",
                "Avg IoU",
                "Avg Dice",
                "Avg Precision",
                "Avg Recall",
                "IoU ≥ 0.50",
                "IoU ≥ 0.75",
                "IoU ≥ 0.90",
            ],
        )
        self.assertEqual(table.iloc[0]["Pipeline"], "SAM1 GT bbox")
        self.assertEqual(table.iloc[1]["Pipeline"], "SAM1 YOLO bbox")
        self.assertEqual(table.iloc[0]["Images"], 512)
        self.assertEqual(table.attrs["instance_count"], 10)
        self.assertEqual(table.iloc[0]["Avg IoU"], "0.800")
        self.assertEqual(table.iloc[1]["Avg IoU"], "0.700 ± 0.100")
        self.assertNotIn("n", table.columns)
        self.assertNotIn("Sahne", table.columns)
        self.assertNotIn("Tekrar", table.columns)
        self.assertNotIn("Boundary IoU", table.columns)

    def test_build_detector_table_formats_seed_mean_and_std(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset_id": "sample",
                    "seed_count": 3,
                    "seed_ids": "42,123,2026",
                    **{
                        f"{prefix}_{suffix}": value
                        for prefix in (
                            "fixed_confidence_threshold",
                            "bbox_AP50",
                            "bbox_AP75",
                            "bbox_AP90",
                            "bbox_AP50_95",
                            "precision_at_bbox_iou50",
                            "recall_at_bbox_iou50",
                            "precision_at_bbox_iou75",
                            "recall_at_bbox_iou75",
                            "precision_at_bbox_iou90",
                            "recall_at_bbox_iou90",
                        )
                        for suffix, value in (("mean", 0.8), ("std", 0.1))
                    },
                }
            ]
        )

        table = build_detector_table(frame, dataset_id="sample")

        self.assertEqual(
            table.iloc[0]["BBox mAP50"],
            "0.800 ± 0.100",
        )
        self.assertEqual(table.iloc[0]["Detector"], "YOLO26x (3 seed)")
        self.assertEqual(table.iloc[0]["Images"], 512)
        self.assertNotIn("Tekrar", table.columns)
        self.assertNotIn("Confidence", table.columns)

    def test_single_seed_tables_do_not_show_fake_standard_deviation(self) -> None:
        rows = []
        for model in ("sam1", "sam2", "sam3"):
            base = {
                "dataset_id": "sample",
                "reference_type": "human",
                "stratum": "overall",
                "model": model,
                "instance_count": 10,
                "source_scene_count": 4,
                "mean_iou": 0.8,
                "mean_dice": 0.9,
                "mean_precision": 0.8,
                "mean_recall": 0.9,
                "success_at_iou_50": 1.0,
                "success_at_iou_75": 0.8,
                "success_at_iou_90": 0.2,
            }
            rows.append({**base, "bbox_source": "gt_bbox", "detector_seed": None})
            rows.append({**base, "bbox_source": "yolo_bbox", "detector_seed": 42})

        table = build_segmentation_table(
            pd.DataFrame(rows),
            dataset_id="sample",
            reference_type="human",
            stratum="overall",
        )

        self.assertEqual(table.iloc[1]["Avg IoU"], "0.800")
        self.assertEqual(table.attrs["detector_seeds"], (42,))

    def test_build_qualitative_examples_creates_four_readable_pages(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            source_path = temporary_root / "source.png"
            Image.new("RGB", (2500, 2000), "white").save(source_path)

            outputs = build_qualitative_examples(
                source_path,
                temporary_root / "qualitative",
            )

            self.assertEqual(len(outputs), 4)
            for _, path in outputs:
                self.assertTrue(path.is_file())
                with Image.open(path) as image:
                    self.assertEqual(image.size, (1800, 1170))


if __name__ == "__main__":
    unittest.main()
