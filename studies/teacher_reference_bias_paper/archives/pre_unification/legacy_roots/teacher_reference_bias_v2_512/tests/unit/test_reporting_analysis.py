from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from teacher_reference_bias.reporting.analysis import (
    collect_canonical_metrics,
    collect_detector_metrics,
    collect_prediction_status_audit,
    collect_training_health,
    detector_seed_summary,
    holm_adjust,
    paired_model_comparisons,
    ranking_comparisons,
    segmentation_seed_summary,
)


class ReportingAnalysisTest(unittest.TestCase):
    def test_holm_adjust_is_monotone_in_sorted_p_values(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_dual_reference_file_is_preferred_over_single_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            columns = {
                "run_id": ["run"],
                "model_id": ["model"],
                "model_version": ["sam1"],
                "prompt_type": ["gt_bbox"],
                "image_id": ["image"],
                "instance_id": ["instance"],
                "source_scene_id": ["scene"],
                "reference_type": ["human"],
                "stratum": ["overlap__low_mask_area"],
                "iou": [0.5],
                "dice": [0.6],
                "precision": [0.7],
                "recall": [0.8],
                "boundary_iou": [0.4],
                "true_positive_pixels": [1],
                "false_positive_pixels": [1],
                "false_negative_pixels": [1],
            }
            single = root / "dataset" / "sam1" / "gt_bbox"
            dual = root / "dataset" / "sam1" / "gt_bbox_dual_reference"
            single.mkdir(parents=True)
            dual.mkdir(parents=True)
            pd.DataFrame(columns).to_csv(single / "metrics_instance.csv", index=False)
            dual_frame = pd.DataFrame(columns)
            pseudo = dual_frame.copy()
            pseudo["reference_type"] = "pseudo_sam1"
            pd.concat([dual_frame, pseudo]).to_csv(
                dual / "metrics_instance.csv",
                index=False,
            )
            metric_path = dual / "metrics_instance.csv"
            source_path = dual / "source.txt"
            source_path.write_text("source\n", encoding="utf-8")
            def fingerprint(path: Path) -> dict[str, object]:
                return {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            manifest_path = dual / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "stage": "matched_prediction_evaluation",
                        "input_drift": [],
                        "input_file_fingerprints_at_finish": {
                            "source": fingerprint(source_path)
                        },
                        "output_file_fingerprints": {
                            "metrics_instance": fingerprint(metric_path)
                        },
                        "parameters": {
                            "known_positive_empty_reference_policy": "score_zero",
                            "primary_granularity": "instance",
                        },
                    }
                ),
                encoding="utf-8",
            )

            metrics, files = collect_canonical_metrics(root)

            self.assertEqual(files, [metric_path, manifest_path])
            self.assertEqual(set(metrics["reference_type"]), {"human", "pseudo_sam1"})

    def test_empty_result_tables_preserve_their_csv_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            detector_metrics, files = collect_detector_metrics(
                Path(temporary_directory)
            )

        detector_summary = detector_seed_summary(detector_metrics)
        segmentation_summary = segmentation_seed_summary(pd.DataFrame())

        self.assertEqual(files, [])
        self.assertIn("bbox_AP50", detector_metrics.columns)
        self.assertIn("bbox_AP50_mean", detector_summary.columns)
        self.assertIn("seed_count", detector_summary.columns)
        self.assertIn("mean_iou_seed_mean", segmentation_summary.columns)
        self.assertIn("seed_count", segmentation_summary.columns)

    def test_paired_wilcoxon_uses_source_scene_means(self) -> None:
        rows = []
        for model, values in {
            "sam1": (0.9, 0.8, 0.7),
            "sam2": (0.7, 0.6, 0.4),
        }.items():
            for instance_index, (scene, value) in enumerate(
                zip(("scene-a", "scene-a", "scene-b"), values, strict=True)
            ):
                rows.append(
                    {
                        "dataset_id": "dataset",
                        "model": model,
                        "bbox_source": "gt_bbox",
                        "detector_seed": pd.NA,
                        "reference_type": "human",
                        "instance_id": f"instance-{instance_index}",
                        "source_scene_id": scene,
                        "stratum": "no_overlap__low_mask_area",
                        "iou": value,
                    }
                )

        comparisons = paired_model_comparisons(
            pd.DataFrame(rows),
            bootstrap_samples=100,
            confidence_level=0.95,
            bootstrap_seed=42,
        )
        overall = comparisons[comparisons["stratum"] == "overall"].iloc[0]

        self.assertEqual(overall["wilcoxon_unit"], "source_scene_mean")
        self.assertEqual(int(overall["wilcoxon_observations"]), 2)
        self.assertEqual(int(overall["paired_instances"]), 3)

    def test_ranking_reports_kendall_and_teacher_advantage_change(self) -> None:
        rows = []
        scores = {
            ("sam1", "human"): 0.50,
            ("sam2", "human"): 0.80,
            ("sam3", "human"): 0.70,
            ("sam1", "pseudo_sam1"): 0.95,
            ("sam2", "pseudo_sam1"): 0.75,
            ("sam3", "pseudo_sam1"): 0.65,
        }
        for (model, reference_type), value in scores.items():
            rows.append(
                {
                    "dataset_id": "dataset",
                    "model": model,
                    "bbox_source": "gt_bbox",
                    "detector_seed": pd.NA,
                    "reference_type": reference_type,
                    "instance_id": "instance",
                    "source_scene_id": "scene",
                    "stratum": "no_overlap__low_mask_area",
                    "iou": value,
                }
            )

        ranking = ranking_comparisons(pd.DataFrame(rows))
        overall = ranking[ranking["stratum"] == "overall"].iloc[0]

        self.assertAlmostEqual(float(overall["kendall_tau"]), -1 / 3)
        self.assertAlmostEqual(
            float(overall["sam1_teacher_advantage_change"]),
            0.50,
        )

    def test_prediction_status_audit_checks_status_and_mask_area(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prediction_file = (
                root
                / "dataset"
                / "sam1"
                / "gt_bbox"
                / "predictions.jsonl"
            )
            prediction_file.parent.mkdir(parents=True)
            prediction_file.write_text(
                json.dumps(
                    {
                        "instance_id": "instance-1",
                        "status": "ok",
                        "input_bbox": [0.0, 0.0, 2.0, 2.0],
                        "predicted_mask_rle": {
                            "size": [2, 2],
                            "counts": "013",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            audit, files = collect_prediction_status_audit(root)

        self.assertEqual(files, [prediction_file])
        self.assertEqual(len(audit), 1)
        self.assertEqual(int(audit.iloc[0]["ok"]), 1)
        self.assertEqual(int(audit.iloc[0]["nonzero_area_masks"]), 1)
        self.assertEqual(
            int(
                audit.iloc[0][
                    "nonempty_masks_without_prompt_overlap"
                ]
            ),
            0,
        )
        self.assertEqual(int(audit.iloc[0]["status_area_mismatches"]), 0)

    def test_prediction_status_audit_detects_mask_outside_prompt_box(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prediction_file = (
                root
                / "dataset"
                / "sam3"
                / "gt_bbox"
                / "predictions.jsonl"
            )
            prediction_file.parent.mkdir(parents=True)
            prediction_file.write_text(
                json.dumps(
                    {
                        "instance_id": "instance-1",
                        "status": "ok",
                        "input_bbox": [1.0, 1.0, 1.0, 1.0],
                        "predicted_mask_rle": {
                            "size": [2, 2],
                            "counts": "013",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            audit, _ = collect_prediction_status_audit(root)

        self.assertEqual(
            int(
                audit.iloc[0][
                    "nonempty_masks_without_prompt_overlap"
                ]
            ),
            1,
        )

    def test_training_health_records_transient_nan_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results_file = (
                root
                / "dataset"
                / "seed_42"
                / "train"
                / "results.csv"
            )
            results_file.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "epoch": 1,
                        "val/box_loss": float("nan"),
                        "val/cls_loss": 0.3,
                        "val/dfl_loss": 0.2,
                        "metrics/precision(B)": 0.6,
                        "metrics/recall(B)": 0.5,
                        "metrics/mAP50(B)": 0.7,
                        "metrics/mAP50-95(B)": 0.4,
                    },
                    {
                        "epoch": 2,
                        "val/box_loss": 0.2,
                        "val/cls_loss": 0.2,
                        "val/dfl_loss": 0.1,
                        "metrics/precision(B)": 0.8,
                        "metrics/recall(B)": 0.7,
                        "metrics/mAP50(B)": 0.9,
                        "metrics/mAP50-95(B)": 0.6,
                    },
                ]
            ).to_csv(results_file, index=False)

            audit, files = collect_training_health(root)

        self.assertEqual(files, [results_file])
        self.assertEqual(int(audit.iloc[0]["epochs_completed"]), 2)
        self.assertEqual(
            int(audit.iloc[0]["nonfinite_validation_loss_cells"]),
            1,
        )
        self.assertTrue(bool(audit.iloc[0]["final_core_metrics_finite"]))
        self.assertAlmostEqual(float(audit.iloc[0]["final_ap50"]), 0.9)


if __name__ == "__main__":
    unittest.main()
