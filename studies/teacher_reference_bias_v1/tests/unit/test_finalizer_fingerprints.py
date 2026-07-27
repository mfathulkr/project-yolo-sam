from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "finalize_matched_study.py"
)
SPEC = importlib.util.spec_from_file_location(
    "finalize_matched_study",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinalizerFingerprintTest(unittest.TestCase):
    def test_detects_changed_declared_run_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.txt"
            output_path = root / "output.txt"
            input_path.write_text("input", encoding="utf-8")
            output_path.write_text("output", encoding="utf-8")
            manifest_path = root / "condition" / "manifest.json"
            manifest_path.parent.mkdir()
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "stage": "test",
                        "status": "completed",
                        "config_hash": "a" * 64,
                        "inputs": {"input": str(input_path)},
                        "outputs": {"output": str(output_path)},
                        "input_file_fingerprints": {
                            "input": self._fingerprint(input_path)
                        },
                        "output_file_fingerprints": {
                            "output": self._fingerprint(output_path)
                        },
                        "input_fingerprint_capture": "start",
                        "input_file_fingerprints_at_finish": {
                            "input": self._fingerprint(input_path)
                        },
                        "input_drift": [],
                    }
                ),
                encoding="utf-8",
            )
            output_path.write_text("changed", encoding="utf-8")
            errors: list[str] = []

            MODULE.validate_completed_run_manifest_fingerprints(
                root,
                required=[],
                errors=errors,
            )

        self.assertEqual(len(errors), 1)
        self.assertIn("outputs.output", errors[0])

    def test_rejects_input_drift_even_when_finish_hash_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.txt"
            source.write_text("finish\n", encoding="utf-8")
            finish_fingerprint = self._fingerprint(source)
            start_fingerprint = {
                **finish_fingerprint,
                "sha256": "a" * 64,
            }
            manifest_path = root / "condition" / "manifest.json"
            manifest_path.parent.mkdir()
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "stage": "test",
                        "status": "completed",
                        "config_hash": "a" * 64,
                        "inputs": {"source": str(source)},
                        "outputs": {},
                        "input_file_fingerprints": {
                            "source": start_fingerprint,
                        },
                        "input_file_fingerprints_at_finish": {
                            "source": finish_fingerprint,
                        },
                        "input_fingerprint_capture": "start",
                        "input_drift": ["source"],
                        "output_file_fingerprints": {},
                    }
                ),
                encoding="utf-8",
            )
            errors: list[str] = []

            MODULE.validate_completed_run_manifest_fingerprints(
                root,
                required=[],
                errors=errors,
            )

        self.assertTrue(
            any("Run sırasında giriş dosyası değişmiş" in error for error in errors)
        )
        self.assertTrue(
            any(
                "başlangıç/bitiş input hash'leri uyuşmuyor" in error
                for error in errors
            )
        )

    def test_canonical_analysis_requires_complete_condition_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            study_root = root / "study"
            analysis_root = study_root / "analysis"
            analysis_root.mkdir(parents=True)
            datasets = [
                SimpleNamespace(
                    dataset_id="human_set",
                    reference_type="human",
                    prepared_root=root / "human",
                ),
                SimpleNamespace(
                    dataset_id="pseudo_set",
                    reference_type="pseudo_sam1",
                    prepared_root=root / "pseudo",
                ),
            ]
            instance_counts = {"human_set": 2, "pseudo_set": 3}
            for dataset in datasets:
                coco_path = (
                    dataset.prepared_root
                    / "test"
                    / "_annotations.coco.json"
                )
                coco_path.parent.mkdir(parents=True)
                coco_path.write_text(
                    json.dumps(
                        {
                            "annotations": [
                                {"id": index}
                                for index in range(
                                    instance_counts[dataset.dataset_id]
                                )
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            protocol = SimpleNamespace(detector_seeds=(42, 123, 2026))

            canonical_rows: list[dict[str, object]] = []
            aggregate_rows: list[dict[str, object]] = []
            for dataset in datasets:
                references = (
                    ("human", "pseudo_sam1")
                    if dataset.reference_type == "human"
                    else ("pseudo_sam1",)
                )
                for model in MODULE.MODELS:
                    for bbox_source, seeds in (
                        ("gt_bbox", (None,)),
                        ("yolo_bbox", protocol.detector_seeds),
                    ):
                        for seed in seeds:
                            for reference_type in references:
                                condition = {
                                    "dataset_id": dataset.dataset_id,
                                    "model": model,
                                    "bbox_source": bbox_source,
                                    "detector_seed": seed,
                                    "reference_type": reference_type,
                                }
                                for instance_index in range(
                                    instance_counts[dataset.dataset_id]
                                ):
                                    canonical_rows.append(
                                        {
                                            **condition,
                                            "instance_id": instance_index,
                                            "iou": 0.5,
                                            "dice": 0.5,
                                            "precision": 0.5,
                                            "recall": 0.5,
                                            "boundary_iou": 0.5,
                                            "true_positive_pixels": 1,
                                            "false_positive_pixels": 1,
                                            "false_negative_pixels": 1,
                                        }
                                    )
                                for stratum in MODULE.STRATA:
                                    aggregate_rows.append(
                                        {
                                            **condition,
                                            "stratum": stratum,
                                            "mean_iou": 0.5,
                                            "mean_dice": 0.5,
                                            "mean_precision": 0.5,
                                            "mean_recall": 0.5,
                                            "mean_boundary_iou": 0.5,
                                            "success_at_iou_50": 1.0,
                                            "success_at_iou_75": 0.0,
                                            "success_at_iou_90": 0.0,
                                            "iou_ci_lower": 0.4,
                                            "iou_ci_upper": 0.6,
                                            "instance_count": 1,
                                            "source_scene_count": 1,
                                            "bootstrap_samples": 10_000,
                                        }
                                    )
            canonical = pd.DataFrame(canonical_rows)
            canonical.to_csv(
                analysis_root / "canonical_instance_metrics.csv",
                index=False,
            )
            pd.DataFrame(aggregate_rows).to_csv(
                analysis_root / "aggregate_metrics.csv",
                index=False,
            )
            detector_rows = []
            for dataset in datasets:
                for seed in protocol.detector_seeds:
                    detector_rows.append(
                        {
                            "dataset_id": dataset.dataset_id,
                            "seed": seed,
                            "split": "test",
                            "confidence_threshold_source_split": "validation",
                            **{
                                column: 0.5
                                for column in (
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
                            },
                        }
                    )
            pd.DataFrame(detector_rows).to_csv(
                analysis_root / "detector_metrics_by_seed.csv",
                index=False,
            )

            errors: list[str] = []
            MODULE.validate_canonical_analysis_content(
                study_root,
                protocol,
                datasets,
                errors=errors,
            )
            self.assertEqual(errors, [])

            canonical.iloc[:-1].to_csv(
                analysis_root / "canonical_instance_metrics.csv",
                index=False,
            )
            errors = []
            MODULE.validate_canonical_analysis_content(
                study_root,
                protocol,
                datasets,
                errors=errors,
            )

        self.assertTrue(
            any("instance sayıları uyuşmuyor" in error for error in errors)
        )

    def test_detector_actual_args_allow_only_dataset_seed_and_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            study_root = root / "study"
            base_weights = root / "base.pt"
            base_weights.touch()
            protocol = SimpleNamespace(
                detector={
                    "base_weights": str(base_weights),
                    "epochs": 100,
                    "patience": 30,
                    "batch": 12,
                    "optimizer": "auto",
                },
                detector_seeds=(42, 123),
                image_size=1024,
            )
            datasets = [
                SimpleNamespace(
                    dataset_id=dataset_id,
                    prepared_root=root / dataset_id,
                )
                for dataset_id in ("first", "second")
            ]
            args_paths: list[Path] = []
            for dataset in datasets:
                (dataset.prepared_root / "data.yaml").parent.mkdir(
                    parents=True
                )
                (dataset.prepared_root / "data.yaml").touch()
                for seed in protocol.detector_seeds:
                    args_path = (
                        study_root
                        / "detectors"
                        / dataset.dataset_id
                        / f"seed_{seed}"
                        / "train"
                        / "args.yaml"
                    )
                    args_path.parent.mkdir(parents=True)
                    args_path.write_text(
                        json.dumps(
                            {
                                "task": "detect",
                                "mode": "train",
                                "model": str(base_weights),
                                "data": str(
                                    dataset.prepared_root / "data.yaml"
                                ),
                                "project": str(args_path.parents[1]),
                                "save_dir": str(args_path.parent),
                                "epochs": 100,
                                "patience": 30,
                                "batch": 12,
                                "imgsz": 1024,
                                "optimizer": "auto",
                                "seed": seed,
                                "deterministic": True,
                                "workers": 4,
                                "val": True,
                                "split": "val",
                                "lr0": 0.01,
                            }
                        ),
                        encoding="utf-8",
                    )
                    args_paths.append(args_path)
            errors: list[str] = []
            MODULE.validate_detector_training_args_matrix(
                study_root,
                protocol,
                datasets,
                required=[],
                errors=errors,
            )
            self.assertEqual(errors, [])

            changed = json.loads(args_paths[-1].read_text(encoding="utf-8"))
            changed["lr0"] = 0.02
            args_paths[-1].write_text(json.dumps(changed), encoding="utf-8")
            errors = []
            MODULE.validate_detector_training_args_matrix(
                study_root,
                protocol,
                datasets,
                required=[],
                errors=errors,
            )

        self.assertTrue(
            any("izin verilen" in error and "lr0" in error for error in errors)
        )

    def test_isaid_rle_sensitivity_archive_is_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            study_root = root / "study"
            audit_root = study_root / "audits"
            archive_root = (
                audit_root / "pre_isaid_lossless_rle_metric_fix"
            )
            archive_root.mkdir(parents=True)
            prepared_root = root / "prepared_isaid"
            test_coco = (
                prepared_root / "test" / "_annotations.coco.json"
            )
            test_coco.parent.mkdir(parents=True)
            test_coco.write_text(
                json.dumps({"annotations": [{"id": 1}, {"id": 2}]}),
                encoding="utf-8",
            )
            sensitivity_path = (
                audit_root / "isaid_rle_reference_sensitivity.json"
            )
            sensitivity_path.write_text(
                json.dumps(
                    {
                        "instances": 2,
                        "old_empty": 1,
                        "new_empty": 0,
                        "mean_reference_iou": 0.8,
                        "median_reference_iou": 0.9,
                        "instances_below_iou_0_90": 1,
                    }
                ),
                encoding="utf-8",
            )
            migration_path = (
                audit_root / "isaid_lossless_rle_migration.json"
            )
            migration_path.write_text(
                json.dumps(
                    {
                        "splits": [
                            {
                                "split": "test",
                                "before_sha256": "a" * 64,
                                "after_sha256": "b" * 64,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            archived_metric = archive_root / "summary.csv"
            archived_metric.write_text("metric,value\niou,0.5\n", encoding="utf-8")
            archive_manifest = archive_root / "manifest.json"
            archive_manifest.write_text(
                json.dumps(
                    {
                        "status": (
                            "superseded_invalid_for_scientific_results"
                        ),
                        "superseded_reference": {
                            "test_coco_sha256": "a" * 64,
                            "replacement_test_coco_sha256": "b" * 64,
                        },
                        "summary_rows": [{"mean_iou": "0.5"}],
                        "files": [
                            {
                                "archive_path": str(archived_metric),
                                "sha256": MODULE.sha256_file(
                                    archived_metric
                                ),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            dataset = SimpleNamespace(
                dataset_id="isaid_plane",
                prepared_root=prepared_root,
            )
            errors: list[str] = []
            MODULE.validate_isaid_rle_sensitivity_and_archive(
                study_root,
                [dataset],
                required=[],
                errors=errors,
            )
            self.assertEqual(errors, [])

            archived_metric.write_text("changed", encoding="utf-8")
            errors = []
            MODULE.validate_isaid_rle_sensitivity_and_archive(
                study_root,
                [dataset],
                required=[],
                errors=errors,
            )

        self.assertTrue(
            any("archive hash'i uyuşmuyor" in error for error in errors)
        )

    def test_legacy_detector_repair_chain_is_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            study_root = root / "study"
            prepared_root = root / "prepared"
            prepared_root.mkdir()
            scoped_path = (
                prepared_root
                / "detector_training_content_manifest.json"
            )
            scoped_path.write_text(
                json.dumps({"tree_sha256": "c" * 64}),
                encoding="utf-8",
            )
            original_path = (
                study_root
                / MODULE.LEGACY_DETECTOR_MANIFEST_ARCHIVE
                / "dataset"
                / "seed_42"
                / "manifest.json"
            )
            original_path.parent.mkdir(parents=True)
            original_path.write_text("legacy\n", encoding="utf-8")
            original_hash = MODULE.sha256_file(original_path)
            live_path = (
                study_root
                / "detectors"
                / "dataset"
                / "seed_42"
                / "manifest.json"
            )
            live_path.parent.mkdir(parents=True)
            live_path.write_text(
                json.dumps(
                    {
                        "input_fingerprint_capture": "provenance_repair",
                        "provenance_repair": {
                            "original_manifest_sha256": original_hash,
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit_path = (
                study_root
                / "audits"
                / "legacy_detector_manifest_repair"
                / "manifest.json"
            )
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "rows": [
                            {
                                "dataset_id": "dataset",
                                "seed": 42,
                                "action": (
                                    "repaired_with_archived_original"
                                ),
                                "manifest_sha256": MODULE.sha256_file(
                                    live_path
                                ),
                                "original_manifest_path": str(
                                    original_path
                                ),
                                "original_manifest_sha256": original_hash,
                                "detector_content_manifest_sha256": (
                                    MODULE.sha256_file(scoped_path)
                                ),
                                "detector_content_tree_sha256": "c" * 64,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            protocol = SimpleNamespace(detector_seeds=(42,))
            dataset = SimpleNamespace(
                dataset_id="dataset",
                prepared_root=prepared_root,
            )
            errors: list[str] = []
            MODULE.validate_legacy_detector_manifest_repair(
                study_root,
                protocol,
                [dataset],
                required=[],
                errors=errors,
            )
            self.assertEqual(errors, [])

            original_path.write_text("changed\n", encoding="utf-8")
            errors = []
            MODULE.validate_legacy_detector_manifest_repair(
                study_root,
                protocol,
                [dataset],
                required=[],
                errors=errors,
            )

        self.assertTrue(
            any(
                "Original detector manifest archive hash'i uyuşmuyor" in error
                for error in errors
            )
        )

    def test_superseded_archive_manifests_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            study_root = Path(temporary_directory)
            original = {
                "run_id": "old_run",
                "stage": "gt_bbox_segmentation",
                "status": "completed",
                "config_hash": "a" * 64,
                "inputs": {},
                "outputs": {},
            }
            archived_manifests = [
                study_root
                / MODULE.SUPERSEDED_ISAID_METRIC_ARCHIVE
                / "predictions"
                / "manifest.json",
                study_root
                / MODULE.LEGACY_DETECTOR_MANIFEST_ARCHIVE
                / "isaid_plane"
                / "seed_42"
                / "manifest.json",
            ]
            before: dict[Path, bytes] = {}
            for archived_manifest in archived_manifests:
                archived_manifest.parent.mkdir(parents=True)
                archived_manifest.write_text(
                    json.dumps(original),
                    encoding="utf-8",
                )
                before[archived_manifest] = archived_manifest.read_bytes()
            dataset = SimpleNamespace(
                dataset_id="isaid_plane",
                prepared_root=study_root / "prepared",
            )

            self.assertEqual(
                MODULE.attach_prepared_content_manifests(
                    study_root,
                    [dataset],
                ),
                [],
            )
            self.assertEqual(
                MODULE.backfill_completed_run_manifests(study_root),
                [],
            )
            errors: list[str] = []
            required: list[Path] = []
            MODULE.validate_completed_run_manifest_fingerprints(
                study_root,
                required=required,
                errors=errors,
            )
            self.assertEqual(errors, [])
            self.assertEqual(required, [])
            for archived_manifest in archived_manifests:
                self.assertEqual(
                    archived_manifest.read_bytes(),
                    before[archived_manifest],
                )

    @staticmethod
    def _fingerprint(path: Path) -> dict[str, object]:
        content = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }


if __name__ == "__main__":
    unittest.main()
