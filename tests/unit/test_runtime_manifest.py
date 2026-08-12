from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yolo_sam.runtime.manifest import (
    declared_file_fingerprints,
    environment_snapshot,
    finish_run_manifest,
    new_run_manifest,
    validate_completed_run_manifest,
)


class RuntimeManifestTest(unittest.TestCase):
    def test_environment_snapshot_records_only_selected_runtime_variables(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "2",
                "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
                "UNRELATED_SECRET": "must-not-be-recorded",
            },
            clear=False,
        ):
            snapshot = environment_snapshot(
                Path(__file__).resolve().parents[2]
            )

        self.assertEqual(
            snapshot["environment_variables"]["CUDA_VISIBLE_DEVICES"],
            "2",
        )
        self.assertEqual(
            snapshot["environment_variables"]["PYTORCH_ALLOC_CONF"],
            "backend:cudaMallocAsync",
        )
        self.assertNotIn(
            "UNRELATED_SECRET",
            snapshot["environment_variables"],
        )

    def test_finish_manifest_records_terminal_state(self) -> None:
        manifest = {
            "status": "running",
            "finished_at_utc": None,
            "error": None,
        }
        finished = finish_run_manifest(manifest, status="completed")
        self.assertEqual(finished["status"], "completed")
        self.assertIsNotNone(finished["finished_at_utc"])

    def test_declared_files_are_hashed_but_directories_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.txt"
            source.write_text("frozen-input\n", encoding="utf-8")

            fingerprints = declared_file_fingerprints(
                {
                    "source": str(source),
                    "directory": str(root),
                    "count": 1,
                }
            )

        self.assertEqual(set(fingerprints), {"source"})
        self.assertEqual(fingerprints["source"]["bytes"], 13)
        self.assertEqual(
            fingerprints["source"]["sha256"],
            "34edc4e89aefd8ca6bb502e219f13d3f444ee8a2a95560275c6c1e3d5cf4c72d",
        )

    def test_finish_manifest_hashes_declared_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.txt"
            output.write_text("result\n", encoding="utf-8")
            manifest = {
                "status": "running",
                "finished_at_utc": None,
                "error": None,
                "inputs": {},
                "outputs": {"result": str(output)},
            }

            finished = finish_run_manifest(manifest, status="completed")

        self.assertEqual(
            set(finished["output_file_fingerprints"]),
            {"result"},
        )

    def test_finish_preserves_start_hash_and_reports_input_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.txt"
            source.write_text("before\n", encoding="utf-8")
            manifest = new_run_manifest(
                project_root=Path(__file__).resolve().parents[2],
                run_id="test-run",
                stage="test",
                config_hash="a" * 64,
                inputs={"source": str(source)},
                parameters={},
            )
            start_hash = manifest["input_file_fingerprints"]["source"][
                "sha256"
            ]
            source.write_text("after\n", encoding="utf-8")

            finished = finish_run_manifest(manifest, status="completed")

        self.assertEqual(
            finished["input_file_fingerprints"]["source"]["sha256"],
            start_hash,
        )
        self.assertNotEqual(
            finished["input_file_fingerprints_at_finish"]["source"][
                "sha256"
            ],
            start_hash,
        )
        self.assertEqual(finished["input_drift"], ["source"])
        self.assertEqual(finished["input_fingerprint_capture"], "start")

    def test_completed_manifest_allows_only_explicit_changed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "mutable-provenance.json"
            output = root / "result.txt"
            manifest_path = root / "manifest.json"
            source.write_text('{"version": 1}\n', encoding="utf-8")
            output.write_text("result\n", encoding="utf-8")
            manifest = new_run_manifest(
                project_root=Path(__file__).resolve().parents[2],
                run_id="test-run",
                stage="test",
                config_hash="a" * 64,
                inputs={"segmenter_provenance": str(source)},
                parameters={},
            )
            manifest["outputs"] = {"result": str(output)}
            finish_run_manifest(manifest, status="completed")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            source.write_text('{"version": 2}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Current input fingerprint mismatch"):
                validate_completed_run_manifest(manifest_path)
            validated = validate_completed_run_manifest(
                manifest_path,
                allow_changed_input_names=frozenset({"segmenter_provenance"}),
            )

        self.assertEqual(validated["run_id"], "test-run")

    def test_completed_manifest_resolves_repository_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / ".git").mkdir()
            source = root / "data" / "source.txt"
            output = root / "results" / "result.txt"
            manifest_path = root / "results" / "run" / "manifest.json"
            source.parent.mkdir()
            output.parent.mkdir()
            manifest_path.parent.mkdir()
            source.write_text("source\n", encoding="utf-8")
            output.write_text("result\n", encoding="utf-8")
            manifest = new_run_manifest(
                project_root=Path(__file__).resolve().parents[2],
                run_id="portable-run",
                stage="test",
                config_hash="a" * 64,
                inputs={"source": str(source)},
                parameters={},
            )
            manifest["outputs"] = {"result": str(output)}
            finish_run_manifest(manifest, status="completed")
            manifest["inputs"]["source"] = "data/source.txt"
            manifest["outputs"]["result"] = "results/result.txt"
            for group, name, relative in (
                ("input_file_fingerprints", "source", "data/source.txt"),
                ("input_file_fingerprints_at_finish", "source", "data/source.txt"),
                ("output_file_fingerprints", "result", "results/result.txt"),
            ):
                manifest[group][name]["path"] = relative
            manifest["path_base"] = "repository_root"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            validated = validate_completed_run_manifest(manifest_path)

        self.assertEqual(validated["run_id"], "portable-run")

    def test_completed_manifest_never_exempts_changed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "mutable-provenance.json"
            output = root / "result.txt"
            manifest_path = root / "manifest.json"
            source.write_text("source\n", encoding="utf-8")
            output.write_text("before\n", encoding="utf-8")
            manifest = new_run_manifest(
                project_root=Path(__file__).resolve().parents[2],
                run_id="test-run",
                stage="test",
                config_hash="a" * 64,
                inputs={"segmenter_provenance": str(source)},
                parameters={},
            )
            manifest["outputs"] = {"segmenter_provenance": str(output)}
            finish_run_manifest(manifest, status="completed")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output.write_text("after\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Current output fingerprint mismatch"):
                validate_completed_run_manifest(
                    manifest_path,
                    allow_changed_input_names=frozenset({"segmenter_provenance"}),
                )

    def test_completed_manifest_accepts_only_declared_runtime_input_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "resume.pt"
            output = root / "result.txt"
            manifest_path = root / "manifest.json"
            source.write_text("before\n", encoding="utf-8")
            output.write_text("result\n", encoding="utf-8")
            manifest = new_run_manifest(
                project_root=Path(__file__).resolve().parents[2],
                run_id="resume-run",
                stage="test",
                config_hash="a" * 64,
                inputs={"resume_checkpoint": str(source)},
                parameters={},
                expected_input_drift=("resume_checkpoint",),
            )
            source.write_text("after\n", encoding="utf-8")
            manifest["outputs"] = {"result": str(output)}
            finish_run_manifest(manifest, status="completed")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            validated = validate_completed_run_manifest(manifest_path)

        self.assertEqual(validated["input_drift"], ["resume_checkpoint"])
        self.assertEqual(validated["expected_input_drift"], ["resume_checkpoint"])

    def test_completed_manifest_rejects_undeclared_runtime_input_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.txt"
            output = root / "result.txt"
            manifest_path = root / "manifest.json"
            source.write_text("before\n", encoding="utf-8")
            output.write_text("result\n", encoding="utf-8")
            manifest = new_run_manifest(
                project_root=Path(__file__).resolve().parents[2],
                run_id="drift-run",
                stage="test",
                config_hash="a" * 64,
                inputs={"source": str(source)},
                parameters={},
            )
            source.write_text("after\n", encoding="utf-8")
            manifest["outputs"] = {"result": str(output)}
            finish_run_manifest(manifest, status="completed")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Run inputs changed"):
                validate_completed_run_manifest(manifest_path)

    def test_completed_manifest_rejects_unknown_expected_drift_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.txt"
            output = root / "result.txt"
            manifest_path = root / "manifest.json"
            source.write_text("source\n", encoding="utf-8")
            output.write_text("result\n", encoding="utf-8")
            manifest = new_run_manifest(
                project_root=Path(__file__).resolve().parents[2],
                run_id="invalid-loaded-run",
                stage="test",
                config_hash="a" * 64,
                inputs={"source": str(source)},
                parameters={},
            )
            manifest["expected_input_drift"] = ["resume_checkpoint"]
            manifest["outputs"] = {"result": str(output)}
            finish_run_manifest(manifest, status="completed")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "declared inputs"):
                validate_completed_run_manifest(manifest_path)

    def test_new_manifest_rejects_unknown_expected_drift_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "declared inputs"):
            new_run_manifest(
                project_root=Path(__file__).resolve().parents[2],
                run_id="invalid-run",
                stage="test",
                config_hash="a" * 64,
                inputs={},
                parameters={},
                expected_input_drift=("resume_checkpoint",),
            )


if __name__ == "__main__":
    unittest.main()
