from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "study.py"
SPEC = importlib.util.spec_from_file_location("study_cli", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load study CLI")
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


class StudyCliTest(unittest.TestCase):
    def test_run_script_sets_allocator_default_without_overriding_user_value(
        self,
    ) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(STUDY.subprocess, "run") as run,
        ):
            STUDY.run_script("worker.py", ["--flag"])

        environment = run.call_args.kwargs["env"]
        self.assertEqual(
            environment["PYTORCH_ALLOC_CONF"],
            "backend:cudaMallocAsync",
        )

        with (
            patch.dict(
                os.environ,
                {"PYTORCH_ALLOC_CONF": "custom"},
                clear=True,
            ),
            patch.object(STUDY.subprocess, "run") as run,
        ):
            STUDY.run_script("worker.py", [])

        self.assertEqual(
            run.call_args.kwargs["env"]["PYTORCH_ALLOC_CONF"],
            "custom",
        )

    def test_stage_is_skipped_only_for_completed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "output.csv"
            manifest = root / "manifest.json"
            artifact.write_text("value\n1\n", encoding="utf-8")
            manifest.write_text(
                json.dumps({"status": "running"}),
                encoding="utf-8",
            )
            self.assertFalse(
                STUDY.skip_completed(
                    artifact,
                    force=False,
                    manifest_path=manifest,
                )
            )

            manifest.write_text(
                json.dumps({"status": "completed"}),
                encoding="utf-8",
            )
            self.assertTrue(
                STUDY.skip_completed(
                    artifact,
                    force=False,
                    manifest_path=manifest,
                )
            )
            self.assertFalse(
                STUDY.skip_completed(
                    artifact,
                    force=True,
                    manifest_path=manifest,
                )
            )


if __name__ == "__main__":
    unittest.main()
