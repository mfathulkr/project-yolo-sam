from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "finalize_after_post_training.py"
)
SPEC = importlib.util.spec_from_file_location("finalization_pipeline", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class FinalizationPipelineTests(unittest.TestCase):
    def test_plan_builds_and_validates_reports_before_bundle_status(self) -> None:
        commands = PIPELINE.command_plan()
        joined = [" ".join(command) for command in commands]
        self.assertEqual(len(commands), 6)
        self.assertIn("analyze", joined[0])
        self.assertIn("figures", joined[1])
        self.assertIn("write_full_metric_reports.py", joined[2])
        self.assertIn("validate_full_metric_reports.py", joined[3])
        self.assertIn("build_portable_bundles.py", joined[4])
        self.assertIn("--strict", joined[5])

    def test_dependencies_cover_both_datasets(self) -> None:
        self.assertEqual(
            [path.parent.name for path in PIPELINE.dependency_manifests()],
            ["isaid_small_vehicle", "samrs_sota_small_vehicle"],
        )


if __name__ == "__main__":
    unittest.main()
