from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "complete_yolo_condition_after_training.py"
)
SPEC = importlib.util.spec_from_file_location("post_training_pipeline", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class PostTrainingPipelineTests(unittest.TestCase):
    def test_command_plan_covers_detector_and_three_segmenters(self) -> None:
        commands = PIPELINE.command_plan(Path("dataset.yaml"), "2")
        self.assertEqual(len(commands), 7)
        self.assertEqual(commands[0][2], "detect")
        self.assertEqual(
            [command[command.index("--model") + 1] for command in commands[1::2]],
            ["sam1", "sam2", "sam3"],
        )
        for command in commands:
            self.assertIn("42", command)

    def test_dataset_id_is_read_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dataset.yaml"
            path.write_text("dataset_id: example\n", encoding="utf-8")
            self.assertEqual(PIPELINE.read_dataset_id(path), "example")


if __name__ == "__main__":
    unittest.main()
