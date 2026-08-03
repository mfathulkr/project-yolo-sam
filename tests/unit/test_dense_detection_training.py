from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from ultralytics.models.yolo.detect.train import DetectionTrainer

from yolo_sam.detection.training import DenseInstanceDetectionTrainer


class DenseInstanceDetectionTrainerTests(TestCase):
    def setUp(self) -> None:
        self.trainer = object.__new__(DenseInstanceDetectionTrainer)
        self.trainer.args = SimpleNamespace(task="detect")

    def test_validation_batch_is_reduced_to_training_batch(self) -> None:
        with patch.object(
            DetectionTrainer,
            "get_dataloader",
            autospec=True,
            return_value="loader",
        ) as parent:
            result = self.trainer.get_dataloader(
                "validation",
                batch_size=24,
                rank=-1,
                mode="val",
            )

        self.assertEqual(result, "loader")
        parent.assert_called_once_with(
            self.trainer,
            "validation",
            batch_size=12,
            rank=-1,
            mode="val",
        )

    def test_training_batch_is_unchanged(self) -> None:
        with patch.object(
            DetectionTrainer,
            "get_dataloader",
            autospec=True,
            return_value="loader",
        ) as parent:
            self.trainer.get_dataloader(
                "train",
                batch_size=12,
                rank=-1,
                mode="train",
            )

        parent.assert_called_once_with(
            self.trainer,
            "train",
            batch_size=12,
            rank=-1,
            mode="train",
        )
