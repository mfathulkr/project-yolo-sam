from __future__ import annotations

from ultralytics.models.yolo.detect.train import DetectionTrainer


class DenseInstanceDetectionTrainer(DetectionTrainer):
    """Keep validation batches at the training size for dense scenes."""

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        if mode == "val" and self.args.task == "detect":
            batch_size = max(1, batch_size // 2)
        return super().get_dataloader(
            dataset_path,
            batch_size=batch_size,
            rank=rank,
            mode=mode,
        )
