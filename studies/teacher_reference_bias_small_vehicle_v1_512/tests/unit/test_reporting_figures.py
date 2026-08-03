from __future__ import annotations

import unittest

import pandas as pd

from teacher_reference_bias.reporting.figures import (
    STRATUM_ORDER,
    _select_representative_images,
)


class ReportingFiguresTest(unittest.TestCase):
    def test_representative_selection_uses_all_instances_and_models(self) -> None:
        rows = []
        values = {
            1: {"sam1": 1.0, "sam2": 0.1, "sam3": 0.1},
            2: {"sam1": 1.0, "sam2": 0.5, "sam3": 0.5},
            3: {"sam1": 1.0, "sam2": 0.9, "sam3": 0.9},
        }
        for stratum in STRATUM_ORDER[1:]:
            for raw_image_id, model_values in values.items():
                image_id = f"dataset:{raw_image_id}"
                for instance_index in range(2):
                    for model, iou in model_values.items():
                        rows.append(
                            {
                                "dataset_id": "dataset",
                                "reference_type": "pseudo_sam1",
                                "bbox_source": "gt_bbox",
                                "model": model,
                                "image_id": image_id,
                                "instance_id": (
                                    f"{stratum}-{image_id}-{instance_index}"
                                ),
                                "source_scene_id": (
                                    f"{stratum}-scene-{image_id}"
                                ),
                                "stratum": stratum,
                                "iou": iou,
                            }
                        )

        selected = _select_representative_images(
            pd.DataFrame(rows),
            dataset_id="dataset",
            reference_type="pseudo_sam1",
        )

        self.assertEqual(selected, ["dataset:2"] * 4)


if __name__ == "__main__":
    unittest.main()
