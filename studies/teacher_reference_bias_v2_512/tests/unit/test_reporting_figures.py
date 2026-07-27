from __future__ import annotations

import unittest

import pandas as pd

from teacher_reference_bias.reporting.figures import (
    STRATUM_ORDER,
    _select_representative_instances,
)


class ReportingFiguresTest(unittest.TestCase):
    def test_representative_selection_uses_all_three_models(self) -> None:
        rows = []
        values = {
            "a": {"sam1": 1.0, "sam2": 0.1, "sam3": 0.1},
            "b": {"sam1": 1.0, "sam2": 0.5, "sam3": 0.5},
            "c": {"sam1": 1.0, "sam2": 0.9, "sam3": 0.9},
        }
        for stratum in STRATUM_ORDER[1:]:
            for suffix, model_values in values.items():
                for model, iou in model_values.items():
                    rows.append(
                        {
                            "dataset_id": "dataset",
                            "reference_type": "pseudo_sam1",
                            "bbox_source": "gt_bbox",
                            "model": model,
                            "instance_id": f"{stratum}-{suffix}",
                            "source_scene_id": f"{stratum}-scene-{suffix}",
                            "stratum": stratum,
                            "iou": iou,
                        }
                    )

        selected = _select_representative_instances(
            pd.DataFrame(rows),
            dataset_id="dataset",
            reference_type="pseudo_sam1",
        )

        self.assertEqual(
            selected,
            [f"{stratum}-b" for stratum in STRATUM_ORDER[1:]],
        )


if __name__ == "__main__":
    unittest.main()
