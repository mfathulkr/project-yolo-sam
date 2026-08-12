from __future__ import annotations

import numpy as np
import pandas as pd

from teacher_reference_bias_multiteacher.analysis import validate_metric_cube
from teacher_reference_bias_multiteacher.paths import DATASETS, MODELS


def test_canonical_metric_cubes_and_coverage_identity() -> None:
    for source in DATASETS.values():
        frame = pd.read_csv(source.analysis_root / "canonical_instance_metrics.csv")
        frame["detector_seed"] = frame["detector_seed"].astype("Int64")
        validate_metric_cube(frame)
        assert len(frame) == source.instance_count * 3 * 2 * 4
        identity_reference = {
            "sam1": "pseudo_sam1" if source.dataset_family == "isaid" else "reproduced_pseudo_sam1",
            "sam2": "pseudo_sam2",
            "sam3": "pseudo_sam3",
        }
        for model in MODELS:
            selected = frame[
                (frame["model"] == model)
                & (frame["bbox_source"] == "gt_bbox")
                & (frame["reference_type"] == identity_reference[model])
            ]
            reference_pixels = selected["true_positive_pixels"] + selected["false_negative_pixels"]
            expected = np.where(reference_pixels > 0, 1.0, 0.0)
            assert np.allclose(selected["iou"], expected)


def test_isaid_yolo_own_reference_effect_is_positive() -> None:
    for experiment_id in ("isaid_plane", "isaid_small_vehicle"):
        source = DATASETS[experiment_id]
        effects = pd.read_csv(source.analysis_root / "paired_reference_effects.csv")
        selected = effects[effects["bbox_source"] == "yolo_bbox"]
        for model in MODELS:
            row = selected[
                (selected["model"] == model)
                & (selected["comparison_reference"] == f"pseudo_{model}")
            ].iloc[0]
            assert float(row["delta_iou"]) > 0
            assert float(row["delta_ci_lower"]) > 0
