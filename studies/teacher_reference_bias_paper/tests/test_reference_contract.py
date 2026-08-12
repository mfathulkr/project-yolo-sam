from __future__ import annotations

import pandas as pd

from teacher_reference_bias_multiteacher.paths import DATASETS, REFERENCES


def test_reference_families_are_not_conflated() -> None:
    assert DATASETS["isaid_plane"].reference_types[0] == "human"
    assert REFERENCES["human"].is_independent_human
    assert DATASETS["samrs_plane"].reference_types[0] == "published_samrs_reference"
    assert REFERENCES["published_samrs_reference"].is_published_samrs
    assert not REFERENCES["published_samrs_reference"].is_independent_human
    assert "reproduced_pseudo_sam1" in DATASETS["samrs_plane"].reference_types


def test_known_positive_empty_reference_policy_is_visible() -> None:
    source = DATASETS["isaid_small_vehicle"]
    empty = pd.read_csv(source.analysis_root / "reference_empty_stats.csv").set_index("reference_type")
    assert int(empty.loc["pseudo_sam1", "empty_count"]) == 19
    assert float(empty.loc["human", "empty_rate"]) == 0.0


def test_published_samrs_is_close_but_not_identical_to_reproduced_sam1() -> None:
    for experiment_id in ("samrs_plane", "samrs_small_vehicle"):
        source = DATASETS[experiment_id]
        frame = pd.read_csv(source.analysis_root / "reference_agreement.csv")
        row = frame[
            (frame["reference_a"] == "published_samrs_reference")
            & (frame["reference_b"] == "reproduced_pseudo_sam1")
        ].iloc[0]
        value = float(row["mean_instance_iou"])
        assert 0.98 < value < 1.0
