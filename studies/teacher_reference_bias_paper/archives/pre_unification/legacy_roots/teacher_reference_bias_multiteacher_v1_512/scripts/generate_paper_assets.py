#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(STUDY_ROOT / "src"))

from teacher_reference_bias_multiteacher.comparison_report import empty_reference_table
from teacher_reference_bias_multiteacher.paper_assets import (
    build_figures,
    build_tables,
    paper_asset_source_paths,
    write_manifest,
)


def main() -> None:
    analysis_root = STUDY_ROOT / "results" / "analysis"
    input_paths = [
        analysis_root / "manifest.json",
        analysis_root / "aggregate_metrics.csv",
        analysis_root / "paired_reference_effects.csv",
        analysis_root / "teacher_advantage.csv",
        analysis_root / "reference_agreement.csv",
        analysis_root / "detector_seed_summary.csv",
        analysis_root / "canonical_instance_metrics.csv",
    ]
    aggregates, effects, advantages, agreements, detector, metrics = (
        pd.read_csv(path) for path in input_paths[1:]
    )
    empty_stats = empty_reference_table().rename(
        columns={
            "Teacher": "teacher_label",
            "Instances": "instance_count",
            "Empty masks": "empty_masks",
            "Empty rate": "empty_rate",
        }
    )
    empty_stats["dataset_id"] = empty_stats["Dataset"].map(
        {"iSAID Plane": "isaid_plane", "iSAID Small Vehicle": "isaid_small_vehicle"}
    )
    empty_stats["empty_rate"] = empty_stats["empty_rate"].astype(float)
    table_paths = build_tables(
        aggregates,
        effects,
        advantages,
        agreements,
        empty_stats,
        detector,
    )
    figure_paths = build_figures(
        aggregates,
        effects,
        agreements,
        empty_stats,
        metrics,
    )
    input_paths.extend(paper_asset_source_paths(metrics))
    manifest = write_manifest(table_paths, figure_paths, input_paths=input_paths)
    print(f"{len(table_paths)} table files, {len(figure_paths)} figure files")
    print(manifest)


if __name__ == "__main__":
    main()
