from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias.reporting.figures import (
    detector_seed_figure,
    gt_bbox_reference_comparison,
    qualitative_gt_bbox_figure,
    reference_inflation_figure,
    shared_human_reference_figure,
    strata_heatmap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical figures for the matched study."
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=STUDY_ROOT / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = args.study_root / "analysis"
    output_root = args.study_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    aggregates = pd.read_csv(analysis_root / "aggregate_metrics.csv")
    inflation = pd.read_csv(analysis_root / "reference_inflation.csv")
    canonical = pd.read_csv(analysis_root / "canonical_instance_metrics.csv")
    try:
        detector_metrics = pd.read_csv(
            analysis_root / "detector_metrics_by_seed.csv"
        )
    except EmptyDataError:
        detector_metrics = pd.DataFrame()
    outputs = [
        gt_bbox_reference_comparison(
            aggregates,
            output_root / "gt_bbox_reference_comparison.png",
        ),
        reference_inflation_figure(
            inflation,
            output_root / "isaid_reference_inflation.png",
        ),
        strata_heatmap(
            aggregates,
            output_root / "gt_bbox_strata_heatmap.png",
        ),
        qualitative_gt_bbox_figure(
            study_root=args.study_root,
            prepared_root=(
                STUDY_ROOT / "data" / "prepared" / "isaid_small_vehicle"
            ),
            dataset_id="isaid_small_vehicle",
            reference_type="human",
            canonical_metrics=canonical,
            output_path=output_root / "isaid_small_vehicle_gt_bbox_qualitative.png",
        ),
        qualitative_gt_bbox_figure(
            study_root=args.study_root,
            prepared_root=(
                STUDY_ROOT / "data" / "prepared" / "isaid_small_vehicle"
            ),
            dataset_id="isaid_small_vehicle",
            reference_type="pseudo_sam1",
            canonical_metrics=canonical,
            output_path=(
                output_root
                / "isaid_small_vehicle_pseudo_sam1_gt_bbox_qualitative.png"
            ),
        ),
        qualitative_gt_bbox_figure(
            study_root=args.study_root,
            prepared_root=(
                STUDY_ROOT
                / "data"
                / "prepared"
                / "samrs_sota_small_vehicle"
            ),
            dataset_id="samrs_sota_small_vehicle",
            reference_type="pseudo_sam1",
            canonical_metrics=canonical,
            output_path=output_root / "samrs_sota_small_vehicle_gt_bbox_qualitative.png",
        ),
    ]
    shared_audit_root = analysis_root / "shared_human_reference_audit"
    shared_summary_path = (
        shared_audit_root / "model_dual_reference_summary.csv"
    )
    shared_intervals_path = (
        shared_audit_root / "model_reference_inflation_ci.json"
    )
    if shared_summary_path.is_file() and shared_intervals_path.is_file():
        outputs.append(
            shared_human_reference_figure(
                pd.read_csv(shared_summary_path),
                json.loads(shared_intervals_path.read_text(encoding="utf-8")),
                output_root / "shared_human_reference_comparison.png",
            )
        )
    detector_output = detector_seed_figure(
        detector_metrics,
        output_root / "detector_seed_metrics.png",
    )
    if detector_output is not None:
        outputs.append(detector_output)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
