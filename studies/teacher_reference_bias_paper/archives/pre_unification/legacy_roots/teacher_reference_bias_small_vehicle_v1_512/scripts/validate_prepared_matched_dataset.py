from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.prepared_validation import (
    build_detector_training_content_manifest,
    build_prepared_content_manifest,
    validate_prepared_matched_dataset,
)
from yolo_sam.data.contracts import BBoxSource, ReferenceType
from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a prepared matched-study dataset before model execution."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=STUDY_ROOT / "configs" / "protocol.yaml",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--expected-test-per-stratum",
        type=int,
        help="Intentional override; defaults to evaluation.max_per_stratum.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_matched_study_config(args.protocol)
    dataset = load_dataset_study_config(args.dataset)
    expected_test_per_stratum = (
        args.expected_test_per_stratum
        if args.expected_test_per_stratum is not None
        else int(protocol.evaluation["max_per_stratum"])
    )
    report = validate_prepared_matched_dataset(
        dataset.prepared_root,
        image_size=protocol.image_size,
        expected_test_per_stratum=expected_test_per_stratum,
        overlap_threshold=float(protocol.evaluation["overlap_threshold"]),
        expected_reference_type=dataset.reference_type,
        expected_bbox_source=(
            BBoxSource.HUMAN_ANNOTATION
            if dataset.reference_type == ReferenceType.HUMAN
            else BBoxSource.ORIGINAL_DETECTION_ANNOTATION
        ),
    )
    output_path = args.output or (
        STUDY_ROOT
        / "results"
        / "audits"
        / f"{dataset.dataset_id}_prepared_dataset.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for split, summary in report.split_summaries.items():
        print(f"{split}: {summary}")
    for finding in report.findings:
        print(f"{finding.severity.upper()} {finding.code}: {finding.message}")
    print(f"passed={report.passed}")
    print(output_path)
    if not report.passed:
        raise SystemExit(1)
    content_manifest_path = dataset.prepared_root / "content_manifest.json"
    content_manifest_path.write_text(
        json.dumps(
            build_prepared_content_manifest(dataset.prepared_root),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(content_manifest_path)
    detector_content_manifest_path = (
        dataset.prepared_root / "detector_training_content_manifest.json"
    )
    detector_content_manifest_path.write_text(
        json.dumps(
            build_detector_training_content_manifest(dataset.prepared_root),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(detector_content_manifest_path)


if __name__ == "__main__":
    main()
