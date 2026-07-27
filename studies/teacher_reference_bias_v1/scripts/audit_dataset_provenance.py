from __future__ import annotations

import argparse
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.profiles import get_dataset_profile
from yolo_sam.data.provenance import (
    audit_isaid_coco_dataset,
    audit_samrs_pickle_dataset,
    write_audit_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dataset identity, category mapping, annotation fields, and source-scene splits."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", default="samrs_sota")
    parser.add_argument("--target-category", default=None)
    parser.add_argument("--declared-target-id", type=int, default=None)
    parser.add_argument(
        "--allow-raw-scene-overlap",
        action="store_true",
        help=(
            "Report source-scene overlap in the publisher split as a warning. "
            "Only use this when the preparation stage rebuilds source-scene-safe splits."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STUDY_ROOT / "results" / "dataset_audits",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_dataset_profile(args.profile)
    if profile.annotation_format == "samrs_pickle_instances":
        report = audit_samrs_pickle_dataset(
            root=args.root,
            profile=profile,
            target_category=args.target_category,
            declared_target_id=args.declared_target_id,
            allow_raw_scene_overlap=args.allow_raw_scene_overlap,
        )
    elif profile.annotation_format == "coco_instance_segmentation":
        report = audit_isaid_coco_dataset(
            root=args.root,
            profile=profile,
            target_category=args.target_category or "plane",
        )
    else:
        raise ValueError(f"Unsupported audit format: {profile.annotation_format}")
    write_audit_report(
        report,
        json_path=args.output_dir / f"{profile.profile_id}_audit.json",
        markdown_path=args.output_dir / f"{profile.profile_id}_audit.md",
    )
    print(f"Audit status: {'PASS' if report.passed else 'FAIL'}")
    for finding in report.findings:
        print(f"[{finding.severity.upper()}] {finding.code}: {finding.message}")
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
