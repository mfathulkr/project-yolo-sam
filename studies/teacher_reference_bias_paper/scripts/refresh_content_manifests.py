from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.prepared_validation import (  # noqa: E402
    build_detector_training_content_manifest,
    build_prepared_content_manifest,
)
from teacher_reference_bias_multiteacher.paths import DATASETS  # noqa: E402


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh portable prepared/master content manifests."
    )
    parser.add_argument(
        "--master-only",
        action="store_true",
        help="Skip matched prepared trees and refresh only full master pools.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for source in DATASETS.values():
        if not args.master_only:
            write_manifest(
                source.prepared_root / "content_manifest.json",
                build_prepared_content_manifest(source.prepared_root),
            )
            write_manifest(
                source.prepared_root / "detector_training_content_manifest.json",
                build_detector_training_content_manifest(source.prepared_root),
            )
        master_root = source.root / "data" / "master"
        if master_root.is_dir():
            write_manifest(
                master_root / "content_manifest.json",
                build_prepared_content_manifest(
                    master_root,
                    splits=("train", "validation", "test_pool", "test"),
                ),
            )
            if not args.master_only:
                write_manifest(
                    master_root / "detector_training_content_manifest.json",
                    build_detector_training_content_manifest(master_root),
                )


if __name__ == "__main__":
    main()
