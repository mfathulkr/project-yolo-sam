from __future__ import annotations

import argparse
import sys
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.comparison_report import (  # noqa: E402
    write_experiment_report,
    write_main_gt_bbox_report,
    write_main_report,
)
from teacher_reference_bias_multiteacher.paths import DATASETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Çapraz analiz raporlarını üret.")
    parser.add_argument("--experiment", choices=tuple(DATASETS))
    parser.add_argument("--skip-main", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = (args.experiment,) if args.experiment else tuple(DATASETS)
    for experiment_id in experiments:
        result = write_experiment_report(DATASETS[experiment_id])
        print(result["pdf"])
    if not args.skip_main and args.experiment is None:
        result = write_main_report(STUDY_ROOT / "analysis")
        print(result["pdf"])
        result = write_main_gt_bbox_report(STUDY_ROOT / "analysis")
        print(result["pdf"])


if __name__ == "__main__":
    main()
