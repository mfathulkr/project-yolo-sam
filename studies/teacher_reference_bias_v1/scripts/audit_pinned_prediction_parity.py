from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias.reporting.analysis import sha256_file
from teacher_reference_bias.reporting.prediction_parity import (
    summarize_prediction_masks,
)
from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
)


DEFAULT_DATASETS = (
    Path("studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml"),
    Path("studies/teacher_reference_bias_v1/configs/datasets/samrs_sota_plane.yaml"),
)
MODELS = ("sam1", "sam2", "sam3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove that explicitly pinned SAM revisions reproduce the existing "
            "GT-bbox masks exactly."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("snapshot", "verify"),
        required=True,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("studies/teacher_reference_bias_v1/configs/protocol.yaml"),
    )
    parser.add_argument("--dataset", type=Path, action="append")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "studies/teacher_reference_bias_v1/results/audits/"
            "pinned_revision_prediction_parity.json"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing baseline snapshot intentionally.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def condition_rows(
    *,
    study_root: Path,
    dataset_paths: list[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_path in dataset_paths:
        dataset = load_dataset_study_config(dataset_path)
        for model in MODELS:
            prediction_path = (
                study_root
                / "predictions"
                / dataset.dataset_id
                / model
                / "gt_bbox"
                / "predictions.jsonl"
            )
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "model": model,
                    "prediction_path": str(prediction_path.resolve()),
                    "summary": summarize_prediction_masks(prediction_path),
                }
            )
    return rows


def snapshot(args: argparse.Namespace) -> None:
    output = project_path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(
            f"Baseline already exists: {output}. Use --force to replace it."
        )
    protocol_path = project_path(args.protocol)
    protocol = load_matched_study_config(protocol_path)
    dataset_paths = [
        project_path(path) for path in (args.dataset or DEFAULT_DATASETS)
    ]
    conditions = condition_rows(
        study_root=STUDY_ROOT / "results",
        dataset_paths=dataset_paths,
    )
    payload = {
        "schema_version": 1,
        "status": "baseline_recorded",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verified_at_utc": None,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "conditions": [
            {
                "dataset_id": row["dataset_id"],
                "model": row["model"],
                "prediction_path": row["prediction_path"],
                "baseline": row["summary"],
                "current": None,
                "exact_mask_parity": None,
            }
            for row in conditions
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)


def verify(args: argparse.Namespace) -> None:
    output = project_path(args.output)
    if not output.is_file():
        raise FileNotFoundError(
            f"Baseline snapshot is missing: {output}. Run --mode snapshot first."
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    protocol_path = project_path(args.protocol)
    protocol = load_matched_study_config(protocol_path)
    dataset_paths = [
        project_path(path) for path in (args.dataset or DEFAULT_DATASETS)
    ]
    current_rows = condition_rows(
        study_root=STUDY_ROOT / "results",
        dataset_paths=dataset_paths,
    )
    current_by_key = {
        (row["dataset_id"], row["model"]): row for row in current_rows
    }
    baseline_keys = {
        (str(row["dataset_id"]), str(row["model"]))
        for row in payload.get("conditions", [])
    }
    if baseline_keys != set(current_by_key):
        raise ValueError(
            "Baseline conditions do not match the requested datasets/models"
        )

    passed = True
    for row in payload["conditions"]:
        key = (str(row["dataset_id"]), str(row["model"]))
        current = current_by_key[key]["summary"]
        baseline = row["baseline"]
        exact = (
            current["row_count"] == baseline["row_count"]
            and current["ok_count"] == baseline["ok_count"]
            and current["canonical_mask_sha256"]
            == baseline["canonical_mask_sha256"]
        )
        row["current"] = current
        row["exact_mask_parity"] = exact
        passed = passed and exact

    payload["status"] = "pass" if passed else "fail"
    payload["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["protocol"] = str(protocol_path.resolve())
    payload["protocol_sha256"] = sha256_file(protocol_path)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    if not passed:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.mode == "snapshot":
        snapshot(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
