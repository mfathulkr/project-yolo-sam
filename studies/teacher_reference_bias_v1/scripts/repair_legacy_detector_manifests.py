from __future__ import annotations

import argparse
import json
import shutil
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

from yolo_sam.data.prepared_validation import (
    validate_detector_training_content_manifest,
)
from teacher_reference_bias.reporting.analysis import sha256_file
from yolo_sam.runtime.manifest import declared_file_fingerprints
from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
)


DEFAULT_DATASETS = (
    Path("studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml"),
    Path("studies/teacher_reference_bias_v1/configs/datasets/samrs_sota_plane.yaml"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive and explicitly repair legacy detector-training "
            "manifests that predate start/finish input fingerprint tracking."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("studies/teacher_reference_bias_v1/configs/protocol.yaml"),
    )
    parser.add_argument("--dataset", type=Path, action="append")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def manifest_needs_repair(payload: dict[str, Any]) -> bool:
    return (
        payload.get("input_fingerprint_capture")
        not in {"start", "provenance_repair"}
        or not isinstance(payload.get("input_file_fingerprints"), dict)
        or not isinstance(
            payload.get("input_file_fingerprints_at_finish"),
            dict,
        )
        or not isinstance(payload.get("output_file_fingerprints"), dict)
    )


def main() -> None:
    args = parse_args()
    protocol = load_matched_study_config(project_path(args.protocol))
    dataset_paths = args.dataset or list(DEFAULT_DATASETS)
    datasets = {
        dataset.dataset_id: dataset
        for dataset in (
            load_dataset_study_config(project_path(path))
            for path in dataset_paths
        )
    }
    study_root = STUDY_ROOT / "results"
    audit_root = study_root / "audits" / "legacy_detector_manifest_repair"
    audit_path = audit_root / "manifest.json"
    if audit_path.exists():
        raise SystemExit(
            f"Repair audit already exists and will not be overwritten: {audit_path}"
        )

    expected: list[tuple[str, int, Path]] = []
    for dataset_id in sorted(datasets):
        for seed in protocol.detector_seeds:
            expected.append(
                (
                    dataset_id,
                    seed,
                    study_root
                    / "detectors"
                    / dataset_id
                    / f"seed_{seed}"
                    / "manifest.json",
                )
            )
    missing = [str(path) for _, _, path in expected if not path.is_file()]
    if missing:
        raise SystemExit(f"Detector manifests are missing: {missing}")

    rows: list[dict[str, Any]] = []
    for dataset_id, seed, manifest_path in expected:
        payload = read_json(manifest_path)
        if payload.get("stage") != "train_detector":
            raise SystemExit(f"Unexpected stage in {manifest_path}")
        if payload.get("status") != "completed":
            raise SystemExit(
                f"Detector run is not completed: {manifest_path}"
            )
        dataset = datasets[dataset_id]
        scoped_manifest_path = (
            dataset.prepared_root
            / "detector_training_content_manifest.json"
        )
        if not scoped_manifest_path.is_file():
            raise SystemExit(
                f"Detector content manifest is missing: {scoped_manifest_path}"
            )
        scoped_manifest = read_json(scoped_manifest_path)
        validation_errors = validate_detector_training_content_manifest(
            dataset.prepared_root,
            scoped_manifest,
        )
        if validation_errors:
            raise SystemExit(
                f"Invalid detector content manifest: {validation_errors}"
            )

        original_hash = sha256_file(manifest_path)
        archive_path = (
            audit_root
            / "originals"
            / dataset_id
            / f"seed_{seed}"
            / "manifest.json"
        )
        if payload.get("input_fingerprint_capture") == "provenance_repair":
            if manifest_needs_repair(payload):
                raise SystemExit(
                    "Partially repaired detector manifest is missing required "
                    f"fingerprints: {manifest_path}"
                )
            repair = payload.get("provenance_repair")
            if not isinstance(repair, dict):
                raise SystemExit(
                    f"Detector provenance repair metadata is missing: "
                    f"{manifest_path}"
                )
            if not archive_path.is_file():
                raise SystemExit(
                    f"Archived original detector manifest is missing: "
                    f"{archive_path}"
                )
            archived_hash = sha256_file(archive_path)
            if (
                repair.get("original_manifest_path") != str(archive_path)
                or repair.get("original_manifest_sha256") != archived_hash
            ):
                raise SystemExit(
                    f"Detector provenance repair chain is invalid: "
                    f"{manifest_path}"
                )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "action": "repaired_with_archived_original",
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": original_hash,
                    "original_manifest_path": str(archive_path),
                    "original_manifest_sha256": archived_hash,
                    "detector_content_manifest_path": str(
                        scoped_manifest_path
                    ),
                    "detector_content_manifest_sha256": sha256_file(
                        scoped_manifest_path
                    ),
                    "detector_content_tree_sha256": scoped_manifest[
                        "tree_sha256"
                    ],
                }
            )
            continue
        if not manifest_needs_repair(payload):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "action": "unchanged_start_fingerprinted",
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": original_hash,
                    "detector_content_manifest_path": str(
                        scoped_manifest_path
                    ),
                    "detector_content_manifest_sha256": sha256_file(
                        scoped_manifest_path
                    ),
                    "detector_content_tree_sha256": scoped_manifest[
                        "tree_sha256"
                    ],
                }
            )
            continue

        archive_path.parent.mkdir(parents=True, exist_ok=False)
        shutil.copy2(manifest_path, archive_path)
        if sha256_file(archive_path) != original_hash:
            raise RuntimeError(f"Manifest archive hash mismatch: {manifest_path}")

        inputs = payload.setdefault("inputs", {})
        inputs.pop("prepared_content_manifest", None)
        inputs["detector_training_content_manifest"] = str(
            scoped_manifest_path
        )
        run_root = manifest_path.parent
        expected_outputs = {
            "best_weights": run_root / "train" / "weights" / "best.pt",
            "results_csv": run_root / "train" / "results.csv",
            "args_yaml": run_root / "train" / "args.yaml",
        }
        missing_outputs = [
            str(path) for path in expected_outputs.values() if not path.is_file()
        ]
        if missing_outputs:
            raise SystemExit(
                f"Detector training outputs are missing: {missing_outputs}"
            )
        payload["outputs"] = {
            name: str(path) for name, path in expected_outputs.items()
        }
        input_fingerprints = declared_file_fingerprints(inputs)
        output_fingerprints = declared_file_fingerprints(
            payload.get("outputs", {})
        )
        repaired_at = datetime.now(timezone.utc).isoformat()
        payload["input_file_fingerprints"] = input_fingerprints
        payload["input_file_fingerprints_at_finish"] = input_fingerprints
        payload["output_file_fingerprints"] = output_fingerprints
        payload["input_fingerprint_capture"] = "provenance_repair"
        payload["input_drift"] = []
        payload["provenance_repair"] = {
            "schema_version": 1,
            "repaired_at_utc": repaired_at,
            "scope": "train_detector_only",
            "reason": (
                "The run began before start/finish fingerprint fields and "
                "the detector-scoped image/label manifest were introduced. "
                "The original manifest is preserved byte-for-byte. The "
                "repair fingerprints only the frozen train/validation image "
                "and YOLO-label tree actually consumed by detector training."
            ),
            "original_manifest_path": str(archive_path),
            "original_manifest_sha256": original_hash,
            "detector_content_tree_sha256": scoped_manifest["tree_sha256"],
            "isaid_rle_migration_outside_repair_scope": (
                dataset_id == "isaid_plane"
            ),
        }
        write_json(manifest_path, payload)
        rows.append(
            {
                "dataset_id": dataset_id,
                "seed": seed,
                "action": "repaired_with_archived_original",
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "original_manifest_path": str(archive_path),
                "original_manifest_sha256": original_hash,
                "detector_content_manifest_path": str(scoped_manifest_path),
                "detector_content_manifest_sha256": sha256_file(
                    scoped_manifest_path
                ),
                "detector_content_tree_sha256": scoped_manifest[
                    "tree_sha256"
                ],
            }
        )

    write_json(
        audit_path,
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "scope": "six_detector_training_manifests",
            "rows": rows,
        },
    )
    print(audit_path)
    print(
        "Repaired:",
        sum(row["action"].startswith("repaired") for row in rows),
        "Unchanged:",
        sum(row["action"].startswith("unchanged") for row in rows),
    )


if __name__ == "__main__":
    main()
