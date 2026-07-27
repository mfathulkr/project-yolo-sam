#!/usr/bin/env python3
"""Migrate frozen study manifests after the repository layout refactor."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.prepared_validation import (  # noqa: E402
    build_detector_training_content_manifest,
    build_prepared_content_manifest,
)
from yolo_sam.runtime.manifest import (  # noqa: E402
    declared_file_fingerprints,
)


MIGRATION_ID = "study_layout_20260726"
RESULTS_ROOT = STUDY_ROOT / "results"
AUDIT_ROOT = RESULTS_ROOT / "audits" / "repository_layout_migration"
AUDIT_PATH = AUDIT_ROOT / "layout_migration.json"
ORIGINALS_ROOT = AUDIT_ROOT / "originals"
PRE_MOVE_MANIFEST = (
    REPO_ROOT / "docs" / "migration" / "study_layout_20260726_pre.json"
)
POST_MOVE_MANIFEST = (
    REPO_ROOT / "docs" / "migration" / "study_layout_20260726_post.json"
)

OLD_REPO = REPO_ROOT
OLD_STUDY_RESULTS = OLD_REPO / "artifacts" / "studies" / STUDY_ROOT.name
OLD_ISAID_PREPARED = OLD_REPO / "data" / "matched" / "isaid_plane"
OLD_SAMRS_PREPARED = (
    OLD_REPO / "data" / "matched" / "samrs_sota_plane"
)

PATH_REPLACEMENTS = (
    (
        str(OLD_STUDY_RESULTS),
        str(RESULTS_ROOT),
    ),
    (
        str(OLD_ISAID_PREPARED),
        str(STUDY_ROOT / "data" / "prepared" / "isaid_plane"),
    ),
    (
        str(OLD_SAMRS_PREPARED),
        str(STUDY_ROOT / "data" / "prepared" / "samrs_sota_plane"),
    ),
    (
        str(OLD_REPO / "data" / "isaid_raw"),
        str(OLD_REPO / "datasets" / "isaid" / "raw"),
    ),
    (
        str(OLD_REPO / "data" / "samrs_raw"),
        str(OLD_REPO / "datasets" / "samrs" / "raw"),
    ),
    (
        str(OLD_REPO / "configs" / "datasets" / "isaid_plane.yaml"),
        str(STUDY_ROOT / "configs" / "datasets" / "isaid_plane.yaml"),
    ),
    (
        str(
            OLD_REPO
            / "configs"
            / "datasets"
            / "samrs_sota_plane.yaml"
        ),
        str(
            STUDY_ROOT
            / "configs"
            / "datasets"
            / "samrs_sota_plane.yaml"
        ),
    ),
    (
        str(
            OLD_REPO
            / "configs"
            / "studies"
            / "teacher_reference_bias_v1.yaml"
        ),
        str(STUDY_ROOT / "configs" / "protocol.yaml"),
    ),
    (
        str(OLD_REPO / "paper_teacher_reference_bias"),
        str(STUDY_ROOT / "reports" / "paper"),
    ),
    (
        str(OLD_REPO / "docs" / "REPRODUCIBILITY_APPENDIX.md"),
        str(STUDY_ROOT / "docs" / "REPRODUCIBILITY_APPENDIX.md"),
    ),
    (
        str(OLD_REPO / "scripts"),
        str(STUDY_ROOT / "scripts"),
    ),
    (
        str(OLD_REPO / "src" / "sam3_bbox_study"),
        str(OLD_REPO / "src" / "yolo_sam"),
    ),
    (
        str(OLD_REPO / "yolo26x.pt"),
        str(OLD_REPO / "models" / "yolo" / "yolo26x.pt"),
    ),
    (
        str(OLD_REPO / "yolo26n.pt"),
        str(OLD_REPO / "models" / "yolo" / "yolo26n.pt"),
    ),
)

ARCHIVE_ROOTS = (
    RESULTS_ROOT / "audits" / "pre_isaid_lossless_rle_fix",
    RESULTS_ROOT / "audits" / "pre_isaid_lossless_rle_metric_fix",
    RESULTS_ROOT
    / "audits"
    / "legacy_detector_manifest_repair"
    / "originals",
    RESULTS_ROOT / "legacy",
    AUDIT_ROOT,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_paths(value: Any) -> Any:
    if isinstance(value, str):
        replaced = value
        for old, new in PATH_REPLACEMENTS:
            replaced = replaced.replace(old, new)
        return replaced
    if isinstance(value, list):
        return [replace_paths(item) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_paths(item)
            for key, item in value.items()
        }
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def is_archived(path: Path) -> bool:
    return any(path.is_relative_to(root) for root in ARCHIVE_ROOTS)


def candidate_json_files() -> list[Path]:
    files = []
    for path in RESULTS_ROOT.rglob("*.json"):
        if is_archived(path):
            continue
        if path == RESULTS_ROOT / "reproducibility_manifest.json":
            continue
        if path.is_relative_to(RESULTS_ROOT / "analysis"):
            continue
        files.append(path)
    files.extend(
        [
            STUDY_ROOT / "configs" / "protocol.yaml",
            STUDY_ROOT / "configs" / "datasets" / "isaid_plane.yaml",
            STUDY_ROOT
            / "configs"
            / "datasets"
            / "samrs_sota_plane.yaml",
            STUDY_ROOT / "data" / "prepared" / "isaid_plane" / "data.yaml",
            STUDY_ROOT
            / "data"
            / "prepared"
            / "isaid_plane"
            / "content_manifest.json",
            STUDY_ROOT
            / "data"
            / "prepared"
            / "isaid_plane"
            / "detector_training_content_manifest.json",
            STUDY_ROOT
            / "data"
            / "prepared"
            / "samrs_sota_plane"
            / "data.yaml",
            STUDY_ROOT
            / "data"
            / "prepared"
            / "samrs_sota_plane"
            / "content_manifest.json",
            STUDY_ROOT
            / "data"
            / "prepared"
            / "samrs_sota_plane"
            / "detector_training_content_manifest.json",
        ]
    )
    return sorted(set(path for path in files if path.is_file()))


def backup(path: Path) -> Path:
    relative = path.relative_to(REPO_ROOT)
    destination = ORIGINALS_ROOT / relative
    destination = destination.with_suffix(destination.suffix + ".original")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)
    return destination


def update_data_yaml() -> list[Path]:
    changed: list[Path] = []
    for dataset_id in ("isaid_plane", "samrs_sota_plane"):
        path = (
            STUDY_ROOT
            / "data"
            / "prepared"
            / dataset_id
            / "data.yaml"
        )
        original = path.read_text(encoding="utf-8")
        migrated = replace_paths(original)
        if migrated != original:
            path.write_text(migrated, encoding="utf-8")
            changed.append(path)
    return changed


def rebuild_prepared_manifests() -> list[Path]:
    changed: list[Path] = []
    for dataset_id in ("isaid_plane", "samrs_sota_plane"):
        dataset_root = STUDY_ROOT / "data" / "prepared" / dataset_id
        prepared_path = dataset_root / "content_manifest.json"
        detector_path = (
            dataset_root / "detector_training_content_manifest.json"
        )
        write_json(
            prepared_path,
            build_prepared_content_manifest(dataset_root),
        )
        write_json(
            detector_path,
            build_detector_training_content_manifest(dataset_root),
        )
        changed.extend((prepared_path, detector_path))
    return changed


def migrate_run_manifest(
    path: Path,
    payload: dict[str, Any],
    original_payload: dict[str, Any],
) -> None:
    previous_capture = original_payload.get("input_fingerprint_capture")
    previous_repair = original_payload.get("provenance_repair")
    migration = {
        "migration_id": MIGRATION_ID,
        "reason": "repository_layout_only",
        "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_input_fingerprint_capture": previous_capture,
        "original_config_hash": original_payload.get("config_hash"),
        "original_input_file_fingerprints": deepcopy(
            original_payload.get("input_file_fingerprints")
        ),
        "original_input_file_fingerprints_at_finish": deepcopy(
            original_payload.get("input_file_fingerprints_at_finish")
        ),
        "original_output_file_fingerprints": deepcopy(
            original_payload.get("output_file_fingerprints")
        ),
        "previous_provenance_repair": deepcopy(previous_repair),
        "pre_move_manifest": {
            "path": str(PRE_MOVE_MANIFEST),
            "sha256": sha256_file(PRE_MOVE_MANIFEST),
        },
        "post_move_manifest": {
            "path": str(POST_MOVE_MANIFEST),
            "sha256": sha256_file(POST_MOVE_MANIFEST),
        },
    }
    payload["layout_migration"] = migration
    inputs = payload.get("inputs")
    outputs = payload.get("outputs")
    if isinstance(inputs, dict):
        current_inputs = declared_file_fingerprints(inputs)
        payload["input_file_fingerprints"] = current_inputs
        payload["input_file_fingerprints_at_finish"] = deepcopy(
            current_inputs
        )
    if isinstance(outputs, dict):
        payload["output_file_fingerprints"] = declared_file_fingerprints(
            outputs
        )
    payload["input_fingerprint_capture"] = "layout_migration"
    payload["input_drift"] = []
    write_json(path, payload)


def migrate_json_files() -> list[Path]:
    changed: list[Path] = []
    for path in candidate_json_files():
        if path.suffix != ".json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        migrated = replace_paths(payload)
        if (
            isinstance(migrated, dict)
            and migrated.get("status") == "completed"
            and "run_id" in migrated
            and "stage" in migrated
        ):
            migrate_run_manifest(path, migrated, payload)
            changed.append(path)
        elif migrated != payload:
            write_json(path, migrated)
            changed.append(path)
    return changed


def update_segmenter_provenance() -> Path:
    path = RESULTS_ROOT / "audits" / "segmenter_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol_path = STUDY_ROOT / "configs" / "protocol.yaml"
    payload = replace_paths(payload)
    payload["protocol"] = str(protocol_path)
    payload["protocol_sha256"] = sha256_file(protocol_path)
    write_json(path, payload)
    return path


def update_detector_repair_audit() -> Path:
    path = (
        RESULTS_ROOT
        / "audits"
        / "legacy_detector_manifest_repair"
        / "manifest.json"
    )
    payload = replace_paths(
        json.loads(path.read_text(encoding="utf-8"))
    )
    for row in payload["rows"]:
        manifest_path = Path(row["manifest_path"])
        row["manifest_sha256"] = sha256_file(manifest_path)
        original_path = row.get("original_manifest_path")
        if original_path:
            row["original_manifest_sha256"] = sha256_file(
                Path(original_path)
            )
        dataset_root = (
            STUDY_ROOT / "data" / "prepared" / row["dataset_id"]
        )
        content_path = (
            dataset_root / "detector_training_content_manifest.json"
        )
        content = json.loads(content_path.read_text(encoding="utf-8"))
        row["detector_content_manifest_path"] = str(content_path)
        row["detector_content_manifest_sha256"] = sha256_file(
            content_path
        )
        row["detector_content_tree_sha256"] = content["tree_sha256"]
    write_json(path, payload)
    return path


def verify_immutable_archives(
    before: dict[str, str],
) -> None:
    after = {
        str(path): sha256_file(path)
        for root in ARCHIVE_ROOTS[:-1]
        for path in root.rglob("*")
        if path.is_file()
    }
    if after != before:
        raise RuntimeError("An immutable scientific archive changed")


def stable_modified_files(paths: set[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        if path.is_relative_to(RESULTS_ROOT / "analysis"):
            continue
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "original_copy": str(backup(path)),
            }
        )
    return rows


def main() -> None:
    if AUDIT_PATH.exists():
        raise FileExistsError(
            f"Layout migration audit already exists: {AUDIT_PATH}"
        )
    if json.loads(POST_MOVE_MANIFEST.read_text(encoding="utf-8")).get(
        "status"
    ) != "verified":
        raise RuntimeError("Post-move repository manifest is not verified")

    candidates = candidate_json_files()
    before_hashes = {path: sha256_file(path) for path in candidates}
    for path in candidates:
        backup(path)
    immutable_before = {
        str(path): sha256_file(path)
        for root in ARCHIVE_ROOTS[:-1]
        for path in root.rglob("*")
        if path.is_file()
    }

    modified: set[Path] = set(update_data_yaml())
    modified.update(rebuild_prepared_manifests())
    modified.update(migrate_json_files())
    modified.add(update_segmenter_provenance())
    modified.add(update_detector_repair_audit())

    verify_immutable_archives(immutable_before)
    rows = []
    for path in sorted(modified):
        before = before_hashes.get(path)
        after = sha256_file(path)
        if before == after:
            continue
        relative = path.relative_to(REPO_ROOT)
        original_copy = (
            ORIGINALS_ROOT / relative
        ).with_suffix(path.suffix + ".original")
        rows.append(
            {
                "path": str(path),
                "before_sha256": before,
                "after_sha256": after,
                "bytes_after": path.stat().st_size,
                "original_copy": str(original_copy),
            }
        )

    audit = {
        "schema_version": 1,
        "migration_id": MIGRATION_ID,
        "status": "pass",
        "scope": "repository_layout_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "path_replacements": [
            {"old": old, "new": new}
            for old, new in PATH_REPLACEMENTS
        ],
        "pre_move_manifest": {
            "path": str(PRE_MOVE_MANIFEST),
            "sha256": sha256_file(PRE_MOVE_MANIFEST),
        },
        "post_move_manifest": {
            "path": str(POST_MOVE_MANIFEST),
            "sha256": sha256_file(POST_MOVE_MANIFEST),
        },
        "modified_file_count": len(rows),
        "immutable_archive_file_count": len(immutable_before),
        "immutable_archive_tree": immutable_before,
        "files": rows,
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(AUDIT_PATH, audit)
    print(AUDIT_PATH)
    print(f"modified_files={len(rows)}")


if __name__ == "__main__":
    main()
