#!/usr/bin/env python3
"""Repair dependency fingerprints written before layout migration settled."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
RESULTS_ROOT = STUDY_ROOT / "results"
MIGRATION_ROOT = (
    RESULTS_ROOT / "audits" / "repository_layout_migration"
)
LAYOUT_AUDIT_PATH = MIGRATION_ROOT / "layout_migration.json"
REPAIR_AUDIT_PATH = MIGRATION_ROOT / "dependency_repair.json"
REPAIR_ORIGINALS_ROOT = MIGRATION_ROOT / "dependency_repair_originals"
MIGRATION_ID = "study_layout_20260726"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def backup_intermediate(path: Path) -> Path:
    relative = path.relative_to(REPO_ROOT)
    destination = (
        REPAIR_ORIGINALS_ROOT / relative
    ).with_suffix(path.suffix + ".before_dependency_repair")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"Dependency-repair backup already exists: {destination}"
        )
    shutil.copy2(path, destination)
    return destination


def update_layout_row(
    layout_rows: dict[str, dict[str, Any]],
    *,
    path: Path,
    before_hash: str,
) -> None:
    row = layout_rows.get(str(path))
    if row is None:
        raise KeyError(f"Layout migration row missing for {path}")
    if row.get("after_sha256") != before_hash:
        raise RuntimeError(
            f"Layout migration intermediate hash mismatch for {path}"
        )
    row["dependency_repair_before_sha256"] = before_hash
    row["dependency_repair_after_sha256"] = sha256_file(path)
    row["after_sha256"] = sha256_file(path)
    row["bytes_after"] = path.stat().st_size


def repair_segmenter_provenance_dependents(
    *,
    layout_rows: dict[str, dict[str, Any]],
    provenance_path: Path,
) -> list[dict[str, Any]]:
    current_fingerprint = fingerprint(provenance_path)
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(
        RESULTS_ROOT.joinpath("predictions").rglob("manifest.json")
    ):
        payload = read_json(manifest_path)
        inputs = payload.get("inputs", {})
        if inputs.get("segmenter_provenance") != str(provenance_path):
            continue
        migration = payload.get("layout_migration")
        if (
            not isinstance(migration, dict)
            or migration.get("migration_id") != MIGRATION_ID
        ):
            raise RuntimeError(
                f"Run manifest has no valid layout migration: {manifest_path}"
            )
        start = payload.get("input_file_fingerprints", {})
        finish = payload.get("input_file_fingerprints_at_finish", {})
        old_start = deepcopy(start.get("segmenter_provenance"))
        old_finish = deepcopy(finish.get("segmenter_provenance"))
        if not isinstance(old_start, dict) or not isinstance(old_finish, dict):
            raise RuntimeError(
                f"Segmenter provenance fingerprint missing: {manifest_path}"
            )
        if old_start != old_finish:
            raise RuntimeError(
                f"Start/finish provenance fingerprints differ: {manifest_path}"
            )
        if old_start.get("sha256") == current_fingerprint["sha256"]:
            raise RuntimeError(
                f"Manifest does not need dependency repair: {manifest_path}"
            )

        before_hash = sha256_file(manifest_path)
        backup_path = backup_intermediate(manifest_path)
        repair = {
            "dependency": "segmenter_provenance",
            "reason": (
                "The segmenter provenance file received its final protocol "
                "path/hash after run manifests were fingerprinted during the "
                "same repository-only migration."
            ),
            "previous_fingerprint": old_start,
            "current_fingerprint": current_fingerprint,
        }
        migration.setdefault("dependency_repairs", []).append(repair)
        start["segmenter_provenance"] = deepcopy(current_fingerprint)
        finish["segmenter_provenance"] = deepcopy(current_fingerprint)
        write_json(manifest_path, payload)
        update_layout_row(
            layout_rows,
            path=manifest_path,
            before_hash=before_hash,
        )
        rows.append(
            {
                "path": str(manifest_path),
                "before_sha256": before_hash,
                "after_sha256": sha256_file(manifest_path),
                "intermediate_copy": str(backup_path),
                "intermediate_copy_sha256": sha256_file(backup_path),
                "dependency": "segmenter_provenance",
            }
        )
    if len(rows) != 24:
        raise RuntimeError(
            f"Expected 24 segmenter dependent manifests, found {len(rows)}"
        )
    return rows


def repair_protocol_bound_audit(
    path: Path,
    *,
    protocol_path: Path,
    layout_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = read_json(path)
    previous_protocol_hash = payload.get("protocol_sha256")
    current_protocol_hash = sha256_file(protocol_path)
    if previous_protocol_hash == current_protocol_hash:
        raise RuntimeError(f"Audit does not need protocol repair: {path}")
    before_hash = sha256_file(path)
    backup_path = backup_intermediate(path)
    payload["layout_migration"] = {
        "migration_id": MIGRATION_ID,
        "reason": "repository_layout_only",
        "dependency": "protocol",
        "previous_protocol_sha256": previous_protocol_hash,
        "current_protocol_path": str(protocol_path),
        "current_protocol_sha256": current_protocol_hash,
    }
    payload["protocol_sha256"] = current_protocol_hash
    write_json(path, payload)
    update_layout_row(layout_rows, path=path, before_hash=before_hash)
    return {
        "path": str(path),
        "before_sha256": before_hash,
        "after_sha256": sha256_file(path),
        "intermediate_copy": str(backup_path),
        "intermediate_copy_sha256": sha256_file(backup_path),
        "dependency": "protocol",
    }


def main() -> None:
    if REPAIR_AUDIT_PATH.exists():
        raise FileExistsError(
            f"Dependency repair audit already exists: {REPAIR_AUDIT_PATH}"
        )
    layout = read_json(LAYOUT_AUDIT_PATH)
    if (
        layout.get("status") != "pass"
        or layout.get("migration_id") != MIGRATION_ID
    ):
        raise RuntimeError("Layout migration audit is not valid")
    layout_rows = {
        str(row["path"]): row for row in layout.get("files", [])
    }
    protocol_path = STUDY_ROOT / "configs" / "protocol.yaml"
    provenance_path = (
        RESULTS_ROOT / "audits" / "segmenter_provenance.json"
    )

    repair_rows = repair_segmenter_provenance_dependents(
        layout_rows=layout_rows,
        provenance_path=provenance_path,
    )
    for audit_name in (
        "pinned_revision_prediction_parity.json",
        "pre_sam3_global_assignment_fix.json",
    ):
        repair_rows.append(
            repair_protocol_bound_audit(
                RESULTS_ROOT / "audits" / audit_name,
                protocol_path=protocol_path,
                layout_rows=layout_rows,
            )
        )

    repair_audit = {
        "schema_version": 1,
        "status": "pass",
        "migration_id": MIGRATION_ID,
        "scope": "layout_migration_dependency_order_only",
        "repaired_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": fingerprint(protocol_path),
        "segmenter_provenance": fingerprint(provenance_path),
        "modified_file_count": len(repair_rows),
        "files": repair_rows,
    }
    write_json(REPAIR_AUDIT_PATH, repair_audit)
    layout["dependency_repair"] = {
        "path": str(REPAIR_AUDIT_PATH),
        "sha256": sha256_file(REPAIR_AUDIT_PATH),
        "scope": repair_audit["scope"],
        "modified_file_count": len(repair_rows),
    }
    write_json(LAYOUT_AUDIT_PATH, layout)
    print(REPAIR_AUDIT_PATH)
    print(f"Repaired dependency fingerprints: {len(repair_rows)}")


if __name__ == "__main__":
    main()
