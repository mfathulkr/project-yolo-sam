#!/usr/bin/env python3
"""Safely migrate repository-owned data into the study-oriented layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_ID = "study_layout_20260726"
MANIFEST_DIR = REPO_ROOT / "docs" / "migration"
PRE_MANIFEST = MANIFEST_DIR / f"{MIGRATION_ID}_pre.json"
POST_MANIFEST = MANIFEST_DIR / f"{MIGRATION_ID}_post.json"
CURRENT_MANIFEST = MANIFEST_DIR / f"{MIGRATION_ID}_current.json"
REWRITE_AUDIT_ROOT = (
    REPO_ROOT
    / "studies"
    / "teacher_reference_bias_v1"
    / "results"
    / "audits"
    / "repository_layout_migration"
)
REWRITE_AUDIT = REWRITE_AUDIT_ROOT / "layout_migration.json"
SUPPLEMENTAL_REWRITE_AUDIT = (
    MANIFEST_DIR / f"{MIGRATION_ID}_supplemental_rewrites.json"
)


@dataclass(frozen=True)
class Move:
    source: str
    destination: str
    owner: str


@dataclass(frozen=True)
class TreeInventory:
    path: str
    file_count: int
    directory_count: int
    total_bytes: int
    sha256: str


def _fixed_moves() -> list[Move]:
    teacher = "studies/teacher_reference_bias_v1"
    isaid = "studies/isaid_vehicle_study"
    samrs = "studies/samrs_sota_plane_study"
    semantic = "studies/semantic_drone_car_study"
    landcover = "studies/landcover_building_study"
    return [
        Move("data/isaid_raw", "datasets/isaid/raw", "shared"),
        Move("data/isaid_raw_downloads", "datasets/isaid/downloads", "shared"),
        Move("data/samrs_raw", "datasets/samrs/raw", "shared"),
        Move(
            "data/matched/isaid_plane",
            f"{teacher}/data/prepared/isaid_plane",
            "teacher_reference_bias_v1",
        ),
        Move(
            "data/matched/samrs_sota_plane",
            f"{teacher}/data/prepared/samrs_sota_plane",
            "teacher_reference_bias_v1",
        ),
        Move(
            "artifacts/studies/teacher_reference_bias_v1",
            f"{teacher}/results",
            "teacher_reference_bias_v1",
        ),
        Move(
            "artifacts/audits/isaid_plane",
            f"{teacher}/results/audits/legacy_dataset_discovery/isaid_plane",
            "teacher_reference_bias_v1",
        ),
        Move(
            "artifacts/audits/local_samrs_claimed_sota",
            f"{teacher}/results/audits/legacy_dataset_discovery/"
            "local_samrs_claimed_sota",
            "teacher_reference_bias_v1",
        ),
        Move(
            "artifacts/legacy/pre_refactor_20260726",
            f"{teacher}/results/legacy/pre_refactor_20260726",
            "teacher_reference_bias_v1",
        ),
        Move(
            "paper_teacher_reference_bias",
            f"{teacher}/reports/paper",
            "teacher_reference_bias_v1",
        ),
        Move(
            "configs/studies/teacher_reference_bias_v1.yaml",
            f"{teacher}/configs/protocol.yaml",
            "teacher_reference_bias_v1",
        ),
        Move(
            "configs/datasets/isaid_plane.yaml",
            f"{teacher}/configs/datasets/isaid_plane.yaml",
            "teacher_reference_bias_v1",
        ),
        Move(
            "configs/datasets/samrs_sota_plane.yaml",
            f"{teacher}/configs/datasets/samrs_sota_plane.yaml",
            "teacher_reference_bias_v1",
        ),
        Move(
            "docs/MATCHED_EXPERIMENT_PLAN.md",
            f"{teacher}/docs/EXPERIMENT_PLAN.md",
            "teacher_reference_bias_v1",
        ),
        Move(
            "docs/LITERATURE_REVIEW.md",
            f"{teacher}/docs/LITERATURE_REVIEW.md",
            "teacher_reference_bias_v1",
        ),
        Move(
            "docs/REPRODUCIBILITY_APPENDIX.md",
            f"{teacher}/docs/REPRODUCIBILITY_APPENDIX.md",
            "teacher_reference_bias_v1",
        ),
        Move(
            "data/isaid_vehicle",
            f"{isaid}/data/prepared",
            "isaid_vehicle_study",
        ),
        Move(
            "presentation_isaid_vehicle_sam3_sam2_study",
            f"{isaid}/reports",
            "isaid_vehicle_study",
        ),
        Move(
            "configs/isaid_vehicle_yolo26x.yaml",
            f"{isaid}/configs/yolo26x.yaml",
            "isaid_vehicle_study",
        ),
        Move(
            "configs/isaid_vehicle_yolo26x_cpu_eval.yaml",
            f"{isaid}/configs/yolo26x_cpu_eval.yaml",
            "isaid_vehicle_study",
        ),
        Move(
            "docs/ISAID_VEHICLE_STUDY_WALKTHROUGH.md",
            f"{isaid}/docs/WALKTHROUGH.md",
            "isaid_vehicle_study",
        ),
        Move(
            "runs/yolo26x_isaid_vehicle_s1024",
            f"{isaid}/results/detector_training/yolo26x_isaid_vehicle_s1024",
            "isaid_vehicle_study",
        ),
        Move(
            "data/samrs_sota_plane",
            f"{samrs}/data/prepared",
            "samrs_sota_plane_study",
        ),
        Move(
            "presentation_samrs_sota_plane_study",
            f"{samrs}/reports",
            "samrs_sota_plane_study",
        ),
        Move(
            "configs/samrs_sota_plane_yolo26x.yaml",
            f"{samrs}/configs/yolo26x.yaml",
            "samrs_sota_plane_study",
        ),
        Move(
            "runs/yolo26x_samrs_sota_plane_s1024",
            f"{samrs}/results/detector_training/"
            "yolo26x_samrs_sota_plane_s1024",
            "samrs_sota_plane_study",
        ),
        Move(
            "configs/semantic_drone_car_yolo26x.yaml",
            f"{semantic}/configs/yolo26x.yaml",
            "semantic_drone_car_study",
        ),
        Move(
            "CODEX_HANDOFF_SEMANTIC_DRONE.md",
            f"{semantic}/docs/HANDOFF.md",
            "semantic_drone_car_study",
        ),
        Move(
            "configs/experiment.yaml",
            f"{landcover}/configs/yolo_sam3.yaml",
            "landcover_building_study",
        ),
        Move("yolo26n.pt", "models/yolo/yolo26n.pt", "shared"),
        Move("yolo26x.pt", "models/yolo/yolo26x.pt", "shared"),
    ]


def _result_moves() -> list[Move]:
    moves: list[Move] = []
    rules = (
        (
            "isaid_vehicle_",
            "studies/isaid_vehicle_study/results/pipelines",
            "isaid_vehicle_study",
        ),
        (
            "samrs_sota_plane_",
            "studies/samrs_sota_plane_study/results/pipelines",
            "samrs_sota_plane_study",
        ),
    )
    results_root = REPO_ROOT / "results"
    if not results_root.is_dir():
        return moves
    for entry in sorted(results_root.iterdir()):
        for prefix, destination_root, owner in rules:
            if entry.name.startswith(prefix):
                suffix = entry.name[len(prefix) :]
                moves.append(
                    Move(
                        str(entry.relative_to(REPO_ROOT)),
                        f"{destination_root}/{suffix}",
                        owner,
                    )
                )
                break
    return moves


def planned_moves() -> list[Move]:
    return [*_fixed_moves(), *_result_moves()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(
    path: Path,
    *,
    excluded_paths: tuple[Path, ...] = (),
    content_overrides: dict[Path, Path] | None = None,
) -> TreeInventory:
    if not path.exists():
        raise FileNotFoundError(path)
    overrides = content_overrides or {}
    digest = hashlib.sha256()
    file_count = 0
    directory_count = 0
    total_bytes = 0

    if path.is_file():
        content_path = overrides.get(path, path)
        stat = content_path.stat()
        content_hash = sha256_file(content_path)
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(content_hash.encode("ascii"))
        return TreeInventory(
            path=str(path),
            file_count=1,
            directory_count=0,
            total_bytes=stat.st_size,
            sha256=digest.hexdigest(),
        )

    for current_root, directory_names, file_names in os.walk(path):
        directory_names.sort()
        file_names.sort()
        current = Path(current_root)
        directory_count += len(directory_names)
        for name in file_names:
            file_path = current / name
            if any(
                file_path == excluded
                or file_path.is_relative_to(excluded)
                for excluded in excluded_paths
            ):
                continue
            relative = file_path.relative_to(path).as_posix()
            content_path = overrides.get(file_path, file_path)
            stat = content_path.stat()
            content_hash = sha256_file(content_path)
            file_count += 1
            total_bytes += stat.st_size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(content_hash.encode("ascii"))
            digest.update(b"\n")

    return TreeInventory(
        path=str(path),
        file_count=file_count,
        directory_count=directory_count,
        total_bytes=total_bytes,
        sha256=digest.hexdigest(),
    )


def _audited_rewrite_overrides() -> dict[Path, Path]:
    audit_paths = (
        path
        for path in (REWRITE_AUDIT, SUPPLEMENTAL_REWRITE_AUDIT)
        if path.is_file()
    )
    overrides: dict[Path, Path] = {}
    for audit_path in audit_paths:
        overrides.update(_audited_rewrite_overrides_from(audit_path))
    return overrides


def _audited_rewrite_overrides_from(audit_path: Path) -> dict[Path, Path]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("migration_id") != MIGRATION_ID or audit.get("status") != "pass":
        raise RuntimeError(f"Invalid metadata rewrite audit: {audit_path}")
    records = audit.get("files")
    if not isinstance(records, list):
        raise RuntimeError(f"Invalid metadata rewrite records: {audit_path}")

    overrides: dict[Path, Path] = {}
    for record in records:
        current = Path(str(record["path"]))
        original = Path(str(record["original_copy"]))
        if not current.is_absolute():
            current = REPO_ROOT / current
        if not original.is_absolute():
            original = REPO_ROOT / original
        if not current.is_file():
            raise FileNotFoundError(current)
        if not original.is_file():
            raise FileNotFoundError(original)
        current_hash = sha256_file(current)
        original_hash = sha256_file(original)
        if current_hash != str(record["after_sha256"]):
            raise RuntimeError(
                f"Audited relocated metadata changed after rewrite: {current}"
            )
        if original_hash != str(record["before_sha256"]):
            raise RuntimeError(
                f"Archived pre-rewrite metadata changed: {original}"
            )
        if current.stat().st_size != int(record["bytes_after"]):
            raise RuntimeError(
                f"Audited relocated metadata size changed: {current}"
            )
        overrides[current] = original

    if len(overrides) != int(audit.get("modified_file_count", -1)):
        raise RuntimeError(
            f"Metadata rewrite audit count mismatch: {audit_path}"
        )
    return overrides


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_pre_manifest() -> dict[str, object]:
    if not PRE_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Pre-migration manifest is missing: {PRE_MANIFEST}"
        )
    return json.loads(PRE_MANIFEST.read_text(encoding="utf-8"))


def _load_historical_post_records() -> dict[str, dict[str, object]]:
    if not POST_MANIFEST.is_file():
        return {}
    manifest = json.loads(POST_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("migration_id") != MIGRATION_ID:
        raise RuntimeError(f"Wrong migration ID in {POST_MANIFEST}")
    if manifest.get("status") != "verified":
        raise RuntimeError(f"Historical post manifest is not verified: {POST_MANIFEST}")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise RuntimeError(f"Invalid historical post manifest: {POST_MANIFEST}")
    return {
        str(record["move"]["destination"]): record
        for record in records
    }


def _comparable_inventory(
    inventory_payload: dict[str, object],
    *,
    include_directory_count: bool,
) -> dict[str, object]:
    keys = ["file_count", "total_bytes", "sha256"]
    if include_directory_count:
        keys.append("directory_count")
    return {key: inventory_payload[key] for key in keys}


def _is_mutable_output_destination(destination: Path) -> bool:
    relative = destination.relative_to(REPO_ROOT)
    return (
        any(part in {"results", "reports"} for part in relative.parts)
        or relative.as_posix()
        == (
            "studies/teacher_reference_bias_v1/docs/"
            "REPRODUCIBILITY_APPENDIX.md"
        )
    )


def _active_moves(moves: Iterable[Move]) -> list[Move]:
    active: list[Move] = []
    for move in moves:
        source = REPO_ROOT / move.source
        destination = REPO_ROOT / move.destination
        if source.exists() and destination.exists():
            raise FileExistsError(
                f"Both source and destination exist: {source} -> {destination}"
            )
        if not source.exists() and not destination.exists():
            raise FileNotFoundError(
                f"Neither source nor destination exists: {source} -> {destination}"
            )
        active.append(move)
    return active


def preflight() -> None:
    moves = _active_moves(planned_moves())
    records: list[dict[str, object]] = []
    for index, move in enumerate(moves, start=1):
        source = REPO_ROOT / move.source
        destination = REPO_ROOT / move.destination
        measured_path = source if source.exists() else destination
        print(f"[{index}/{len(moves)}] inventory {measured_path}")
        records.append(
            {
                "move": asdict(move),
                "inventory": asdict(inventory(measured_path)),
                "state": "source" if source.exists() else "destination",
            }
        )
    payload: dict[str, object] = {
        "migration_id": MIGRATION_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPO_ROOT),
        "record_count": len(records),
        "records": records,
    }
    _write_json(PRE_MANIFEST, payload)
    print(f"Wrote {PRE_MANIFEST}")


def apply_moves() -> None:
    manifest = _load_pre_manifest()
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Invalid pre-migration manifest")
    for index, record in enumerate(records, start=1):
        move_data = record["move"]
        source = REPO_ROOT / str(move_data["source"])
        destination = REPO_ROOT / str(move_data["destination"])
        if destination.exists() and not source.exists():
            print(f"[{index}/{len(records)}] already moved {destination}")
            continue
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(records)}] move {source} -> {destination}")
        source.rename(destination)
    print("All planned paths were moved.")


def verify() -> None:
    manifest = _load_pre_manifest()
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Invalid pre-migration manifest")
    verified: list[dict[str, object]] = []
    rewrite_overrides = _audited_rewrite_overrides()
    historical_post = _load_historical_post_records()
    used_historical_mutable_proof = False
    all_destinations = [
        REPO_ROOT / str(record["move"]["destination"])
        for record in records
    ]
    for index, record in enumerate(records, start=1):
        move_data = record["move"]
        expected = record["inventory"]
        source = REPO_ROOT / str(move_data["source"])
        destination = REPO_ROOT / str(move_data["destination"])
        if source.exists():
            raise RuntimeError(f"Source still exists after migration: {source}")
        if not destination.exists():
            raise FileNotFoundError(destination)
        print(f"[{index}/{len(records)}] verify {destination}")
        nested_destinations = tuple(
            candidate
            for candidate in all_destinations
            if candidate != destination and candidate.is_relative_to(destination)
        )
        excluded_paths = nested_destinations
        if REWRITE_AUDIT_ROOT.is_relative_to(destination):
            excluded_paths = (*excluded_paths, REWRITE_AUDIT_ROOT)
        actual = inventory(
            destination,
            excluded_paths=excluded_paths,
            content_overrides={
                current: original
                for current, original in rewrite_overrides.items()
                if current == destination or current.is_relative_to(destination)
            },
        )
        include_directory_count = not nested_destinations
        comparable = _comparable_inventory(
            asdict(actual),
            include_directory_count=include_directory_count,
        )
        expected_comparable = _comparable_inventory(
            expected,
            include_directory_count=include_directory_count,
        )
        verification_status = "verified_current"
        if comparable != expected_comparable:
            if not _is_mutable_output_destination(destination):
                raise RuntimeError(
                    f"Inventory mismatch for {destination}: "
                    f"expected={expected_comparable}, actual={comparable}"
                )
            historical_record = historical_post.get(str(move_data["destination"]))
            if historical_record is None:
                raise RuntimeError(
                    "Mutable output differs from the pre-migration inventory "
                    f"without a historical verified post record: {destination}"
                )
            historical_comparable = _comparable_inventory(
                historical_record["inventory"],
                include_directory_count=include_directory_count,
            )
            if historical_comparable != expected_comparable:
                raise RuntimeError(
                    "Historical move proof does not match the pre-migration "
                    f"inventory for {destination}"
                )
            verification_status = "historically_verified_then_regenerated"
            used_historical_mutable_proof = True
            print(
                "  mutable study output changed after its historically verified "
                "move; owner-level validators are required"
            )
        verified.append(
            {
                "move": move_data,
                "comparison_inventory": asdict(actual),
                "expected_pre_move_inventory": expected,
                "status": verification_status,
            }
        )
    payload: dict[str, object] = {
        "migration_id": MIGRATION_ID,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPO_ROOT),
        "record_count": len(verified),
        "records": verified,
        "status": (
            "verified_with_regenerated_mutable_outputs"
            if used_historical_mutable_proof
            else "verified"
        ),
        "historical_post_manifest": str(POST_MANIFEST),
        "metadata_rewrite_audit": (
            str(REWRITE_AUDIT) if rewrite_overrides else None
        ),
    }
    output_path = CURRENT_MANIFEST if POST_MANIFEST.exists() else POST_MANIFEST
    _write_json(output_path, payload)
    print(f"Wrote {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "apply", "verify"),
        help="Create the inventory, perform moves, or verify moved content.",
    )
    return parser


def main() -> None:
    command = build_parser().parse_args().command
    if command == "preflight":
        preflight()
    elif command == "apply":
        apply_moves()
    else:
        verify()


if __name__ == "__main__":
    main()
