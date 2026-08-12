from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
AUDIT_PATH = STUDY_ROOT / "docs" / "RUN_MANIFEST_MIGRATION_AUDIT.json"
MIGRATION_ID = "teacher_reference_bias_paper_unification_v1"

LEGACY_LAYOUTS = {
    "isaid_plane": ("teacher_reference_bias_v2_512", "isaid_plane"),
    "samrs_plane": ("teacher_reference_bias_v2_512", "samrs_sota_plane"),
    "isaid_small_vehicle": (
        "teacher_reference_bias_small_vehicle_v1_512",
        "isaid_small_vehicle",
    ),
    "samrs_small_vehicle": (
        "teacher_reference_bias_small_vehicle_v1_512",
        "samrs_sota_small_vehicle",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def path_replacements(experiment_id: str) -> tuple[tuple[str, str], ...]:
    legacy_study, legacy_dataset = LEGACY_LAYOUTS[experiment_id]
    old_root = REPO_ROOT / "studies" / legacy_study
    new_root = STUDY_ROOT / "experiments" / experiment_id
    replacements = (
        (
            str(old_root / "configs" / "datasets" / f"{legacy_dataset}.yaml"),
            str(new_root / "config.yaml"),
        ),
        (
            str(old_root / "configs" / "protocol.yaml"),
            str(STUDY_ROOT / "configs" / "protocol.yaml"),
        ),
        (
            str(old_root / "data" / "prepared" / legacy_dataset),
            str(new_root / "data" / "prepared"),
        ),
        (
            str(old_root / "results" / "detectors" / legacy_dataset),
            str(new_root / "results" / "detector"),
        ),
        (
            str(old_root / "results" / "predictions" / legacy_dataset),
            str(new_root / "results" / "predictions"),
        ),
        (
            str(old_root / "results" / "references" / legacy_dataset),
            str(new_root / "results" / "references"),
        ),
        (
            str(old_root / "results" / "evaluation" / legacy_dataset),
            str(new_root / "results" / "evaluation"),
        ),
        (
            str(old_root / "results" / "dataset_audits" / legacy_dataset),
            str(new_root / "results" / "audits"),
        ),
    )
    relative = tuple(
        (
            str(Path(old).relative_to(REPO_ROOT)),
            str(Path(new).relative_to(REPO_ROOT)),
        )
        for old, new in replacements
    )
    return tuple(sorted((*replacements, *relative), key=lambda item: -len(item[0])))


def rewrite_string(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    rewritten = value
    for old, new in replacements:
        if rewritten == old or rewritten.startswith(f"{old}/"):
            rewritten = f"{new}{rewritten[len(old):]}"
            break
    repository_prefix = f"{REPO_ROOT.resolve()}/"
    if rewritten.startswith(repository_prefix):
        rewritten = rewritten[len(repository_prefix) :]
    return rewritten


def rewrite_value(value: Any, replacements: tuple[tuple[str, str], ...]) -> Any:
    if isinstance(value, str):
        return rewrite_string(value, replacements)
    if isinstance(value, list):
        return [rewrite_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: rewrite_value(item, replacements)
            for key, item in value.items()
        }
    return value


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def fingerprint(path: Path, recorded_path: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": recorded_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def rewrite_companion_files(
    experiment_root: Path,
    replacements: tuple[tuple[str, str], ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidates = sorted(
        list(experiment_root.glob("results/predictions/**/effective_config.input.json"))
        + list(
            experiment_root.glob(
                "results/predictions/**/segmenter_provenance.input.json"
            )
        )
    )
    for path in candidates:
        before_hash = sha256_file(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rewritten = rewrite_value(payload, replacements)
        action = "validated_existing"
        if rewritten != payload:
            path.write_text(
                json.dumps(rewritten, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            action = "migrated"
        rows.append(
            {
                "path": portable(path),
                "action": action,
                "before_sha256": before_hash,
                "after_sha256": sha256_file(path),
            }
        )

    args_path = experiment_root / "results" / "detector" / "seed_42" / "train" / "args.yaml"
    if args_path.is_file():
        before_hash = sha256_file(args_path)
        payload = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        rewritten = rewrite_value(payload, replacements)
        action = "validated_existing"
        if rewritten != payload:
            args_path.write_text(
                yaml.safe_dump(rewritten, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            action = "migrated"
        rows.append(
            {
                "path": portable(args_path),
                "action": action,
                "before_sha256": before_hash,
                "after_sha256": sha256_file(args_path),
            }
        )
    return rows


def repair_run_manifests(
    experiment_root: Path,
    replacements: tuple[tuple[str, str], ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(experiment_root.glob("results/**/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("input_file_fingerprints_at_finish"), dict):
            continue
        original_hash = sha256_file(path)
        rewritten = rewrite_value(payload, replacements)
        fingerprint_changes: list[dict[str, object]] = []
        for group_name in (
            "input_file_fingerprints",
            "input_file_fingerprints_at_finish",
            "output_file_fingerprints",
        ):
            group = rewritten.get(group_name)
            if not isinstance(group, dict):
                continue
            for name, previous in list(group.items()):
                if not isinstance(previous, dict) or not isinstance(
                    previous.get("path"), str
                ):
                    raise ValueError(f"Invalid fingerprint {group_name}/{name}: {path}")
                recorded_path = rewrite_string(str(previous["path"]), replacements)
                current = fingerprint(resolve_recorded_path(recorded_path), recorded_path)
                if previous != current:
                    fingerprint_changes.append(
                        {
                            "group": group_name,
                            "name": name,
                            "previous": previous,
                            "current": current,
                        }
                    )
                group[name] = current

        prior_migration = rewritten.get("paper_study_layout_migration")
        if prior_migration is None:
            rewritten["paper_study_layout_migration"] = {
                "schema_version": 1,
                "migration_id": MIGRATION_ID,
                "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
                "reason": "repository_layout_only_no_model_rerun",
                "original_manifest_sha256": original_hash,
                "path_base": "repository_root",
                "fingerprint_refresh_count": len(fingerprint_changes),
            }
        rewritten["path_base"] = "repository_root"
        action = "validated_existing"
        if rewritten != payload:
            path.write_text(
                json.dumps(rewritten, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            action = "migrated_or_refreshed"
        rows.append(
            {
                "path": portable(path),
                "action": action,
                "before_sha256": original_hash,
                "after_sha256": sha256_file(path),
                "original_manifest_sha256": rewritten[
                    "paper_study_layout_migration"
                ]["original_manifest_sha256"],
                "fingerprint_refresh_count": len(fingerprint_changes),
            }
        )
    return rows


def main() -> None:
    companion_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for experiment_id in LEGACY_LAYOUTS:
        experiment_root = STUDY_ROOT / "experiments" / experiment_id
        replacements = path_replacements(experiment_id)
        companion_rows.extend(rewrite_companion_files(experiment_root, replacements))
        manifest_rows.extend(repair_run_manifests(experiment_root, replacements))

    audit = {
        "schema_version": 1,
        "status": "pass",
        "migration_id": MIGRATION_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "portable canonical run manifests and generated path metadata",
        "scientific_effect": "none; no prediction, reference, or metric values changed",
        "companion_file_count": len(companion_rows),
        "run_manifest_count": len(manifest_rows),
        "companion_files": companion_rows,
        "run_manifests": manifest_rows,
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(AUDIT_PATH)
    print(f"Companion files: {len(companion_rows)}")
    print(f"Run manifests: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
