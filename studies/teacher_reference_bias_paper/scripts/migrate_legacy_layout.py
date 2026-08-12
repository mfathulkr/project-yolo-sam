from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
V2 = REPO_ROOT / "studies" / "teacher_reference_bias_v2_512"
SMALL = REPO_ROOT / "studies" / "teacher_reference_bias_small_vehicle_v1_512"
MULTI = REPO_ROOT / "studies" / "teacher_reference_bias_multiteacher_v1_512"
EXPERIMENTS = STUDY_ROOT / "experiments"
ARCHIVE = STUDY_ROOT / "archives" / "pre_unification"


@dataclass(frozen=True)
class Move:
    source: Path
    destination: Path
    role: str


def moves() -> list[Move]:
    rows = [
        Move(V2 / "data/prepared/isaid_plane", EXPERIMENTS / "isaid_plane/data/prepared", "prepared_data"),
        Move(V2 / "data/prepared/samrs_sota_plane", EXPERIMENTS / "samrs_plane/data/prepared", "prepared_data"),
        Move(SMALL / "data/prepared/isaid_small_vehicle", EXPERIMENTS / "isaid_small_vehicle/data/prepared", "prepared_data"),
        Move(SMALL / "data/prepared/samrs_sota_small_vehicle", EXPERIMENTS / "samrs_small_vehicle/data/prepared", "prepared_data"),
        Move(SMALL / "data/master/isaid_small_vehicle", EXPERIMENTS / "isaid_small_vehicle/data/master", "master_data"),
        Move(SMALL / "data/master/samrs_sota_small_vehicle", EXPERIMENTS / "samrs_small_vehicle/data/master", "master_data"),
    ]
    source_studies = {
        "isaid_plane": (V2, "isaid_plane"),
        "samrs_plane": (V2, "samrs_sota_plane"),
        "isaid_small_vehicle": (SMALL, "isaid_small_vehicle"),
        "samrs_small_vehicle": (SMALL, "samrs_sota_small_vehicle"),
    }
    for experiment_id, (legacy_root, dataset_id) in source_studies.items():
        experiment_root = EXPERIMENTS / experiment_id
        rows.extend(
            [
                Move(
                    legacy_root / "results/detectors" / dataset_id / "seed_42",
                    experiment_root / "results/detector/seed_42",
                    "canonical_detector_seed_42",
                ),
                Move(
                    legacy_root / "results/predictions" / dataset_id,
                    experiment_root / "results/predictions",
                    "frozen_predictions",
                ),
                Move(
                    legacy_root / "results/evaluation" / dataset_id,
                    experiment_root / "archives/pre_unification/evaluation",
                    "legacy_evaluation",
                ),
                Move(
                    legacy_root / "results/dataset_audits" / dataset_id,
                    experiment_root / "results/audits/raw_dataset",
                    "dataset_audit",
                ),
            ]
        )

    rows.extend(
        [
            Move(V2 / "results/references/isaid_plane", EXPERIMENTS / "isaid_plane/archives/pre_unification/references", "legacy_reference"),
            Move(SMALL / "results/references/isaid_small_vehicle", EXPERIMENTS / "isaid_small_vehicle/archives/pre_unification/references", "legacy_reference"),
            Move(V2 / "reports/full_metrics/isaid_plane_human", EXPERIMENTS / "isaid_plane/reports/full_metrics/human", "full_metric_report"),
            Move(V2 / "reports/full_metrics/isaid_plane_pseudo_sam1", EXPERIMENTS / "isaid_plane/reports/full_metrics/pseudo_sam1", "full_metric_report"),
            Move(V2 / "reports/full_metrics/samrs_sota_plane", EXPERIMENTS / "samrs_plane/reports/full_metrics/published_samrs_reference", "full_metric_report"),
            Move(SMALL / "reports/full_metrics/isaid_small_vehicle_human", EXPERIMENTS / "isaid_small_vehicle/reports/full_metrics/human", "full_metric_report"),
            Move(SMALL / "reports/full_metrics/isaid_small_vehicle_pseudo_sam1", EXPERIMENTS / "isaid_small_vehicle/reports/full_metrics/pseudo_sam1", "full_metric_report"),
            Move(SMALL / "reports/full_metrics/samrs_sota_small_vehicle", EXPERIMENTS / "samrs_small_vehicle/reports/full_metrics/published_samrs_reference", "full_metric_report"),
        ]
    )
    for dataset_id, experiment_id in (
        ("isaid_plane", "isaid_plane"),
        ("isaid_small_vehicle", "isaid_small_vehicle"),
    ):
        rows.extend(
            [
                Move(
                    MULTI / "results/references" / dataset_id,
                    EXPERIMENTS / experiment_id / "archives/pre_unification/multiteacher_references",
                    "legacy_reference",
                ),
                Move(
                    MULTI / "results/evaluation" / dataset_id,
                    EXPERIMENTS / experiment_id / "archives/pre_unification/multiteacher_evaluation",
                    "legacy_evaluation",
                ),
            ]
        )
        for teacher in ("sam2", "sam3"):
            slug = f"{dataset_id}_pseudo_{teacher}"
            rows.append(
                Move(
                    MULTI / "reports/full_metrics" / slug,
                    EXPERIMENTS / experiment_id / "reports/full_metrics" / f"pseudo_{teacher}",
                    "full_metric_report",
                )
            )

    rows.extend(
        [
            Move(V2 / "results/analysis", ARCHIVE / "plane_pair/results/analysis", "legacy_combined_analysis"),
            Move(V2 / "results/figures", ARCHIVE / "plane_pair/results/figures", "legacy_combined_figures"),
            Move(V2 / "results/historical_noncanonical_seeds", ARCHIVE / "plane_pair/results/historical_noncanonical_seeds", "noncanonical_seed_archive"),
            Move(V2 / "results/detectors/isaid_plane/seed_123", ARCHIVE / "plane_pair/detectors/isaid_plane/seed_123", "noncanonical_detector_seed"),
            Move(V2 / "results/detectors/isaid_plane/seed_2026", ARCHIVE / "plane_pair/detectors/isaid_plane/seed_2026", "noncanonical_detector_seed"),
            Move(V2 / "results/detectors/samrs_sota_plane/seed_123", ARCHIVE / "plane_pair/detectors/samrs_sota_plane/seed_123", "noncanonical_detector_seed"),
            Move(V2 / "results/detectors/samrs_sota_plane/seed_2026", ARCHIVE / "plane_pair/detectors/samrs_sota_plane/seed_2026", "noncanonical_detector_seed"),
            Move(V2 / "results/audits", ARCHIVE / "plane_pair/results/audits", "legacy_audits"),
            Move(V2 / "results/smoke", ARCHIVE / "plane_pair/results/smoke", "legacy_smoke"),
            Move(V2 / "results/preflight.json", ARCHIVE / "plane_pair/results/preflight.json", "legacy_preflight"),
            Move(V2 / "bundles", ARCHIVE / "plane_pair/bundles", "legacy_bundle"),
            Move(SMALL / "results/analysis", ARCHIVE / "small_vehicle_pair/results/analysis", "legacy_combined_analysis"),
            Move(SMALL / "results/figures", ARCHIVE / "small_vehicle_pair/results/figures", "legacy_combined_figures"),
            Move(SMALL / "results/audits", ARCHIVE / "small_vehicle_pair/results/audits", "legacy_audits"),
            Move(SMALL / "results/smoke", ARCHIVE / "small_vehicle_pair/results/smoke", "legacy_smoke"),
            Move(SMALL / "results/post_training", ARCHIVE / "small_vehicle_pair/results/post_training", "legacy_worker_state"),
            Move(SMALL / "results/finalization", ARCHIVE / "small_vehicle_pair/results/finalization", "legacy_worker_state"),
            Move(SMALL / "results/preflight.json", ARCHIVE / "small_vehicle_pair/results/preflight.json", "legacy_preflight"),
            Move(SMALL / "bundles", ARCHIVE / "small_vehicle_pair/bundles", "legacy_bundle"),
            Move(MULTI / "results/analysis", ARCHIVE / "multiteacher/results/analysis", "legacy_combined_analysis"),
            Move(MULTI / "results/figures", ARCHIVE / "multiteacher/results/figures", "legacy_combined_figures"),
            Move(MULTI / "reports/teacher_comparison", ARCHIVE / "multiteacher/reports/teacher_comparison", "legacy_comparison_report"),
            Move(MULTI / "paper/assets", STUDY_ROOT / "paper_writing/assets", "paper_assets"),
            Move(MULTI / "paper/overleaf", STUDY_ROOT / "paper_writing/overleaf", "overleaf_source"),
        ]
    )
    return rows


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> tuple[dict[str, object], dict[str, tuple[int, int]]]:
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    root = path.parent if path.is_file() else path
    digest = hashlib.sha256()
    identities: dict[str, tuple[int, int]] = {}
    total_bytes = 0
    for index, file_path in enumerate(files, 1):
        relative = file_path.relative_to(root).as_posix()
        stat = file_path.stat()
        content_hash = digest_file(file_path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
        identities[relative] = (stat.st_ino, stat.st_size)
        total_bytes += stat.st_size
        if index % 1000 == 0:
            print(f"HASH {path}: {index}/{len(files)}", flush=True)
    return (
        {
            "file_count": len(files),
            "total_bytes": total_bytes,
            "tree_sha256": digest.hexdigest(),
        },
        identities,
    )


def verify_identity(path: Path, identities: dict[str, tuple[int, int]]) -> None:
    root = path.parent if path.is_file() else path
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    actual = {}
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        stat = file_path.stat()
        actual[relative] = (stat.st_ino, stat.st_size)
    if actual != identities:
        raise RuntimeError(f"Taşıma sonrası dosya kimliği uyuşmuyor: {path}")


def portable(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def main() -> None:
    records: list[dict[str, object]] = []
    for row in moves():
        if not row.source.exists():
            if row.destination.exists():
                print(f"SKIP already migrated: {row.destination}", flush=True)
                continue
            raise FileNotFoundError(row.source)
        if row.destination.exists():
            raise FileExistsError(row.destination)
        print(f"FINGERPRINT {row.source}", flush=True)
        summary, identities = fingerprint(row.source)
        row.destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(row.source, row.destination)
        verify_identity(row.destination, identities)
        records.append(
            {
                "source": portable(row.source),
                "destination": portable(row.destination),
                "role": row.role,
                **summary,
                "verification": "same_inode_and_size_after_atomic_rename",
            }
        )
        print(f"MOVED {row.destination}", flush=True)

    output = STUDY_ROOT / "docs" / "MIGRATION_MANIFEST.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "method": "sha256_before_atomic_rename_then_inode_and_size_verification",
                "moves": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
