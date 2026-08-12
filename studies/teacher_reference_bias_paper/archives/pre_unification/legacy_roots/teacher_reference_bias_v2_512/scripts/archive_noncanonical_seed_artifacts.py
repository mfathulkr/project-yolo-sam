from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = STUDY_ROOT / "results"
ARCHIVE_ROOT = RESULTS_ROOT / "historical_noncanonical_seeds"
DATASETS = ("isaid_plane", "samrs_sota_plane")
MODELS = ("sam1", "sam2", "sam3")
SEEDS = (123, 2026)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requested_moves() -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for dataset_id in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                source = (
                    RESULTS_ROOT
                    / "predictions"
                    / dataset_id
                    / model
                    / "yolo_bbox"
                    / f"seed_{seed}"
                )
                target = (
                    ARCHIVE_ROOT
                    / "predictions"
                    / dataset_id
                    / model
                    / "yolo_bbox"
                    / f"seed_{seed}"
                )
                moves.append((source, target))
                evaluation_name = (
                    f"seed_{seed}_dual_reference"
                    if dataset_id == "isaid_plane"
                    else f"seed_{seed}"
                )
                source = (
                    RESULTS_ROOT
                    / "evaluation"
                    / dataset_id
                    / model
                    / "yolo_bbox"
                    / evaluation_name
                )
                target = (
                    ARCHIVE_ROOT
                    / "evaluation"
                    / dataset_id
                    / model
                    / "yolo_bbox"
                    / evaluation_name
                )
                moves.append((source, target))
    return moves


def main() -> None:
    moved: list[str] = []
    archived: list[str] = []
    for source, target in requested_moves():
        if source.exists() and target.exists():
            raise FileExistsError(f"Both active and archived paths exist: {source}, {target}")
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append(str(target.relative_to(STUDY_ROOT)))
        elif not target.exists():
            raise FileNotFoundError(f"Neither active nor archived artifact exists: {source}")
        archived.append(str(target.relative_to(STUDY_ROOT)))

    files = sorted(path for path in ARCHIVE_ROOT.rglob("*") if path.is_file())
    inventory = [
        {
            "path": str(path.relative_to(STUDY_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
        if path.name != "archive_manifest.json"
    ]
    payload = {
        "schema_version": 1,
        "status": "historical_noncanonical",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_detector_seed": 42,
        "archived_detector_seeds": list(SEEDS),
        "scientific_use": "excluded_from_all_canonical_analysis_and_reports",
        "sam1_sam2_validity": "historical_noncanonical",
        "sam3_validity": "invalid_legacy_pcs_interface_do_not_use",
        "archived_directories": archived,
        "moved_during_this_invocation": moved,
        "file_count": len(inventory),
        "files": inventory,
    }
    manifest_path = ARCHIVE_ROOT / "archive_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Archived {len(moved)} directories and indexed {len(inventory)} files")


if __name__ == "__main__":
    main()
