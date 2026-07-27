from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
DEFAULT_STUDY_ROOT = (
    STUDY_ROOT / "results"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive iSAID predictions and metrics superseded by the "
            "lossless-reference migration."
        )
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_summary_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("summary_instance.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "source": path.relative_to(root).as_posix(),
                        **row,
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    study_root = args.study_root.resolve()
    archive_root = (
        study_root
        / "audits"
        / "pre_isaid_lossless_rle_metric_fix"
    )
    manifest_path = archive_root / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(
            f"Archive already exists and will not be overwritten: {manifest_path}"
        )

    source_roots = {
        "evaluation": study_root / "evaluation" / "isaid_plane",
        "predictions": study_root / "predictions" / "isaid_plane",
    }
    missing = [str(path) for path in source_roots.values() if not path.is_dir()]
    if missing:
        raise SystemExit(f"Missing source directories: {missing}")

    archive_root.mkdir(parents=True, exist_ok=False)
    files: list[dict[str, Any]] = []
    for label, source_root in source_roots.items():
        destination_root = archive_root / label
        shutil.copytree(source_root, destination_root)
        for source_path in sorted(source_root.rglob("*")):
            if not source_path.is_file():
                continue
            relative = source_path.relative_to(source_root)
            archive_path = destination_root / relative
            source_hash = sha256_file(source_path)
            archive_hash = sha256_file(archive_path)
            if source_hash != archive_hash:
                raise RuntimeError(
                    f"Archive copy hash mismatch: {source_path}"
                )
            files.append(
                {
                    "source_path": str(source_path),
                    "archive_path": str(archive_path),
                    "bytes": source_path.stat().st_size,
                    "sha256": source_hash,
                }
            )

    migration_path = study_root / "audits" / "isaid_lossless_rle_migration.json"
    if not migration_path.is_file():
        raise SystemExit(f"Missing RLE migration audit: {migration_path}")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    test_migration = next(
        row for row in migration["splits"] if row["split"] == "test"
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "superseded_invalid_for_scientific_results",
        "dataset_id": "isaid_plane",
        "reason": (
            "These metrics used contour-reconstructed COCO polygons rather "
            "than the exact raster masks. They are retained only as an audit "
            "trail and must not be reported as final results."
        ),
        "superseded_reference": {
            "test_coco_sha256": test_migration["before_sha256"],
            "replacement_test_coco_sha256": test_migration["after_sha256"],
        },
        "source_roots": {
            label: str(path) for label, path in source_roots.items()
        },
        "summary_rows": read_summary_rows(source_roots["evaluation"]),
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(f"Archived files: {len(files)}")


if __name__ == "__main__":
    main()
