from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
PLANE_STUDY = REPO_ROOT / "studies" / "teacher_reference_bias_v2_512"
ARCHIVE_ROOT = PLANE_STUDY / "results" / "historical_noncanonical_seeds"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invalid_manifests() -> list[Path]:
    manifests: list[Path] = []
    for dataset_id in ("isaid_plane", "samrs_sota_plane"):
        for seed in (123, 2026):
            manifests.append(
                ARCHIVE_ROOT
                / "predictions"
                / dataset_id
                / "sam3"
                / "yolo_bbox"
                / f"seed_{seed}"
                / "manifest.json"
            )
            evaluation_suffix = (
                f"seed_{seed}_dual_reference"
                if dataset_id == "isaid_plane"
                else f"seed_{seed}"
            )
            manifests.append(
                ARCHIVE_ROOT
                / "evaluation"
                / dataset_id
                / "sam3"
                / "yolo_bbox"
                / evaluation_suffix
                / "manifest.json"
            )
    return manifests


def main() -> None:
    records = []
    for manifest_path in invalid_manifests():
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "superseded_invalid_for_scientific_results"
        manifest["invalidation"] = {
            "reason": (
                "SAM3 bbox inference used PCS visual-exemplar semantics and a "
                "0.5 output filter instead of one-instance-per-box PVS inference."
            ),
            "canonical_replacement": "seed_42 Sam3Tracker PVS artifacts",
            "must_not_enter_analysis": True,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        records.append(
            {
                "manifest": str(manifest_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(manifest_path),
            }
        )

    index_path = ARCHIVE_ROOT / "invalid_sam3_pcs_manifest.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scientific_role": "historical_invalid_artifact_index",
                "reason": "Noncanonical SAM3 PCS artifacts are excluded from all analyses.",
                "artifact_count": len(records),
                "artifacts": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(index_path)


if __name__ == "__main__":
    main()
