from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias.reporting.analysis import sha256_file
from yolo_sam.data.prepared_validation import (
    validate_detector_training_content_manifest,
    validate_prepared_content_manifest,
)
from yolo_sam.runtime.manifest import (
    declared_file_fingerprints,
    environment_snapshot,
)
from teacher_reference_bias.config import (
    DatasetStudyConfig,
    MatchedStudyConfig,
    load_dataset_study_config,
    load_matched_study_config,
)


MODELS = ("sam1", "sam2", "sam3")
STRATA = (
    "overall",
    "no_overlap__low_mask_area",
    "no_overlap__high_mask_area",
    "overlap__low_mask_area",
    "overlap__high_mask_area",
)
MASK_METRICS = (
    "iou",
    "dice",
    "precision",
    "recall",
    "boundary_iou",
)
DEFAULT_DATASETS = (
    Path("studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml"),
    Path("studies/teacher_reference_bias_v1/configs/datasets/samrs_sota_plane.yaml"),
)
ANALYSIS_FILES = (
    "canonical_instance_metrics.csv",
    "aggregate_metrics.csv",
    "paired_model_comparisons.csv",
    "reference_inflation.csv",
    "ranking_comparisons.csv",
    "detector_metrics_by_seed.csv",
    "detector_seed_summary.csv",
    "segmentation_seed_summary.csv",
    "prediction_status_audit.csv",
    "training_health_audit.csv",
    "manifest.json",
)
FIGURE_FILES = (
    "gt_bbox_reference_comparison.png",
    "isaid_reference_inflation.png",
    "shared_human_reference_comparison.png",
    "gt_bbox_strata_heatmap.png",
    "detector_seed_metrics.png",
    "isaid_plane_gt_bbox_qualitative.png",
    "samrs_sota_plane_gt_bbox_qualitative.png",
)
PAPER_FILES = (
    "teacher_reference_bias_paper.md",
    "teacher_reference_bias_paper.docx",
    "teacher_reference_bias_paper_6pages.pdf",
    "paper_manifest.json",
)
FULL_METRIC_REPORTS = {
    "isaid_plane": {
        "slug": "isaid_plane",
        "min_pdf_pages": 18,
        "table_count": 12,
        "reference_types": ("human", "pseudo_sam1"),
    },
    "samrs_sota_plane": {
        "slug": "samrs_sota_plane",
        "min_pdf_pages": 13,
        "table_count": 7,
        "reference_types": ("pseudo_sam1",),
    },
}
SUPERSEDED_ISAID_METRIC_ARCHIVE = (
    Path("audits") / "pre_isaid_lossless_rle_metric_fix"
)
LEGACY_DETECTOR_MANIFEST_ARCHIVE = (
    Path("audits") / "legacy_detector_manifest_repair" / "originals"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final completeness, provenance, and artifact QA."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("studies/teacher_reference_bias_v1/configs/protocol.yaml"),
    )
    parser.add_argument("--dataset", type=Path, action="append")
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("studies/teacher_reference_bias_v1/reports/paper"),
    )
    parser.add_argument(
        "--appendix",
        type=Path,
        default=Path("studies/teacher_reference_bias_v1/docs/REPRODUCIBILITY_APPENDIX.md"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def normalize_layout_path_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    replacements = (
        (
            str(ROOT / "artifacts" / "studies" / "teacher_reference_bias_v1"),
            str(STUDY_ROOT / "results"),
        ),
        (
            str(ROOT / "data" / "matched" / "isaid_plane"),
            str(STUDY_ROOT / "data" / "prepared" / "isaid_plane"),
        ),
        (
            str(ROOT / "data" / "matched" / "samrs_sota_plane"),
            str(STUDY_ROOT / "data" / "prepared" / "samrs_sota_plane"),
        ),
        (
            str(ROOT / "yolo26x.pt"),
            str(ROOT / "models" / "yolo" / "yolo26x.pt"),
        ),
        (
            str(ROOT / "yolo26n.pt"),
            str(ROOT / "models" / "yolo" / "yolo26n.pt"),
        ),
    )
    normalized = value
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    if normalized == "yolo26x.pt":
        return "models/yolo/yolo26x.pt"
    if normalized == "yolo26n.pt":
        return "models/yolo/yolo26n.pt"
    return normalized


def migrated_archive_path(path: Path) -> Path:
    return Path(str(normalize_layout_path_text(str(path))))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configuration_tree_sha256(
    root: Path,
    rows: list[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for expected in rows:
        path = root / str(expected["path"])
        if not path.is_file():
            return ""
        actual = {
            "path": str(expected["path"]),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        digest.update(
            json.dumps(
                actual,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def require_file(
    path: Path,
    *,
    label: str,
    required: list[Path],
    errors: list[str],
    allow_empty: bool = False,
) -> None:
    if not path.is_file() or (path.stat().st_size == 0 and not allow_empty):
        errors.append(f"{label} eksik veya boş: {path}")
        return
    required.append(path)


def require_completed_manifest(
    path: Path,
    *,
    label: str,
    required: list[Path],
    errors: list[str],
) -> dict[str, Any] | None:
    require_file(path, label=label, required=required, errors=errors)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    payload = read_json(path)
    if payload.get("status") != "completed":
        errors.append(
            f"{label} completed değil: {path} "
            f"(status={payload.get('status')!r})"
        )
    return payload


def validate_unit_interval_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
    errors: list[str],
) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        errors.append(f"{label} metrik kolonları eksik: {sorted(missing)}")
        return
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        errors.append(f"{label} sonlu olmayan metrik içeriyor")
    if ((numeric < 0.0) | (numeric > 1.0)).any().any():
        errors.append(f"{label} [0,1] dışında metrik içeriyor")


def validate_canonical_analysis_content(
    study_root: Path,
    protocol: MatchedStudyConfig,
    datasets: list[DatasetStudyConfig],
    *,
    errors: list[str],
) -> None:
    analysis_root = study_root / "analysis"
    canonical_path = analysis_root / "canonical_instance_metrics.csv"
    aggregate_path = analysis_root / "aggregate_metrics.csv"
    detector_path = analysis_root / "detector_metrics_by_seed.csv"
    if not all(
        path.is_file() and path.stat().st_size
        for path in (canonical_path, aggregate_path, detector_path)
    ):
        return

    canonical = pd.read_csv(canonical_path)
    condition_columns = (
        "dataset_id",
        "model",
        "bbox_source",
        "detector_seed",
        "reference_type",
    )
    missing = set(condition_columns) - set(canonical.columns)
    if missing:
        errors.append(
            "Canonical instance metrics koşul kolonları eksik: "
            f"{sorted(missing)}"
        )
        return
    validate_unit_interval_columns(
        canonical,
        MASK_METRICS,
        label="Canonical instance metrics",
        errors=errors,
    )
    pixel_columns = (
        "true_positive_pixels",
        "false_positive_pixels",
        "false_negative_pixels",
    )
    missing_pixels = set(pixel_columns) - set(canonical.columns)
    if missing_pixels:
        errors.append(
            "Canonical instance metrics piksel kolonları eksik: "
            f"{sorted(missing_pixels)}"
        )
    else:
        pixels = canonical[list(pixel_columns)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if (
            pixels.isna().any().any()
            or not np.isfinite(pixels.to_numpy(dtype=float)).all()
            or (pixels < 0).any().any()
        ):
            errors.append(
                "Canonical instance metrics geçersiz piksel sayısı içeriyor"
            )

    expected_counts: dict[tuple[str, str, str, int | None, str], int] = {}
    for dataset in datasets:
        coco_path = dataset.prepared_root / "test" / "_annotations.coco.json"
        if not coco_path.is_file():
            errors.append(
                f"{dataset.dataset_id} test COCO dosyası bulunamadı: {coco_path}"
            )
            continue
        instance_count = len(read_json(coco_path).get("annotations", []))
        if instance_count <= 0:
            errors.append(
                f"{dataset.dataset_id} test COCO instance sayısı geçersiz"
            )
            continue
        reference_types = (
            ("human", "pseudo_sam1")
            if dataset.reference_type == "human"
            else (dataset.reference_type,)
        )
        for model in MODELS:
            for reference_type in reference_types:
                expected_counts[
                    (
                        dataset.dataset_id,
                        model,
                        "gt_bbox",
                        None,
                        reference_type,
                    )
                ] = instance_count
                for seed in protocol.detector_seeds:
                    expected_counts[
                        (
                            dataset.dataset_id,
                            model,
                            "yolo_bbox",
                            int(seed),
                            reference_type,
                        )
                    ] = instance_count

    actual_counts: dict[tuple[str, str, str, int | None, str], int] = {}
    for key, group in canonical.groupby(
        list(condition_columns),
        dropna=False,
        sort=True,
    ):
        seed_value = key[3]
        normalized_key = (
            str(key[0]),
            str(key[1]),
            str(key[2]),
            None if pd.isna(seed_value) else int(seed_value),
            str(key[4]),
        )
        actual_counts[normalized_key] = int(len(group))

    missing_conditions = set(expected_counts) - set(actual_counts)
    extra_conditions = set(actual_counts) - set(expected_counts)
    if missing_conditions:
        errors.append(
            "Canonical instance metrics koşulları eksik: "
            + ", ".join(sorted(map(str, missing_conditions)))
        )
    if extra_conditions:
        errors.append(
            "Canonical instance metrics beklenmeyen koşul içeriyor: "
            + ", ".join(sorted(map(str, extra_conditions)))
        )
    bad_counts = {
        key: {"expected": expected_counts[key], "actual": actual_counts.get(key)}
        for key in expected_counts
        if actual_counts.get(key) != expected_counts[key]
    }
    if bad_counts:
        errors.append(
            "Canonical instance metrics instance sayıları uyuşmuyor: "
            f"{bad_counts}"
        )

    aggregate = pd.read_csv(aggregate_path)
    aggregate_metrics = (
        "mean_iou",
        "mean_dice",
        "mean_precision",
        "mean_recall",
        "mean_boundary_iou",
        "success_at_iou_50",
        "success_at_iou_75",
        "success_at_iou_90",
        "iou_ci_lower",
        "iou_ci_upper",
    )
    validate_unit_interval_columns(
        aggregate,
        aggregate_metrics,
        label="Aggregate metrics",
        errors=errors,
    )
    aggregate_key_columns = (*condition_columns, "stratum")
    if set(aggregate_key_columns).issubset(aggregate.columns):
        actual_aggregate_keys = {
            (
                str(row.dataset_id),
                str(row.model),
                str(row.bbox_source),
                None
                if pd.isna(row.detector_seed)
                else int(row.detector_seed),
                str(row.reference_type),
                str(row.stratum),
            )
            for row in aggregate.itertuples(index=False)
        }
        expected_aggregate_keys = {
            (*condition, stratum)
            for condition in expected_counts
            for stratum in STRATA
        }
        if actual_aggregate_keys != expected_aggregate_keys:
            errors.append(
                "Aggregate metrics tam koşul x stratum matrisini içermiyor"
            )
        if len(aggregate) != len(expected_aggregate_keys):
            errors.append(
                "Aggregate metrics satır sayısı beklenen değerde değil: "
                f"{len(aggregate)} != {len(expected_aggregate_keys)}"
            )
    else:
        errors.append("Aggregate metrics koşul kolonları eksik")
    for count_column in ("instance_count", "source_scene_count"):
        if count_column not in aggregate:
            errors.append(f"Aggregate metrics {count_column} kolonu eksik")
            continue
        counts = pd.to_numeric(aggregate[count_column], errors="coerce")
        if counts.isna().any() or (counts <= 0).any():
            errors.append(f"Aggregate metrics {count_column} geçersiz")
    if {"iou_ci_lower", "iou_ci_upper"}.issubset(aggregate.columns):
        lower = pd.to_numeric(aggregate["iou_ci_lower"], errors="coerce")
        upper = pd.to_numeric(aggregate["iou_ci_upper"], errors="coerce")
        if (lower > upper).any():
            errors.append("Aggregate metrics IoU güven aralığı ters")
    if "bootstrap_samples" not in aggregate or not (
        pd.to_numeric(aggregate["bootstrap_samples"], errors="coerce")
        == 10_000
    ).all():
        errors.append("Aggregate metrics bütün satırlarda 10.000 bootstrap değil")

    detector = pd.read_csv(detector_path)
    expected_detector_keys = {
        (dataset.dataset_id, int(seed))
        for dataset in datasets
        for seed in protocol.detector_seeds
    }
    if {"dataset_id", "seed"}.issubset(detector.columns):
        actual_detector_keys = {
            (str(row.dataset_id), int(row.seed))
            for row in detector.itertuples(index=False)
        }
        if (
            actual_detector_keys != expected_detector_keys
            or len(detector) != len(expected_detector_keys)
        ):
            errors.append("Detector metrics iki dataset x üç seed matrisini içermiyor")
    else:
        errors.append("Detector metrics dataset_id/seed kolonları eksik")
    detector_metrics = (
        "fixed_confidence_threshold",
        "bbox_AP50",
        "bbox_AP75",
        "bbox_AP90",
        "bbox_AP50_95",
        "precision_at_bbox_iou50",
        "recall_at_bbox_iou50",
        "precision_at_bbox_iou75",
        "recall_at_bbox_iou75",
        "precision_at_bbox_iou90",
        "recall_at_bbox_iou90",
    )
    validate_unit_interval_columns(
        detector,
        detector_metrics,
        label="Detector metrics",
        errors=errors,
    )
    if "split" not in detector or not (detector["split"] == "test").all():
        errors.append("Detector metrics yalnız test split'inden oluşmuyor")
    if (
        "confidence_threshold_source_split" not in detector
        or not (
            detector["confidence_threshold_source_split"] == "validation"
        ).all()
    ):
        errors.append(
            "Detector metrics confidence eşiği kaynağı validation değil"
        )


def validate_isaid_rle_migration(
    study_root: Path,
    datasets: list[DatasetStudyConfig],
    *,
    required: list[Path],
    errors: list[str],
) -> None:
    matches = [
        dataset for dataset in datasets if dataset.dataset_id == "isaid_plane"
    ]
    if len(matches) != 1:
        errors.append("iSAID RLE migration denetimi için tek dataset bulunamadı")
        return
    dataset = matches[0]
    audit_path = study_root / "audits" / "isaid_lossless_rle_migration.json"
    require_file(
        audit_path,
        label="iSAID lossless RLE migration audit",
        required=required,
        errors=errors,
    )
    if not audit_path.is_file() or not audit_path.stat().st_size:
        return
    payload = read_json(audit_path)
    if payload.get("status") != "pass":
        errors.append("iSAID lossless RLE migration audit pass değil")
    rows = {
        str(row.get("split")): row
        for row in payload.get("splits", [])
    }
    if set(rows) != {"train", "validation", "test"}:
        errors.append("iSAID RLE migration üç split içermiyor")
        return
    for split, row in rows.items():
        current_path = (
            dataset.prepared_root / split / "_annotations.coco.json"
        )
        archive_path = Path(str(row.get("archive_path", "")))
        require_file(
            archive_path,
            label=f"iSAID {split} pre-RLE-fix COCO archive",
            required=required,
            errors=errors,
        )
        if (
            archive_path.is_file()
            and sha256_file(archive_path) != row.get("before_sha256")
        ):
            errors.append(f"iSAID {split} pre-fix archive hash'i uyuşmuyor")
        if (
            current_path.is_file()
            and sha256_file(current_path) != row.get("after_sha256")
        ):
            errors.append(f"iSAID {split} post-fix COCO hash'i uyuşmuyor")
        if int(row.get("empty_masks_after", -1)) != 0:
            errors.append(f"iSAID {split} post-fix boş maske içeriyor")
        if int(row.get("area_mismatches_after", -1)) != 0:
            errors.append(f"iSAID {split} post-fix alan uyuşmazlığı içeriyor")
        if row.get("before_sha256") == row.get("after_sha256"):
            errors.append(f"iSAID {split} RLE migration içeriği değiştirmemiş")
    for source in payload.get("source_files", []):
        source_path = Path(str(source.get("path", "")))
        require_file(
            source_path,
            label="iSAID RLE migration official source annotation",
            required=required,
            errors=errors,
        )
        if (
            source_path.is_file()
            and sha256_file(source_path) != source.get("sha256")
        ):
            errors.append(
                f"iSAID RLE migration source hash'i uyuşmuyor: {source_path}"
            )


def validate_isaid_rle_sensitivity_and_archive(
    study_root: Path,
    datasets: list[DatasetStudyConfig],
    *,
    required: list[Path],
    errors: list[str],
) -> None:
    matches = [
        dataset for dataset in datasets if dataset.dataset_id == "isaid_plane"
    ]
    if len(matches) != 1:
        errors.append(
            "iSAID RLE sensitivity denetimi için tek dataset bulunamadı"
        )
        return
    dataset = matches[0]
    test_coco = read_json(
        dataset.prepared_root / "test" / "_annotations.coco.json"
    )
    expected_instances = len(test_coco.get("annotations", []))

    sensitivity_path = (
        study_root / "audits" / "isaid_rle_reference_sensitivity.json"
    )
    require_file(
        sensitivity_path,
        label="iSAID RLE reference sensitivity audit",
        required=required,
        errors=errors,
    )
    if sensitivity_path.is_file() and sensitivity_path.stat().st_size:
        payload = read_json(sensitivity_path)
        if int(payload.get("instances", -1)) != expected_instances:
            errors.append(
                "iSAID RLE sensitivity instance sayısı test split'iyle "
                "uyuşmuyor"
            )
        if int(payload.get("old_empty", -1)) <= 0:
            errors.append(
                "iSAID RLE sensitivity eski referanstaki boş maskeyi "
                "göstermiyor"
            )
        if int(payload.get("new_empty", -1)) != 0:
            errors.append(
                "iSAID RLE sensitivity yeni referansta boş maske gösteriyor"
            )
        mean_iou = float(payload.get("mean_reference_iou", np.nan))
        median_iou = float(payload.get("median_reference_iou", np.nan))
        if not (np.isfinite(mean_iou) and 0.0 < mean_iou < 1.0):
            errors.append("iSAID RLE sensitivity mean IoU geçersiz")
        if not (np.isfinite(median_iou) and 0.0 < median_iou < 1.0):
            errors.append("iSAID RLE sensitivity median IoU geçersiz")
        if int(payload.get("instances_below_iou_0_90", -1)) <= 0:
            errors.append(
                "iSAID RLE sensitivity 0,90 altındaki örnekleri "
                "kaydetmiyor"
            )

    migration_path = study_root / "audits" / "isaid_lossless_rle_migration.json"
    migration = read_json(migration_path) if migration_path.is_file() else {}
    split_rows = {
        str(row.get("split")): row for row in migration.get("splits", [])
    }
    test_migration = split_rows.get("test", {})
    archive_manifest_path = (
        study_root
        / "audits"
        / "pre_isaid_lossless_rle_metric_fix"
        / "manifest.json"
    )
    require_file(
        archive_manifest_path,
        label="iSAID pre-RLE-fix metric archive manifest",
        required=required,
        errors=errors,
    )
    if not archive_manifest_path.is_file():
        return
    archive = read_json(archive_manifest_path)
    if archive.get("status") != "superseded_invalid_for_scientific_results":
        errors.append("iSAID pre-RLE-fix metric archive geçersiz etiketlenmiş")
    reference = archive.get("superseded_reference", {})
    if reference.get("test_coco_sha256") != test_migration.get("before_sha256"):
        errors.append("iSAID pre-RLE-fix archive eski COCO hash'i uyuşmuyor")
    if (
        reference.get("replacement_test_coco_sha256")
        != test_migration.get("after_sha256")
    ):
        errors.append("iSAID pre-RLE-fix archive yeni COCO hash'i uyuşmuyor")
    archived_files = archive.get("files", [])
    if not archived_files:
        errors.append("iSAID pre-RLE-fix metric archive dosya içermiyor")
    if not archive.get("summary_rows"):
        errors.append("iSAID pre-RLE-fix metric archive özet metrik içermiyor")
    for row in archived_files:
        original_path = Path(str(row.get("archive_path", "")))
        path = migrated_archive_path(original_path)
        require_file(
            path,
            label=(
                "iSAID pre-RLE-fix archived artifact "
                f"(manifest path: {original_path})"
            ),
            required=required,
            errors=errors,
        )
        if path.is_file() and sha256_file(path) != row.get("sha256"):
            errors.append(f"iSAID pre-RLE-fix archive hash'i uyuşmuyor: {path}")


def validate_legacy_detector_manifest_repair(
    study_root: Path,
    protocol: MatchedStudyConfig,
    datasets: list[DatasetStudyConfig],
    *,
    required: list[Path],
    errors: list[str],
) -> None:
    audit_path = (
        study_root
        / "audits"
        / "legacy_detector_manifest_repair"
        / "manifest.json"
    )
    require_file(
        audit_path,
        label="Legacy detector manifest repair audit",
        required=required,
        errors=errors,
    )
    if not audit_path.is_file():
        return
    audit = read_json(audit_path)
    if audit.get("status") != "pass":
        errors.append("Legacy detector manifest repair audit pass değil")
    rows = {
        (str(row.get("dataset_id")), int(row.get("seed", -1))): row
        for row in audit.get("rows", [])
    }
    expected = {
        (dataset.dataset_id, int(seed))
        for dataset in datasets
        for seed in protocol.detector_seeds
    }
    if set(rows) != expected:
        errors.append(
            "Legacy detector manifest repair audit altı koşulu kapsamıyor"
        )
        return
    dataset_by_id = {dataset.dataset_id: dataset for dataset in datasets}
    for (dataset_id, seed), row in rows.items():
        manifest_path = (
            study_root
            / "detectors"
            / dataset_id
            / f"seed_{seed}"
            / "manifest.json"
        )
        require_file(
            manifest_path,
            label="Detector training run manifest",
            required=required,
            errors=errors,
        )
        if (
            manifest_path.is_file()
            and sha256_file(manifest_path) != row.get("manifest_sha256")
        ):
            errors.append(
                f"Detector manifest repair sonrası değişmiş: {manifest_path}"
            )
        scoped_path = (
            dataset_by_id[dataset_id].prepared_root
            / "detector_training_content_manifest.json"
        )
        require_file(
            scoped_path,
            label="Detector-scoped prepared content manifest",
            required=required,
            errors=errors,
        )
        if (
            scoped_path.is_file()
            and sha256_file(scoped_path)
            != row.get("detector_content_manifest_sha256")
        ):
            errors.append(
                f"Detector content manifest repair sonrası değişmiş: {scoped_path}"
            )
        if scoped_path.is_file():
            scoped = read_json(scoped_path)
            if (
                scoped.get("tree_sha256")
                != row.get("detector_content_tree_sha256")
            ):
                errors.append(
                    f"Detector content tree hash audit ile uyuşmuyor: {scoped_path}"
                )
        action = row.get("action")
        if action == "repaired_with_archived_original":
            archive_path = Path(str(row.get("original_manifest_path", "")))
            require_file(
                archive_path,
                label="Original legacy detector manifest archive",
                required=required,
                errors=errors,
            )
            if (
                archive_path.is_file()
                and sha256_file(archive_path)
                != row.get("original_manifest_sha256")
            ):
                errors.append(
                    f"Original detector manifest archive hash'i uyuşmuyor: "
                    f"{archive_path}"
                )
            if manifest_path.is_file():
                repaired = read_json(manifest_path)
                provenance_repair = repaired.get("provenance_repair", {})
                capture = repaired.get("input_fingerprint_capture")
                migration = repaired.get("layout_migration", {})
                capture_is_valid = capture == "provenance_repair" or (
                    capture == "layout_migration"
                    and migration.get("previous_input_fingerprint_capture")
                    == "provenance_repair"
                )
                if not capture_is_valid:
                    errors.append(
                        f"Repaired detector manifest etiketi eksik: {manifest_path}"
                    )
                if (
                    provenance_repair.get("original_manifest_sha256")
                    != row.get("original_manifest_sha256")
                ):
                    errors.append(
                        f"Detector provenance repair zinciri kopuk: {manifest_path}"
                    )
        elif action == "unchanged_start_fingerprinted":
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
                capture = manifest.get("input_fingerprint_capture")
                migration = manifest.get("layout_migration", {})
                capture_is_valid = capture == "start" or (
                    capture == "layout_migration"
                    and migration.get("previous_input_fingerprint_capture")
                    == "start"
                )
                if not capture_is_valid:
                    errors.append(
                        f"Unchanged detector manifest start hash'li değil: "
                        f"{manifest_path}"
                    )
        else:
            errors.append(
                f"Bilinmeyen detector manifest repair action: {action}"
            )


def is_archived_run_manifest(
    path: Path,
    study_root: Path,
) -> bool:
    return any(
        path.is_relative_to(study_root / relative_root)
        for relative_root in (
            SUPERSEDED_ISAID_METRIC_ARCHIVE,
            LEGACY_DETECTOR_MANIFEST_ARCHIVE,
        )
    )


def backfill_completed_run_manifests(study_root: Path) -> list[Path]:
    updated: list[Path] = []
    for path in sorted(study_root.rglob("manifest.json")):
        if is_archived_run_manifest(path, study_root):
            continue
        payload = read_json(path)
        if (
            "run_id" not in payload
            or "stage" not in payload
            or payload.get("status") != "completed"
        ):
            continue
        inputs = payload.get("inputs")
        outputs = payload.get("outputs")
        if isinstance(inputs, dict):
            payload["input_file_fingerprints"] = declared_file_fingerprints(inputs)
        if isinstance(outputs, dict):
            payload["output_file_fingerprints"] = declared_file_fingerprints(
                outputs
            )
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        updated.append(path)
    return updated


def attach_prepared_content_manifests(
    study_root: Path,
    datasets: list[DatasetStudyConfig],
) -> list[Path]:
    updated: list[Path] = []
    for path in sorted(study_root.rglob("manifest.json")):
        if is_archived_run_manifest(path, study_root):
            continue
        payload = read_json(path)
        if (
            "run_id" not in payload
            or "stage" not in payload
            or payload.get("status") != "completed"
        ):
            continue
        matches = [
            dataset
            for dataset in datasets
            if dataset.dataset_id in path.parts
            or dataset.dataset_id in str(payload.get("run_id", ""))
        ]
        if len(matches) != 1:
            continue
        content_manifest = matches[0].prepared_root / "content_manifest.json"
        inputs = payload.setdefault("inputs", {})
        inputs["prepared_content_manifest"] = str(content_manifest)
        if payload.get("stage") in {
            "gt_bbox_segmentation",
            "yolo_bbox_segmentation",
        }:
            inputs["segmenter_provenance"] = str(
                study_root / "audits" / "segmenter_provenance.json"
            )
        serialized = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        path.write_text(serialized + "\n", encoding="utf-8")
        updated.append(path)
    return updated


def validate_completed_run_manifest_fingerprints(
    study_root: Path,
    *,
    required: list[Path],
    errors: list[str],
) -> None:
    for manifest_path in sorted(study_root.rglob("manifest.json")):
        if is_archived_run_manifest(
            manifest_path,
            study_root,
        ):
            continue
        payload = read_json(manifest_path)
        if (
            "run_id" not in payload
            or "stage" not in payload
            or payload.get("status") != "completed"
        ):
            continue
        required.append(manifest_path)
        capture = payload.get("input_fingerprint_capture")
        repair = payload.get("provenance_repair")
        if capture not in {
            "start",
            "provenance_repair",
            "layout_migration",
        }:
            errors.append(
                "Run manifest giriş hash'inin ne zaman yakalandığını "
                f"kanıtlamıyor: {manifest_path}"
            )
        if capture == "provenance_repair" and not isinstance(repair, dict):
            errors.append(
                f"Run manifest provenance repair kaydı eksik: {manifest_path}"
            )
        if capture == "layout_migration":
            migration = payload.get("layout_migration")
            if (
                not isinstance(migration, dict)
                or migration.get("migration_id") != "study_layout_20260726"
                or migration.get("reason") != "repository_layout_only"
            ):
                errors.append(
                    "Run manifest geçerli layout migration kaydı içermiyor: "
                    f"{manifest_path}"
                )
        drift = payload.get("input_drift")
        if drift not in ([], None):
            errors.append(
                f"Run sırasında giriş dosyası değişmiş: {manifest_path}: "
                f"{drift}"
            )
        config_hash = str(payload.get("config_hash", ""))
        if len(config_hash) != 64 or any(
            character not in "0123456789abcdef" for character in config_hash
        ):
            errors.append(
                f"Run manifest geçerli config hash içermiyor: {manifest_path}"
            )

        for value_key, fingerprint_key in (
            ("inputs", "input_file_fingerprints"),
            ("outputs", "output_file_fingerprints"),
        ):
            values = payload.get(value_key)
            fingerprints = payload.get(fingerprint_key)
            if not isinstance(values, dict) or not isinstance(
                fingerprints,
                dict,
            ):
                errors.append(
                    f"Run manifest {fingerprint_key} alanı eksik: "
                    f"{manifest_path}"
                )
                continue
            for name, value in values.items():
                if not isinstance(value, str):
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = ROOT / path
                if not path.is_file():
                    continue
                fingerprint = fingerprints.get(name)
                if not isinstance(fingerprint, dict):
                    errors.append(
                        f"Run manifest {value_key}.{name} için hash içermiyor: "
                        f"{manifest_path}"
                    )
                    continue
                if (
                    int(fingerprint.get("bytes", -1)) != path.stat().st_size
                    or fingerprint.get("sha256") != sha256_file(path)
                ):
                    errors.append(
                        f"Run manifest {value_key}.{name} hash'i mevcut "
                        f"dosyayla uyuşmuyor: {manifest_path}"
                    )
                required.append(path)
        start_fingerprints = payload.get("input_file_fingerprints")
        finish_fingerprints = payload.get(
            "input_file_fingerprints_at_finish"
        )
        if (
            isinstance(start_fingerprints, dict)
            and isinstance(finish_fingerprints, dict)
            and start_fingerprints != finish_fingerprints
        ):
            errors.append(
                "Run manifest başlangıç/bitiş input hash'leri uyuşmuyor: "
                f"{manifest_path}"
            )


def validate_repository_layout_migration(
    study_root: Path,
    *,
    required: list[Path],
    errors: list[str],
) -> None:
    audit_path = (
        study_root
        / "audits"
        / "repository_layout_migration"
        / "layout_migration.json"
    )
    require_file(
        audit_path,
        label="Repository layout migration audit",
        required=required,
        errors=errors,
    )
    if not audit_path.is_file():
        return
    payload = read_json(audit_path)
    if (
        payload.get("status") != "pass"
        or payload.get("migration_id") != "study_layout_20260726"
        or payload.get("scope") != "repository_layout_only"
    ):
        errors.append("Repository layout migration audit geçersiz")
        return

    for manifest_key in ("pre_move_manifest", "post_move_manifest"):
        row = payload.get(manifest_key, {})
        path = Path(str(row.get("path", "")))
        require_file(
            path,
            label=f"Layout migration {manifest_key}",
            required=required,
            errors=errors,
        )
        if path.is_file() and sha256_file(path) != row.get("sha256"):
            errors.append(f"Layout migration {manifest_key} hash'i uyuşmuyor")

    rows = payload.get("files", [])
    if int(payload.get("modified_file_count", -1)) != len(rows):
        errors.append("Layout migration dosya sayısı uyuşmuyor")
    for row in rows:
        path = Path(str(row.get("path", "")))
        original = Path(str(row.get("original_copy", "")))
        require_file(
            path,
            label="Layout-migrated file",
            required=required,
            errors=errors,
        )
        require_file(
            original,
            label="Layout-migration original copy",
            required=required,
            errors=errors,
        )
        if path.is_file() and sha256_file(path) != row.get("after_sha256"):
            errors.append(f"Layout-migrated file hash'i uyuşmuyor: {path}")
        if (
            original.is_file()
            and sha256_file(original) != row.get("before_sha256")
        ):
            errors.append(
                f"Layout migration original hash'i uyuşmuyor: {original}"
            )

    dependency_row = payload.get("dependency_repair", {})
    dependency_path = Path(str(dependency_row.get("path", "")))
    require_file(
        dependency_path,
        label="Layout migration dependency repair audit",
        required=required,
        errors=errors,
    )
    if not dependency_path.is_file():
        return
    if sha256_file(dependency_path) != dependency_row.get("sha256"):
        errors.append("Layout migration dependency repair hash'i uyuşmuyor")
        return

    dependency = read_json(dependency_path)
    dependency_rows = dependency.get("files", [])
    if (
        dependency.get("status") != "pass"
        or dependency.get("migration_id") != "study_layout_20260726"
        or dependency.get("scope") != "layout_migration_dependency_order_only"
        or int(dependency.get("modified_file_count", -1))
        != len(dependency_rows)
        or len(dependency_rows) != 26
    ):
        errors.append("Layout migration dependency repair audit geçersiz")
        return

    migration_rows = {
        str(row.get("path")): row
        for row in rows
        if isinstance(row, dict)
    }
    for row in dependency_rows:
        path = Path(str(row.get("path", "")))
        intermediate = Path(str(row.get("intermediate_copy", "")))
        require_file(
            path,
            label="Dependency-repaired layout file",
            required=required,
            errors=errors,
        )
        require_file(
            intermediate,
            label="Dependency repair intermediate copy",
            required=required,
            errors=errors,
        )
        if path.is_file() and sha256_file(path) != row.get("after_sha256"):
            errors.append(
                f"Dependency-repaired file hash'i uyuşmuyor: {path}"
            )
        if (
            intermediate.is_file()
            and sha256_file(intermediate) != row.get("before_sha256")
        ):
            errors.append(
                f"Dependency repair intermediate hash'i uyuşmuyor: "
                f"{intermediate}"
            )
        migration_row = migration_rows.get(str(path))
        if migration_row is None:
            errors.append(
                f"Dependency repair ana migration auditinde yok: {path}"
            )
            continue
        if (
            migration_row.get("dependency_repair_before_sha256")
            != row.get("before_sha256")
            or migration_row.get("dependency_repair_after_sha256")
            != row.get("after_sha256")
            or migration_row.get("after_sha256") != row.get("after_sha256")
        ):
            errors.append(
                f"Dependency repair zinciri ana audit ile uyuşmuyor: {path}"
            )


def validate_detector_training_args_matrix(
    study_root: Path,
    protocol: MatchedStudyConfig,
    datasets: list[DatasetStudyConfig],
    *,
    required: list[Path],
    errors: list[str],
) -> None:
    normalized_rows: dict[tuple[str, int], dict[str, Any]] = {}
    allowed_differences = {"data", "project", "save_dir", "seed"}
    base_weights = Path(str(protocol.detector["base_weights"]))
    if not base_weights.is_absolute():
        base_weights = ROOT / base_weights
    for dataset in datasets:
        for seed in protocol.detector_seeds:
            path = (
                study_root
                / "detectors"
                / dataset.dataset_id
                / f"seed_{seed}"
                / "train"
                / "args.yaml"
            )
            require_file(
                path,
                label=(
                    f"{dataset.dataset_id} detector seed {seed} actual args"
                ),
                required=required,
                errors=errors,
            )
            if not path.is_file():
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                errors.append(f"Detector args YAML mapping değil: {path}")
                continue
            expected_values = {
                "task": "detect",
                "mode": "train",
                "model": str(base_weights.resolve()),
                "data": str(
                    (dataset.prepared_root / "data.yaml").resolve()
                ),
                "epochs": int(protocol.detector["epochs"]),
                "patience": int(protocol.detector["patience"]),
                "batch": int(protocol.detector["batch"]),
                "imgsz": int(protocol.image_size),
                "optimizer": str(protocol.detector["optimizer"]),
                "seed": int(seed),
                "deterministic": True,
                "workers": 4,
                "val": True,
                "split": "val",
            }
            mismatches = {
                key: {
                    "expected": expected,
                    "actual": payload.get(key),
                }
                for key, expected in expected_values.items()
                if normalize_layout_path_text(payload.get(key))
                != normalize_layout_path_text(expected)
            }
            if mismatches:
                errors.append(
                    f"{dataset.dataset_id} seed {seed} actual args frozen "
                    f"protokolle uyuşmuyor: {mismatches}"
                )
            normalized_rows[(dataset.dataset_id, int(seed))] = {
                key: value
                for key, value in payload.items()
                if key not in allowed_differences
            }
    if not normalized_rows:
        return
    baseline_key = sorted(normalized_rows)[0]
    baseline = normalized_rows[baseline_key]
    for key, row in normalized_rows.items():
        if row != baseline:
            differing_keys = sorted(
                name
                for name in set(row) | set(baseline)
                if row.get(name) != baseline.get(name)
            )
            errors.append(
                "Detector actual args izin verilen dataset/seed/output "
                f"alanları dışında farklı: {baseline_key} vs {key}: "
                f"{differing_keys}"
            )


def expected_pipeline_files(
    study_root: Path,
    protocol: MatchedStudyConfig,
    datasets: list[DatasetStudyConfig],
    *,
    required: list[Path],
    errors: list[str],
) -> None:
    for dataset in datasets:
        dataset_id = dataset.dataset_id
        is_human = dataset.reference_type.value == "human"
        require_file(
            dataset.prepared_root / "data.yaml",
            label=f"{dataset_id} prepared data config",
            required=required,
            errors=errors,
        )
        content_manifest_path = (
            dataset.prepared_root / "content_manifest.json"
        )
        require_file(
            content_manifest_path,
            label=f"{dataset_id} prepared content manifest",
            required=required,
            errors=errors,
        )
        if (
            content_manifest_path.is_file()
            and content_manifest_path.stat().st_size
        ):
            for error in validate_prepared_content_manifest(
                dataset.prepared_root,
                read_json(content_manifest_path),
            ):
                errors.append(f"{dataset_id}: {error}")
        detector_content_manifest_path = (
            dataset.prepared_root
            / "detector_training_content_manifest.json"
        )
        require_file(
            detector_content_manifest_path,
            label=f"{dataset_id} detector training content manifest",
            required=required,
            errors=errors,
        )
        if (
            detector_content_manifest_path.is_file()
            and detector_content_manifest_path.stat().st_size
        ):
            for error in validate_detector_training_content_manifest(
                dataset.prepared_root,
                read_json(detector_content_manifest_path),
            ):
                errors.append(f"{dataset_id}: {error}")
        for split in ("train", "validation", "test"):
            for name in ("_annotations.coco.json", "metadata.csv"):
                require_file(
                    dataset.prepared_root / split / name,
                    label=f"{dataset_id}/{split}/{name}",
                    required=required,
                    errors=errors,
                )
        if is_human:
            reference_root = study_root / "references" / dataset_id
            require_file(
                reference_root / "sam1_gt_bbox_pseudo.jsonl",
                label=f"{dataset_id} controlled SAM1 pseudo reference",
                required=required,
                errors=errors,
            )
            require_completed_manifest(
                reference_root / "sam1_gt_bbox_pseudo.manifest.json",
                label=(
                    f"{dataset_id} controlled SAM1 pseudo-reference manifest"
                ),
                required=required,
                errors=errors,
            )
        for model in MODELS:
            gt_prediction_root = (
                study_root / "predictions" / dataset_id / model / "gt_bbox"
            )
            require_completed_manifest(
                gt_prediction_root / "manifest.json",
                label=f"{dataset_id}/{model} GT-bbox prediction manifest",
                required=required,
                errors=errors,
            )
            require_file(
                gt_prediction_root / "predictions.jsonl",
                label=f"{dataset_id}/{model} GT-bbox predictions",
                required=required,
                errors=errors,
            )
            gt_mode = "gt_bbox_dual_reference" if is_human else "gt_bbox"
            gt_evaluation_root = (
                study_root / "evaluation" / dataset_id / model / gt_mode
            )
            require_completed_manifest(
                gt_evaluation_root / "manifest.json",
                label=f"{dataset_id}/{model}/{gt_mode} evaluation manifest",
                required=required,
                errors=errors,
            )
            for name in (
                "metrics_instance.csv",
                "summary_instance.csv",
                "metrics_image_union.csv",
            ):
                require_file(
                    gt_evaluation_root / name,
                    label=f"{dataset_id}/{model}/{gt_mode}/{name}",
                    required=required,
                    errors=errors,
                )

        for seed in protocol.detector_seeds:
            detector_root = study_root / "detectors" / dataset_id / f"seed_{seed}"
            training_manifest = require_completed_manifest(
                detector_root / "manifest.json",
                label=f"{dataset_id} detector seed {seed} training manifest",
                required=required,
                errors=errors,
            )
            if training_manifest is not None:
                parameters = training_manifest.get("parameters", {})
                expected_parameters = {
                    **protocol.detector,
                    "image_size": protocol.image_size,
                }
                mismatches = {
                    key: {
                        "expected": value,
                        "actual": parameters.get(key),
                    }
                    for key, value in expected_parameters.items()
                    if normalize_layout_path_text(parameters.get(key))
                    != normalize_layout_path_text(value)
                }
                if mismatches:
                    errors.append(
                        f"{dataset_id} detector seed {seed} frozen "
                        f"parametre farkı var: {mismatches}"
                    )
            require_file(
                detector_root / "train" / "weights" / "best.pt",
                label=f"{dataset_id} detector seed {seed} checkpoint",
                required=required,
                errors=errors,
            )
            require_file(
                detector_root / "train" / "results.csv",
                label=f"{dataset_id} detector seed {seed} training results",
                required=required,
                errors=errors,
            )
            require_file(
                detector_root / "train" / "args.yaml",
                label=f"{dataset_id} detector seed {seed} training args",
                required=required,
                errors=errors,
            )
            validation_eval_root = (
                detector_root / "evaluation" / "validation"
            )
            require_completed_manifest(
                validation_eval_root / "manifest.json",
                label=(
                    f"{dataset_id} detector seed {seed} validation manifest"
                ),
                required=required,
                errors=errors,
            )
            for name in (
                "detections_coco.json",
                "metrics.json",
                "selected_confidence_threshold.json",
            ):
                require_file(
                    validation_eval_root / name,
                    label=(
                        f"{dataset_id} detector seed {seed} validation {name}"
                    ),
                    required=required,
                    errors=errors,
                )
            threshold_path = (
                validation_eval_root / "selected_confidence_threshold.json"
            )
            threshold_selection = (
                read_json(threshold_path)
                if threshold_path.is_file() and threshold_path.stat().st_size
                else None
            )
            if threshold_selection is not None:
                if threshold_selection.get("selection_split") != "validation":
                    errors.append(
                        f"{dataset_id} seed {seed} confidence eşiği "
                        "validation üzerinde seçilmemiş"
                    )
                selected_threshold = float(
                    threshold_selection.get(
                        "selected_confidence_threshold",
                        -1.0,
                    )
                )
                if not 0 <= selected_threshold <= 1:
                    errors.append(
                        f"{dataset_id} seed {seed} confidence eşiği geçersiz"
                    )

            detector_eval_root = detector_root / "evaluation" / "test"
            require_completed_manifest(
                detector_eval_root / "manifest.json",
                label=f"{dataset_id} detector seed {seed} evaluation manifest",
                required=required,
                errors=errors,
            )
            for name in ("detections_coco.json", "metrics.json"):
                require_file(
                    detector_eval_root / name,
                    label=f"{dataset_id} detector seed {seed} {name}",
                    required=required,
                    errors=errors,
                )
            test_metrics_path = detector_eval_root / "metrics.json"
            if (
                threshold_selection is not None
                and test_metrics_path.is_file()
                and test_metrics_path.stat().st_size
            ):
                test_metrics = read_json(test_metrics_path)
                if (
                    test_metrics.get("confidence_threshold_source_split")
                    != "validation"
                ):
                    errors.append(
                        f"{dataset_id} seed {seed} test confidence eşiğinin "
                        "kaynağı validation değil"
                    )
                if abs(
                    float(test_metrics.get("fixed_confidence_threshold", -1.0))
                    - float(
                        threshold_selection[
                            "selected_confidence_threshold"
                        ]
                    )
                ) > 1e-12:
                    errors.append(
                        f"{dataset_id} seed {seed} test ve validation "
                        "confidence eşikleri uyuşmuyor"
                    )

            for model in MODELS:
                prediction_root = (
                    study_root
                    / "predictions"
                    / dataset_id
                    / model
                    / "yolo_bbox"
                    / f"seed_{seed}"
                )
                require_completed_manifest(
                    prediction_root / "manifest.json",
                    label=(
                        f"{dataset_id}/{model} YOLO-bbox seed {seed} "
                        "prediction manifest"
                    ),
                    required=required,
                    errors=errors,
                )
                for name in (
                    "predictions.jsonl",
                    "unmatched_detector_predictions.jsonl",
                ):
                    require_file(
                        prediction_root / name,
                        label=f"{dataset_id}/{model}/seed {seed}/{name}",
                        required=required,
                        errors=errors,
                        allow_empty=name.startswith("unmatched_"),
                    )

                seed_mode = (
                    f"seed_{seed}_dual_reference"
                    if is_human
                    else f"seed_{seed}"
                )
                evaluation_root = (
                    study_root
                    / "evaluation"
                    / dataset_id
                    / model
                    / "yolo_bbox"
                    / seed_mode
                )
                require_completed_manifest(
                    evaluation_root / "manifest.json",
                    label=(
                        f"{dataset_id}/{model}/yolo_bbox/{seed_mode} "
                        "evaluation manifest"
                    ),
                    required=required,
                    errors=errors,
                )
                for name in (
                    "metrics_instance.csv",
                    "summary_instance.csv",
                    "metrics_image_union.csv",
                ):
                    require_file(
                        evaluation_root / name,
                        label=(
                            f"{dataset_id}/{model}/yolo_bbox/{seed_mode}/{name}"
                        ),
                        required=required,
                        errors=errors,
                    )


def validate_analysis_and_paper(
    study_root: Path,
    paper_dir: Path,
    protocol: MatchedStudyConfig,
    datasets: list[DatasetStudyConfig],
    *,
    required: list[Path],
    errors: list[str],
) -> None:
    analysis_root = study_root / "analysis"
    for name in ANALYSIS_FILES:
        require_file(
            analysis_root / name,
            label=f"Canonical analysis {name}",
            required=required,
            errors=errors,
        )
    analysis_manifest_path = analysis_root / "manifest.json"
    if analysis_manifest_path.is_file():
        manifest = read_json(analysis_manifest_path)
        if int(manifest.get("parameters", {}).get("bootstrap_samples", 0)) != 10_000:
            errors.append("Canonical analysis 10.000 bootstrap ile üretilmemiş")
    validate_canonical_analysis_content(
        study_root,
        protocol,
        datasets,
        errors=errors,
    )
    prediction_audit_path = analysis_root / "prediction_status_audit.csv"
    if prediction_audit_path.is_file() and prediction_audit_path.stat().st_size:
        prediction_audit = pd.read_csv(prediction_audit_path)
        matched = prediction_audit[
            prediction_audit["row_kind"] == "matched_ground_truth"
        ]
        unmatched = prediction_audit[
            prediction_audit["row_kind"] == "unmatched_detector"
        ]
        if len(matched) != 24 or len(unmatched) != 18:
            errors.append(
                "Prediction status audit koşul sayıları beklenen 24 matched + "
                f"18 unmatched değil: {len(matched)} + {len(unmatched)}"
            )
        for column in (
            "duplicate_instance_ids",
            "status_area_mismatches",
            "nonempty_masks_without_prompt_overlap",
            "inference_error",
        ):
            if int(prediction_audit[column].sum()) != 0:
                errors.append(
                    f"Prediction status audit {column} toplamı sıfır değil"
                )
        if (matched["total_rows"] <= 0).any():
            errors.append("Prediction status audit boş matched prediction içeriyor")
        status_total = prediction_audit[
            ["ok", "empty_mask", "missing_bbox", "inference_error"]
        ].sum(axis=1)
        if not (status_total == prediction_audit["total_rows"]).all():
            errors.append("Prediction status audit durum toplamları satır sayısıyla uyuşmuyor")
    training_health_path = analysis_root / "training_health_audit.csv"
    if training_health_path.is_file() and training_health_path.stat().st_size:
        training_health = pd.read_csv(training_health_path)
        expected_training_runs = 2 * 3
        if len(training_health) != expected_training_runs:
            errors.append(
                "Training health audit altı detector koşulu içermiyor: "
                f"{len(training_health)}"
            )
        if (
            training_health["dataset_id"].nunique() != 2
            or training_health["seed"].nunique() != 3
        ):
            errors.append("Training health audit dataset/seed matrisi eksik")
        finite_flags = training_health["final_core_metrics_finite"]
        if not pd.api.types.is_bool_dtype(finite_flags):
            finite_flags = (
                finite_flags.astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False})
            )
        if finite_flags.isna().any() or not finite_flags.all():
            errors.append("Training health audit final detector metriği geçersiz")
        if (training_health["epochs_completed"] != 100).any():
            errors.append("Training health audit bütün koşullarda 100 epok değil")
        for column in (
            "final_precision",
            "final_recall",
            "final_ap50",
            "final_ap50_95",
        ):
            values = pd.to_numeric(training_health[column], errors="coerce")
            if values.isna().any() or not values.between(0.0, 1.0).all():
                errors.append(
                    f"Training health audit {column} [0,1] dışında"
                )

    shared_root = analysis_root / "shared_human_reference_audit"
    shared_manifest = require_completed_manifest(
        shared_root / "manifest.json",
        label="Shared human-reference audit manifest",
        required=required,
        errors=errors,
    )
    if shared_manifest is not None:
        if int(shared_manifest.get("parameters", {}).get("bootstrap_samples", 0)) != 10_000:
            errors.append("Shared human-reference audit 10.000 bootstrap değil")
        shared_counts = shared_manifest.get("counts", {})
        if int(shared_counts.get("matched_instances", 0)) != 1033:
            errors.append("Shared human-reference audit eşleşen instance sayısı 1.033 değil")
        if int(shared_counts.get("unique_human_objects", 0)) != 770:
            errors.append(
                "Shared human-reference audit benzersiz insan nesnesi sayısı "
                "770 değil"
            )
        mapped_images = int(shared_counts.get("mapped_images", 0))
        pixel_exact_images = int(shared_counts.get("pixel_exact_images", 0))
        if mapped_images != 126:
            errors.append(
                "Shared human-reference audit eşleşen görüntü sayısı 126 değil"
            )
        if pixel_exact_images != mapped_images:
            errors.append(
                "Shared human-reference audit eşleşmelerinin tamamı piksel düzeyinde "
                "birebir aynı değil"
            )
    for name in (
        "reference_quality_summary.csv",
        "model_dual_reference_summary.csv",
        "model_reference_inflation_ci.json",
        "ranking_comparison.json",
        "unique_human_object_sensitivity.csv",
    ):
        require_file(
            shared_root / name,
            label=f"Shared audit {name}",
            required=required,
            errors=errors,
        )

    for name in FIGURE_FILES:
        require_file(
            study_root / "figures" / name,
            label=f"Figure {name}",
            required=required,
            errors=errors,
        )

    for name in PAPER_FILES:
        require_file(
            paper_dir / name,
            label=f"Paper output {name}",
            required=required,
            errors=errors,
        )
    paper_manifest_path = paper_dir / "paper_manifest.json"
    if paper_manifest_path.is_file():
        paper_manifest = read_json(paper_manifest_path)
        if paper_manifest.get("status") != "completed":
            errors.append("Paper manifest final completed durumunda değil")
        if paper_manifest.get("completion_issues"):
            errors.append("Paper manifest çözülmemiş completion issue içeriyor")

    pdf_path = paper_dir / "teacher_reference_bias_paper_6pages.pdf"
    if pdf_path.is_file():
        completed = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        pages = next(
            (
                int(line.split(":", 1)[1].strip())
                for line in completed.stdout.splitlines()
                if line.startswith("Pages:")
            ),
            0,
        )
        if pages != 6:
            errors.append(f"Paper PDF 6 sayfa değil: {pages}")

    docx_path = paper_dir / "teacher_reference_bias_paper.docx"
    if docx_path.is_file():
        with zipfile.ZipFile(docx_path) as archive:
            corrupt = archive.testzip()
        if corrupt is not None:
            errors.append(f"DOCX zip bütünlüğü bozuk: {corrupt}")


def validate_full_metric_reports(
    *,
    required: list[Path],
    errors: list[str],
) -> None:
    reports_root = STUDY_ROOT / "reports" / "full_metrics"
    strata = (
        "overall",
        "no_overlap__low_mask_area",
        "no_overlap__high_mask_area",
        "overlap__low_mask_area",
        "overlap__high_mask_area",
    )
    metric_columns = (
        "Avg IoU",
        "Avg Dice",
        "Avg Precision",
        "Avg Recall",
        "IoU ≥ 0.50",
        "IoU ≥ 0.75",
        "IoU ≥ 0.90",
    )
    detector_columns = (
        "BBox mAP50",
        "BBox mAP75",
        "BBox mAP90",
        "BBox mAP50-95",
        "BBox Precision@0.50",
        "BBox Recall@0.50",
        "BBox Precision@0.75",
        "BBox Recall@0.75",
        "BBox Precision@0.90",
        "BBox Recall@0.90",
    )
    for dataset_id, definition in FULL_METRIC_REPORTS.items():
        slug = str(definition["slug"])
        report_dir = reports_root / slug
        markdown_path = report_dir / f"{slug}_full_metric_document.md"
        docx_path = report_dir / f"{slug}_full_metric_document_colored.docx"
        pdf_path = report_dir / f"{slug}_full_metric_document_colored.pdf"
        manifest_path = report_dir / "report_manifest.json"
        for path, label in (
            (markdown_path, "Markdown"),
            (docx_path, "DOCX"),
            (pdf_path, "PDF"),
            (manifest_path, "manifest"),
        ):
            require_file(
                path,
                label=f"{dataset_id} tam metrik {label}",
                required=required,
                errors=errors,
            )

        tables_dir = report_dir / "tables"
        expected_tables = {
            tables_dir / "detector_summary.csv",
            tables_dir / "reference_sensitivity.csv",
            *{
                tables_dir / f"{reference_type}__{stratum}.csv"
                for reference_type in definition["reference_types"]
                for stratum in strata
            },
        }
        if len(expected_tables) != int(definition["table_count"]):
            errors.append(
                f"{dataset_id} tam metrik tablo tanımı beklenmeyen sayıda"
            )
        for table_path in sorted(expected_tables):
            require_file(
                table_path,
                label=f"{dataset_id} tam metrik tablo {table_path.name}",
                required=required,
                errors=errors,
            )
            if "__" not in table_path.name or not table_path.is_file():
                continue
            table = pd.read_csv(table_path)
            if len(table) != 6:
                errors.append(
                    f"{dataset_id}/{table_path.name} altı pipeline satırı "
                    f"içermiyor: {len(table)}"
                )
            missing_columns = sorted(set(metric_columns) - set(table.columns))
            if missing_columns:
                errors.append(
                    f"{dataset_id}/{table_path.name} metrik kolonları eksik: "
                    f"{missing_columns}"
                )
            for column in metric_columns:
                if column not in table:
                    continue
                values = pd.to_numeric(
                    table[column].astype(str).str.split("±").str[0].str.strip(),
                    errors="coerce",
                )
                if values.isna().any() or not values.between(0.0, 1.0).all():
                    errors.append(
                        f"{dataset_id}/{table_path.name}/{column} geçersiz"
                    )

        detector_table_path = tables_dir / "detector_summary.csv"
        if detector_table_path.is_file():
            detector_table = pd.read_csv(detector_table_path)
            missing_detector_columns = sorted(
                set(detector_columns) - set(detector_table.columns)
            )
            if missing_detector_columns:
                errors.append(
                    f"{dataset_id} detector tablosu kolonları eksik: "
                    f"{missing_detector_columns}"
                )
            for column in detector_columns:
                if column not in detector_table:
                    continue
                values = pd.to_numeric(
                    detector_table[column]
                    .astype(str)
                    .str.split("±")
                    .str[0]
                    .str.strip(),
                    errors="coerce",
                )
                if (
                    values.isna().any()
                    or not values.between(0.0, 1.0).all()
                ):
                    errors.append(
                        f"{dataset_id}/detector_summary.csv/{column} geçersiz"
                    )

        if markdown_path.is_file():
            markdown = markdown_path.read_text(encoding="utf-8")
            for forbidden in (
                "mAP proxy",
                "Boundary IoU",
                "Success@IoU",
                "| n |",
                "| Sahne |",
                "| Tekrar |",
            ):
                if forbidden in markdown:
                    errors.append(
                        f"{dataset_id} tam metrik raporunda yasaklı ifade var: "
                        f"{forbidden}"
                    )
            for heading in (
                "Overall",
                "No Overlap × Low Mask Area",
                "No Overlap × High Mask Area",
                "Overlap × Low Mask Area",
                "Overlap × High Mask Area",
            ):
                if heading not in markdown:
                    errors.append(
                        f"{dataset_id} tam metrik raporunda bölüm eksik: {heading}"
                    )

        qualitative_dir = report_dir / "qualitative"
        expected_qualitative = {
            qualitative_dir / f"{stratum}.png"
            for stratum in strata
            if stratum != "overall"
        }
        for image_path in sorted(expected_qualitative):
            require_file(
                image_path,
                label=(
                    f"{dataset_id} tam metrik nitel örnek "
                    f"{image_path.name}"
                ),
                required=required,
                errors=errors,
            )

        if docx_path.is_file():
            with zipfile.ZipFile(docx_path) as archive:
                corrupt = archive.testzip()
                document_xml = archive.read("word/document.xml")
            if corrupt is not None:
                errors.append(
                    f"{dataset_id} tam metrik DOCX zip bozuk: {corrupt}"
                )
            table_count = document_xml.count(b"<w:tbl>")
            if table_count != int(definition["table_count"]):
                errors.append(
                    f"{dataset_id} tam metrik DOCX tablo sayısı "
                    f"{table_count}, beklenen {definition['table_count']}"
                )

        if pdf_path.is_file():
            completed = subprocess.run(
                ["pdfinfo", str(pdf_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            pages = next(
                (
                    int(line.split(":", 1)[1].strip())
                    for line in completed.stdout.splitlines()
                    if line.startswith("Pages:")
                ),
                0,
            )
            if pages < int(definition["min_pdf_pages"]):
                errors.append(
                    f"{dataset_id} tam metrik PDF {pages} sayfa; "
                    "beklenen asgari yapı "
                    f"{definition['min_pdf_pages']} sayfa"
                )

        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if manifest.get("dataset_id") != dataset_id:
                errors.append(
                    f"{dataset_id} tam metrik manifest dataset kimliği yanlış"
                )
            if tuple(manifest.get("strata", ())) != strata:
                errors.append(
                    f"{dataset_id} tam metrik manifest stratum sırası yanlış"
                )
            if (
                manifest.get("report_format")
                != "legacy_samrs_full_metric_colored"
            ):
                errors.append(
                    f"{dataset_id} tam metrik manifest rapor biçimi yanlış"
                )
            for category in ("inputs", "outputs"):
                rows = manifest.get(category, {})
                if not isinstance(rows, dict) or not rows:
                    errors.append(
                        f"{dataset_id} tam metrik manifest {category} boş"
                    )
                    continue
                for raw_path, expected_hash in rows.items():
                    path = Path(str(raw_path))
                    if not path.is_file():
                        errors.append(
                            f"{dataset_id} tam metrik manifest dosyası eksik: "
                            f"{path}"
                        )
                        continue
                    required.append(path)
                    if sha256_file(path) != expected_hash:
                        errors.append(
                            f"{dataset_id} tam metrik manifest hash'i değişmiş: "
                            f"{path}"
                        )


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Sonuç yok._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def dataset_split_table(datasets: list[DatasetStudyConfig]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for split in ("train", "validation", "test"):
            split_root = dataset.prepared_root / split
            coco = read_json(split_root / "_annotations.coco.json")
            metadata = pd.read_csv(split_root / "metadata.csv")
            rows.append(
                {
                    "Veri": dataset.dataset_id,
                    "Split": split,
                    "Görüntü": len(coco["images"]),
                    "Instance": len(coco["annotations"]),
                    "Kaynak sahne": metadata["source_scene_id"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def write_appendix(
    path: Path,
    *,
    protocol_path: Path,
    protocol: MatchedStudyConfig,
    datasets: list[DatasetStudyConfig],
    study_root: Path,
    reproducibility_manifest: Path,
) -> None:
    detector = pd.read_csv(study_root / "analysis" / "detector_seed_summary.csv")
    segmentation = pd.read_csv(
        study_root / "analysis" / "segmentation_seed_summary.csv"
    )
    prediction_status = pd.read_csv(
        study_root / "analysis" / "prediction_status_audit.csv"
    )
    training_health = pd.read_csv(
        study_root / "analysis" / "training_health_audit.csv"
    )
    prediction_status_summary = (
        prediction_status[
            prediction_status["row_kind"] == "matched_ground_truth"
        ]
        .groupby("bbox_source", as_index=False)[
            [
                "total_rows",
                "ok",
                "empty_mask",
                "missing_bbox",
                "inference_error",
            ]
        ]
        .sum()
        .rename(
            columns={
                "bbox_source": "Bbox kaynağı",
                "total_rows": "Toplam",
                "ok": "Başarılı",
                "empty_mask": "Boş maske",
                "missing_bbox": "Eksik bbox",
                "inference_error": "Inference hatası",
            }
        )
    )
    dataset_rows = pd.DataFrame(
        [
            {
                "Veri": dataset.dataset_id,
                "Sürüm": dataset.version,
                "Referans": dataset.reference_type.value,
                "Sınıf": dataset.target_category,
                "Arşiv SHA-256": dataset.archive_sha256 or "configte yok",
                "Prepared tree SHA-256": read_json(
                    dataset.prepared_root / "content_manifest.json"
                )["tree_sha256"],
                "Detector train tree SHA-256": read_json(
                    dataset.prepared_root
                    / "detector_training_content_manifest.json"
                )["tree_sha256"],
            }
            for dataset in datasets
        ]
    )
    model_audit = read_json(
        study_root / "audits" / "segmenter_provenance.json"
    )
    model_rows = pd.DataFrame(
        [
            {
                "Model": row["model"],
                "Model kimliği": row["model_id"],
                "Revision": row["revision"],
                "Checkpoint SHA-256": row["actual_sha256"],
                "Processor/config tree SHA-256": row[
                    "configuration_tree_sha256"
                ],
            }
            for row in model_audit["models"]
        ]
    )
    rle_migration = read_json(
        study_root / "audits" / "isaid_lossless_rle_migration.json"
    )
    detector_repair = read_json(
        study_root
        / "audits"
        / "legacy_detector_manifest_repair"
        / "manifest.json"
    )
    repaired_detector_manifests = sum(
        row.get("action") == "repaired_with_archived_original"
        for row in detector_repair["rows"]
    )
    unchanged_detector_manifests = sum(
        row.get("action") == "unchanged_start_fingerprinted"
        for row in detector_repair["rows"]
    )
    text = f"""# Reproducibility Appendix

## Durum

Bu ek, `teacher_reference_bias_v1` çalışmasının final QA kapısından sonra
otomatik üretilmiştir.

- Üretim zamanı: `{datetime.now(timezone.utc).isoformat()}`
- Frozen protokol: `{protocol_path.relative_to(ROOT)}`
- Protokol kimliği: `{protocol.study_id}`
- Görüntü boyutu: `{protocol.image_size}×{protocol.image_size}`
- Detector seed'leri: `{", ".join(str(seed) for seed in protocol.detector_seeds)}`
- Bootstrap: `{protocol.evaluation["bootstrap_samples"]}`
- Tam hash envanteri: `{reproducibility_manifest.relative_to(ROOT)}`

## Veri Kaynakları

{markdown_table(dataset_rows)}

SAMRS arşiv kimliği numeric class ID, RBox ve RHBox geometri düzeyinde resmi
detection anotasyonlarıyla exhaustive olarak doğrulanmıştır. Pickle içindeki
metin kategori alanı authoritative kaynak değildir.

## Veri ve Manifest Migration Kayıtları

- iSAID insan maskeleri resmi polygonlardan OpenCV rastera ve kayıpsız
  compressed COCO RLE'ye geçirilmiştir. Train/validation/test için migration
  sonrası boş maske ve decoded-area uyuşmazlığı sıfırdır.
- Migration audit durumu: `{rle_migration["status"]}`.
- Detector eğitim girdisi yalnız train/validation görüntü ve YOLO label
  ağacıdır; test split'i ile segmentation maskeleri bu kapsama girmez.
- Start/finish fingerprint şemasından önce başlayan
  `{repaired_detector_manifests}` detector manifesti, byte düzeyindeki özgün
  manifest arşivi korunarak açık provenance repair auditinden geçirilmiştir.
- Başlangıç fingerprint'i zaten bulunan ve değişmeden bırakılan detector
  manifesti sayısı: `{unchanged_detector_manifests}`.
- Finalizer run manifestlerini değiştirmez; input drift veya kopuk repair
  zinciri final hatasıdır.

## Model Kimlikleri

{markdown_table(model_rows)}

## Split Özeti

{markdown_table(dataset_split_table(datasets))}

Split birimi tile değil kaynak sahnedir. Train, validation ve test kaynak
sahne kesişimi iki veri setinde de sıfırdır. Testte dört
`overlap × mask area` katmanının her birinde 32 görüntü vardır.

## Deney Matrisi

- GT-bbox: 2 veri seti × 3 SAM modeli = 6 koşul.
- YOLO-bbox: 2 veri seti × 3 detector seed × 3 SAM modeli = 18 koşul.
- iSAID değerlendirmesi: aynı tahmin üzerinde insan + SAM1 pseudo referans.
- SAMRS değerlendirmesi: resmi SAM1 pseudo referansı.
    - Ortak görüntü denetimi: 126 görüntü, 1.033 tile-instance görünümü,
      770 benzersiz insan-anotasyonlu uçak ve 35 kaynak sahne.

## Detector Özeti

{markdown_table(detector)}

## Detector Eğitim Sağlığı

Ara validation loss kayıtlarındaki geçici non-finite hücreler ayrıca
gösterilir; paylaşılabilir sonuç için son precision, recall ve AP değerlerinin
tamamı sonlu ve `[0,1]` aralığında olmalıdır.

{markdown_table(training_health)}

## YOLO-bbox Segmentation Özeti

{markdown_table(segmentation)}

## Prediction Durum Denetimi

{markdown_table(prediction_status_summary)}

## Yeniden Üretim Sırası

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py preflight \\
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml \\
  --dataset studies/teacher_reference_bias_v1/configs/datasets/samrs_sota_plane.yaml

.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py analyze
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py figures
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py paper
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py finalize
```

Model, veri seti ve seed bazındaki ayrıntılı stage komutları `README.md`
içindedir. Her final run manifesti resolved config hash'i ile giriş/çıkış dosya
hash'lerini içerir.

## Değerlendirme Notu

YOLO-bbox instance tablosunda eşleşmeyen GT boş maskeyle sıfır skor alır.
Eşleşmeyen detector tahmini detector AP hesabında false positive ve
image-level union maskesinde tahmin olarak korunur. Instance mask tablosu COCO
mask AP değildir.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    protocol_path = project_path(args.protocol)
    protocol = load_matched_study_config(protocol_path)
    dataset_paths = [project_path(path) for path in (args.dataset or DEFAULT_DATASETS)]
    datasets = [load_dataset_study_config(path) for path in dataset_paths]
    study_root = STUDY_ROOT / "results"
    paper_dir = project_path(args.paper_dir)
    appendix_path = project_path(args.appendix)

    required: list[Path] = [protocol_path, *dataset_paths]
    errors: list[str] = []
    for documentation_path in (
        ROOT / "README.md",
        ROOT / "REPORT.md",
        ROOT / "requirements.txt",
        ROOT / "docs" / "REFACTOR_PLAN.md",
        ROOT / "docs" / "REPOSITORY_ARCHITECTURE.md",
        ROOT / "docs" / "WORKLOG.md",
        ROOT / "docs" / "LEGACY_STATUS.md",
        STUDY_ROOT / "docs" / "EXPERIMENT_PLAN.md",
        STUDY_ROOT / "docs" / "LITERATURE_REVIEW.md",
    ):
        require_file(
            documentation_path,
            label=f"Çalışma dokümanı {documentation_path.name}",
            required=required,
            errors=errors,
        )
    validate_repository_layout_migration(
        study_root,
        required=required,
        errors=errors,
    )
    validate_completed_run_manifest_fingerprints(
        study_root,
        required=required,
        errors=errors,
    )
    require_file(
        study_root / "preflight.json",
        label="Preflight report",
        required=required,
        errors=errors,
    )
    preflight_path = study_root / "preflight.json"
    if preflight_path.is_file() and read_json(preflight_path).get("status") != "pass":
        errors.append("Preflight status pass değil")
    validate_isaid_rle_migration(
        study_root,
        datasets,
        required=required,
        errors=errors,
    )
    validate_isaid_rle_sensitivity_and_archive(
        study_root,
        datasets,
        required=required,
        errors=errors,
    )
    validate_legacy_detector_manifest_repair(
        study_root,
        protocol,
        datasets,
        required=required,
        errors=errors,
    )
    segmenter_provenance_path = (
        study_root / "audits" / "segmenter_provenance.json"
    )
    require_file(
        segmenter_provenance_path,
        label="Segmenter model provenance audit",
        required=required,
        errors=errors,
    )
    if (
        segmenter_provenance_path.is_file()
        and segmenter_provenance_path.stat().st_size
    ):
        model_provenance = read_json(segmenter_provenance_path)
        if model_provenance.get("status") != "pass":
            errors.append("Segmenter model provenance audit pass değil")
        if model_provenance.get("protocol_sha256") != sha256_file(
            protocol_path
        ):
            errors.append(
                "Segmenter model provenance audit frozen protokolle uyuşmuyor"
            )
        for row in model_provenance.get("models", []):
            checkpoint_path = Path(str(row.get("checkpoint_path", "")))
            if not checkpoint_path.is_file():
                errors.append(
                    f"{row.get('model')} checkpoint dosyası bulunamadı: "
                    f"{checkpoint_path}"
                )
                continue
            if sha256_file(checkpoint_path) != row.get("actual_sha256"):
                errors.append(
                    f"{row.get('model')} checkpoint hash'i provenance "
                    "auditinden sonra değişmiş"
                )
            configuration_files = row.get("configuration_files")
            if not isinstance(configuration_files, list):
                errors.append(
                    f"{row.get('model')} configuration provenance eksik"
                )
                continue
            actual_configuration_hash = configuration_tree_sha256(
                Path(str(row.get("snapshot_path", ""))),
                configuration_files,
            )
            if (
                not actual_configuration_hash
                or actual_configuration_hash
                != row.get("configuration_tree_sha256")
            ):
                errors.append(
                    f"{row.get('model')} processor/config tree hash'i "
                    "provenance auditinden sonra değişmiş"
                )
    prediction_parity_path = (
        study_root / "audits" / "pinned_revision_prediction_parity.json"
    )
    pre_assignment_fix_path = (
        study_root
        / "audits"
        / "pre_sam3_global_assignment_fix.json"
    )
    require_file(
        pre_assignment_fix_path,
        label="SAM3 global assignment düzeltmesi öncesi parity snapshot",
        required=required,
        errors=errors,
    )
    require_file(
        prediction_parity_path,
        label="Pinned revision prediction parity audit",
        required=required,
        errors=errors,
    )
    if prediction_parity_path.is_file() and prediction_parity_path.stat().st_size:
        prediction_parity = read_json(prediction_parity_path)
        if prediction_parity.get("status") != "pass":
            errors.append("Pinned revision prediction parity audit pass değil")
        if prediction_parity.get("protocol_sha256") != sha256_file(
            protocol_path
        ):
            errors.append(
                "Pinned revision prediction parity audit frozen protokolle "
                "uyuşmuyor"
            )
        for row in prediction_parity.get("conditions", []):
            if row.get("exact_mask_parity") is not True:
                errors.append(
                    "Pinned revision prediction parity sağlanmadı: "
                    f"{row.get('dataset_id')}/{row.get('model')}"
                )
        if (
            pre_assignment_fix_path.is_file()
            and pre_assignment_fix_path.stat().st_size
        ):
            pre_assignment_fix = read_json(pre_assignment_fix_path)
            if pre_assignment_fix.get("protocol_sha256") != sha256_file(
                protocol_path
            ):
                errors.append(
                    "SAM3 assignment düzeltmesi öncesi snapshot frozen "
                    "protokolle uyuşmuyor"
                )
            old_by_key = {
                (str(row["dataset_id"]), str(row["model"])): row["baseline"]
                for row in pre_assignment_fix.get("conditions", [])
            }
            new_by_key = {
                (str(row["dataset_id"]), str(row["model"])): row["baseline"]
                for row in prediction_parity.get("conditions", [])
            }
            if set(old_by_key) != set(new_by_key):
                errors.append(
                    "SAM3 assignment düzeltmesi öncesi ve sonrası parity "
                    "koşulları uyuşmuyor"
                )
            else:
                for key in sorted(old_by_key):
                    old_hash = old_by_key[key].get(
                        "canonical_mask_sha256"
                    )
                    new_hash = new_by_key[key].get(
                        "canonical_mask_sha256"
                    )
                    if key[1] == "sam3" and old_hash == new_hash:
                        errors.append(
                            "SAM3 global assignment düzeltmesi tahmin hash'ini "
                            f"değiştirmemiş: {key[0]}"
                        )
                    if key[1] != "sam3" and old_hash != new_hash:
                        errors.append(
                            "SAM3 global assignment düzeltmesi başka modeli "
                            f"değiştirmiş: {key[0]}/{key[1]}"
                        )

    expected_pipeline_files(
        study_root,
        protocol,
        datasets,
        required=required,
        errors=errors,
    )
    validate_detector_training_args_matrix(
        study_root,
        protocol,
        datasets,
        required=required,
        errors=errors,
    )
    validate_analysis_and_paper(
        study_root,
        paper_dir,
        protocol,
        datasets,
        required=required,
        errors=errors,
    )
    validate_full_metric_reports(
        required=required,
        errors=errors,
    )
    if errors:
        raise RuntimeError(
            "Final study QA başarısız:\n- " + "\n- ".join(errors)
        )

    reproducibility_manifest = study_root / "reproducibility_manifest.json"
    write_appendix(
        appendix_path,
        protocol_path=protocol_path,
        protocol=protocol,
        datasets=datasets,
        study_root=study_root,
        reproducibility_manifest=reproducibility_manifest,
    )
    required.append(appendix_path)

    code_files = sorted(
        {
            *ROOT.joinpath("src", "yolo_sam").rglob("*.py"),
            *STUDY_ROOT.joinpath("src", "teacher_reference_bias").rglob("*.py"),
            *STUDY_ROOT.joinpath("scripts").glob("*.py"),
            *STUDY_ROOT.joinpath("tests").rglob("*.py"),
            *ROOT.joinpath("tests").rglob("*.py"),
        }
    )
    inventory_files = sorted(set(required) | set(code_files))
    payload = {
        "schema_version": 1,
        "status": "completed",
        "study_id": protocol.study_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_snapshot(ROOT),
        "counts": {
            "files": len(inventory_files),
            "datasets": len(datasets),
            "detector_runs": len(datasets) * len(protocol.detector_seeds),
            "gt_bbox_segmentations": len(datasets) * len(MODELS),
            "yolo_bbox_segmentations": (
                len(datasets) * len(protocol.detector_seeds) * len(MODELS)
            ),
        },
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in inventory_files
        ],
    }
    reproducibility_manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(reproducibility_manifest)
    print(appendix_path)


if __name__ == "__main__":
    main()
