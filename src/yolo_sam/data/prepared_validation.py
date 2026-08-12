from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pycocotools import mask as mask_utils

from yolo_sam.data.contracts import BBoxSource, ReferenceType


EXPECTED_TEST_STRATA = frozenset(
    {
        "no_overlap__low_mask_area",
        "no_overlap__high_mask_area",
        "overlap__low_mask_area",
        "overlap__high_mask_area",
    }
)


def _segmentation_area(
    segmentation: object,
    *,
    height: int,
    width: int,
) -> int:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        encoded = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        encoded = segmentation
    else:
        raise ValueError(
            f"Unsupported COCO segmentation type: {type(segmentation).__name__}"
        )
    return int(mask_utils.area(encoded))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_content_files(
    dataset_root: Path,
    *,
    splits: tuple[str, ...],
) -> set[Path]:
    files: set[Path] = set()
    for name in (
        "data.yaml",
        "source_scene_split.csv",
        "source_scene_split.json",
        "split_manifest.json",
    ):
        path = dataset_root / name
        if path.is_file():
            files.add(path)
    for split in splits:
        split_root = dataset_root / split
        for name in ("_annotations.coco.json", "metadata.csv"):
            path = split_root / name
            if path.is_file():
                files.add(path)
        for directory_name in ("images", "labels"):
            directory = split_root / directory_name
            if directory.is_dir():
                files.update(path for path in directory.rglob("*") if path.is_file())
    return files


def _detector_training_content_files(
    dataset_root: Path,
    *,
    splits: tuple[str, ...],
) -> set[Path]:
    files: set[Path] = set()
    data_yaml = dataset_root / "data.yaml"
    if data_yaml.is_file():
        files.add(data_yaml)
    for split in splits:
        split_root = dataset_root / split
        for directory_name in ("images", "labels"):
            directory = split_root / directory_name
            if directory.is_dir():
                files.update(
                    path for path in directory.rglob("*") if path.is_file()
                )
    return files


def _build_file_tree_manifest(
    dataset_root: Path,
    *,
    files: set[Path],
    splits: tuple[str, ...],
    scope: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    tree_digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(files):
        relative_path = path.relative_to(dataset_root).as_posix()
        size = path.stat().st_size
        digest = _sha256_file(path)
        total_bytes += size
        tree_digest.update(
            f"{relative_path}\0{size}\0{digest}\n".encode("utf-8")
        )
        entries.append(
            {
                "path": relative_path,
                "bytes": size,
                "sha256": digest,
            }
        )
    return {
        "schema_version": 1,
        "status": "completed",
        "scope": scope,
        # Manifest dosyası veri kümesi kökünde tutulur; mutlak makine yolu
        # taşınabilir değildir ve içerik doğrulaması için gerekli değildir.
        "dataset_root": ".",
        "splits": list(splits),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
        "files": entries,
    }


def build_prepared_content_manifest(
    dataset_root: Path,
    *,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    files = _prepared_content_files(dataset_root, splits=splits)
    return _build_file_tree_manifest(
        dataset_root,
        files=files,
        splits=splits,
        scope="full_prepared_dataset",
    )


def build_detector_training_content_manifest(
    dataset_root: Path,
    *,
    splits: tuple[str, ...] = ("train", "validation"),
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    files = _detector_training_content_files(
        dataset_root,
        splits=splits,
    )
    return _build_file_tree_manifest(
        dataset_root,
        files=files,
        splits=splits,
        scope="detector_training_images_and_labels",
    )


def validate_prepared_content_manifest(
    dataset_root: Path,
    manifest: dict[str, Any],
) -> list[str]:
    dataset_root = dataset_root.resolve()
    errors: list[str] = []
    if manifest.get("status") != "completed":
        errors.append("content manifest status completed değil")
    if manifest.get("scope") != "full_prepared_dataset":
        errors.append("content manifest scope geçersiz")
    seen: set[str] = set()
    tree_digest = hashlib.sha256()
    total_bytes = 0
    for entry in manifest.get("files", []):
        relative_path = str(entry["path"])
        if relative_path in seen:
            errors.append(f"tekrarlı content manifest yolu: {relative_path}")
            continue
        seen.add(relative_path)
        path = dataset_root / relative_path
        if not path.is_file():
            errors.append(f"content manifest dosyası eksik: {relative_path}")
            continue
        size = path.stat().st_size
        digest = _sha256_file(path)
        if size != int(entry["bytes"]) or digest != str(entry["sha256"]):
            errors.append(f"content manifest hash uyuşmazlığı: {relative_path}")
        total_bytes += size
        tree_digest.update(
            f"{relative_path}\0{size}\0{digest}\n".encode("utf-8")
        )
    splits = tuple(str(value) for value in manifest.get("splits", []))
    actual_paths = {
        path.relative_to(dataset_root).as_posix()
        for path in _prepared_content_files(dataset_root, splits=splits)
    }
    unexpected = sorted(actual_paths - seen)
    missing = sorted(seen - actual_paths)
    if unexpected:
        errors.append(
            "content manifestte olmayan dosyalar var: "
            + ", ".join(unexpected[:10])
        )
    if missing:
        errors.append(
            "content manifestte kayıtlı dosyalar eksik: "
            + ", ".join(missing[:10])
        )
    if len(seen) != int(manifest.get("file_count", -1)):
        errors.append("content manifest file_count uyuşmuyor")
    if total_bytes != int(manifest.get("total_bytes", -1)):
        errors.append("content manifest total_bytes uyuşmuyor")
    if tree_digest.hexdigest() != str(manifest.get("tree_sha256")):
        errors.append("content manifest tree_sha256 uyuşmuyor")
    return errors


def validate_detector_training_content_manifest(
    dataset_root: Path,
    manifest: dict[str, Any],
) -> list[str]:
    dataset_root = dataset_root.resolve()
    errors: list[str] = []
    if manifest.get("status") != "completed":
        errors.append("detector content manifest status completed değil")
    if manifest.get("scope") != "detector_training_images_and_labels":
        errors.append("detector content manifest scope geçersiz")
    splits = tuple(str(value) for value in manifest.get("splits", []))
    if splits != ("train", "validation"):
        errors.append("detector content manifest split kapsamı geçersiz")
    expected_paths = {
        path.relative_to(dataset_root).as_posix()
        for path in _detector_training_content_files(
            dataset_root,
            splits=splits,
        )
    }
    seen: set[str] = set()
    tree_digest = hashlib.sha256()
    total_bytes = 0
    for entry in manifest.get("files", []):
        relative_path = str(entry["path"])
        if relative_path in seen:
            errors.append(
                f"tekrarlı detector content manifest yolu: {relative_path}"
            )
            continue
        seen.add(relative_path)
        path = dataset_root / relative_path
        if not path.is_file():
            errors.append(
                f"detector content manifest dosyası eksik: {relative_path}"
            )
            continue
        size = path.stat().st_size
        digest = _sha256_file(path)
        if size != int(entry["bytes"]) or digest != str(entry["sha256"]):
            errors.append(
                f"detector content manifest hash uyuşmazlığı: {relative_path}"
            )
        total_bytes += size
        tree_digest.update(
            f"{relative_path}\0{size}\0{digest}\n".encode("utf-8")
        )
    unexpected = sorted(expected_paths - seen)
    missing = sorted(seen - expected_paths)
    if unexpected:
        errors.append(
            "detector content manifestte olmayan dosyalar var: "
            + ", ".join(unexpected[:10])
        )
    if missing:
        errors.append(
            "detector content manifestte kayıtlı dosyalar eksik: "
            + ", ".join(missing[:10])
        )
    if len(seen) != int(manifest.get("file_count", -1)):
        errors.append("detector content manifest file_count uyuşmuyor")
    if total_bytes != int(manifest.get("total_bytes", -1)):
        errors.append("detector content manifest total_bytes uyuşmuyor")
    if tree_digest.hexdigest() != str(manifest.get("tree_sha256")):
        errors.append("detector content manifest tree_sha256 uyuşmuyor")
    return errors


@dataclass(frozen=True)
class PreparedValidationFinding:
    code: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedDatasetValidation:
    dataset_root: str
    split_summaries: dict[str, dict[str, Any]]
    findings: list[PreparedValidationFinding]

    @property
    def passed(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "passed": self.passed,
            "split_summaries": self.split_summaries,
            "findings": [asdict(finding) for finding in self.findings],
        }


def _error(
    findings: list[PreparedValidationFinding],
    code: str,
    message: str,
    **details: Any,
) -> None:
    findings.append(
        PreparedValidationFinding(
            code=code,
            severity="error",
            message=message,
            details=details,
        )
    )


def _load_split(
    dataset_root: Path,
    split: str,
    findings: list[PreparedValidationFinding],
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    split_root = dataset_root / split
    annotation_path = split_root / "_annotations.coco.json"
    metadata_path = split_root / "metadata.csv"
    missing = [
        str(path)
        for path in (
            annotation_path,
            metadata_path,
            split_root / "images",
            split_root / "labels",
        )
        if not path.exists()
    ]
    if missing:
        _error(
            findings,
            "PREPARED_SPLIT_FILES_MISSING",
            f"Prepared split {split!r} is incomplete.",
            split=split,
            missing=missing,
        )
        return None
    return (
        json.loads(annotation_path.read_text(encoding="utf-8")),
        pd.read_csv(metadata_path),
    )


def _validate_split(
    dataset_root: Path,
    split: str,
    coco: dict[str, Any],
    metadata: pd.DataFrame,
    image_size: int,
    expected_reference_type: ReferenceType,
    expected_bbox_source: BBoxSource,
    findings: list[PreparedValidationFinding],
) -> dict[str, Any]:
    split_root = dataset_root / split
    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))
    image_ids = [int(image["id"]) for image in images]
    image_names = [str(image["file_name"]) for image in images]
    annotation_ids = [int(annotation["id"]) for annotation in annotations]

    if len(image_ids) != len(set(image_ids)) or len(image_names) != len(set(image_names)):
        _error(
            findings,
            "DUPLICATE_IMAGE_ID_OR_NAME",
            f"Split {split!r} contains duplicate COCO image IDs or names.",
            split=split,
        )
    if len(annotation_ids) != len(set(annotation_ids)):
        _error(
            findings,
            "DUPLICATE_ANNOTATION_ID",
            f"Split {split!r} contains duplicate COCO annotation IDs.",
            split=split,
        )

    metadata_ids = (
        {int(value) for value in metadata["image_id"].tolist()}
        if "image_id" in metadata
        else set()
    )
    metadata_names = (
        {str(value) for value in metadata["file_name"].tolist()}
        if "file_name" in metadata
        else set()
    )
    if set(image_ids) != metadata_ids or set(image_names) != metadata_names:
        _error(
            findings,
            "COCO_METADATA_MISMATCH",
            f"Split {split!r} COCO images and metadata rows differ.",
            split=split,
            coco_images=len(images),
            metadata_rows=len(metadata),
        )

    actual_image_names = {
        path.name for path in (split_root / "images").iterdir() if path.is_file()
    }
    expected_label_names = {Path(name).with_suffix(".txt").name for name in image_names}
    actual_label_names = {
        path.name for path in (split_root / "labels").iterdir() if path.is_file()
    }
    if set(image_names) != actual_image_names:
        _error(
            findings,
            "COCO_IMAGE_FILES_MISMATCH",
            f"Split {split!r} COCO records and image files differ.",
            split=split,
            missing=sorted(set(image_names) - actual_image_names)[:20],
            unexpected=sorted(actual_image_names - set(image_names))[:20],
        )
    if expected_label_names != actual_label_names:
        _error(
            findings,
            "YOLO_LABEL_FILES_MISMATCH",
            f"Split {split!r} image records and YOLO label files differ.",
            split=split,
            missing=sorted(expected_label_names - actual_label_names)[:20],
            unexpected=sorted(actual_label_names - expected_label_names)[:20],
        )

    image_id_set = set(image_ids)
    image_size_by_id = {
        int(image["id"]): (int(image["height"]), int(image["width"]))
        for image in images
    }
    annotation_counts: Counter[int] = Counter()
    invalid_boxes = 0
    invalid_provenance = 0
    invalid_masks = 0
    mask_area_mismatches = 0
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        annotation_counts[image_id] += 1
        if image_id not in image_id_set:
            _error(
                findings,
                "ORPHAN_COCO_ANNOTATION",
                f"Split {split!r} contains an annotation for an unknown image.",
                split=split,
                annotation_id=int(annotation["id"]),
                image_id=image_id,
            )
        x, y, width, height = (float(value) for value in annotation["bbox"])
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > image_size + 1e-6
            or y + height > image_size + 1e-6
        ):
            invalid_boxes += 1
        if (
            annotation.get("bbox_source") != expected_bbox_source.value
            or annotation.get("reference_type") != expected_reference_type.value
            or (
                expected_reference_type == ReferenceType.HUMAN
                and "source_annotation_id" not in annotation
            )
            or (
                expected_reference_type == ReferenceType.PSEUDO_SAM1
                and "source_label" not in annotation
            )
        ):
            invalid_provenance += 1
        try:
            height, width = image_size_by_id[image_id]
            decoded_area = _segmentation_area(
                annotation.get("segmentation"),
                height=height,
                width=width,
            )
            declared_area = int(annotation["area"])
            if decoded_area <= 0:
                invalid_masks += 1
            if decoded_area != declared_area:
                mask_area_mismatches += 1
        except (KeyError, TypeError, ValueError):
            invalid_masks += 1
    if invalid_boxes:
        _error(
            findings,
            "INVALID_OR_OUT_OF_BOUNDS_BBOX",
            f"Split {split!r} contains invalid or out-of-bounds bounding boxes.",
            split=split,
            count=invalid_boxes,
        )
    if invalid_provenance:
        _error(
            findings,
            "REFERENCE_PROVENANCE_MISMATCH",
            f"Split {split!r} contains annotations with unexpected reference or bbox provenance.",
            split=split,
            count=invalid_provenance,
            expected_reference_type=expected_reference_type.value,
            expected_bbox_source=expected_bbox_source.value,
        )
    if invalid_masks:
        _error(
            findings,
            "INVALID_OR_EMPTY_REFERENCE_MASK",
            f"Split {split!r} contains invalid or empty reference masks.",
            split=split,
            count=invalid_masks,
        )
    if mask_area_mismatches:
        _error(
            findings,
            "REFERENCE_MASK_AREA_MISMATCH",
            (
                f"Split {split!r} contains decoded masks whose pixel area "
                "differs from the declared COCO area."
            ),
            split=split,
            count=mask_area_mismatches,
        )

    bad_dimensions = sum(
        int(image["width"]) != image_size or int(image["height"]) != image_size
        for image in images
    )
    if bad_dimensions:
        _error(
            findings,
            "IMAGE_DIMENSION_MISMATCH",
            f"Split {split!r} contains records outside the frozen image size.",
            split=split,
            expected=image_size,
            count=bad_dimensions,
        )

    metadata_counts = {
        int(row["image_id"]): int(row["num_objects"])
        for _, row in metadata.iterrows()
    }
    count_mismatches = sum(
        metadata_counts.get(image_id) != annotation_counts.get(image_id, 0)
        for image_id in image_ids
    )
    if count_mismatches:
        _error(
            findings,
            "METADATA_ANNOTATION_COUNT_MISMATCH",
            f"Split {split!r} metadata object counts do not match COCO.",
            split=split,
            count=count_mismatches,
        )

    label_count_mismatches = 0
    invalid_yolo_rows = 0
    for image in images:
        image_id = int(image["id"])
        label_path = split_root / "labels" / Path(str(image["file_name"])).with_suffix(
            ".txt"
        ).name
        rows = [
            line.split()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != annotation_counts.get(image_id, 0):
            label_count_mismatches += 1
        for row in rows:
            if len(row) != 5:
                invalid_yolo_rows += 1
                continue
            try:
                class_id = int(row[0])
                coordinates = [float(value) for value in row[1:]]
            except ValueError:
                invalid_yolo_rows += 1
                continue
            if (
                class_id != 0
                or any(value < 0 or value > 1 for value in coordinates)
                or coordinates[2] <= 0
                or coordinates[3] <= 0
            ):
                invalid_yolo_rows += 1
    if label_count_mismatches or invalid_yolo_rows:
        _error(
            findings,
            "INVALID_YOLO_LABELS",
            f"Split {split!r} contains invalid or count-mismatched YOLO labels.",
            split=split,
            count_mismatched_files=label_count_mismatches,
            invalid_rows=invalid_yolo_rows,
        )

    return {
        "images": len(images),
        "positive_images": sum(count > 0 for count in annotation_counts.values()),
        "negative_images": len(images)
        - sum(count > 0 for count in annotation_counts.values()),
        "annotations": len(annotations),
        "source_scenes": int(metadata["source_scene_id"].nunique()),
    }


def validate_prepared_matched_dataset(
    dataset_root: Path,
    *,
    image_size: int,
    expected_test_per_stratum: int,
    overlap_threshold: float,
    expected_reference_type: ReferenceType = ReferenceType.HUMAN,
    expected_bbox_source: BBoxSource = BBoxSource.HUMAN_ANNOTATION,
    required_splits: tuple[str, ...] = ("train", "validation", "test"),
) -> PreparedDatasetValidation:
    dataset_root = dataset_root.resolve()
    findings: list[PreparedValidationFinding] = []
    summaries: dict[str, dict[str, Any]] = {}
    metadata_by_split: dict[str, pd.DataFrame] = {}

    for split in required_splits:
        loaded = _load_split(dataset_root, split, findings)
        if loaded is None:
            continue
        coco, metadata = loaded
        metadata_by_split[split] = metadata
        summaries[split] = _validate_split(
            dataset_root,
            split,
            coco,
            metadata,
            image_size,
            expected_reference_type,
            expected_bbox_source,
            findings,
        )

    scene_sets = {
        split: set(metadata["source_scene_id"].astype(str))
        for split, metadata in metadata_by_split.items()
    }
    split_names = sorted(scene_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = scene_sets[left] & scene_sets[right]
            if overlap:
                _error(
                    findings,
                    "SOURCE_SCENE_SPLIT_LEAKAGE",
                    "Prepared splits contain tiles from the same source scene.",
                    left_split=left,
                    right_split=right,
                    count=len(overlap),
                    examples=sorted(overlap)[:20],
                )

    test_metadata = metadata_by_split.get("test")
    if test_metadata is not None:
        if "stratum" not in test_metadata:
            _error(
                findings,
                "TEST_STRATUM_COLUMN_MISSING",
                "Test metadata has no stratum column.",
            )
        else:
            counts = {
                str(key): int(value)
                for key, value in test_metadata["stratum"].value_counts().items()
            }
            expected = {
                stratum: expected_test_per_stratum
                for stratum in EXPECTED_TEST_STRATA
            }
            if counts != expected:
                _error(
                    findings,
                    "TEST_STRATUM_COUNT_MISMATCH",
                    "Test split does not contain the frozen number of images per stratum.",
                    expected=expected,
                    actual=counts,
                )

            invalid_overlap_rows = 0
            invalid_area_rows = 0
            for _, row in test_metadata.iterrows():
                stratum = str(row["stratum"])
                pair_iou = float(row["max_pair_bbox_iou"])
                area_ratio = float(row["mask_area_ratio"])
                area_threshold = float(row["area_threshold"])
                if (
                    stratum.startswith("no_overlap__")
                    and pair_iou > 0.0
                ) or (
                    stratum.startswith("overlap__")
                    and pair_iou < overlap_threshold
                ):
                    invalid_overlap_rows += 1
                if (
                    stratum.endswith("__low_mask_area")
                    and area_ratio >= area_threshold
                ) or (
                    stratum.endswith("__high_mask_area")
                    and area_ratio < area_threshold
                ):
                    invalid_area_rows += 1
            if invalid_overlap_rows:
                _error(
                    findings,
                    "TEST_OVERLAP_STRATUM_MISMATCH",
                    "Some test rows violate the frozen overlap definition.",
                    count=invalid_overlap_rows,
                )
            if invalid_area_rows:
                _error(
                    findings,
                    "TEST_AREA_STRATUM_MISMATCH",
                    "Some test rows violate the frozen mask-area definition.",
                    count=invalid_area_rows,
                )

    return PreparedDatasetValidation(
        dataset_root=str(dataset_root),
        split_summaries=summaries,
        findings=findings,
    )
