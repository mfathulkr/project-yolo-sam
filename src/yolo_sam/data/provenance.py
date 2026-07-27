from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from yolo_sam.data.profiles import DatasetProfile, normalize_category_name


SOURCE_TILE_PATTERN = re.compile(r"^(?P<scene>.+)_\d{4}$")


@dataclass(frozen=True)
class AuditFinding:
    code: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetAuditReport:
    root: str
    profile_id: str
    annotation_format: str
    annotation_directory: str | None
    image_directory: str | None
    annotation_files: int
    image_files: int
    instances: int
    category_mapping: dict[str, list[str]]
    split_file_counts: dict[str, int]
    source_scene_counts: dict[str, int]
    findings: list[AuditFinding]

    @property
    def passed(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_scene_id(stem: str) -> str:
    match = SOURCE_TILE_PATTERN.match(stem)
    return match.group("scene") if match else stem


def find_single_directory(root: Path, names: Iterable[str]) -> Path | None:
    candidates: list[Path] = []
    for name in names:
        candidates.extend(path for path in root.rglob(name) if path.is_dir())
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0]
    return None


def read_split_stems(path: Path) -> list[str]:
    return [
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _category_mapping(
    pkl_paths: list[Path],
    expected_keys: frozenset[str],
) -> tuple[dict[int, set[str]], int, dict[str, int]]:
    category_mapping: dict[int, set[str]] = defaultdict(set)
    missing_key_counts: dict[str, int] = defaultdict(int)
    instances = 0

    for pkl_path in pkl_paths:
        with pkl_path.open("rb") as handle:
            records = pickle.load(handle)
        if not isinstance(records, list):
            missing_key_counts["invalid_pickle_payload"] += 1
            continue

        for record in records:
            instances += 1
            if not isinstance(record, dict):
                missing_key_counts["invalid_instance_record"] += 1
                continue
            for key in expected_keys:
                if key not in record:
                    missing_key_counts[key] += 1
            if "label" in record and "category" in record:
                category_mapping[int(record["label"])].add(str(record["category"]))

    return category_mapping, instances, dict(sorted(missing_key_counts.items()))


def _parse_authoritative_rdet_line(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[np.ndarray, str, int]:
    fields = line.split()
    if len(fields) != 10:
        raise ValueError(
            f"{path}:{line_number}: expected 10 fields, found {len(fields)}"
        )
    try:
        rbox = np.asarray([float(value) for value in fields[:8]], dtype=np.float64).reshape(4, 2)
        label = int(fields[9])
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid numeric coordinate or category ID"
        ) from exc
    return rbox, fields[8], label


def _validate_authoritative_rdet_labels(
    pkl_paths: list[Path],
    rdet_dir: Path,
) -> tuple[dict[int, set[str]], dict[str, Any], list[AuditFinding]]:
    findings: list[AuditFinding] = []
    mapping: dict[int, set[str]] = defaultdict(set)
    rdet_paths = sorted(rdet_dir.glob("*.txt"))
    pkl_by_stem = {path.stem: path for path in pkl_paths}
    rdet_by_stem = {path.stem: path for path in rdet_paths}

    missing_rdet = sorted(set(pkl_by_stem) - set(rdet_by_stem))
    unexpected_rdet = sorted(set(rdet_by_stem) - set(pkl_by_stem))
    if missing_rdet or unexpected_rdet:
        findings.append(
            AuditFinding(
                code="AUTHORITATIVE_RDET_STEM_MISMATCH",
                severity="error",
                message=(
                    "Authoritative rotated-detection labels and instance pickles "
                    "do not cover the same image stems."
                ),
                details={
                    "pickles_without_rdet": len(missing_rdet),
                    "rdet_without_pickle": len(unexpected_rdet),
                    "pickle_examples": missing_rdet[:20],
                    "rdet_examples": unexpected_rdet[:20],
                },
            )
        )

    counters: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    checked_files = 0
    checked_instances = 0
    for stem in sorted(set(pkl_by_stem) & set(rdet_by_stem)):
        pkl_path = pkl_by_stem[stem]
        rdet_path = rdet_by_stem[stem]
        with pkl_path.open("rb") as handle:
            records = pickle.load(handle)
        lines = [
            line.strip()
            for line in rdet_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not isinstance(records, list):
            counters["invalid_pickle_payload"] += 1
            if len(examples["invalid_pickle_payload"]) < 20:
                examples["invalid_pickle_payload"].append(stem)
            continue
        if len(records) != len(lines):
            counters["instance_count_mismatch"] += 1
            if len(examples["instance_count_mismatch"]) < 20:
                examples["instance_count_mismatch"].append(
                    f"{stem}: pickle={len(records)}, rdet={len(lines)}"
                )
            continue

        checked_files += 1
        for index, (record, line) in enumerate(zip(records, lines, strict=True), start=1):
            checked_instances += 1
            try:
                rdet_rbox, category, label = _parse_authoritative_rdet_line(
                    line,
                    path=rdet_path,
                    line_number=index,
                )
            except ValueError as exc:
                counters["invalid_rdet_row"] += 1
                if len(examples["invalid_rdet_row"]) < 20:
                    examples["invalid_rdet_row"].append(str(exc))
                continue

            mapping[label].add(category)
            if not isinstance(record, dict):
                counters["invalid_instance_record"] += 1
                if len(examples["invalid_instance_record"]) < 20:
                    examples["invalid_instance_record"].append(f"{stem}:{index}")
                continue
            if int(record.get("label", -1)) != label:
                counters["label_mismatch"] += 1
                if len(examples["label_mismatch"]) < 20:
                    examples["label_mismatch"].append(
                        f"{stem}:{index}: pickle={record.get('label')}, rdet={label}"
                    )

            record_rbox = np.asarray(record.get("rbox", []), dtype=np.float64)
            if record_rbox.shape != (4, 2) or not np.allclose(
                record_rbox,
                rdet_rbox,
                rtol=0.0,
                atol=1e-6,
            ):
                counters["rbox_geometry_mismatch"] += 1
                if len(examples["rbox_geometry_mismatch"]) < 20:
                    examples["rbox_geometry_mismatch"].append(f"{stem}:{index}")

            expected_rhbox = np.asarray(
                [
                    rdet_rbox[:, 0].min(),
                    rdet_rbox[:, 1].min(),
                    rdet_rbox[:, 0].max(),
                    rdet_rbox[:, 1].max(),
                ],
                dtype=np.float64,
            )
            record_rhbox = np.asarray(record.get("rhbox", []), dtype=np.float64)
            if record_rhbox.shape != (4,) or not np.allclose(
                record_rhbox,
                expected_rhbox,
                rtol=0.0,
                atol=1e-6,
            ):
                counters["rhbox_geometry_mismatch"] += 1
                if len(examples["rhbox_geometry_mismatch"]) < 20:
                    examples["rhbox_geometry_mismatch"].append(f"{stem}:{index}")

    inconsistent_mapping = {
        str(label): sorted(categories)
        for label, categories in mapping.items()
        if len(categories) != 1
    }
    if inconsistent_mapping:
        counters["inconsistent_category_mapping"] = len(inconsistent_mapping)
        examples["inconsistent_category_mapping"] = [
            f"{label}: {categories}"
            for label, categories in list(inconsistent_mapping.items())[:20]
        ]

    if counters:
        findings.append(
            AuditFinding(
                code="AUTHORITATIVE_RDET_VALIDATION_FAILED",
                severity="error",
                message=(
                    "Instance pickles are not an exact label-and-geometry match for "
                    "the authoritative rotated-detection annotations."
                ),
                details={
                    "counts": dict(sorted(counters.items())),
                    "examples": dict(sorted(examples.items())),
                },
            )
        )
    elif pkl_paths:
        findings.append(
            AuditFinding(
                code="AUTHORITATIVE_RDET_VALIDATED",
                severity="info",
                message=(
                    "Every instance label and original RBox/RHBox geometry matches "
                    "the authoritative rotated-detection annotations."
                ),
                details={
                    "files": checked_files,
                    "instances": checked_instances,
                    "rdet_directory": str(rdet_dir),
                },
            )
        )

    summary = {
        "files": checked_files,
        "instances": checked_instances,
        "rdet_directory": str(rdet_dir),
    }
    return mapping, summary, findings


def audit_samrs_pickle_dataset(
    root: Path,
    profile: DatasetProfile,
    target_category: str | None = None,
    declared_target_id: int | None = None,
    *,
    authoritative_rdet_dir: Path | None = None,
    allow_raw_scene_overlap: bool = False,
) -> DatasetAuditReport:
    root = root.resolve()
    findings: list[AuditFinding] = []
    pkl_dir = find_single_directory(root, ("ins",))
    image_dir = find_single_directory(root, ("images",))

    if pkl_dir is None:
        findings.append(
            AuditFinding(
                code="PKL_DIRECTORY_NOT_UNIQUE",
                severity="error",
                message="Exactly one SAMRS instance pickle directory named 'ins' is required.",
            )
        )
        pkl_paths: list[Path] = []
    else:
        pkl_paths = sorted(pkl_dir.glob("*.pkl"))

    if image_dir is None:
        findings.append(
            AuditFinding(
                code="IMAGE_DIRECTORY_NOT_UNIQUE",
                severity="error",
                message="Exactly one image directory is required.",
            )
        )
        image_paths: list[Path] = []
    else:
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        )

    pickle_mapping, instances, missing_key_counts = _category_mapping(
        pkl_paths,
        expected_keys=profile.expected_instance_keys,
    )
    if authoritative_rdet_dir is None:
        authoritative_rdet_dir = find_single_directory(root, ("rbbtxts",))
    authoritative_mapping: dict[int, set[str]] = {}
    authoritative_valid = False
    if authoritative_rdet_dir is not None:
        authoritative_mapping, _, authoritative_findings = _validate_authoritative_rdet_labels(
            pkl_paths,
            authoritative_rdet_dir.resolve(),
        )
        findings.extend(authoritative_findings)
        authoritative_valid = not any(
            finding.severity == "error" for finding in authoritative_findings
        )

    mapping = authoritative_mapping if authoritative_valid else pickle_mapping
    normalized_actual = {
        normalize_category_name(category)
        for categories in mapping.values()
        for category in categories
    }
    normalized_expected = {normalize_category_name(category) for category in profile.categories}

    if normalized_actual != normalized_expected:
        findings.append(
            AuditFinding(
                code="CATEGORY_PROFILE_MISMATCH",
                severity="error",
                message=(
                    "Annotation categories do not match the declared dataset profile. "
                    "This dataset must not be used under the requested profile."
                ),
                details={
                    "missing_expected": sorted(normalized_expected - normalized_actual),
                    "unexpected_actual": sorted(normalized_actual - normalized_expected),
                },
            )
        )

    inconsistent_labels = {
        label: sorted(categories)
        for label, categories in mapping.items()
        if len(categories) != 1
    }
    if inconsistent_labels:
        findings.append(
            AuditFinding(
                code="INCONSISTENT_LABEL_MAPPING",
                severity="error",
                message="One or more numeric labels map to multiple category names.",
                details={"labels": inconsistent_labels},
            )
        )

    if authoritative_valid and pickle_mapping != authoritative_mapping:
        findings.append(
            AuditFinding(
                code="PICKLE_CATEGORY_STRINGS_IGNORED",
                severity="warning",
                message=(
                    "Pickle category-name strings do not describe this SOTA subset. "
                    "Numeric labels and geometry were exhaustively validated against "
                    "the authoritative detection annotations, whose category mapping is used."
                ),
                details={
                    "pickle_mapping": {
                        str(label): sorted(categories)
                        for label, categories in sorted(pickle_mapping.items())
                    },
                    "authoritative_mapping": {
                        str(label): sorted(categories)
                        for label, categories in sorted(authoritative_mapping.items())
                    },
                },
            )
        )

    if missing_key_counts:
        findings.append(
            AuditFinding(
                code="MISSING_INSTANCE_KEYS",
                severity="error",
                message="Some instance records are incomplete or invalid.",
                details=missing_key_counts,
            )
        )

    if pkl_paths and image_paths:
        pkl_stems = {path.stem for path in pkl_paths}
        image_stems = {path.stem for path in image_paths}
        if pkl_stems != image_stems:
            findings.append(
                AuditFinding(
                    code="IMAGE_ANNOTATION_STEM_MISMATCH",
                    severity="error",
                    message="Image and pickle annotation stems do not match.",
                    details={
                        "images_without_pkl": len(image_stems - pkl_stems),
                        "pkls_without_image": len(pkl_stems - image_stems),
                    },
                )
            )

    if target_category is not None and declared_target_id is not None:
        actual_categories = sorted(mapping.get(declared_target_id, set()))
        normalized_target = normalize_category_name(target_category)
        if not actual_categories or normalized_target not in {
            normalize_category_name(category) for category in actual_categories
        }:
            findings.append(
                AuditFinding(
                    code="TARGET_ID_CATEGORY_MISMATCH",
                    severity="error",
                    message="The declared target category ID does not map to the requested category.",
                    details={
                        "target_category": target_category,
                        "declared_target_id": declared_target_id,
                        "actual_categories": actual_categories,
                    },
                )
            )

    split_file_counts: dict[str, int] = {}
    source_scene_counts: dict[str, int] = {}
    split_scene_ids: dict[str, set[str]] = {}
    for split_name, file_name in (("train", "train.txt"), ("valid", "valid.txt")):
        matches = sorted(root.rglob(file_name))
        if len(matches) != 1:
            findings.append(
                AuditFinding(
                    code="SPLIT_FILE_NOT_UNIQUE",
                    severity="error",
                    message=f"Expected exactly one {file_name}.",
                    details={"matches": [str(path) for path in matches]},
                )
            )
            continue
        stems = read_split_stems(matches[0])
        scenes = {source_scene_id(stem) for stem in stems}
        split_file_counts[split_name] = len(stems)
        source_scene_counts[split_name] = len(scenes)
        split_scene_ids[split_name] = scenes

    if {"train", "valid"} <= split_scene_ids.keys():
        overlap = split_scene_ids["train"] & split_scene_ids["valid"]
        if overlap:
            findings.append(
                AuditFinding(
                    code="SOURCE_SCENE_SPLIT_LEAKAGE",
                    severity="warning" if allow_raw_scene_overlap else "error",
                    message=(
                        "The raw train and validation lists contain tiles from common "
                        "source scenes. A source-scene-safe resplit is required."
                    ),
                    details={
                        "overlap_count": len(overlap),
                        "examples": sorted(overlap)[:20],
                    },
                )
            )

    if not pkl_paths:
        findings.append(
            AuditFinding(
                code="NO_PICKLE_ANNOTATIONS",
                severity="error",
                message="No instance pickle annotations were found.",
            )
        )
    if not image_paths:
        findings.append(
            AuditFinding(
                code="NO_IMAGES",
                severity="error",
                message="No input images were found.",
            )
        )

    return DatasetAuditReport(
        root=str(root),
        profile_id=profile.profile_id,
        annotation_format=profile.annotation_format,
        annotation_directory=str(pkl_dir) if pkl_dir else None,
        image_directory=str(image_dir) if image_dir else None,
        annotation_files=len(pkl_paths),
        image_files=len(image_paths),
        instances=instances,
        category_mapping={
            str(label): sorted(categories)
            for label, categories in sorted(mapping.items())
        },
        split_file_counts=split_file_counts,
        source_scene_counts=source_scene_counts,
        findings=findings,
    )


def audit_isaid_coco_dataset(
    root: Path,
    profile: DatasetProfile,
    target_category: str = "plane",
) -> DatasetAuditReport:
    root = root.resolve()
    findings: list[AuditFinding] = []
    combined_mapping: dict[str, set[str]] = defaultdict(set)
    split_file_counts: dict[str, int] = {}
    source_scene_counts: dict[str, int] = {}
    split_scenes: dict[str, set[str]] = {}
    target_ids_by_split: dict[str, list[int]] = {}
    annotation_files = 0
    image_files = 0
    instances = 0
    annotation_directory: Path | None = None
    image_directory: Path | None = None

    for split in ("train", "val"):
        annotation_path = root / split / "Annotations" / f"iSAID_{split}.json"
        images_root = root / split / "images"
        if not annotation_path.exists():
            findings.append(
                AuditFinding(
                    code="MISSING_ISAID_ANNOTATION",
                    severity="error",
                    message=f"Missing iSAID {split} COCO annotation file.",
                    details={"path": str(annotation_path)},
                )
            )
            continue
        if not images_root.exists():
            findings.append(
                AuditFinding(
                    code="MISSING_ISAID_IMAGES",
                    severity="error",
                    message=f"Missing iSAID {split} image directory.",
                    details={"path": str(images_root)},
                )
            )
            continue

        annotation_files += 1
        annotation_directory = annotation_path.parent
        image_directory = images_root
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        split_mapping = {
            int(category["id"]): str(category["name"])
            for category in payload.get("categories", [])
        }
        for label, category in split_mapping.items():
            combined_mapping[f"{split}:{label}"].add(category)
        target_ids_by_split[split] = [
            label
            for label, category in split_mapping.items()
            if normalize_category_name(category) == normalize_category_name(target_category)
        ]

        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        instances += len(annotations)
        split_file_counts[split] = len(images)
        scenes = {source_scene_id(Path(str(image["file_name"])).stem) for image in images}
        split_scenes[split] = scenes
        source_scene_counts[split] = len(scenes)

        expected_image_names = {str(image["file_name"]) for image in images}
        actual_image_names = {
            path.name
            for path in images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        }
        image_files += len(actual_image_names)
        if not expected_image_names <= actual_image_names:
            findings.append(
                AuditFinding(
                    code="MISSING_ANNOTATED_IMAGES",
                    severity="error",
                    message=f"Some annotated iSAID {split} images are missing.",
                    details={
                        "missing_count": len(expected_image_names - actual_image_names),
                        "examples": sorted(expected_image_names - actual_image_names)[:20],
                    },
                )
            )

        missing_fields = Counter()
        image_ids = {int(image["id"]) for image in images}
        for annotation in annotations:
            for key in profile.expected_instance_keys:
                if key not in annotation:
                    missing_fields[key] += 1
            if "image_id" in annotation and int(annotation["image_id"]) not in image_ids:
                missing_fields["unknown_image_id"] += 1
        if missing_fields:
            findings.append(
                AuditFinding(
                    code="INVALID_COCO_ANNOTATIONS",
                    severity="error",
                    message=f"Some iSAID {split} annotations are incomplete.",
                    details=dict(missing_fields),
                )
            )

    normalized_actual = {
        normalize_category_name(category)
        for categories in combined_mapping.values()
        for category in categories
    }
    normalized_expected = {normalize_category_name(category) for category in profile.categories}
    if normalized_actual != normalized_expected:
        findings.append(
            AuditFinding(
                code="CATEGORY_PROFILE_MISMATCH",
                severity="error",
                message="iSAID categories do not match the declared official profile.",
                details={
                    "missing_expected": sorted(normalized_expected - normalized_actual),
                    "unexpected_actual": sorted(normalized_actual - normalized_expected),
                },
            )
        )

    invalid_target_splits = {
        split: target_ids
        for split, target_ids in target_ids_by_split.items()
        if len(target_ids) != 1
    }
    if invalid_target_splits:
        findings.append(
            AuditFinding(
                code="TARGET_CATEGORY_NOT_UNIQUE",
                severity="error",
                message="The requested iSAID target category does not map to exactly one ID per split.",
                details={
                    "target_category": target_category,
                    "invalid_splits": invalid_target_splits,
                },
            )
        )
    elif len({target_ids[0] for target_ids in target_ids_by_split.values()}) > 1:
        findings.append(
            AuditFinding(
                code="CATEGORY_IDS_DIFFER_BY_SPLIT",
                severity="warning",
                message=(
                    "The target category uses different numeric IDs across iSAID splits. "
                    "Category selection must use each split's name-to-ID mapping."
                ),
                details={
                    split: target_ids[0]
                    for split, target_ids in target_ids_by_split.items()
                },
            )
        )

    if {"train", "val"} <= split_scenes.keys():
        overlap = split_scenes["train"] & split_scenes["val"]
        if overlap:
            findings.append(
                AuditFinding(
                    code="SOURCE_SCENE_SPLIT_LEAKAGE",
                    severity="error",
                    message="iSAID train and validation contain common source scenes.",
                    details={
                        "overlap_count": len(overlap),
                        "examples": sorted(overlap)[:20],
                    },
                )
            )

    return DatasetAuditReport(
        root=str(root),
        profile_id=profile.profile_id,
        annotation_format=profile.annotation_format,
        annotation_directory=str(annotation_directory) if annotation_directory else None,
        image_directory=str(image_directory) if image_directory else None,
        annotation_files=annotation_files,
        image_files=image_files,
        instances=instances,
        category_mapping={
            label: sorted(categories)
            for label, categories in sorted(combined_mapping.items())
        },
        split_file_counts=split_file_counts,
        source_scene_counts=source_scene_counts,
        findings=findings,
    )


def write_audit_report(
    report: DatasetAuditReport,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    status = "GEÇTİ" if report.passed else "BAŞARISIZ"
    lines = [
        "# Dataset Provenance Audit",
        "",
        f"- Durum: **{status}**",
        f"- Profil: `{report.profile_id}`",
        f"- Kök: `{report.root}`",
        f"- Anotasyon formatı: `{report.annotation_format}`",
        f"- Görüntü: {report.image_files}",
        f"- Anotasyon dosyası: {report.annotation_files}",
        f"- Instance: {report.instances}",
        "",
        "## Category Mapping",
        "",
        "| Split / ID | Category |",
        "|---:|---|",
    ]
    for label, categories in report.category_mapping.items():
        lines.append(f"| {label} | {', '.join(categories)} |")

    lines.extend(["", "## Bulgular", ""])
    if not report.findings:
        lines.append("Kritik bulgu yok.")
    for finding in report.findings:
        lines.extend(
            [
                f"### {finding.severity.upper()}: `{finding.code}`",
                "",
                finding.message,
                "",
            ]
        )
        if finding.details:
            lines.extend(
                [
                    "```json",
                    json.dumps(finding.details, indent=2, ensure_ascii=False),
                    "```",
                    "",
                ]
            )

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
