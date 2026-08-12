from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from yolo_sam.data.contracts import BBoxSource, ReferenceType


STUDY_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = STUDY_ROOT.parents[1]


@dataclass(frozen=True)
class DatasetStudyConfig:
    dataset_id: str
    version: str
    profile_id: str
    raw_root: Path
    prepared_root: Path
    reference_type: ReferenceType
    target_category: str
    source_url: str
    license_name: str
    annotation_format: str
    archive_sha256: str | None
    master_prepared_root: Path | None
    area_threshold: float | None
    experiment_root: Path

    @property
    def results_root(self) -> Path:
        return self.experiment_root / "results"


@dataclass(frozen=True)
class MatchedStudyConfig:
    study_id: str
    image_size: int
    split_fractions: dict[str, float]
    split_seed: int
    detector_seeds: tuple[int, ...]
    detector: dict[str, Any]
    segmenters: tuple[str, ...]
    segmenter_configs: dict[str, dict[str, Any]]
    bbox_sources: tuple[BBoxSource, ...]
    evaluation: dict[str, Any]

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if abs(sum(self.split_fractions.values()) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1.0")
        if not self.detector_seeds:
            raise ValueError("The matched study requires at least one detector seed")
        if len(set(self.detector_seeds)) != len(self.detector_seeds):
            raise ValueError("detector_seeds must be unique")
        if set(self.segmenters) != {"sam1", "sam2", "sam3"}:
            raise ValueError("The primary matched study must contain SAM1, SAM2, and SAM3")
        if set(self.segmenter_configs) != set(self.segmenters):
            raise ValueError("Every segmenter requires exactly one segmenter_configs entry")
        for model in ("sam1", "sam2"):
            config = self.segmenter_configs[model]
            if not str(config.get("revision", "")).strip():
                raise ValueError(f"{model} requires a pinned revision")
            checkpoint_hash = str(config.get("checkpoint_sha256", ""))
            if len(checkpoint_hash) != 64:
                raise ValueError(f"{model} requires a SHA-256 checkpoint hash")
        sam3_config = self.segmenter_configs["sam3"]
        if not str(sam3_config.get("checkpoint_file", "")).strip():
            raise ValueError("sam3 requires checkpoint_file provenance")
        if len(str(sam3_config.get("checkpoint_sha256", ""))) != 64:
            raise ValueError("sam3 requires a SHA-256 checkpoint hash")
        if sam3_config.get("inference_interface") != "sam3_tracker_pvs":
            raise ValueError("sam3 bbox inference requires sam3_tracker_pvs")
        if float(sam3_config.get("mask_threshold", 1.0)) != 0.0:
            raise ValueError("sam3_tracker_pvs requires a logit mask threshold of 0.0")
        if int(sam3_config.get("box_batch_size", 0)) <= 0:
            raise ValueError("sam3 box_batch_size must be positive")
        if BBoxSource.MASK_DERIVED in self.bbox_sources:
            raise ValueError("mask_derived bbox is forbidden in the primary matched study")
        overlap_threshold = float(self.evaluation["overlap_threshold"])
        if not 0 <= overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be in [0, 1]")
        yolo_match_iou = float(self.evaluation["yolo_instance_match_iou"])
        if not 0 <= yolo_match_iou <= 1:
            raise ValueError("yolo_instance_match_iou must be in [0, 1]")
        max_per_stratum = int(self.evaluation["max_per_stratum"])
        if max_per_stratum <= 0:
            raise ValueError("max_per_stratum must be positive")


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def load_dataset_study_config(path: Path) -> DatasetStudyConfig:
    payload = _read_yaml(path)
    required = (
        "dataset_id",
        "version",
        "profile_id",
        "raw_root",
        "prepared_root",
        "reference_type",
        "target_category",
        "source_url",
        "license_name",
        "annotation_format",
        "experiment_root",
    )
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"Dataset config {path} is missing required fields: {missing}")
    raw_root = Path(payload["raw_root"])
    prepared_root = Path(payload["prepared_root"])
    master_prepared_value = payload.get("master_prepared_root")
    experiment_root_value = Path(payload["experiment_root"])
    master_prepared_root = (
        Path(master_prepared_value)
        if master_prepared_value is not None
        else None
    )
    return DatasetStudyConfig(
        dataset_id=str(payload["dataset_id"]),
        version=str(payload["version"]),
        profile_id=str(payload["profile_id"]),
        raw_root=raw_root if raw_root.is_absolute() else REPO_ROOT / raw_root,
        prepared_root=(
            prepared_root
            if prepared_root.is_absolute()
            else REPO_ROOT / prepared_root
        ),
        reference_type=ReferenceType(str(payload["reference_type"])),
        target_category=str(payload["target_category"]),
        source_url=str(payload["source_url"]),
        license_name=str(payload["license_name"]),
        annotation_format=str(payload["annotation_format"]),
        archive_sha256=payload.get("archive_sha256"),
        master_prepared_root=(
            master_prepared_root
            if master_prepared_root is None or master_prepared_root.is_absolute()
            else REPO_ROOT / master_prepared_root
        ),
        area_threshold=(
            float(payload["area_threshold"])
            if payload.get("area_threshold") is not None
            else None
        ),
        experiment_root=(
            experiment_root_value
            if experiment_root_value.is_absolute()
            else REPO_ROOT / experiment_root_value
        ),
    )


def load_matched_study_config(path: Path) -> MatchedStudyConfig:
    payload = _read_yaml(path)
    return MatchedStudyConfig(
        study_id=str(payload["study_id"]),
        image_size=int(payload["image_size"]),
        split_fractions={
            str(name): float(fraction)
            for name, fraction in payload["split_fractions"].items()
        },
        split_seed=int(payload["split_seed"]),
        detector_seeds=tuple(int(seed) for seed in payload["detector_seeds"]),
        detector=dict(payload["detector"]),
        segmenters=tuple(str(value) for value in payload["segmenters"]),
        segmenter_configs={
            str(name): dict(config)
            for name, config in payload["segmenter_configs"].items()
        },
        bbox_sources=tuple(BBoxSource(str(value)) for value in payload["bbox_sources"]),
        evaluation=dict(payload["evaluation"]),
    )


def resolved_config_hash(
    protocol: MatchedStudyConfig,
    dataset: DatasetStudyConfig,
) -> str:
    payload = {
        "protocol": {
            "study_id": protocol.study_id,
            "image_size": protocol.image_size,
            "split_fractions": protocol.split_fractions,
            "split_seed": protocol.split_seed,
            "detector_seeds": protocol.detector_seeds,
            "detector": protocol.detector,
            "segmenters": protocol.segmenters,
            "segmenter_configs": protocol.segmenter_configs,
            "bbox_sources": [source.value for source in protocol.bbox_sources],
            "evaluation": protocol.evaluation,
        },
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "version": dataset.version,
            "profile_id": dataset.profile_id,
            "raw_root": str(dataset.raw_root),
            "prepared_root": str(dataset.prepared_root),
            "reference_type": dataset.reference_type.value,
            "target_category": dataset.target_category,
            "source_url": dataset.source_url,
            "license_name": dataset.license_name,
            "annotation_format": dataset.annotation_format,
            "archive_sha256": dataset.archive_sha256,
            "master_prepared_root": (
                str(dataset.master_prepared_root)
                if dataset.master_prepared_root is not None
                else None
            ),
            "area_threshold": dataset.area_threshold,
            "experiment_root": str(dataset.experiment_root),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
