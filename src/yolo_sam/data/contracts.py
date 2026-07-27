from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class ReferenceType(StrEnum):
    HUMAN = "human"
    PSEUDO_SAM1 = "pseudo_sam1"
    PSEUDO_OTHER = "pseudo_other"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class BBoxSource(StrEnum):
    HUMAN_ANNOTATION = "human_annotation"
    ORIGINAL_DETECTION_ANNOTATION = "original_detection_annotation"
    YOLO_PREDICTION = "yolo_prediction"
    MASK_DERIVED = "mask_derived"


class PromptType(StrEnum):
    GT_BBOX = "gt_bbox"
    YOLO_BBOX = "yolo_bbox"


class PredictionStatus(StrEnum):
    OK = "ok"
    EMPTY_MASK = "empty_mask"
    MISSING_BBOX = "missing_bbox"
    INFERENCE_ERROR = "inference_error"


@dataclass(frozen=True)
class BBoxXYWH:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(float("-inf") < float(value) < float("inf") for value in values):
            raise ValueError(f"BBox contains a non-finite value: {values}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"BBox width and height must be positive: {values}")

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, ...]) -> "BBoxXYWH":
        if len(values) != 4:
            raise ValueError(f"Expected four bbox values, got {len(values)}")
        return cls(*(float(value) for value in values))

    @property
    def area(self) -> float:
        return self.width * self.height

    def clipped(self, image_width: int, image_height: int) -> "BBoxXYWH":
        x1 = min(max(self.x, 0.0), float(image_width))
        y1 = min(max(self.y, 0.0), float(image_height))
        x2 = min(max(self.x + self.width, 0.0), float(image_width))
        y2 = min(max(self.y + self.height, 0.0), float(image_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("BBox lies outside image bounds after clipping")
        return BBoxXYWH(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.width, self.height]


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str
    dataset_version: str
    source_url: str
    license_name: str
    annotation_format: str
    reference_type: ReferenceType
    archive_sha256: str | None = None

    def __post_init__(self) -> None:
        required = {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "source_url": self.source_url,
            "license_name": self.license_name,
            "annotation_format": self.annotation_format,
        }
        missing = [key for key, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"Dataset identity has empty required fields: {missing}")
        if self.reference_type is ReferenceType.UNKNOWN:
            raise ValueError("Reference provenance must be known for a final study dataset")


@dataclass(frozen=True)
class ImageRecord:
    dataset_id: str
    image_id: str
    source_scene_id: str
    image_path: Path
    width: int
    height: int
    split: str | None = None
    crop_x: int | None = None
    crop_y: int | None = None
    crop_width: int | None = None
    crop_height: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.image_id or not self.source_scene_id:
            raise ValueError("dataset_id, image_id, and source_scene_id are required")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")


@dataclass(frozen=True)
class MaskReference:
    reference_type: ReferenceType
    annotation_source: str
    rle: dict[str, Any] | None = None
    mask_path: Path | None = None
    generator_model: str | None = None
    generator_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.rle is None and self.mask_path is None:
            raise ValueError("Mask reference requires either RLE data or a mask path")
        if not self.annotation_source:
            raise ValueError("Mask annotation_source is required")
        if self.reference_type is ReferenceType.PSEUDO_SAM1 and not self.generator_model:
            raise ValueError("SAM1 pseudo masks require generator_model provenance")


@dataclass(frozen=True)
class InstanceRecord:
    instance_id: str
    image_id: str
    category_id: int
    category_name: str
    bbox: BBoxXYWH
    bbox_source: BBoxSource
    mask_references: tuple[MaskReference, ...]
    area_pixels: int
    area_fraction: float
    crowding_overlap: float

    def __post_init__(self) -> None:
        if not self.instance_id or not self.image_id or not self.category_name:
            raise ValueError("instance_id, image_id, and category_name are required")
        if not self.mask_references:
            raise ValueError("At least one mask reference is required")
        if self.area_pixels <= 0:
            raise ValueError("area_pixels must be positive")
        if not 0 < self.area_fraction <= 1:
            raise ValueError("area_fraction must be in (0, 1]")
        if not 0 <= self.crowding_overlap <= 1:
            raise ValueError("crowding_overlap must be in [0, 1]")

    def reference(self, reference_type: ReferenceType) -> MaskReference:
        matches = [
            reference
            for reference in self.mask_references
            if reference.reference_type is reference_type
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {reference_type} reference for {self.instance_id}, got {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    model_id: str
    model_version: str
    image_id: str
    instance_id: str
    prompt_type: PromptType
    prompt_source: BBoxSource
    input_bbox: BBoxXYWH | None
    predicted_mask_rle: dict[str, Any] | None
    status: PredictionStatus
    runtime_ms: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.model_id or not self.instance_id:
            raise ValueError("run_id, model_id, and instance_id are required")
        if self.status is PredictionStatus.OK:
            if self.input_bbox is None or self.predicted_mask_rle is None:
                raise ValueError("Successful predictions require input_bbox and predicted_mask_rle")
        if self.runtime_ms is not None and self.runtime_ms < 0:
            raise ValueError("runtime_ms cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.input_bbox is not None:
            payload["input_bbox"] = self.input_bbox.to_list()
        return payload


def validate_primary_bbox_source(source: BBoxSource) -> None:
    if source is BBoxSource.MASK_DERIVED:
        raise ValueError("Mask-derived boxes are forbidden in the primary matched experiment")
