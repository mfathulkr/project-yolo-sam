from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = STUDY_ROOT.parents[1]


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: str
    target_label: str
    canonical_study: Path
    teacher_instance_count: int

    @property
    def prepared_root(self) -> Path:
        return self.canonical_study / "data" / "prepared" / self.dataset_id

    @property
    def coco_path(self) -> Path:
        return self.prepared_root / "test" / "_annotations.coco.json"

    @property
    def images_root(self) -> Path:
        return self.prepared_root / "test" / "images"

    @property
    def predictions_root(self) -> Path:
        return self.canonical_study / "results" / "predictions" / self.dataset_id

    @property
    def canonical_analysis_root(self) -> Path:
        return self.canonical_study / "results" / "analysis"


DATASETS = {
    "isaid_plane": DatasetSource(
        dataset_id="isaid_plane",
        target_label="uçak",
        canonical_study=REPO_ROOT / "studies" / "teacher_reference_bias_v2_512",
        teacher_instance_count=5_447,
    ),
    "isaid_small_vehicle": DatasetSource(
        dataset_id="isaid_small_vehicle",
        target_label="küçük araç",
        canonical_study=(
            REPO_ROOT / "studies" / "teacher_reference_bias_small_vehicle_v1_512"
        ),
        teacher_instance_count=12_051,
    ),
}

MODELS = ("sam1", "sam2", "sam3")
TEACHERS = ("sam1", "sam2", "sam3")
BBOX_SOURCES = ("gt_bbox", "yolo_bbox")
REFERENCE_TYPES = ("human", "pseudo_sam1", "pseudo_sam2", "pseudo_sam3")
STRATA = (
    "overall",
    "no_overlap__low_mask_area",
    "no_overlap__high_mask_area",
    "overlap__low_mask_area",
    "overlap__high_mask_area",
)


def prediction_path(source: DatasetSource, model: str, bbox_source: str) -> Path:
    root = source.predictions_root / model / bbox_source
    if bbox_source == "yolo_bbox":
        root = root / "seed_42"
    return root / "predictions.jsonl"


def reference_path(dataset_id: str, teacher: str) -> Path:
    return (
        STUDY_ROOT
        / "results"
        / "references"
        / dataset_id
        / f"{teacher}_gt_bbox_pseudo.jsonl"
    )


def evaluation_path(
    dataset_id: str,
    model: str,
    bbox_source: str,
) -> Path:
    suffix = "gt_bbox" if bbox_source == "gt_bbox" else "yolo_bbox_seed_42"
    return (
        STUDY_ROOT
        / "results"
        / "evaluation"
        / dataset_id
        / model
        / suffix
        / "metrics_instance.csv"
    )
