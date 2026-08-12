from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = STUDY_ROOT.parents[1]
EXPERIMENTS_ROOT = STUDY_ROOT / "experiments"


@dataclass(frozen=True)
class ReferenceSource:
    reference_type: str
    display_name: str
    teacher: str | None
    is_independent_human: bool = False
    is_published_samrs: bool = False


@dataclass(frozen=True)
class ExperimentSource:
    experiment_id: str
    dataset_id: str
    dataset_family: str
    target_label: str
    target_category: str
    instance_count: int
    area_threshold: float
    reference_types: tuple[str, ...]

    @property
    def root(self) -> Path:
        return EXPERIMENTS_ROOT / self.experiment_id

    @property
    def prepared_root(self) -> Path:
        return self.root / "data" / "prepared"

    @property
    def coco_path(self) -> Path:
        return self.prepared_root / "test" / "_annotations.coco.json"

    @property
    def images_root(self) -> Path:
        return self.prepared_root / "test" / "images"

    @property
    def results_root(self) -> Path:
        return self.root / "results"

    @property
    def predictions_root(self) -> Path:
        return self.results_root / "predictions"

    @property
    def references_root(self) -> Path:
        return self.results_root / "references"

    @property
    def evaluation_root(self) -> Path:
        return self.results_root / "evaluation"

    @property
    def analysis_root(self) -> Path:
        return self.results_root / "analysis"

    @property
    def figures_root(self) -> Path:
        return self.results_root / "figures"

    @property
    def detector_root(self) -> Path:
        return self.results_root / "detector"

    @property
    def reports_root(self) -> Path:
        return self.root / "reports"


ISAID_REFERENCES = (
    "human",
    "pseudo_sam1",
    "pseudo_sam2",
    "pseudo_sam3",
)
SAMRS_REFERENCES = (
    "published_samrs_reference",
    "reproduced_pseudo_sam1",
    "pseudo_sam2",
    "pseudo_sam3",
)

REFERENCES = {
    "human": ReferenceSource(
        "human", "İnsan referansı", None, is_independent_human=True
    ),
    "published_samrs_reference": ReferenceSource(
        "published_samrs_reference",
        "Yayınlanmış SAMRS referansı",
        "sam1",
        is_published_samrs=True,
    ),
    "reproduced_pseudo_sam1": ReferenceSource(
        "reproduced_pseudo_sam1", "Yeniden üretilmiş SAM1 pseudo referansı", "sam1"
    ),
    "pseudo_sam1": ReferenceSource(
        "pseudo_sam1", "SAM1 pseudo referansı", "sam1"
    ),
    "pseudo_sam2": ReferenceSource(
        "pseudo_sam2", "SAM2 pseudo referansı", "sam2"
    ),
    "pseudo_sam3": ReferenceSource(
        "pseudo_sam3", "SAM3 pseudo referansı", "sam3"
    ),
}

DATASETS = {
    "isaid_plane": ExperimentSource(
        experiment_id="isaid_plane",
        dataset_id="isaid_plane",
        dataset_family="isaid",
        target_label="uçak",
        target_category="plane",
        instance_count=5_447,
        area_threshold=0.0167140960693359,
        reference_types=ISAID_REFERENCES,
    ),
    "isaid_small_vehicle": ExperimentSource(
        experiment_id="isaid_small_vehicle",
        dataset_id="isaid_small_vehicle",
        dataset_family="isaid",
        target_label="küçük araç",
        target_category="Small_Vehicle",
        instance_count=12_051,
        area_threshold=0.0018463134765625,
        reference_types=ISAID_REFERENCES,
    ),
    "samrs_plane": ExperimentSource(
        experiment_id="samrs_plane",
        dataset_id="samrs_sota_plane",
        dataset_family="samrs",
        target_label="uçak",
        target_category="plane",
        instance_count=3_713,
        area_threshold=0.011932373046875,
        reference_types=SAMRS_REFERENCES,
    ),
    "samrs_small_vehicle": ExperimentSource(
        experiment_id="samrs_small_vehicle",
        dataset_id="samrs_sota_small_vehicle",
        dataset_family="samrs",
        target_label="küçük araç",
        target_category="small-vehicle",
        instance_count=7_659,
        area_threshold=0.0065670013427734,
        reference_types=SAMRS_REFERENCES,
    ),
}

DATASETS_BY_DATASET_ID = {source.dataset_id: source for source in DATASETS.values()}
MODELS = ("sam1", "sam2", "sam3")
BBOX_SOURCES = ("gt_bbox", "yolo_bbox")
STRATA = (
    "overall",
    "no_overlap__low_mask_area",
    "no_overlap__high_mask_area",
    "overlap__low_mask_area",
    "overlap__high_mask_area",
)


def prediction_path(source: ExperimentSource, model: str, bbox_source: str) -> Path:
    root = source.predictions_root / model / bbox_source
    if bbox_source == "yolo_bbox":
        root = root / "seed_42"
    return root / "predictions.jsonl"


def unmatched_prediction_path(source: ExperimentSource, model: str) -> Path:
    return (
        source.predictions_root
        / model
        / "yolo_bbox"
        / "seed_42"
        / "unmatched_detector_predictions.jsonl"
    )


def reference_path(source: ExperimentSource, reference_type: str) -> Path:
    if reference_type in {"human", "published_samrs_reference"}:
        return source.coco_path
    return source.references_root / f"{reference_type}.jsonl"


def evaluation_path(
    source: ExperimentSource,
    reference_type: str,
    model: str,
    bbox_source: str,
) -> Path:
    suffix = "gt_bbox" if bbox_source == "gt_bbox" else "yolo_bbox_seed_42"
    return (
        source.evaluation_root
        / reference_type
        / model
        / suffix
        / "metrics_instance.csv"
    )
