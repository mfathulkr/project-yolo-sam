from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.analysis import (  # noqa: E402
    validate_metric_cube,
)
from teacher_reference_bias_multiteacher.io import read_jsonl  # noqa: E402
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    BBOX_SOURCES,
    DATASETS,
    MODELS,
    REPO_ROOT as PATHS_REPO_ROOT,
    prediction_path,
    reference_path,
    unmatched_prediction_path,
)
from teacher_reference_bias_multiteacher.pseudo_reference import (  # noqa: E402
    validate_pseudo_reference_identity,
)
from teacher_reference_bias_multiteacher.rle_metrics import (  # noqa: E402
    normalized_rle,
)
from teacher_reference_bias.config import (  # noqa: E402
    load_dataset_study_config,
    load_matched_study_config,
)
from yolo_sam.runtime.manifest import (  # noqa: E402
    validate_completed_run_manifest,
)


EXPECTED_REFERENCE_TYPES = {
    "isaid_plane": ("human", "pseudo_sam1", "pseudo_sam2", "pseudo_sam3"),
    "isaid_small_vehicle": (
        "human",
        "pseudo_sam1",
        "pseudo_sam2",
        "pseudo_sam3",
    ),
    "samrs_plane": (
        "published_samrs_reference",
        "reproduced_pseudo_sam1",
        "pseudo_sam2",
        "pseudo_sam3",
    ),
    "samrs_small_vehicle": (
        "published_samrs_reference",
        "reproduced_pseudo_sam1",
        "pseudo_sam2",
        "pseudo_sam3",
    ),
}

REPORT_STRATA = (
    "overall",
    "no_overlap__low_mask_area",
    "no_overlap__high_mask_area",
    "overlap__low_mask_area",
    "overlap__high_mask_area",
)
REPORT_METRICS = {
    "Avg IoU": "mean_iou",
    "Avg Dice": "mean_dice",
    "Avg Precision": "mean_precision",
    "Avg Recall": "mean_recall",
    "IoU ≥ 0.50": "success_at_iou_50",
    "IoU ≥ 0.75": "success_at_iou_75",
    "IoU ≥ 0.90": "success_at_iou_90",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, str]] = []

    def pass_(self, name: str, detail: str = "") -> None:
        self.checks.append({"name": name, "status": "PASS", "detail": detail})

    def fail(self, name: str, detail: str) -> None:
        self.checks.append({"name": name, "status": "FAIL", "detail": detail})

    def run(self, name: str, function: Callable[[], str | None]) -> None:
        try:
            detail = function() or ""
        except Exception as exc:  # noqa: BLE001 - audit must collect all failures
            self.fail(name, f"{type(exc).__name__}: {exc}")
        else:
            self.pass_(name, detail)

    @property
    def failures(self) -> list[dict[str, str]]:
        return [row for row in self.checks if row["status"] == "FAIL"]


def assert_file(path: Path, minimum_bytes: int = 1) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size < minimum_bytes:
        raise ValueError(f"Dosya beklenenden küçük: {path} ({path.stat().st_size})")


def validate_prepared(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    experiment_root = STUDY_ROOT / "experiments" / experiment_id
    dataset = load_dataset_study_config(experiment_root / "config.yaml")
    master = load_dataset_study_config(experiment_root / "master_config.yaml")
    if dataset.master_prepared_root != master.prepared_root:
        raise ValueError("Matched config ile master config aynı kaynak havuzunu göstermiyor")
    if dataset.master_prepared_root is None:
        raise ValueError("master_prepared_root tanımlı değil")
    master_manifest_path = dataset.master_prepared_root / "content_manifest.json"
    assert_file(master_manifest_path)
    provenance_path = source.prepared_root / "master_provenance.json"
    assert_file(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    recorded_root = Path(str(provenance["master_prepared_root"]))
    if not recorded_root.is_absolute():
        recorded_root = REPO_ROOT / recorded_root
    if recorded_root.resolve() != dataset.master_prepared_root.resolve():
        raise ValueError("master_provenance canonical master kökünü göstermiyor")
    recorded_manifest = Path(str(provenance["master_content_manifest"]))
    if not recorded_manifest.is_absolute():
        recorded_manifest = REPO_ROOT / recorded_manifest
    if recorded_manifest.resolve() != master_manifest_path.resolve():
        raise ValueError(
            "master_provenance canonical master content manifest yolunu göstermiyor"
        )
    expected_master_hash = str(provenance["master_content_manifest_sha256"])
    actual_master_hash = sha256_file(master_manifest_path)
    if expected_master_hash != actual_master_hash:
        raise ValueError(
            "master_provenance content manifest SHA-256 uyuşmuyor: "
            f"{expected_master_hash} != {actual_master_hash}"
        )
    master_manifest = json.loads(master_manifest_path.read_text(encoding="utf-8"))
    if tuple(master_manifest.get("splits", ())) != (
        "train",
        "validation",
        "test_pool",
        "test",
    ):
        raise ValueError("Master content manifest test_pool dahil dört split'i kapsamıyor")
    coco = COCO(str(source.coco_path))
    image_count = len(coco.getImgIds())
    instance_count = len(coco.getAnnIds())
    if image_count != 512 or instance_count != source.instance_count:
        raise ValueError(
            f"images/instances={image_count}/{instance_count}, "
            f"expected=512/{source.instance_count}"
        )
    metadata = pd.read_csv(source.prepared_root / "test" / "metadata.csv")
    counts = metadata["stratum"].value_counts().to_dict()
    if set(counts.values()) != {128} or len(counts) != 4:
        raise ValueError(f"Tabaka dağılımı 4×128 değil: {counts}")
    scenes = int(metadata["source_scene_id"].nunique())
    for required in (
        source.prepared_root / "content_manifest.json",
        source.prepared_root / "detector_training_content_manifest.json",
        source.prepared_root / "data.yaml",
    ):
        assert_file(required)
    data_yaml = (source.prepared_root / "data.yaml").read_text(encoding="utf-8")
    if "path:" in data_yaml:
        raise ValueError("data.yaml mutlak/özel path alanı içeriyor")
    return f"512 görüntü, {instance_count} instance, {scenes} kaynak sahne"


def manifest_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in manifest_strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in manifest_strings(child)]
    return []


def validate_run_manifests(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    manifests: list[Path] = []
    for path in sorted(source.results_root.glob("**/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("input_file_fingerprints_at_finish"), dict):
            continue
        if payload.get("path_base") != "repository_root":
            raise ValueError(f"Run manifest repository-relative değil: {path}")
        strings = manifest_strings(
            {
                "inputs": payload.get("inputs"),
                "outputs": payload.get("outputs"),
                "input_fingerprints": payload.get(
                    "input_file_fingerprints_at_finish"
                ),
                "output_fingerprints": payload.get("output_file_fingerprints"),
            }
        )
        absolute = [value for value in strings if value.startswith("/")]
        if absolute:
            raise ValueError(f"Run manifest mutlak yol içeriyor: {path}: {absolute[:2]}")
        validate_completed_run_manifest(path)
        manifests.append(path)
    if len(manifests) != 9:
        raise ValueError(f"9 yerine {len(manifests)} strict run manifest bulundu")

    legacy_studies = (
        "teacher_reference_bias_v2_512",
        "teacher_reference_bias_small_vehicle_v1_512",
    )
    legacy_tokens = tuple(f"studies/{name}" for name in legacy_studies)
    companion_paths = sorted(
        list(source.results_root.glob("predictions/**/effective_config.input.json"))
        + list(
            source.results_root.glob(
                "predictions/**/segmenter_provenance.input.json"
            )
        )
        + list(source.results_root.glob("detector/*/train/args.yaml"))
    )
    for path in companion_paths:
        content = path.read_text(encoding="utf-8")
        hits = [token for token in legacy_tokens if token in content]
        if hits:
            raise ValueError(f"Taşınmamış companion yol kaydı: {path}: {hits}")
    return f"{len(manifests)} strict ve taşınabilir run manifest"


def _bbox_iou(left: list[float], right: list[float]) -> float:
    left_x2 = float(left[0]) + float(left[2])
    left_y2 = float(left[1]) + float(left[3])
    right_x2 = float(right[0]) + float(right[2])
    right_y2 = float(right[1]) + float(right[3])
    width = max(0.0, min(left_x2, right_x2) - max(float(left[0]), float(right[0])))
    height = max(0.0, min(left_y2, right_y2) - max(float(left[1]), float(right[1])))
    intersection = width * height
    union = float(left[2]) * float(left[3]) + float(right[2]) * float(right[3]) - intersection
    return intersection / union if union > 0 else 0.0


def _same_numeric_sequence(left: object, right: object, tolerance: float = 1e-9) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    return all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right, strict=True)
    )


def _independent_yolo_assignment(
    *,
    coco: COCO,
    dataset_id: str,
    detections_path: Path,
    confidence_threshold: float,
    match_iou: float,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    """Rebuild the detector-to-instance assignment without pipeline helpers."""
    image_ids = {int(value) for value in coco.getImgIds()}
    detections_by_image: dict[int, list[dict[str, object]]] = {
        image_id: [] for image_id in image_ids
    }
    detections = json.loads(detections_path.read_text(encoding="utf-8"))
    if not isinstance(detections, list):
        raise ValueError(f"Detection çıktısı liste değil: {detections_path}")
    for original_index, detection in enumerate(detections):
        image_id = int(detection["image_id"])
        if image_id not in detections_by_image:
            raise ValueError(f"Detection bilinmeyen görüntüyü gösteriyor: {image_id}")
        score = float(detection["score"])
        bbox = [float(value) for value in detection["bbox"]]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"Geçersiz detector bbox: {detection}")
        if score < confidence_threshold:
            continue
        detections_by_image[image_id].append(
            {
                "bbox": bbox,
                "score": score,
                "original_index": original_index,
            }
        )
    for image_detections in detections_by_image.values():
        image_detections.sort(key=lambda row: float(row["score"]), reverse=True)

    matched: dict[str, dict[str, object]] = {}
    unmatched: dict[str, dict[str, object]] = {}
    for image_id in sorted(image_ids):
        annotations = sorted(
            coco.loadAnns(coco.getAnnIds(imgIds=[image_id])),
            key=lambda row: int(row["id"]),
        )
        image_detections = detections_by_image[image_id]
        remaining_ground_truth = set(range(len(annotations)))
        remaining_detections = set(range(len(image_detections)))
        for detection_index, detection in enumerate(image_detections):
            if not remaining_ground_truth:
                break
            candidates = [
                (
                    _bbox_iou(
                        [float(value) for value in annotations[ground_truth_index]["bbox"]],
                        detection["bbox"],
                    ),
                    ground_truth_index,
                )
                for ground_truth_index in remaining_ground_truth
            ]
            best_iou, best_ground_truth_index = max(
                candidates,
                key=lambda item: (item[0], -item[1]),
            )
            if best_iou < match_iou:
                continue
            annotation = annotations[best_ground_truth_index]
            instance_id = f"{dataset_id}:{image_id}:{int(annotation['id'])}"
            matched[instance_id] = {
                **detection,
                "bbox_match_iou": best_iou,
                "detection_index": detection_index,
            }
            remaining_ground_truth.remove(best_ground_truth_index)
            remaining_detections.remove(detection_index)
        for detection_index in sorted(remaining_detections):
            detection = image_detections[detection_index]
            unmatched[f"det:{image_id}:{detection_index}"] = {
                **detection,
                "image_id": image_id,
                "detection_index": detection_index,
            }
    return matched, unmatched


def _validate_mask_row(row: dict[str, object], *, image_size: int) -> int:
    rle = normalized_rle(row["predicted_mask_rle"])
    if [int(value) for value in rle["size"]] != [image_size, image_size]:
        raise ValueError(f"Maske boyutu yanlış: {row['instance_id']}: {rle['size']}")
    mask_pixels = int(mask_utils.area(rle))
    status = str(row["status"])
    if status == "ok" and mask_pixels <= 0:
        raise ValueError(f"ok durumunda boş maske: {row['instance_id']}")
    if status in {"empty_mask", "missing_bbox"} and mask_pixels != 0:
        raise ValueError(f"Boş durumunda dolu maske: {row['instance_id']}")
    if status not in {"ok", "empty_mask", "missing_bbox"}:
        raise ValueError(f"Beklenmeyen prediction status: {status}")
    return mask_pixels


def _validate_prediction_identity(
    *,
    row: dict[str, object],
    model: str,
    model_config: dict[str, object],
    run_id: str,
    metadata_row: dict[str, object],
    image_id: int,
    image_size: int,
) -> None:
    expected_model_version = str(
        model_config["revision"]
        if model in {"sam1", "sam2"}
        else model_config["checkpoint_sha256"]
    )
    expected = {
        "model_id": str(model_config["model_id"]),
        "model_version": expected_model_version,
        "run_id": run_id,
        "image_id": f"{metadata_row['dataset_id']}:{image_id}",
        "source_file_name": str(metadata_row["source_file_name"]),
        "source_scene_id": str(metadata_row["source_scene_id"]),
        "stratum": str(metadata_row["stratum"]),
    }
    mismatches = {
        key: (row.get(key), value)
        for key, value in expected.items()
        if str(row.get(key)) != str(value)
    }
    if mismatches:
        raise ValueError(f"Prediction kimliği/metadata uyuşmuyor: {row.get('instance_id')}: {mismatches}")
    confidence = row.get("confidence")
    if row.get("status") == "missing_bbox":
        if confidence is not None or row.get("runtime_ms") is not None:
            raise ValueError(f"missing_bbox confidence/runtime taşıyor: {row['instance_id']}")
    else:
        if confidence is None or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"Geçersiz segmenter confidence: {row['instance_id']}")
        if row.get("runtime_ms") is None or float(row["runtime_ms"]) < 0:
            raise ValueError(f"Geçersiz runtime: {row['instance_id']}")
    _validate_mask_row(row, image_size=image_size)


def validate_predictions(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    protocol = load_matched_study_config(STUDY_ROOT / "configs" / "protocol.yaml")
    coco = COCO(str(source.coco_path))
    metadata = pd.read_csv(source.prepared_root / "test" / "metadata.csv")
    metadata_by_image_id: dict[int, dict[str, object]] = {}
    for _, metadata_series in metadata.iterrows():
        row = metadata_series.to_dict()
        image_id = int(row["image_id"])
        row["dataset_id"] = source.dataset_id
        metadata_by_image_id[image_id] = row
    if set(metadata_by_image_id) != {int(value) for value in coco.getImgIds()}:
        raise ValueError("Metadata ve COCO görüntü kümeleri farklı")

    annotations_by_instance: dict[str, dict[str, object]] = {}
    for annotation in coco.loadAnns(coco.getAnnIds()):
        image_id = int(annotation["image_id"])
        instance_id = f"{source.dataset_id}:{image_id}:{int(annotation['id'])}"
        annotations_by_instance[instance_id] = annotation
    if len(annotations_by_instance) != source.instance_count:
        raise ValueError("COCO instance ID'leri tekil değil")

    threshold_path = (
        source.detector_root
        / "seed_42"
        / "evaluation"
        / "validation"
        / "selected_confidence_threshold.json"
    )
    threshold_selection = json.loads(threshold_path.read_text(encoding="utf-8"))
    if (
        threshold_selection.get("selection_split") != "validation"
        or threshold_selection.get("selection_method") != "max_f1"
        or threshold_selection.get("dataset_id") != source.dataset_id
        or int(threshold_selection.get("seed", -1)) != 42
        or float(threshold_selection.get("selection_iou_threshold", -1)) != 0.5
    ):
        raise ValueError("Detector confidence eşiği validation/max-F1 sözleşmesine uymuyor")
    selected_threshold = float(threshold_selection["selected_confidence_threshold"])
    detections_path = (
        source.detector_root
        / "seed_42"
        / "evaluation"
        / "test"
        / "detections_coco.json"
    )
    matched_yolo, unmatched_yolo = _independent_yolo_assignment(
        coco=coco,
        dataset_id=source.dataset_id,
        detections_path=detections_path,
        confidence_threshold=selected_threshold,
        match_iou=float(protocol.evaluation["yolo_instance_match_iou"]),
    )
    expected_instances = set(annotations_by_instance)
    for model in MODELS:
        for bbox_source in BBOX_SOURCES:
            path = prediction_path(source, model, bbox_source)
            assert_file(path, 100)
            rows = read_jsonl(path)
            if len(rows) != source.instance_count:
                raise ValueError(
                    f"{model}/{bbox_source}: {len(rows)} != {source.instance_count}"
                )
            rows_by_instance = {str(row["instance_id"]): row for row in rows}
            if len(rows_by_instance) != len(rows) or set(rows_by_instance) != expected_instances:
                raise ValueError(f"{model}/{bbox_source}: instance kümesi COCO ile uyuşmuyor")
            manifest = json.loads(
                (path.parent / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("status") != "completed":
                raise ValueError(f"Tamamlanmamış prediction manifest: {path.parent}")
            expected_stage = f"{bbox_source}_segmentation"
            if manifest.get("stage") != expected_stage:
                raise ValueError(f"Yanlış inference stage: {path.parent}")
            if manifest.get("parameters", {}).get("model") != model:
                raise ValueError(f"Manifest model kimliği yanlış: {path.parent}")
            run_id = str(manifest["run_id"])
            effective = json.loads(
                (path.parent / "effective_config.input.json").read_text(encoding="utf-8")
            )
            if (
                effective.get("model") != model
                or effective.get("bbox_source") != bbox_source
                or effective.get("dataset", {}).get("dataset_id") != source.dataset_id
                or effective.get("split") != "test"
                or int(effective.get("image_size", 0)) != protocol.image_size
                or effective.get("model_config") != protocol.segmenter_configs[model]
            ):
                raise ValueError(f"Effective config semantiği yanlış: {path.parent}")
            provenance = json.loads(
                (path.parent / "segmenter_provenance.input.json").read_text(encoding="utf-8")
            )
            provenance_rows = {
                str(row["model"]): row for row in provenance.get("models", [])
            }
            model_provenance = provenance_rows.get(model, {})
            if (
                model_provenance.get("expected_sha256")
                != protocol.segmenter_configs[model]["checkpoint_sha256"]
                or model_provenance.get("actual_sha256")
                != protocol.segmenter_configs[model]["checkpoint_sha256"]
                or not bool(model_provenance.get("passed"))
            ):
                raise ValueError(f"Run segmenter provenance semantiği yanlış: {path.parent}")

            for instance_id, row in rows_by_instance.items():
                annotation = annotations_by_instance[instance_id]
                image_id = int(annotation["image_id"])
                _validate_prediction_identity(
                    row=row,
                    model=model,
                    model_config=protocol.segmenter_configs[model],
                    run_id=run_id,
                    metadata_row=metadata_by_image_id[image_id],
                    image_id=image_id,
                    image_size=protocol.image_size,
                )
                if bbox_source == "gt_bbox":
                    expected_prompt_source = (
                        "human_annotation"
                        if source.dataset_family == "isaid"
                        else "original_detection_annotation"
                    )
                    if (
                        row.get("prompt_type") != "gt_bbox"
                        or row.get("prompt_source") != expected_prompt_source
                        or not _same_numeric_sequence(row.get("input_bbox"), annotation["bbox"])
                        or row.get("status") == "missing_bbox"
                    ):
                        raise ValueError(f"GT-bbox semantiği yanlış: {instance_id}")
                    continue
                if row.get("prompt_type") != "yolo_bbox" or row.get("prompt_source") != "yolo_prediction":
                    raise ValueError(f"YOLO-bbox prompt semantiği yanlış: {instance_id}")
                expected_match = matched_yolo.get(instance_id)
                if expected_match is None:
                    if (
                        row.get("status") != "missing_bbox"
                        or row.get("input_bbox") is not None
                        or row.get("detector_confidence") is not None
                        or float(row.get("bbox_match_iou", -1)) != 0.0
                    ):
                        raise ValueError(f"Bağımsız eşleştirmeye göre missing_bbox değil: {instance_id}")
                elif (
                    row.get("status") == "missing_bbox"
                    or not _same_numeric_sequence(row.get("input_bbox"), expected_match["bbox"])
                    or not math.isclose(
                        float(row["detector_confidence"]),
                        float(expected_match["score"]),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or not math.isclose(
                        float(row["bbox_match_iou"]),
                        float(expected_match["bbox_match_iou"]),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                ):
                    raise ValueError(f"YOLO eşleştirmesi yeniden hesapla uyuşmuyor: {instance_id}")

            if bbox_source == "gt_bbox":
                if int(manifest.get("outputs", {}).get("prediction_count", -1)) != source.instance_count:
                    raise ValueError(f"GT prediction_count yanlış: {path.parent}")
                continue
            if not math.isclose(
                float(effective.get("selected_confidence_threshold", -1)),
                selected_threshold,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"Testte validation confidence eşiği kullanılmamış: {path.parent}")
            unmatched_path = unmatched_prediction_path(source, model)
            unmatched_rows = read_jsonl(unmatched_path)
            unmatched_by_instance = {
                str(row["instance_id"]): row for row in unmatched_rows
            }
            if (
                len(unmatched_by_instance) != len(unmatched_rows)
                or set(unmatched_by_instance) != set(unmatched_yolo)
            ):
                raise ValueError(f"Unmatched detector kümesi yeniden hesapla uyuşmuyor: {model}")
            for detection_id, row in unmatched_by_instance.items():
                expected_detection = unmatched_yolo[detection_id]
                image_id = int(expected_detection["image_id"])
                _validate_prediction_identity(
                    row=row,
                    model=model,
                    model_config=protocol.segmenter_configs[model],
                    run_id=run_id,
                    metadata_row=metadata_by_image_id[image_id],
                    image_id=image_id,
                    image_size=protocol.image_size,
                )
                if (
                    row.get("prompt_type") != "yolo_bbox"
                    or row.get("prompt_source") != "yolo_prediction"
                    or not _same_numeric_sequence(row.get("input_bbox"), expected_detection["bbox"])
                    or not math.isclose(
                        float(row["detector_confidence"]),
                        float(expected_detection["score"]),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                ):
                    raise ValueError(f"Unmatched detector semantiği yanlış: {detection_id}")
            outputs = manifest.get("outputs", {})
            expected_counts = {
                "matched_ground_truth": len(matched_yolo),
                "missing_ground_truth": source.instance_count - len(matched_yolo),
                "unmatched_detections": len(unmatched_yolo),
            }
            if any(int(outputs.get(key, -1)) != value for key, value in expected_counts.items()):
                raise ValueError(f"YOLO manifest eşleştirme sayıları yanlış: {path.parent}")
    return (
        f"6 prediction kümesi; {len(matched_yolo)} eşleşen, "
        f"{source.instance_count - len(matched_yolo)} kaçan ve "
        f"{len(unmatched_yolo)} eşleşmeyen detection bağımsız doğrulandı"
    )


def validate_references(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    if tuple(source.reference_types) != EXPECTED_REFERENCE_TYPES[experiment_id]:
        raise ValueError(f"Referans sözleşmesi yanlış: {source.reference_types}")
    coco = COCO(str(source.coco_path))
    categories = coco.loadCats(coco.getCatIds())
    if len(categories) != 1 or str(categories[0]["name"]) != source.target_category:
        raise ValueError(f"COCO hedef kategorisi yanlış: {categories}")
    image_ids = {int(value) for value in coco.getImgIds()}
    if len(image_ids) != 512:
        raise ValueError("Kaynak referans 512 görüntü içermiyor")
    source_annotations = coco.loadAnns(coco.getAnnIds())
    if len(source_annotations) != source.instance_count:
        raise ValueError("Kaynak referans instance sayısı yanlış")
    expected_native_reference = (
        "human" if source.dataset_family == "isaid" else "pseudo_sam1"
    )
    expected_bbox_source = (
        "human_annotation"
        if source.dataset_family == "isaid"
        else "original_detection_annotation"
    )
    referenced_images: set[int] = set()
    for annotation in source_annotations:
        image_id = int(annotation["image_id"])
        referenced_images.add(image_id)
        if image_id not in image_ids:
            raise ValueError(f"Referans bilinmeyen görüntüyü gösteriyor: {annotation['id']}")
        if (
            annotation.get("reference_type") != expected_native_reference
            or annotation.get("bbox_source") != expected_bbox_source
        ):
            raise ValueError(f"Kaynak referans provenance alanı yanlış: {annotation['id']}")
        rle = normalized_rle(annotation["segmentation"])
        if [int(value) for value in rle["size"]] != [1024, 1024]:
            raise ValueError(f"Kaynak referans maske boyutu yanlış: {annotation['id']}")
        pixels = int(mask_utils.area(rle))
        if pixels <= 0 or int(annotation["area"]) != pixels:
            raise ValueError(f"Kaynak referans maske alanı yanlış: {annotation['id']}")
        bbox = [float(value) for value in annotation["bbox"]]
        if (
            len(bbox) != 4
            or bbox[0] < 0
            or bbox[1] < 0
            or bbox[2] <= 0
            or bbox[3] <= 0
            or bbox[0] + bbox[2] > 1024 + 1e-6
            or bbox[1] + bbox[3] > 1024 + 1e-6
        ):
            raise ValueError(f"Kaynak referans bbox geçersiz: {annotation['id']}")
    if referenced_images != image_ids:
        raise ValueError("Test kümesi hedef-negatif görüntü içeriyor")

    for reference_type in source.reference_types:
        path = reference_path(source, reference_type)
        assert_file(path, 100)
        if reference_type in {"human", "published_samrs_reference"}:
            continue
        teacher = (
            "sam1" if reference_type == "reproduced_pseudo_sam1"
            else reference_type.removeprefix("pseudo_")
        )
        prediction_file = prediction_path(source, teacher, "gt_bbox")
        prediction_manifest = prediction_file.with_name("manifest.json")
        predictions = read_jsonl(prediction_file)
        references = read_jsonl(path)
        if len(references) != source.instance_count:
            raise ValueError(f"{reference_type}: {len(references)} referans satırı")
        identity_rows = [
            dict(row, reference_type=f"pseudo_{teacher}") for row in references
        ]
        validate_pseudo_reference_identity(
            predictions,
            identity_rows,
            teacher=teacher,
        )
        for row in references:
            if (
                row.get("reference_type") != reference_type
                or row.get("dataset_id") != source.dataset_id
                or row.get("construction") != "frozen_gt_bbox_prediction_identity"
            ):
                raise ValueError(
                    f"Pseudo referans kimliği/üretim türü yanlış: {row.get('instance_id')}"
                )
        manifest_path = path.with_suffix(".manifest.json")
        validate_manifest(manifest_path, verify_hashes=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_inputs = {
            prediction_file.relative_to(REPO_ROOT).as_posix(): sha256_file(prediction_file),
            prediction_manifest.relative_to(REPO_ROOT).as_posix(): sha256_file(
                prediction_manifest
            ),
        }
        expected_outputs = {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        }
        empty_count = sum(bool(row["reference_is_empty"]) for row in references)
        if (
            manifest.get("reference_type") != reference_type
            or manifest.get("teacher") != teacher
            or manifest.get("teacher_prompt_type") != "gt_bbox"
            or manifest.get("construction") != "frozen_gt_bbox_prediction_identity"
            or manifest.get("known_positive_empty_reference_policy") != "score_zero"
            or int(manifest.get("instance_count", -1)) != source.instance_count
            or int(manifest.get("empty_reference_count", -1)) != empty_count
            or manifest.get("inputs") != expected_inputs
            or manifest.get("outputs") != expected_outputs
        ):
            raise ValueError(f"Pseudo referans manifest semantiği yanlış: {manifest_path}")
    empty_stats = pd.read_csv(source.analysis_root / "reference_empty_stats.csv")
    if set(empty_stats["reference_type"]) != set(source.reference_types):
        raise ValueError("Boş maske denetimi bütün referansları kapsamıyor")
    return (
        f"{source.instance_count} native referans ve üç GT-bbox pseudo zinciri "
        "RLE düzeyinde doğrulandı"
    )


def validate_analysis(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    cube_path = source.analysis_root / "canonical_instance_metrics.csv"
    metrics = pd.read_csv(cube_path)
    metrics["detector_seed"] = metrics["detector_seed"].astype("Int64")
    validate_metric_cube(metrics)
    expected_rows = source.instance_count * 3 * 2 * 4
    if len(metrics) != expected_rows:
        raise ValueError(f"Metric cube {len(metrics)} != {expected_rows}")
    for filename in (
        "aggregate_metrics.csv",
        "paired_reference_effects.csv",
        "paired_teacher_affinity_contrasts.csv",
        "ranking_by_reference.csv",
        "teacher_advantage.csv",
        "reference_agreement.csv",
        "reference_empty_stats.csv",
        "detector_summary.csv",
        "manifest.json",
        "metric_cube_manifest.json",
    ):
        assert_file(source.analysis_root / filename)
    affinity = pd.read_csv(
        source.analysis_root / "paired_teacher_affinity_contrasts.csv"
    )
    expected_affinity_rows = len(MODELS) * len(BBOX_SOURCES) * len(REPORT_STRATA)
    if len(affinity) != expected_affinity_rows:
        raise ValueError(
            f"Teacher-affinity {len(affinity)} != {expected_affinity_rows} satır"
        )
    required_columns = {
        "self_vs_cross_iou",
        "self_vs_cross_ci_lower",
        "self_vs_cross_ci_upper",
        "relative_advantage_did",
        "relative_advantage_did_ci_lower",
        "relative_advantage_did_ci_upper",
        "bootstrap_samples",
        "confidence_level",
    }
    if not required_columns.issubset(affinity.columns):
        raise ValueError(
            "Teacher-affinity kolonları eksik: "
            f"{sorted(required_columns - set(affinity.columns))}"
        )
    if set(affinity["bootstrap_samples"].astype(int)) != {10_000}:
        raise ValueError("Teacher-affinity 10.000 bootstrap ile hesaplanmamış")
    if set(affinity["confidence_level"].astype(float)) != {0.95}:
        raise ValueError("Teacher-affinity güven düzeyi %95 değil")
    yolo_overall = affinity[
        (affinity["bbox_source"] == "yolo_bbox")
        & (affinity["stratum"] == "overall")
    ]
    if set(yolo_overall["model"]) != set(MODELS):
        raise ValueError("YOLO Overall teacher-affinity üç modeli kapsamıyor")
    if set(yolo_overall["instance_count"].astype(int)) != {
        source.instance_count
    }:
        raise ValueError("YOLO Overall teacher-affinity bütün instance'ları kapsamıyor")
    if (yolo_overall["self_vs_cross_ci_lower"].astype(float) <= 0).any():
        raise ValueError("YOLO self-vs-cross teacher-affinity CI sıfırı kesiyor")
    if experiment_id.startswith("isaid_") and (
        yolo_overall["relative_advantage_did_ci_lower"].astype(float) <= 0
    ).any():
        raise ValueError("iSAID YOLO relative-advantage DiD CI sıfırı kesiyor")
    validate_manifest(source.analysis_root / "manifest.json", verify_hashes=True)
    return (
        f"{len(metrics)} nesne-metrik ve "
        f"{len(affinity)} ek-IoU istatistik satırı"
    )


def validate_manifest(path: Path, *, verify_hashes: bool = False) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"Manifest tamamlanmamış: {path}")
    records: list[dict[str, object]] = []
    for key in ("inputs", "outputs"):
        value = payload.get(key, [])
        if isinstance(value, dict):
            records.extend(
                {"path": item_path, "sha256": item_hash}
                for item_path, item_hash in value.items()
            )
        elif isinstance(value, list):
            records.extend(value)
    checked = 0
    for record in records:
        item_path = str(record["path"])
        if item_path.startswith("/") or item_path.startswith("home/"):
            raise ValueError(f"Portable olmayan manifest yolu: {item_path}")
        item = PATHS_REPO_ROOT / item_path
        assert_file(item)
        if verify_hashes and record.get("sha256"):
            if sha256_file(item) != str(record["sha256"]):
                raise ValueError(f"Manifest hash uyuşmazlığı: {item}")
        checked += 1
    return f"{checked} portable dependency/output"


def pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"PDF page count okunamadı: {path}")


def pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    if "\ufffd" in result.stdout:
        raise ValueError(f"PDF replacement character içeriyor: {path}")
    return result.stdout


def validate_full_reports(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    aggregate = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
    detector = pd.read_csv(source.analysis_root / "detector_summary.csv").iloc[0]
    for reference_type in source.reference_types:
        root = source.reports_root / "full_metrics" / reference_type
        slug = f"{source.experiment_id}_{reference_type}"
        markdown = root / f"{slug}_full_metric_document.md"
        docx = root / f"{slug}_full_metric_document_colored.docx"
        pdf = root / f"{slug}_full_metric_document_colored.pdf"
        for path in (markdown, docx, pdf):
            assert_file(path, 1000)
        if pdf_pages(pdf) < 13:
            raise ValueError(f"Full-metric PDF sayfa sayısı düşük: {pdf}")
        extracted = pdf_text(pdf)
        for token in (source.display_name, "görüntü", "değerlendirme", "çözünürlüğü"):
            if token not in extracted:
                raise ValueError(f"PDF Unicode/text katmanı eksik ({token}): {pdf}")
        with zipfile.ZipFile(docx) as archive:
            if "word/document.xml" not in archive.namelist():
                raise ValueError(f"Geçersiz DOCX: {docx}")
        content = markdown.read_text(encoding="utf-8")
        for token in (
            "Overall",
            "No Overlap × Low Mask Area",
            "No Overlap × High Mask Area",
            "Overlap × Low Mask Area",
            "Overlap × High Mask Area",
            "Avg Precision",
            "Avg Recall",
            "BBox mAP50-95",
            "bütün GT",
        ):
            if token not in content:
                raise ValueError(f"Rapor metni eksik ({token}): {markdown}")
        validate_manifest(root / "report_manifest.json", verify_hashes=True)
        for stratum in REPORT_STRATA:
            table = pd.read_csv(
                root / "tables" / f"{reference_type}__{stratum}.csv"
            )
            if len(table) != len(MODELS) * len(BBOX_SOURCES):
                raise ValueError(f"Rapor tablosu 6 pipeline içermiyor: {root}/{stratum}")
            for _, report_row in table.iterrows():
                model_label, bbox_label = str(report_row["Pipeline"]).split(" ", 1)
                model = model_label.lower()
                bbox_source = "gt_bbox" if bbox_label == "GT bbox" else "yolo_bbox"
                selected = aggregate[
                    (aggregate["reference_type"] == reference_type)
                    & (aggregate["stratum"] == stratum)
                    & (aggregate["model"] == model)
                    & (aggregate["bbox_source"] == bbox_source)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Aggregate eşleşmesi tekil değil: {experiment_id}/"
                        f"{reference_type}/{stratum}/{model}/{bbox_source}"
                    )
                expected = selected.iloc[0]
                expected_images = 512 if stratum == "overall" else 128
                if int(report_row["Images"]) != expected_images:
                    raise ValueError(f"Rapor görüntü sayısı yanlış: {root}/{stratum}")
                for report_column, aggregate_column in REPORT_METRICS.items():
                    if abs(
                        float(report_row[report_column])
                        - round(float(expected[aggregate_column]), 3)
                    ) > 1e-9:
                        raise ValueError(
                            f"Rapor metriği aggregate ile uyuşmuyor: {root}/"
                            f"{stratum}/{report_row['Pipeline']}/{report_column}"
                        )
        detector_table = pd.read_csv(root / "tables" / "detector_summary.csv").iloc[0]
        detector_columns = {
            "BBox mAP50": "bbox_AP50_mean",
            "BBox mAP75": "bbox_AP75_mean",
            "BBox mAP90": "bbox_AP90_mean",
            "BBox mAP50-95": "bbox_AP50_95_mean",
            "BBox Precision@0.50": "precision_at_bbox_iou50_mean",
            "BBox Recall@0.50": "recall_at_bbox_iou50_mean",
            "BBox Precision@0.75": "precision_at_bbox_iou75_mean",
            "BBox Recall@0.75": "recall_at_bbox_iou75_mean",
            "BBox Precision@0.90": "precision_at_bbox_iou90_mean",
            "BBox Recall@0.90": "recall_at_bbox_iou90_mean",
        }
        for report_column, analysis_column in detector_columns.items():
            if abs(
                float(detector_table[report_column])
                - round(float(detector[analysis_column]), 3)
            ) > 1e-9:
                raise ValueError(
                    f"Detector rapor metriği yanlış: {root}/{report_column}"
                )
    return f"{len(source.reference_types)} full-metric MD/DOCX/PDF"


def validate_cross_report(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    root = source.reports_root / "cross_analysis"
    slug = f"{source.experiment_id}_cross_reference_analysis"
    paths = (
        root / f"{slug}.md",
        root / f"{slug}_colored.docx",
        root / f"{slug}_colored.pdf",
    )
    for path in paths:
        assert_file(path, 1000)
    if pdf_pages(paths[2]) < 15:
        raise ValueError("Cross-analysis PDF beklenen kapsamdan kısa")
    extracted = pdf_text(paths[2])
    for token in (
        "Model Kendi Etiketiyle Ne Kadar Ek Puan Alıyor?",
        "Referans Maskeler Birbirine Ne Kadar Benziyor?",
    ):
        if token not in extracted:
            raise ValueError(f"Cross PDF Unicode/text katmanı eksik: {token}")
    content = paths[0].read_text(encoding="utf-8")
    if content.count("Avg IoU") < 10:
        raise ValueError("Cross-analysis 5×2 matris kapsamı eksik")
    for token in (
        "Kendi Etiketiyle IoU",
        "Diğer SAM Etiketleriyle Ortalama IoU",
        "Ek IoU",
        "Güven aralıkları",
        "önceden kaydedilmiş doğrulayıcı",
        "aynı dondurulmuş checkpoint",
    ):
        if token not in content:
            raise ValueError(f"Cross-analysis bilimsel ayrımı eksik: {token}")
    for forbidden in (
        "Doğrudan Teacher-Affinity",
        "Kendi − Çapraz",
        "Göreli Avantaj DiD",
        "[%95 GA]",
    ):
        if forbidden in content:
            raise ValueError(f"Cross-analysis okunamaz teknik tablo içeriyor: {forbidden}")
    validate_manifest(root / "report_manifest.json", verify_hashes=True)
    return f"{pdf_pages(paths[2])} sayfa"


def validate_figures(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    expected = [
        source.figures_root / f"{reference}_gt_bbox_qualitative.png"
        for reference in source.reference_types
    ] + [
        source.figures_root / "model_reference_iou_matrix.png",
        source.figures_root / "reference_effect_with_ci.png",
    ]
    for path in expected:
        assert_file(path, 50_000)
    manifest = json.loads(
        (source.figures_root / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("qualitative_scope") != "all_target_instances_in_selected_images":
        raise ValueError("Nitel figür bütün hedef instance kapsamını ilan etmiyor")
    selections = manifest.get("qualitative_selection", [])
    if len(selections) != len(source.reference_types) * 4:
        raise ValueError("Nitel figür seçim kaydı her referans × dört tabakayı kapsamıyor")
    for reference_type in source.reference_types:
        selected = [
            row for row in selections if row.get("reference_type") == reference_type
        ]
        if {row.get("stratum") for row in selected} != set(REPORT_STRATA[1:]):
            raise ValueError(f"Nitel figür tabaka kapsamı eksik: {reference_type}")
        if any(
            int(row.get("target_instance_count", 0)) < 1
            or row.get("target_instance_count") != row.get("prompt_count_per_model")
            or row.get("display_scope") != "all_target_instances"
            or row.get("selection_method")
            != "model_and_reference_independent_stratum_median_mask_area"
            for row in selected
        ):
            raise ValueError(f"Nitel figür bütün hedeflere prompt vermiyor: {reference_type}")
        if len({str(row.get("source_scene_id")) for row in selected}) != 4:
            raise ValueError(
                f"Nitel figür dört ayrı kaynak sahne kullanmıyor: {reference_type}"
            )
    selected_ids_by_reference = {
        reference_type: tuple(
            row["canonical_image_id"]
            for row in selections
            if row["reference_type"] == reference_type
        )
        for reference_type in source.reference_types
    }
    if len(set(selected_ids_by_reference.values())) != 1:
        raise ValueError("Nitel figürler referanslar arasında aynı görüntüleri kullanmıyor")
    validate_manifest(source.figures_root / "manifest.json", verify_hashes=True)
    return "4 nitel + 2 analiz figürü"


def validate_paper_outputs() -> str:
    main_root = STUDY_ROOT / "analysis"
    for name in (
        "main_cross_analysis.md",
        "main_cross_analysis_colored.docx",
        "main_cross_analysis_colored.pdf",
        "report_manifest.json",
    ):
        assert_file(main_root / name, 100 if name.endswith(".json") else 1000)
    validate_manifest(main_root / "report_manifest.json", verify_hashes=True)
    for name in (
        "main_cross_analysis_gt_bbox.md",
        "main_cross_analysis_gt_bbox_colored.docx",
        "main_cross_analysis_gt_bbox_colored.pdf",
        "main_cross_analysis_gt_bbox_report_manifest.json",
    ):
        assert_file(main_root / name, 100 if name.endswith(".json") else 1000)
    validate_manifest(
        main_root / "main_cross_analysis_gt_bbox_report_manifest.json",
        verify_hashes=True,
    )
    gt_pdf_text = pdf_text(
        main_root / "main_cross_analysis_gt_bbox_colored.pdf"
    )
    for token in ("GT bbox", "özdeşlik kontrolüdür", "Sınırlılıklar"):
        if token not in gt_pdf_text:
            raise ValueError(f"GT-bbox main PDF text katmanı eksik: {token}")
    main_pdf_text = pdf_text(main_root / "main_cross_analysis_colored.pdf")
    for token in (
        "Model Kendi Etiketiyle Ne Kadar Ek Puan Alıyor?",
        "Sınırlılıklar",
        "hedef-pozitif",
    ):
        if token not in main_pdf_text:
            raise ValueError(f"Main PDF Unicode/text katmanı eksik: {token}")
    main_text = (main_root / "main_cross_analysis.md").read_text(encoding="utf-8")
    for token in (
        "Kendi Etiketiyle IoU",
        "Diğer SAM Etiketleriyle Ortalama IoU",
        "Ek IoU",
        "hedef-pozitif olarak seçilmiştir",
        "bağımsız dört replikasyon",
        "YOLO yanlış pozitifleri",
        "önceden kaydedilmiş doğrulayıcı",
        "tam otomatik pseudo-etiketleme hattı değildir",
        "model ailesi düzeyinde genelleme",
    ):
        if token not in main_text:
            raise ValueError(f"Main report zorunlu sınırlaması eksik: {token}")
    assets = STUDY_ROOT / "paper_writing" / "assets"
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("scope") != "four_experiments_no_cross_dataset_pooling":
        raise ValueError("Paper asset scope dört deneyi ayırmıyor")
    validate_manifest(assets / "manifest.json", verify_hashes=True)
    expected_figures = {
        "figure_1_study_design.pdf",
        "figure_2_isaid_own_label_comparison.pdf",
        "figure_3_reference_dependent_model_selection.pdf",
        "figure_4_stratified_own_label_extra_iou.pdf",
    }
    expected_tables = {
        "table_1_experimental_design.tex",
        "table_2_baseline_reference_results.tex",
        "table_3_direct_teacher_affinity.tex",
        "table_4_raw_own_reference_effect.tex",
        "table_5_samrs_reference_integrity.tex",
        "table_6_detector_control.tex",
        "table_s1_stratified_reference_effect.tex",
    }
    figures = {path.name for path in (assets / "figures").glob("*.pdf")}
    tables = {path.name for path in (assets / "tables").glob("*.tex")}
    if figures != expected_figures or tables != expected_tables:
        raise ValueError(f"Paper assets eksik: {len(figures)} figür/{len(tables)} tablo")
    direct_affinity = pd.read_csv(
        assets / "tables" / "table_3_direct_teacher_affinity.csv"
    )
    if len(direct_affinity) != 6 or set(direct_affinity["Experiment"]) != {
        "iSAID Plane",
        "iSAID Small Vehicle",
    }:
        raise ValueError(
            "Ana direct-affinity tablosu yalnız iki insan-kontrollü iSAID "
            "deneyini içermiyor"
        )
    expected_direct_columns = {
        "Experiment",
        "Model",
        "Own-label IoU",
        "Other-SAM-label mean IoU",
        "Extra IoU",
    }
    if set(direct_affinity.columns) != expected_direct_columns:
        raise ValueError("Ana karşılaştırma tablosu okunabilir sütunları içermiyor")
    samrs_integrity = pd.read_csv(
        assets / "tables" / "table_5_samrs_reference_integrity.csv"
    )
    if len(samrs_integrity) != 2 or set(samrs_integrity["Experiment"]) != {
        "SAMRS Plane",
        "SAMRS Small Vehicle",
    }:
        raise ValueError("SAMRS integrity tablosu iki destekleyici deneyi içermiyor")
    for required in (
        STUDY_ROOT / "paper_writing" / "overleaf" / "main.tex",
        STUDY_ROOT / "paper_writing" / "overleaf" / "ref.bib",
        STUDY_ROOT / "paper_writing" / "overleaf" / "README.md",
        STUDY_ROOT / "paper_writing" / "PAPER_STRUCTURE.md",
        STUDY_ROOT / "literature_review" / "LITERATURE_REVIEW.md",
        STUDY_ROOT / "literature_review" / "SEARCH_AUDIT.md",
    ):
        assert_file(required, 500)
    main_tex_path = STUDY_ROOT / "paper_writing" / "overleaf" / "main.tex"
    bib_path = STUDY_ROOT / "paper_writing" / "overleaf" / "ref.bib"
    main_tex = main_tex_path.read_text(encoding="utf-8")
    bibliography = bib_path.read_text(encoding="utf-8")
    if "booktabs" not in main_tex:
        raise ValueError("Overleaf main.tex üretilmiş tablo paketini yüklemiyor")
    for figure_name in expected_figures:
        if figure_name not in main_tex:
            raise ValueError(f"Overleaf figür referansı eksik: {figure_name}")
    bib_keys = re.findall(r"@\w+\{([^,]+),", bibliography)
    if len(bib_keys) != len(set(bib_keys)):
        raise ValueError("BibTeX yinelenen citation key içeriyor")
    cited_keys = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", main_tex)
        for key in group.replace("%", "").split(",")
    }
    missing_keys = cited_keys - set(bib_keys)
    if missing_keys:
        raise ValueError(f"Overleaf citation BibTeX'te yok: {sorted(missing_keys)}")
    if bibliography.count("{") != bibliography.count("}"):
        raise ValueError("BibTeX süslü parantezleri dengeli değil")
    for table_name in expected_tables:
        table_text = (assets / "tables" / table_name).read_text(encoding="utf-8")
        if table_text.count("{") != table_text.count("}"):
            raise ValueError(f"LaTeX tablo parantezleri dengeli değil: {table_name}")
        if re.search(r"\d+\.\d{4,}", table_text):
            raise ValueError(f"LaTeX tablo üçten fazla ondalık içeriyor: {table_name}")
        for forbidden in ("CI lower", "CI upper", "Relative advantage", "Own vs cross"):
            if forbidden in table_text:
                raise ValueError(f"LaTeX tablo teknik ara sütun içeriyor: {table_name}")
    required_figure_text = {
        "figure_1_study_design.pdf": (
            "Controlled comparison",
            "only the evaluation mask changes",
            "Extra IoU on own label",
        ),
        "figure_2_isaid_own_label_comparison.pdf": (
            "Mean on the other two SAM labels",
            "IoU on the model's own label",
            "extra IoU",
        ),
        "figure_3_reference_dependent_model_selection.pdf": (
            "Baseline reference",
            "highest-scoring frozen model",
            "SAMRS Small Vehicle",
        ),
        "figure_4_stratified_own_label_extra_iou.pdf": (
            "No overlap / Low area",
            "Extra IoU on the model's own label",
            "95% confidence interval",
        ),
    }
    for figure_name, tokens in required_figure_text.items():
        figure_text = pdf_text(assets / "figures" / figure_name)
        for token in tokens:
            if token.lower() not in figure_text.lower():
                raise ValueError(f"Figür açıklaması eksik: {figure_name}: {token}")
        info = subprocess.run(
            ["pdfinfo", str(assets / "figures" / figure_name)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        match = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info)
        if match is None:
            raise ValueError(f"Figür sayfa boyutu okunamadı: {figure_name}")
        width_pt, height_pt = (float(value) for value in match.groups())
        if width_pt > 16 / 2.54 * 72 + 1 or height_pt > 20 / 2.54 * 72 + 1:
            raise ValueError(f"Figür dergi 16 x 20 cm sınırını aşıyor: {figure_name}")
    return "4 açıklamalı dergi-boyutlu figür, 7 tablo, main report ve Overleaf iskeleti"


def validate_canonical_segmenter_provenance() -> str:
    path = STUDY_ROOT / "provenance" / "segmenter_provenance.json"
    assert_file(path, 1000)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise ValueError("Canonical segmenter provenance PASS değil")
    if payload.get("path_base") != "repository_root":
        raise ValueError("Canonical segmenter provenance taşınabilir değil")
    protocol_path = STUDY_ROOT / "configs" / "protocol.yaml"
    expected_protocol = protocol_path.relative_to(REPO_ROOT).as_posix()
    if payload.get("protocol") != expected_protocol:
        raise ValueError("Canonical provenance yanlış protocol yolunu gösteriyor")
    if payload.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("Canonical provenance protocol hash'i güncel değil")
    protocol = load_matched_study_config(protocol_path)
    rows = {str(row["model"]): row for row in payload.get("models", [])}
    if set(rows) != set(MODELS):
        raise ValueError("Canonical provenance SAM1/SAM2/SAM3'ü kapsamıyor")
    for model in MODELS:
        row = rows[model]
        if not bool(row.get("passed")):
            raise ValueError(f"{model} checkpoint doğrulaması başarısız")
        if row.get("expected_sha256") != row.get("actual_sha256"):
            raise ValueError(f"{model} checkpoint SHA-256 uyuşmuyor")
        config = protocol.segmenter_configs[model]
        if row.get("expected_sha256") != config["checkpoint_sha256"]:
            raise ValueError(f"{model} provenance/protocol checkpoint uyuşmuyor")
        for locator_key in ("snapshot_locator", "checkpoint_locator"):
            locator = str(row.get(locator_key, ""))
            if locator.startswith("/") or locator.startswith("home/"):
                raise ValueError(f"{model} mutlak locator içeriyor: {locator}")
    return "SAM1/SAM2/SAM3 checkpoint ve protocol hash'leri doğrulandı"


def validate_deep_scientific_audit() -> str:
    json_path = STUDY_ROOT / "docs" / "DEEP_SCIENTIFIC_AUDIT.json"
    md_path = STUDY_ROOT / "docs" / "DEEP_SCIENTIFIC_AUDIT.md"
    overlap_path = STUDY_ROOT / "analysis" / "cross_experiment_overlap_audit.csv"
    agreement_path = (
        STUDY_ROOT / "analysis" / "exploratory_cross_source_mask_agreement.csv"
    )
    detector_path = STUDY_ROOT / "analysis" / "detector_recomputation_audit.csv"
    for path in (json_path, md_path, overlap_path, agreement_path, detector_path):
        assert_file(path, 100)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed_with_declared_limitations":
        raise ValueError("Deep scientific audit sınırlamalarıyla tamamlanmamış")
    split_rows = payload.get("split_audits", [])
    if {row.get("experiment_id") for row in split_rows} != set(DATASETS):
        raise ValueError("Deep audit dört deneyi kapsamıyor")
    for row in split_rows:
        if any(int(value) != 0 for value in row["source_scene_overlap_counts"].values()):
            raise ValueError(f"Split scene leakage: {row['experiment_id']}")
        if int(row["image_counts"]["test"]) != 512:
            raise ValueError(f"Deep audit test sayısı yanlış: {row['experiment_id']}")
        if set(int(value) for value in row["test_stratum_counts"].values()) != {128}:
            raise ValueError(f"Deep audit 4×128 değil: {row['experiment_id']}")
        if not bool(row["test_is_target_positive_only"]):
            raise ValueError(f"Test kapsamı beklenmedik: {row['experiment_id']}")
    overlap = pd.read_csv(overlap_path)
    if len(overlap) != 6 or int(overlap["exact_rgb_test_image_overlap"].sum()) <= 0:
        raise ValueError("Deneyler arası DOTA bağımlılığı kaydedilmemiş")
    agreement = pd.read_csv(agreement_path)
    if set(agreement["target"]) != {"plane", "small_vehicle"}:
        raise ValueError("Exploratory human-SAMRS audit iki hedefi kapsamıyor")
    detector = pd.read_csv(detector_path)
    if set(detector["experiment_id"]) != set(DATASETS):
        raise ValueError("Detector recomputation dört deneyi kapsamıyor")
    if set(detector["status"]) != {"exact_match"}:
        raise ValueError("Detector recomputation exact match değil")
    if (detector["maximum_absolute_metric_difference"].astype(float) > 1e-12).any():
        raise ValueError("Detector recomputation kaydedilmiş metriklerle uyuşmuyor")
    return "split leakage yok; cross-experiment bağımlılık ve audit sınırlamaları kayıtlı"


def validate_active_paths() -> str:
    retired_paths = [
        REPO_ROOT / "studies" / "teacher_reference_bias_v1",
        STUDY_ROOT / "archives",
        *(source.root / "archives" for source in DATASETS.values()),
    ]
    for retired in retired_paths:
        if retired.exists():
            raise ValueError(f"Emekliye ayrılmış kopya hâlâ mevcut: {retired}")
    forbidden = (
        "teacher_reference_bias_v2_512",
        "teacher_reference_bias_small_vehicle_v1_512",
        "teacher_reference_bias_multiteacher_v1_512",
        "/home/ssyzai/",
    )
    scanned = 0
    for root in (
        STUDY_ROOT / "src",
        STUDY_ROOT / "scripts",
        STUDY_ROOT / "configs",
        STUDY_ROOT / "experiments",
        STUDY_ROOT / "docs",
        STUDY_ROOT / "paper_writing",
        STUDY_ROOT / "literature_review",
    ):
        for path in root.rglob("*"):
            if not path.is_file() or "archives" in path.parts or "results" in path.parts:
                continue
            if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".tex", ".bib", ".json"}:
                continue
            if path.name in {
                "MIGRATION_MANIFEST.json",
                "RUN_MANIFEST_MIGRATION_AUDIT.json",
                "QA_REPORT.json",
                "QA_REPORT.md",
                "validate_paper_study.py",
            }:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            hits = [token for token in forbidden if token in text]
            if hits:
                raise ValueError(f"Eski/mutlak yol {hits} dosyada bulundu: {path}")
            scanned += 1
    return f"{scanned} aktif metin/kod dosyası"


def write_report(audit: Audit) -> None:
    payload = {
        "schema_version": 1,
        "status": "failed" if audit.failures else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": audit.checks,
        "pass_count": len(audit.checks) - len(audit.failures),
        "fail_count": len(audit.failures),
    }
    json_path = STUDY_ROOT / "docs" / "QA_REPORT.json"
    md_path = STUDY_ROOT / "docs" / "QA_REPORT.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# QA Report",
        "",
        f"Durum: **{payload['status']}**",
        "",
        f"PASS: {payload['pass_count']} | FAIL: {payload['fail_count']}",
        "",
        "| Kontrol | Durum | Ayrıntı |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['name']} | {row['status']} | {row['detail'].replace('|', '/')} |"
        for row in audit.checks
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


def main() -> None:
    audit = Audit()
    if REPO_ROOT != PATHS_REPO_ROOT:
        audit.fail("repo_root_contract", f"{REPO_ROOT} != {PATHS_REPO_ROOT}")
    else:
        audit.pass_(
            "repo_root_contract",
            "script ve kanonik path modülü aynı repository kökünü çözdü",
        )
    for experiment_id in DATASETS:
        audit.run(f"{experiment_id}:prepared", lambda name=experiment_id: validate_prepared(name))
        audit.run(f"{experiment_id}:predictions", lambda name=experiment_id: validate_predictions(name))
        audit.run(f"{experiment_id}:run_manifests", lambda name=experiment_id: validate_run_manifests(name))
        audit.run(f"{experiment_id}:references", lambda name=experiment_id: validate_references(name))
        audit.run(f"{experiment_id}:analysis", lambda name=experiment_id: validate_analysis(name))
        audit.run(f"{experiment_id}:figures", lambda name=experiment_id: validate_figures(name))
        audit.run(f"{experiment_id}:full_reports", lambda name=experiment_id: validate_full_reports(name))
        audit.run(f"{experiment_id}:cross_report", lambda name=experiment_id: validate_cross_report(name))
    audit.run("paper_outputs", validate_paper_outputs)
    audit.run("canonical_segmenter_provenance", validate_canonical_segmenter_provenance)
    audit.run("deep_scientific_audit", validate_deep_scientific_audit)
    audit.run("active_paths", validate_active_paths)
    write_report(audit)
    for row in audit.checks:
        print(f"{row['status']} {row['name']}: {row['detail']}")
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
