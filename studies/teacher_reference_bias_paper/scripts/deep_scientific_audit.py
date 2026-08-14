from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.paths import DATASETS  # noqa: E402


def rgb_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(rgb.shape, dtype=np.int64).tobytes())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def bbox_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = (float(value) for value in left)
    rx1, ry1, rw, rh = (float(value) for value in right)
    lx2, ly2, rx2, ry2 = lx1 + lw, ly1 + lh, rx1 + rw, ry1 + rh
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return intersection / union if union > 0 else 0.0


def test_image_index(experiment_id: str) -> pd.DataFrame:
    source = DATASETS[experiment_id]
    metadata = pd.read_csv(source.prepared_root / "test" / "metadata.csv")
    rows = []
    for row in metadata.itertuples(index=False):
        path = source.images_root / str(row.file_name)
        rows.append(
            {
                "experiment_id": experiment_id,
                "file_name": str(row.file_name),
                "source_scene_id": str(row.source_scene_id),
                "rgb_sha256": rgb_sha256(path),
            }
        )
    output = pd.DataFrame(rows)
    if output["file_name"].duplicated().any():
        raise ValueError(f"Yinelenen test dosya adı: {experiment_id}")
    if output["rgb_sha256"].duplicated().any():
        raise ValueError(f"Deney içinde yinelenen RGB test görüntüsü: {experiment_id}")
    return output


def split_audit(experiment_id: str) -> dict[str, Any]:
    source = DATASETS[experiment_id]
    metadata = {
        split: pd.read_csv(source.prepared_root / split / "metadata.csv")
        for split in ("train", "validation", "test")
    }
    scenes = {
        split: set(frame["source_scene_id"].astype(str))
        for split, frame in metadata.items()
    }
    overlaps = {
        f"{left}__{right}": len(scenes[left] & scenes[right])
        for left, right in combinations(scenes, 2)
    }
    test = metadata["test"]
    stratum_counts = {
        str(key): int(value)
        for key, value in test["stratum"].value_counts().sort_index().items()
    }
    positive = bool(
        (test["num_objects"].astype(int) >= 1).all()
        and (test["mask_area_ratio"].astype(float) > 0).all()
    )
    if any(overlaps.values()):
        raise ValueError(f"Kaynak sahne split sızıntısı: {experiment_id}/{overlaps}")
    if stratum_counts and set(stratum_counts.values()) != {128}:
        raise ValueError(f"Tabaka sayıları 4×128 değil: {experiment_id}/{stratum_counts}")
    if not positive:
        raise ValueError(f"Testte hedef-negatif satır bulundu: {experiment_id}")
    return {
        "experiment_id": experiment_id,
        "image_counts": {split: int(len(frame)) for split, frame in metadata.items()},
        "source_scene_counts": {
            split: int(len(values)) for split, values in scenes.items()
        },
        "source_scene_overlap_counts": overlaps,
        "test_stratum_counts": stratum_counts,
        "test_is_target_positive_only": positive,
        "area_threshold": float(source.area_threshold),
    }


def cross_experiment_overlap(
    indices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for left, right in combinations(DATASETS, 2):
        left_frame, right_frame = indices[left], indices[right]
        rows.append(
            {
                "experiment_a": left,
                "experiment_b": right,
                "exact_rgb_test_image_overlap": int(
                    len(set(left_frame["rgb_sha256"]) & set(right_frame["rgb_sha256"]))
                ),
                "source_scene_id_overlap": int(
                    len(
                        set(left_frame["source_scene_id"])
                        & set(right_frame["source_scene_id"])
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def annotations_by_rgb_hash(
    experiment_id: str,
    index: pd.DataFrame,
) -> tuple[COCO, dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]]:
    source = DATASETS[experiment_id]
    coco = COCO(str(source.coco_path))
    image_by_name = {str(image["file_name"]): image for image in coco.dataset["images"]}
    output = {}
    for row in index.itertuples(index=False):
        image = image_by_name[str(row.file_name)]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[int(image["id"])]))
        output[str(row.rgb_sha256)] = (image, annotations)
    return coco, output


def cross_source_mask_agreement(
    target: str,
    human_experiment: str,
    samrs_experiment: str,
    indices: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    human_coco, human = annotations_by_rgb_hash(
        human_experiment, indices[human_experiment]
    )
    samrs_coco, samrs = annotations_by_rgb_hash(
        samrs_experiment, indices[samrs_experiment]
    )
    common = sorted(set(human) & set(samrs))
    matched_ious: list[float] = []
    human_instances = 0
    samrs_instances = 0
    matched_instances = 0
    for image_hash in common:
        _, human_annotations = human[image_hash]
        _, samrs_annotations = samrs[image_hash]
        human_instances += len(human_annotations)
        samrs_instances += len(samrs_annotations)
        if not human_annotations or not samrs_annotations:
            continue
        matrix = np.asarray(
            [
                [bbox_iou(left["bbox"], right["bbox"]) for right in samrs_annotations]
                for left in human_annotations
            ],
            dtype=np.float64,
        )
        left_indices, right_indices = linear_sum_assignment(1.0 - matrix)
        for left_index, right_index in zip(left_indices, right_indices, strict=True):
            if matrix[left_index, right_index] < 0.50:
                continue
            matched_instances += 1
            matched_ious.append(
                mask_iou(
                    human_coco.annToMask(human_annotations[left_index]).astype(bool),
                    samrs_coco.annToMask(samrs_annotations[right_index]).astype(bool),
                )
            )
    return {
        "target": target,
        "exact_rgb_image_count": len(common),
        "human_instance_count": human_instances,
        "samrs_instance_count": samrs_instances,
        "bbox_matched_instance_count_at_iou_50": matched_instances,
        "human_unmatched_instance_count": human_instances - matched_instances,
        "samrs_unmatched_instance_count": samrs_instances - matched_instances,
        "matched_mask_mean_iou": float(np.mean(matched_ious)) if matched_ious else None,
        "matched_mask_success_at_iou_50": float(np.mean(np.asarray(matched_ious) >= 0.50))
        if matched_ious
        else None,
        "matched_mask_success_at_iou_75": float(np.mean(np.asarray(matched_ious) >= 0.75))
        if matched_ious
        else None,
        "matched_mask_success_at_iou_90": float(np.mean(np.asarray(matched_ious) >= 0.90))
        if matched_ious
        else None,
        "interpretation": (
            "Exploratory exact-image subset only; post-hoc and not a representative "
            "cross-dataset benchmark. Unmatched instances expose annotation coverage "
            "differences that matched-mask IoU alone omits."
        ),
    }


def historical_metadata_errata() -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for experiment_id, source in DATASETS.items():
        for split in ("train", "validation", "test"):
            path = source.prepared_root / split / "_annotations.coco.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            description = str(payload.get("info", {}).get("description", ""))
            supercategory = str(payload.get("categories", [{}])[0].get("supercategory", ""))
            if source.target_category.lower().replace("_", "-") not in description.lower():
                issues.append(
                    {
                        "experiment_id": experiment_id,
                        "split": split,
                        "field": "info.description",
                        "historical_value": description,
                        "impact": "nonfunctional_metadata_only",
                    }
                )
            if supercategory == "aircraft" and source.target_category != "plane":
                issues.append(
                    {
                        "experiment_id": experiment_id,
                        "split": split,
                        "field": "categories[0].supercategory",
                        "historical_value": supercategory,
                        "impact": "nonfunctional_metadata_only",
                    }
                )
    return issues


def independently_recompute_detector_metrics(experiment_id: str) -> dict[str, Any]:
    """Recompute detector metrics from frozen COCO boxes without project helpers."""

    source = DATASETS[experiment_id]
    evaluation_root = source.detector_root / "seed_42" / "evaluation" / "test"
    recorded = json.loads(
        (evaluation_root / "metrics.json").read_text(encoding="utf-8")
    )
    detections = json.loads(
        (evaluation_root / "detections_coco.json").read_text(encoding="utf-8")
    )
    coco = COCO(str(source.coco_path))
    image_ids = set(int(value) for value in coco.getImgIds())
    if any(int(row["image_id"]) not in image_ids for row in detections):
        raise ValueError(f"Detector bilinmeyen image_id içeriyor: {experiment_id}")
    if any(int(row["category_id"]) != 1 for row in detections):
        raise ValueError(f"Detector target dışı category_id içeriyor: {experiment_id}")
    if len(detections) != int(recorded["detections_for_ap"]):
        raise ValueError(f"Detector AP detection sayısı uyuşmuyor: {experiment_id}")
    if detections and min(float(row["score"]) for row in detections) < float(
        recorded["ap_confidence_floor"]
    ) - 1e-12:
        raise ValueError(f"AP confidence floor ihlali: {experiment_id}")
    per_image_counts = pd.Series(
        [int(row["image_id"]) for row in detections], dtype="int64"
    ).value_counts()
    if len(per_image_counts) and int(per_image_counts.max()) > 500:
        raise ValueError(f"max_detections=500 ihlali: {experiment_id}")

    coco_detections = coco.loadRes(detections)
    evaluator = COCOeval(coco, coco_detections, "bbox")
    evaluator.params.maxDets = [1, 10, 500]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    thresholds = np.asarray(evaluator.params.iouThrs)

    recomputed: dict[str, float | int] = {}
    for threshold in (0.50, 0.75, 0.90):
        index = int(np.argmin(np.abs(thresholds - threshold)))
        values = precision[index, :, :, 0, -1]
        valid = values[values > -1]
        recomputed[f"bbox_AP{int(threshold * 100)}"] = (
            float(valid.mean()) if valid.size else 0.0
        )
    values = precision[:, :, :, 0, -1]
    valid = values[values > -1]
    recomputed["bbox_AP50_95"] = float(valid.mean()) if valid.size else 0.0

    gt_by_image = {
        int(image_id): [
            [float(value) for value in coco.anns[ann_id]["bbox"]]
            for ann_id in coco.getAnnIds(imgIds=[int(image_id)])
        ]
        for image_id in coco.getImgIds()
    }
    selected_by_image: dict[int, list[dict[str, Any]]] = {}
    fixed_confidence = float(recorded["fixed_confidence_threshold"])
    for row in detections:
        if float(row["score"]) >= fixed_confidence:
            selected_by_image.setdefault(int(row["image_id"]), []).append(row)
    total_gt = sum(len(rows) for rows in gt_by_image.values())
    for threshold in (0.50, 0.75, 0.90):
        true_positive = 0
        false_positive = 0
        for image_id, gt_boxes in gt_by_image.items():
            matched: set[int] = set()
            ordered = sorted(
                selected_by_image.get(image_id, []),
                key=lambda row: float(row["score"]),
                reverse=True,
            )
            for detection in ordered:
                candidates = [
                    (bbox_iou(detection["bbox"], gt_box), gt_index)
                    for gt_index, gt_box in enumerate(gt_boxes)
                    if gt_index not in matched
                ]
                best_iou, best_index = max(candidates, default=(0.0, -1))
                if best_index >= 0 and best_iou >= threshold:
                    matched.add(best_index)
                    true_positive += 1
                else:
                    false_positive += 1
        false_negative = total_gt - true_positive
        suffix = int(threshold * 100)
        recomputed[f"true_positive_at_bbox_iou{suffix}"] = true_positive
        recomputed[f"false_positive_at_bbox_iou{suffix}"] = false_positive
        recomputed[f"false_negative_at_bbox_iou{suffix}"] = false_negative
        recomputed[f"precision_at_bbox_iou{suffix}"] = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recomputed[f"recall_at_bbox_iou{suffix}"] = (
            true_positive / total_gt if total_gt else 0.0
        )

    metric_names = tuple(recomputed)
    differences = {
        name: abs(float(recomputed[name]) - float(recorded[name]))
        for name in metric_names
    }
    maximum_difference = max(differences.values(), default=0.0)
    if maximum_difference > 1e-12:
        raise ValueError(
            f"Detector bağımsız recomputation uyuşmuyor: "
            f"{experiment_id}/{maximum_difference}"
        )
    return {
        "experiment_id": experiment_id,
        "images": len(image_ids),
        "ground_truth_instances": total_gt,
        "detections_for_ap": len(detections),
        "maximum_absolute_metric_difference": maximum_difference,
        "status": "exact_match",
    }


def markdown_table(frame: pd.DataFrame) -> str:
    def clean(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            value = f"{float(value):.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(column) for column in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join((header, divider, *rows))


def write_outputs(
    split_results: list[dict[str, Any]],
    overlap: pd.DataFrame,
    agreement: pd.DataFrame,
    detector_audit: pd.DataFrame,
    errata: list[dict[str, str]],
) -> None:
    docs_root = STUDY_ROOT / "docs"
    analysis_root = STUDY_ROOT / "analysis"
    overlap_path = analysis_root / "cross_experiment_overlap_audit.csv"
    agreement_path = analysis_root / "exploratory_cross_source_mask_agreement.csv"
    detector_path = analysis_root / "detector_recomputation_audit.csv"
    overlap.to_csv(overlap_path, index=False)
    agreement.to_csv(agreement_path, index=False)
    detector_audit.to_csv(detector_path, index=False)
    payload = {
        "schema_version": 1,
        "status": "completed_with_declared_limitations",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_audits": split_results,
        "cross_experiment_overlap": overlap.to_dict(orient="records"),
        "exploratory_cross_source_mask_agreement": agreement.to_dict(
            orient="records"
        ),
        "detector_recomputation_audit": detector_audit.to_dict(orient="records"),
        "historical_metadata_errata": errata,
        "scientific_conclusions": [
            "No source-scene leakage exists among train/validation/test within an experiment.",
            "Every 512-image test set is target-positive by design; detector AP is not an official full-distribution benchmark.",
            "The four experiments are not independent replications because DOTA-derived source scenes and exact test images partially overlap.",
            "The historical COCO description/supercategory issues do not affect image pixels, category ids, boxes, masks, model inputs, or metrics; immutable run inputs are retained for hash-valid provenance.",
            "Frozen detector COCO AP and fixed-threshold precision/recall values were independently recomputed and exactly matched the recorded metrics.",
        ],
    }
    (docs_root / "DEEP_SCIENTIFIC_AUDIT.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Deep Scientific Audit",
        "",
        "Durum: **completed_with_declared_limitations**",
        "",
        "Bu denetim testlerin yalnızca çalışmasını değil; split sızıntısını, test kapsamını, deneyler arası bağımlılığı ve anotasyon kaynakları arasındaki doğrudan anlaşmayı kontrol eder.",
        "",
        "## Split ve Örnekleme",
        "",
        "| Deney | Train/Val/Test görüntü | Train/Val/Test sahne | Test kapsamı | Tabakalar |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in split_results:
        images = row["image_counts"]
        scenes = row["source_scene_counts"]
        lines.append(
            f"| {row['experiment_id']} | {images['train']}/{images['validation']}/{images['test']} | "
            f"{scenes['train']}/{scenes['validation']}/{scenes['test']} | yalnız hedef-pozitif | 4×128 |"
        )
    lines.extend(
        [
            "",
            "Bütün deneylerde train/validation/test kaynak sahne kesişimi sıfırdır. Ancak detector test kümeleri hedef-negatif görüntü içermez; AP sonuçları resmi benchmark sonucu değil, seçilmiş pozitif test kümesindeki detector kontrolüdür.",
            "",
            "## Deneyler Arası Bağımlılık",
            "",
            markdown_table(overlap),
            "",
            "Dört deney tek bir ortalamada birleştirilmemeli ve bağımsız dört replikasyon gibi sunulmamalıdır. iSAID ile SAMRS farklı anotasyon ürünleri olsa da DOTA kökenini ve bazı görüntüleri paylaşır.",
            "",
            "## Exploratory Human–SAMRS Anlaşması",
            "",
            markdown_table(agreement),
            "",
            "Bu analiz yalnız iki testte piksel olarak birebir aynı çıkan post-hoc görüntü alt kümesidir. Temsili benchmark değildir. Hem eşleşmiş maskelerin IoU'su hem de eşleşmeyen instance sayıları birlikte okunmalıdır.",
            "",
            "## Detector Metriklerini Bağımsız Yeniden Hesaplama",
            "",
            markdown_table(detector_audit),
            "",
            "Dondurulmuş COCO detection dosyalarından AP50/AP75/AP90/AP50-95 ile sabit validation eşiğindeki precision/recall değerleri proje evaluator yardımcısı kullanılmadan yeniden hesaplanmıştır. Bütün değerler kaydedilmiş JSON ile tam eşleşmiştir.",
            "",
            "## Tarihsel Metadata Errata",
            "",
            f"{len(errata)} işlevsiz COCO açıklama/supercategory alanı tarihsel girişlerde hatalı adlandırılmıştır. Bu alanlar model girdisi veya metrik değildir. Eski run-manifest hash'lerini bozmamak için deney girdileri değiştirilmemiş, yeniden üretim kodu düzeltilmiş ve hata burada açıkça kaydedilmiştir.",
            "",
        ]
    )
    (docs_root / "DEEP_SCIENTIFIC_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    split_results = [split_audit(experiment_id) for experiment_id in DATASETS]
    indices = {experiment_id: test_image_index(experiment_id) for experiment_id in DATASETS}
    overlap = cross_experiment_overlap(indices)
    agreement = pd.DataFrame(
        [
            cross_source_mask_agreement(
                "plane", "isaid_plane", "samrs_plane", indices
            ),
            cross_source_mask_agreement(
                "small_vehicle",
                "isaid_small_vehicle",
                "samrs_small_vehicle",
                indices,
            ),
        ]
    )
    detector_audit = pd.DataFrame(
        [
            independently_recompute_detector_metrics(experiment_id)
            for experiment_id in DATASETS
        ]
    )
    errata = historical_metadata_errata()
    write_outputs(split_results, overlap, agreement, detector_audit, errata)
    print(STUDY_ROOT / "docs" / "DEEP_SCIENTIFIC_AUDIT.md")


if __name__ == "__main__":
    main()
