from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.contracts import (
    BBoxSource,
    BBoxXYWH,
    PredictionRecord,
    PredictionStatus,
    PromptType,
)
from yolo_sam.detection.matching import (
    Detection,
    match_detections_to_ground_truth,
)
from yolo_sam.runtime.manifest import (
    finish_run_manifest,
    new_run_manifest,
    write_run_manifest,
)
from yolo_sam.segmentation.factory import create_box_segmenter
from yolo_sam.segmentation.runner import (
    SegmentationTask,
    encode_binary_mask,
    run_segmentation_tasks,
)
from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
    resolved_config_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched YOLO bbox + SAM instance segmentation."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=STUDY_ROOT / "configs" / "protocol.yaml",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", choices=("sam1", "sam2", "sam3"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--segmenter-device", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def xyxy_to_xywh(values: list[float]) -> BBoxXYWH:
    x1, y1, x2, y2 = values
    return BBoxXYWH(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def append_jsonl(path: Path, rows: list[dict[str, Any]], mode: str) -> str:
    if not rows:
        return mode
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return "a"


def main() -> None:
    args = parse_args()
    protocol = load_matched_study_config(args.protocol)
    dataset = load_dataset_study_config(args.dataset)
    if args.seed not in protocol.detector_seeds:
        raise ValueError(f"Seed {args.seed} is not in the frozen protocol")

    split_root = dataset.prepared_root / args.split
    annotation_path = split_root / "_annotations.coco.json"
    images_root = split_root / "images"
    metadata_path = split_root / "metadata.csv"
    content_manifest = dataset.prepared_root / "content_manifest.json"
    segmenter_provenance = (
        STUDY_ROOT
        / "results"
        / "audits"
        / "segmenter_provenance.json"
    )
    detector_root = (
        STUDY_ROOT
        / "results"
        / "detectors"
        / dataset.dataset_id
        / f"seed_{args.seed}"
    )
    detections_input_path = (
        detector_root / "evaluation" / args.split / "detections_coco.json"
    )
    detector_manifest_path = (
        detector_root / "evaluation" / args.split / "manifest.json"
    )
    threshold_selection_path = (
        detector_root
        / "evaluation"
        / "validation"
        / "selected_confidence_threshold.json"
    )
    for required in (
        annotation_path,
        images_root,
        metadata_path,
        content_manifest,
        segmenter_provenance,
        detections_input_path,
        detector_manifest_path,
        threshold_selection_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    threshold_selection = json.loads(
        threshold_selection_path.read_text(encoding="utf-8")
    )
    if threshold_selection.get("selection_split") != "validation":
        raise ValueError(
            "YOLO-bbox inference requires a validation-selected confidence threshold"
        )
    selected_confidence_threshold = float(
        threshold_selection["selected_confidence_threshold"]
    )

    output_root = (
        STUDY_ROOT
        / "results"
        / "predictions"
        / dataset.dataset_id
        / args.model
        / "yolo_bbox"
        / f"seed_{args.seed}"
    )
    predictions_path = output_root / "predictions.jsonl"
    unmatched_path = output_root / "unmatched_detector_predictions.jsonl"
    manifest_path = output_root / "manifest.json"
    if predictions_path.exists() and not args.force:
        raise FileExistsError(
            f"{predictions_path} already exists. Use --force only for an intentional rerun."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{protocol.study_id}-{dataset.dataset_id}-{args.model}-"
        f"yolo-bbox-seed-{args.seed}"
    )
    manifest = new_run_manifest(
        project_root=ROOT,
        run_id=run_id,
        stage="yolo_bbox_segmentation",
        config_hash=resolved_config_hash(protocol, dataset),
        inputs={
            "dataset_config": str(args.dataset.resolve()),
            "annotation_path": str(annotation_path),
            "prepared_content_manifest": str(content_manifest),
            "segmenter_provenance": str(segmenter_provenance),
            "detections": str(detections_input_path),
            "detector_manifest": str(detector_manifest_path),
            "validation_threshold_selection": str(
                threshold_selection_path
            ),
        },
        parameters={
            "model": args.model,
            "seed": args.seed,
            "detector": protocol.detector,
            "segmenter": protocol.segmenter_configs[args.model],
            "match_iou": protocol.evaluation["yolo_instance_match_iou"],
            "selected_confidence_threshold": selected_confidence_threshold,
            "confidence_threshold_selection_method": str(
                threshold_selection["selection_method"]
            ),
        },
    )
    write_run_manifest(manifest_path, manifest)

    coco = COCO(str(annotation_path))
    images = coco.loadImgs(sorted(coco.getImgIds()))
    detection_rows = json.loads(
        detections_input_path.read_text(encoding="utf-8")
    )
    detections_by_image_id: dict[int, list[Detection]] = {
        int(image["id"]): [] for image in images
    }
    for row in detection_rows:
        confidence = float(row["score"])
        if confidence < selected_confidence_threshold:
            continue
        image_id = int(row["image_id"])
        if image_id not in detections_by_image_id:
            raise ValueError(f"Detection references unknown image ID {image_id}")
        detections_by_image_id[image_id].append(
            Detection(
                bbox=BBoxXYWH.from_sequence(row["bbox"]),
                confidence=confidence,
                class_id=0,
            )
        )
    for detections in detections_by_image_id.values():
        detections.sort(key=lambda detection: detection.confidence, reverse=True)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    segmenter = create_box_segmenter(
        args.model,
        protocol.segmenter_configs[args.model],
        device=args.segmenter_device,
        project_root=ROOT,
        hf_token=hf_token,
    )
    metadata = pd.read_csv(metadata_path)
    metadata_by_name = {
        str(row["file_name"]): row.to_dict()
        for _, row in metadata.iterrows()
    }
    predictions_path.write_text("", encoding="utf-8")
    unmatched_path.write_text("", encoding="utf-8")
    prediction_mode = "a"
    unmatched_mode = "a"
    matched_count = 0
    missing_count = 0
    false_positive_count = 0

    try:
        for image_index, image_record in enumerate(
            tqdm(images, desc=f"{args.model} YOLO bbox"),
            start=1,
        ):
            image_id = int(image_record["id"])
            file_name = str(image_record["file_name"])
            metadata_row = metadata_by_name[file_name]
            annotations = sorted(
                coco.loadAnns(coco.getAnnIds(imgIds=[image_id])),
                key=lambda row: int(row["id"]),
            )
            ground_truth_boxes = [
                BBoxXYWH.from_sequence(annotation["bbox"])
                for annotation in annotations
            ]
            detections = detections_by_image_id[image_id]
            matching = match_detections_to_ground_truth(
                ground_truth_boxes,
                detections,
                iou_threshold=float(protocol.evaluation["yolo_instance_match_iou"]),
            )

            with Image.open(images_root / file_name) as source:
                image = source.convert("RGB")
            detector_tasks = [
                SegmentationTask(
                    image_id=f"{dataset.dataset_id}:{image_id}",
                    instance_id=f"det:{image_id}:{detection_index}",
                    bbox=detection.bbox,
                    bbox_source=BBoxSource.YOLO_PREDICTION,
                    prompt_type=PromptType.YOLO_BBOX,
                )
                for detection_index, detection in enumerate(detections)
            ]
            segmented_detections = (
                run_segmentation_tasks(
                    run_id=run_id,
                    image=image,
                    tasks=detector_tasks,
                    segmenter=segmenter,
                )
                if detector_tasks
                else []
            )
            completed_by_detection_index = {
                index: completed
                for index, completed in enumerate(segmented_detections)
            }
            match_by_ground_truth = {
                match.ground_truth_index: match
                for match in matching.matches
            }

            prediction_rows = []
            for ground_truth_index, annotation in enumerate(annotations):
                instance_id = (
                    f"{dataset.dataset_id}:{image_id}:{int(annotation['id'])}"
                )
                if ground_truth_index in match_by_ground_truth:
                    match = match_by_ground_truth[ground_truth_index]
                    completed = completed_by_detection_index[match.detection_index]
                    payload = completed.record.to_dict()
                    payload["instance_id"] = instance_id
                    payload["bbox_match_iou"] = match.bbox_iou
                    payload["detector_confidence"] = detections[
                        match.detection_index
                    ].confidence
                    matched_count += 1
                else:
                    record = PredictionRecord(
                        run_id=run_id,
                        model_id=segmenter.model_id,
                        model_version=segmenter.model_version,
                        image_id=f"{dataset.dataset_id}:{image_id}",
                        instance_id=instance_id,
                        prompt_type=PromptType.YOLO_BBOX,
                        prompt_source=BBoxSource.YOLO_PREDICTION,
                        input_bbox=None,
                        predicted_mask_rle=encode_binary_mask(
                            np.zeros((image.height, image.width), dtype=bool)
                        ),
                        status=PredictionStatus.MISSING_BBOX,
                    )
                    payload = record.to_dict()
                    payload["bbox_match_iou"] = 0.0
                    payload["detector_confidence"] = None
                    missing_count += 1
                payload["source_scene_id"] = str(metadata_row["source_scene_id"])
                payload["stratum"] = str(metadata_row["stratum"])
                payload["source_file_name"] = str(metadata_row["source_file_name"])
                prediction_rows.append(payload)
            prediction_mode = append_jsonl(
                predictions_path,
                prediction_rows,
                prediction_mode,
            )

            unmatched_rows = []
            for detection_index in matching.unmatched_detection_indices:
                completed = completed_by_detection_index[detection_index]
                payload = completed.record.to_dict()
                payload["source_scene_id"] = str(metadata_row["source_scene_id"])
                payload["stratum"] = str(metadata_row["stratum"])
                payload["source_file_name"] = str(metadata_row["source_file_name"])
                payload["detector_confidence"] = detections[detection_index].confidence
                unmatched_rows.append(payload)
                false_positive_count += 1
            unmatched_mode = append_jsonl(
                unmatched_path,
                unmatched_rows,
                unmatched_mode,
            )

            underlying = getattr(segmenter, "segmenter", None)
            if underlying is not None and hasattr(underlying, "maybe_clear_cuda_cache"):
                underlying.maybe_clear_cuda_cache(image_index)
    except Exception as exc:
        finish_run_manifest(manifest, status="failed", error=str(exc))
        write_run_manifest(manifest_path, manifest)
        raise

    manifest["outputs"] = {
        "predictions": str(predictions_path),
        "unmatched_detector_predictions": str(unmatched_path),
        "detections_coco": str(detections_input_path),
        "matched_ground_truth": matched_count,
        "missing_ground_truth": missing_count,
        "unmatched_detections": false_positive_count,
    }
    finish_run_manifest(manifest, status="completed")
    write_run_manifest(manifest_path, manifest)
    print(predictions_path)


if __name__ == "__main__":
    main()
