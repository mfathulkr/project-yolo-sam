from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from ultralytics import YOLO

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.detection.coco_metrics import (
    evaluate_coco_bbox_detections,
    select_f1_confidence_threshold,
)
from yolo_sam.runtime.manifest import (
    finish_run_manifest,
    new_run_manifest,
    write_run_manifest,
)
from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
    resolved_config_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen YOLO detector and compute real COCO bbox metrics."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=STUDY_ROOT / "configs" / "protocol.yaml",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def xyxy_to_xywh(values: list[float]) -> list[float]:
    x1, y1, x2, y2 = values
    return [x1, y1, x2 - x1, y2 - y1]


def stream_detector_results(
    detector: Any,
    *,
    image_paths: list[str],
    image_size: int,
    confidence_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    device: str,
    batch_size: int,
) -> Iterable[Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        yield from detector.predict(
            source=batch_paths,
            imgsz=image_size,
            conf=confidence_threshold,
            iou=nms_iou_threshold,
            max_det=max_detections,
            device=device,
            batch=batch_size,
            verbose=False,
            stream=True,
        )


def main() -> None:
    args = parse_args()
    protocol = load_matched_study_config(args.protocol)
    dataset = load_dataset_study_config(args.dataset)
    if args.seed not in protocol.detector_seeds:
        raise ValueError(f"Seed {args.seed} is not in the frozen protocol")

    split_root = dataset.prepared_root / args.split
    annotation_path = split_root / "_annotations.coco.json"
    images_root = split_root / "images"
    content_manifest = dataset.prepared_root / "content_manifest.json"
    detector_root = (
        dataset.results_root
        / "detector"
        / f"seed_{args.seed}"
    )
    detector_weights = detector_root / "train" / "weights" / "best.pt"
    threshold_selection_path = (
        detector_root
        / "evaluation"
        / "validation"
        / "selected_confidence_threshold.json"
    )
    required_inputs = [
        annotation_path,
        images_root,
        content_manifest,
        detector_weights,
    ]
    if args.split == "test":
        required_inputs.append(threshold_selection_path)
    for required in required_inputs:
        if not required.exists():
            raise FileNotFoundError(required)

    output_root = detector_root / "evaluation" / args.split
    detections_path = output_root / "detections_coco.json"
    metrics_path = output_root / "metrics.json"
    manifest_path = output_root / "manifest.json"
    if metrics_path.exists() and not args.force:
        raise FileExistsError(
            f"{metrics_path} already exists. Use --force for an intentional rerun."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{protocol.study_id}-{dataset.dataset_id}-detector-"
        f"seed-{args.seed}-{args.split}"
    )
    manifest_inputs = {
        "dataset_config": str(args.dataset.resolve()),
        "annotations": str(annotation_path),
        "prepared_content_manifest": str(content_manifest),
        "detector_weights": str(detector_weights),
    }
    if args.split == "test":
        manifest_inputs["validation_threshold_selection"] = str(
            threshold_selection_path
        )
    manifest = new_run_manifest(
        project_root=ROOT,
        run_id=run_id,
        stage="detector_evaluation",
        config_hash=resolved_config_hash(protocol, dataset),
        inputs=manifest_inputs,
        parameters={
            "seed": args.seed,
            "split": args.split,
            "device": args.device,
            "detector": protocol.detector,
            "image_size": protocol.image_size,
            "evaluation_batch_size": int(protocol.detector["batch"]),
            "streaming_inference": True,
        },
    )
    write_run_manifest(manifest_path, manifest)

    try:
        coco = COCO(str(annotation_path))
        images = coco.loadImgs(sorted(coco.getImgIds()))
        image_paths = [
            str(images_root / str(image["file_name"])) for image in images
        ]
        detector = YOLO(str(detector_weights))
        results = stream_detector_results(
            detector,
            image_paths=image_paths,
            image_size=protocol.image_size,
            confidence_threshold=float(
                protocol.detector["evaluation_confidence_threshold"]
            ),
            nms_iou_threshold=float(protocol.detector["nms_iou_threshold"]),
            max_detections=int(protocol.detector["max_detections"]),
            device=args.device,
            batch_size=int(protocol.detector["batch"]),
        )
        detections: list[dict[str, object]] = []
        for image, result in zip(images, results, strict=True):
            if result.boxes is None:
                continue
            xyxy_rows = result.boxes.xyxy.detach().cpu().tolist()
            confidence_rows = result.boxes.conf.detach().cpu().tolist()
            for xyxy, confidence in zip(
                xyxy_rows,
                confidence_rows,
                strict=True,
            ):
                detections.append(
                    {
                        "image_id": int(image["id"]),
                        "category_id": 1,
                        "bbox": xyxy_to_xywh([float(value) for value in xyxy]),
                        "score": float(confidence),
                    }
                )
        detections_path.write_text(
            json.dumps(detections, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.split == "validation":
            threshold_selection = select_f1_confidence_threshold(
                coco,
                detections,
                iou_threshold=float(
                    protocol.evaluation["yolo_instance_match_iou"]
                ),
            )
            threshold_selection.update(
                {
                    "dataset_id": dataset.dataset_id,
                    "seed": args.seed,
                    "selection_split": "validation",
                }
            )
            threshold_selection_path.write_text(
                json.dumps(
                    threshold_selection,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        elif args.split == "test":
            threshold_selection = json.loads(
                threshold_selection_path.read_text(encoding="utf-8")
            )
            if threshold_selection.get("selection_split") != "validation":
                raise ValueError(
                    "Detector confidence threshold was not selected on validation"
                )
        else:
            threshold_selection = {
                "selection_method": "frozen_protocol_fallback",
                "selection_split": args.split,
                "selected_confidence_threshold": float(
                    protocol.detector["confidence_threshold"]
                ),
            }
        selected_confidence_threshold = float(
            threshold_selection["selected_confidence_threshold"]
        )
        metrics = evaluate_coco_bbox_detections(
            coco,
            detections,
            fixed_confidence_threshold=selected_confidence_threshold,
            max_detections=int(protocol.detector["max_detections"]),
        )
        metrics.update(
            {
                "dataset_id": dataset.dataset_id,
                "seed": args.seed,
                "split": args.split,
                "ap_confidence_floor": float(
                    protocol.detector["evaluation_confidence_threshold"]
                ),
                "confidence_threshold_source_split": str(
                    threshold_selection["selection_split"]
                ),
                "confidence_threshold_selection_method": str(
                    threshold_selection["selection_method"]
                ),
            }
        )
        metrics_path.write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        finish_run_manifest(manifest, status="failed", error=str(exc))
        write_run_manifest(manifest_path, manifest)
        raise

    manifest["outputs"] = {
        "detections": str(detections_path),
        "metrics": str(metrics_path),
    }
    if args.split == "validation":
        manifest["outputs"]["selected_confidence_threshold"] = str(
            threshold_selection_path
        )
    finish_run_manifest(manifest, status="completed")
    write_run_manifest(manifest_path, manifest)
    print(json.dumps(metrics, indent=2))
    print(output_root)


if __name__ == "__main__":
    main()
