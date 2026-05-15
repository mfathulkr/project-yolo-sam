from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3_bbox_study.config import load_config, resolve_path
from sam3_bbox_study.io_utils import ensure_dir, list_images
from sam3_bbox_study.pipelines.yolo_sam3 import first_predict_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector bbox AP on the iSAID eval split.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaid_vehicle_yolo26x_cpu_eval.yaml")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--ap-conf", type=float, default=0.001, help="Low score threshold used for COCO AP.")
    parser.add_argument("--fixed-conf", type=float, default=None, help="Score threshold used for fixed precision/recall.")
    parser.add_argument("--yolo-device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "isaid_vehicle_detector_metrics")
    return parser.parse_args()


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def bbox_iou_xywh(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    lx2, ly2 = lx + lw, ly + lh
    rx2, ry2 = rx + rw, ry + rh
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (lw * lh) + (rw * rh) - inter
    return float(inter / union) if union > 0 else 0.0


def coco_ap_at(coco_eval: COCOeval, threshold: float) -> float:
    iou_thresholds = np.asarray(coco_eval.params.iouThrs)
    index = int(np.argmin(np.abs(iou_thresholds - threshold)))
    precision = coco_eval.eval["precision"][index, :, :, 0, -1]
    valid = precision[precision > -1]
    if valid.size == 0:
        return 0.0
    return float(valid.mean())


def coco_ap50_95(coco_eval: COCOeval) -> float:
    precision = coco_eval.eval["precision"][:, :, :, 0, -1]
    valid = precision[precision > -1]
    if valid.size == 0:
        return 0.0
    return float(valid.mean())


def fixed_precision_recall(
    coco: COCO,
    detections: list[dict[str, object]],
    iou_threshold: float,
    confidence_threshold: float,
) -> tuple[float, float, int, int, int]:
    gt_by_image: dict[int, list[list[float]]] = {}
    for image_id in coco.getImgIds():
        annotation_ids = coco.getAnnIds(imgIds=[image_id])
        gt_by_image[image_id] = [coco.anns[ann_id]["bbox"] for ann_id in annotation_ids]

    det_by_image: dict[int, list[dict[str, object]]] = {}
    for det in detections:
        if float(det["score"]) < confidence_threshold:
            continue
        det_by_image.setdefault(int(det["image_id"]), []).append(det)

    tp = 0
    fp = 0
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    for image_id, gt_boxes in gt_by_image.items():
        matched: set[int] = set()
        image_detections = sorted(det_by_image.get(image_id, []), key=lambda item: float(item["score"]), reverse=True)
        for det in image_detections:
            det_box = det["bbox"]
            best_iou = 0.0
            best_index = -1
            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in matched:
                    continue
                iou = bbox_iou_xywh(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_index = gt_index
            if best_iou >= iou_threshold and best_index >= 0:
                tp += 1
                matched.add(best_index)
            else:
                fp += 1
    fn = total_gt - tp
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(total_gt) if total_gt > 0 else 0.0
    return precision, recall, tp, fp, fn


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = config["dataset"]["eval_split"]
    prepared_dir = resolve_path(config["paths"]["prepared_dataset_dir"])
    images_dir = prepared_dir / split / "images"
    annotation_path = prepared_dir / split / "_annotations.coco.json"
    output_dir = ensure_dir(args.output_dir)

    coco = COCO(str(annotation_path))
    category_ids = coco.getCatIds()
    if len(category_ids) != 1:
        raise RuntimeError(f"Expected one category, got {category_ids}")
    category_id = int(category_ids[0])
    image_id_by_name = {image["file_name"]: int(image["id"]) for image in coco.dataset["images"]}

    detector = YOLO(args.weights or config["yolo"]["trained_weights"])
    detections: list[dict[str, object]] = []
    for image_path in tqdm(list_images(images_dir), desc="YOLO detector eval"):
        predict_kwargs = {
            "source": str(image_path),
            "conf": args.ap_conf,
            "imgsz": config["yolo"]["imgsz"],
            "device": first_predict_device(args.yolo_device or config["yolo"]["device"]),
            "verbose": False,
        }
        if config["yolo"].get("max_det") is not None:
            predict_kwargs["max_det"] = config["yolo"]["max_det"]
        result = detector.predict(**predict_kwargs)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
        scores = boxes.conf.detach().cpu().numpy().tolist()
        image_id = image_id_by_name[image_path.name]
        for box, score in zip(xyxy, scores, strict=True):
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": xyxy_to_xywh(box),
                    "score": float(score),
                }
            )

    detections_path = output_dir / "yolo_detector_eval_detections.json"
    detections_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")

    if detections:
        coco_dt = coco.loadRes(str(detections_path))
        coco_eval = COCOeval(coco, coco_dt, "bbox")
        coco_eval.params.maxDets = [1, 10, int(config["yolo"].get("max_det") or 100)]
        coco_eval.evaluate()
        coco_eval.accumulate()
        ap50_95 = coco_ap50_95(coco_eval)
        ap50 = coco_ap_at(coco_eval, 0.50)
        ap75 = coco_ap_at(coco_eval, 0.75)
        ap90 = coco_ap_at(coco_eval, 0.90)
    else:
        ap50_95 = ap50 = ap75 = ap90 = 0.0

    fixed_conf = float(args.fixed_conf if args.fixed_conf is not None else config["yolo"]["conf"])
    rows: list[dict[str, object]] = [
        {
            "split": split,
            "images": len(image_id_by_name),
            "detections_for_ap": len(detections),
            "ap_conf_threshold": args.ap_conf,
            "fixed_conf_threshold": fixed_conf,
            "bbox_mAP50": ap50,
            "bbox_mAP75": ap75,
            "bbox_mAP90": ap90,
            "bbox_mAP50_95": ap50_95,
        }
    ]
    for threshold in [0.50, 0.75, 0.90]:
        precision, recall, tp, fp, fn = fixed_precision_recall(coco, detections, threshold, fixed_conf)
        rows[0][f"precision_at_iou{int(threshold * 100)}"] = precision
        rows[0][f"recall_at_iou{int(threshold * 100)}"] = recall
        rows[0][f"tp_at_iou{int(threshold * 100)}"] = tp
        rows[0][f"fp_at_iou{int(threshold * 100)}"] = fp
        rows[0][f"fn_at_iou{int(threshold * 100)}"] = fn

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "yolo_detector_eval_metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
