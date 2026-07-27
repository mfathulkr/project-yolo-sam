from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.contracts import BBoxXYWH
from yolo_sam.segmentation.factory import create_box_segmenter
from yolo_sam.segmentation.runner import bbox_xywh_to_xyxy
from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real bbox inference as a matched-study model smoke test."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=STUDY_ROOT / "configs" / "protocol.yaml",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", choices=("sam1", "sam2", "sam3"), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def first_annotated_image(
    coco: COCO,
) -> tuple[dict[str, object], dict[str, object]]:
    for image_id in sorted(coco.getImgIds()):
        annotation_ids = coco.getAnnIds(imgIds=[image_id])
        if annotation_ids:
            return coco.loadImgs([image_id])[0], coco.loadAnns(annotation_ids)[0]
    raise RuntimeError("No annotated image is available for the smoke test.")


def main() -> None:
    args = parse_args()
    protocol = load_matched_study_config(args.protocol)
    dataset = load_dataset_study_config(args.dataset)
    split_root = dataset.prepared_root / args.split
    annotation_path = split_root / "_annotations.coco.json"
    images_root = split_root / "images"
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)

    coco = COCO(str(annotation_path))
    image_record, annotation = first_annotated_image(coco)
    image_path = images_root / str(image_record["file_name"])
    bbox = BBoxXYWH.from_sequence(annotation["bbox"])
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    segmenter = create_box_segmenter(
        args.model,
        protocol.segmenter_configs[args.model],
        device=args.device,
        project_root=ROOT,
        hf_token=hf_token,
    )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    result = segmenter.segment_box(image, bbox_xywh_to_xyxy(bbox))
    mask = np.asarray(result.mask, dtype=bool)
    if mask.shape != (image.height, image.width):
        raise RuntimeError(
            f"{args.model} returned mask shape {mask.shape}; "
            f"expected {(image.height, image.width)}."
        )

    report = {
        "passed": True,
        "dataset_id": dataset.dataset_id,
        "model": args.model,
        "model_id": segmenter.model_id,
        "model_version": segmenter.model_version,
        "device": args.device,
        "split": args.split,
        "image_id": int(image_record["id"]),
        "file_name": str(image_record["file_name"]),
        "annotation_id": int(annotation["id"]),
        "bbox_xywh": bbox.to_list(),
        "mask_shape": list(mask.shape),
        "mask_pixels": int(mask.sum()),
        "score": result.score,
        "metadata": result.metadata,
    }
    output_path = args.output or (
        STUDY_ROOT
        / "results"
        / "smoke"
        / dataset.dataset_id
        / f"{args.model}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    print(output_path)


if __name__ == "__main__":
    main()
