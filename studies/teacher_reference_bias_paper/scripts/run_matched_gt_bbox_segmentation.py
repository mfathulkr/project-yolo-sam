from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

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
    PromptType,
)
from yolo_sam.runtime.manifest import (
    acquire_run_lock,
    finish_run_manifest,
    new_run_manifest,
    write_run_manifest,
)
from yolo_sam.runtime.effective_config import (
    BBOX_SEGMENTATION_CONFIG_SCHEMA,
    bbox_segmentation_effective_config,
    effective_config_hash,
    write_effective_config_snapshot,
)
from yolo_sam.segmentation.factory import create_box_segmenter
from yolo_sam.segmentation.runner import (
    SegmentationTask,
    run_segmentation_tasks,
)
from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run instance-preserving GT bbox segmentation for the matched study."
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
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def coco_tasks_for_image(
    dataset_id: str,
    image: dict[str, object],
    annotations: list[dict[str, object]],
) -> list[SegmentationTask]:
    tasks = []
    for annotation in sorted(annotations, key=lambda row: int(row["id"])):
        bbox_source = BBoxSource(
            str(annotation.get("bbox_source", BBoxSource.HUMAN_ANNOTATION.value))
        )
        tasks.append(
            SegmentationTask(
                image_id=f"{dataset_id}:{image['id']}",
                instance_id=f"{dataset_id}:{image['id']}:{annotation['id']}",
                bbox=BBoxXYWH.from_sequence(annotation["bbox"]),
                bbox_source=bbox_source,
                prompt_type=PromptType.GT_BBOX,
            )
        )
    return tasks


def main() -> None:
    args = parse_args()
    protocol = load_matched_study_config(args.protocol)
    dataset = load_dataset_study_config(args.dataset)
    split_root = dataset.prepared_root / args.split
    annotation_path = split_root / "_annotations.coco.json"
    images_root = split_root / "images"
    metadata_path = split_root / "metadata.csv"
    content_manifest = dataset.prepared_root / "content_manifest.json"
    segmenter_provenance = (
        STUDY_ROOT
        / "provenance"
        / "segmenter_provenance.json"
    )
    for required in (
        annotation_path,
        images_root,
        metadata_path,
        content_manifest,
        segmenter_provenance,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    output_root = (
        dataset.results_root
        / "predictions"
        / args.model
        / "gt_bbox"
    )
    predictions_path = output_root / "predictions.jsonl"
    temporary_predictions_path = output_root / "predictions.jsonl.tmp"
    manifest_path = output_root / "manifest.json"
    if predictions_path.exists() and not args.force:
        raise FileExistsError(
            f"{predictions_path} already exists. Use --force only for an intentional rerun."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    writer_lock = acquire_run_lock(output_root / ".writer.lock")
    temporary_predictions_path.unlink(missing_ok=True)
    provenance_snapshot = output_root / "segmenter_provenance.input.json"
    shutil.copyfile(segmenter_provenance, provenance_snapshot)
    effective_config_snapshot = output_root / "effective_config.input.json"
    effective_config = bbox_segmentation_effective_config(
        study_id=protocol.study_id,
        image_size=protocol.image_size,
        dataset={
            "dataset_id": dataset.dataset_id,
            "version": dataset.version,
            "profile_id": dataset.profile_id,
            "reference_type": dataset.reference_type.value,
            "target_category": dataset.target_category,
            "area_threshold": dataset.area_threshold,
        },
        model=args.model,
        model_config=protocol.segmenter_configs[args.model],
        bbox_source="gt_bbox",
        split=args.split,
    )
    write_effective_config_snapshot(effective_config_snapshot, effective_config)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    segmenter = create_box_segmenter(
        args.model,
        protocol.segmenter_configs[args.model],
        device=args.device,
        project_root=ROOT,
        hf_token=hf_token,
    )
    config_hash = effective_config_hash(effective_config)
    run_id = (
        f"{protocol.study_id}-{dataset.dataset_id}-{args.model}-gt-bbox-"
        f"{config_hash[:12]}"
    )
    manifest = new_run_manifest(
        project_root=ROOT,
        run_id=run_id,
        stage="gt_bbox_segmentation",
        config_hash=config_hash,
        inputs={
            "dataset_config": str(args.dataset.resolve()),
            "annotation_path": str(annotation_path),
            "images_root": str(images_root),
            "prepared_content_manifest": str(content_manifest),
            "segmenter_provenance": str(provenance_snapshot),
            "effective_config": str(effective_config_snapshot),
        },
        parameters={
            "model": args.model,
            "model_config": protocol.segmenter_configs[args.model],
            "split": args.split,
            "device": args.device,
            "prompt_type": PromptType.GT_BBOX.value,
        },
    )
    manifest["config_hash_scope"] = BBOX_SEGMENTATION_CONFIG_SCHEMA
    write_run_manifest(manifest_path, manifest)

    coco = COCO(str(annotation_path))
    metadata = pd.read_csv(metadata_path)
    metadata_by_name = {
        str(row["file_name"]): row.to_dict()
        for _, row in metadata.iterrows()
    }
    images = coco.loadImgs(coco.getImgIds())
    mode = "w"
    prediction_count = 0
    try:
        for image_index, image_record in enumerate(
            tqdm(images, desc=f"{args.model} GT bbox"),
            start=1,
        ):
            file_name = str(image_record["file_name"])
            image_path = images_root / file_name
            annotations = coco.loadAnns(
                coco.getAnnIds(imgIds=[int(image_record["id"])])
            )
            tasks = coco_tasks_for_image(
                dataset.dataset_id,
                image_record,
                annotations,
            )
            if not tasks:
                continue
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            completed = run_segmentation_tasks(
                run_id=run_id,
                image=image,
                tasks=tasks,
                segmenter=segmenter,
            )
            metadata_row = metadata_by_name[file_name]
            with temporary_predictions_path.open(mode, encoding="utf-8") as handle:
                for prediction in completed:
                    payload = prediction.record.to_dict()
                    payload["source_scene_id"] = str(metadata_row["source_scene_id"])
                    payload["stratum"] = str(metadata_row["stratum"])
                    payload["source_file_name"] = str(metadata_row["source_file_name"])
                    handle.write(
                        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    prediction_count += 1
            mode = "a"
            underlying = getattr(segmenter, "segmenter", None)
            if underlying is not None and hasattr(underlying, "maybe_clear_cuda_cache"):
                underlying.maybe_clear_cuda_cache(image_index)
    except Exception as exc:
        temporary_predictions_path.unlink(missing_ok=True)
        finish_run_manifest(manifest, status="failed", error=str(exc))
        write_run_manifest(manifest_path, manifest)
        writer_lock.close()
        raise

    temporary_predictions_path.replace(predictions_path)
    manifest["outputs"] = {
        "predictions": str(predictions_path),
        "prediction_count": prediction_count,
    }
    finish_run_manifest(manifest, status="completed")
    write_run_manifest(manifest_path, manifest)
    writer_lock.close()
    print(predictions_path)


if __name__ == "__main__":
    main()
