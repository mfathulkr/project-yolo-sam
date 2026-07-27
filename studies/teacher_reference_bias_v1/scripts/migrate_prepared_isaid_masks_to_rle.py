from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.isaid import encode_coco_rle
from teacher_reference_bias.config import load_dataset_study_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace lossy prepared iSAID contour polygons with exact RLE "
            "masks reconstructed from the official source polygons."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=STUDY_ROOT / "configs" / "datasets" / "isaid_plane.yaml",
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=(
            STUDY_ROOT
            / "results"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_annotations(
    raw_root: Path,
    target_category: str,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[Path]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    source_files: list[Path] = []
    for split in ("train", "val"):
        path = raw_root / split / "Annotations" / f"iSAID_{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_files.append(path)
        target_ids = {
            int(category["id"])
            for category in payload["categories"]
            if str(category["name"]).strip().lower()
            == target_category.strip().lower()
        }
        if len(target_ids) != 1:
            raise ValueError(
                f"Expected one {target_category!r} category in {path}, "
                f"got {target_ids}"
            )
        file_name_by_image_id = {
            int(image["id"]): str(image["file_name"])
            for image in payload["images"]
        }
        for annotation in payload["annotations"]:
            if int(annotation["category_id"]) not in target_ids:
                continue
            key = (
                file_name_by_image_id[int(annotation["image_id"])],
                int(annotation["id"]),
            )
            if key in output:
                raise ValueError(f"Duplicate source annotation key: {key}")
            output[key] = annotation
    return output, source_files


def rasterize_source_annotation(
    annotation: dict[str, Any],
    *,
    tile_x: int,
    tile_y: int,
    tile_size: int,
) -> np.ndarray:
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    for polygon in annotation.get("segmentation", []):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        points[:, 0] -= tile_x
        points[:, 1] -= tile_y
        cv2.fillPoly(
            mask,
            [np.rint(points).astype(np.int32)],
            color=1,
        )
    return mask


def decoded_area(segmentation: object, height: int, width: int) -> int:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        encoded = mask_utils.merge(rles)
    else:
        encoded = segmentation
    return int(mask_utils.decode(encoded).sum())


def migrate_split(
    *,
    split_root: Path,
    source_by_key: dict[tuple[str, int], dict[str, Any]],
    archive_root: Path,
) -> dict[str, object]:
    coco_path = split_root / "_annotations.coco.json"
    metadata_path = split_root / "metadata.csv"
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    metadata = pd.read_csv(metadata_path)
    metadata_by_image_id = {
        int(row["image_id"]): row
        for _, row in metadata.iterrows()
    }
    image_size_by_id = {
        int(image["id"]): (int(image["height"]), int(image["width"]))
        for image in payload["images"]
    }
    before_empty = 0
    for annotation in payload["annotations"]:
        height, width = image_size_by_id[int(annotation["image_id"])]
        if decoded_area(annotation["segmentation"], height, width) == 0:
            before_empty += 1

    archive_path = archive_root / f"{split_root.name}_annotations.coco.json"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise FileExistsError(
            f"Pre-fix archive already exists and will not be overwritten: "
            f"{archive_path}"
        )
    shutil.copy2(coco_path, archive_path)
    before_sha256 = sha256_file(coco_path)

    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        row = metadata_by_image_id[image_id]
        key = (
            str(row["source_file_name"]),
            int(annotation["source_annotation_id"]),
        )
        source = source_by_key.get(key)
        if source is None:
            raise KeyError(f"Missing official source annotation: {key}")
        tile_size = int(row["tile_size"])
        mask = rasterize_source_annotation(
            source,
            tile_x=int(row["tile_x"]),
            tile_y=int(row["tile_y"]),
            tile_size=tile_size,
        )
        actual_area = int(mask.sum())
        declared_area = int(annotation["area"])
        if actual_area != declared_area:
            raise ValueError(
                f"Source reconstruction area mismatch for {key}: "
                f"{actual_area} != {declared_area}"
            )
        annotation["segmentation"] = encode_coco_rle(mask)

    temporary_path = coco_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(coco_path)

    after_empty = 0
    area_mismatches = 0
    for annotation in payload["annotations"]:
        height, width = image_size_by_id[int(annotation["image_id"])]
        area = decoded_area(annotation["segmentation"], height, width)
        after_empty += int(area == 0)
        area_mismatches += int(area != int(annotation["area"]))
    if after_empty or area_mismatches:
        raise RuntimeError(
            f"Post-migration validation failed for {split_root.name}: "
            f"empty={after_empty}, area_mismatches={area_mismatches}"
        )

    return {
        "split": split_root.name,
        "annotations": len(payload["annotations"]),
        "empty_masks_before": before_empty,
        "empty_masks_after": after_empty,
        "area_mismatches_after": area_mismatches,
        "before_sha256": before_sha256,
        "after_sha256": sha256_file(coco_path),
        "archive_path": str(archive_path),
    }


def main() -> None:
    args = parse_args()
    dataset = load_dataset_study_config(args.dataset)
    if dataset.profile_id != "isaid":
        raise ValueError("This migration only supports the iSAID profile")
    source_by_key, source_files = source_annotations(
        dataset.raw_root,
        dataset.target_category,
    )
    archive_root = (
        args.study_root / "audits" / "pre_isaid_lossless_rle_fix"
    )
    rows = [
        migrate_split(
            split_root=dataset.prepared_root / split,
            source_by_key=source_by_key,
            archive_root=archive_root,
        )
        for split in ("train", "validation", "test")
    ]
    audit_path = (
        args.study_root / "audits" / "isaid_lossless_rle_migration.json"
    )
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "dataset_id": dataset.dataset_id,
                "method": (
                    "official source polygon -> OpenCV tile raster -> "
                    "compressed COCO RLE"
                ),
                "source_files": [
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                    for path in source_files
                ],
                "splits": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(audit_path)


if __name__ == "__main__":
    main()
