from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.isaid import (
    bbox_iou_xywh,
    clip_bbox_to_tile,
    normalize_category_name,
)
from yolo_sam.evaluation.instance_metrics import (
    InstanceMetricRow,
    aggregate_instance_metrics,
    binary_mask_metrics,
    evaluate_prediction_references,
    reference_inflation_rows,
)
from yolo_sam.evaluation.statistics import (
    clustered_bootstrap_mean,
    clustered_inflation_interval,
    compare_model_rankings,
)
from teacher_reference_bias.reporting.analysis import sha256_file
from yolo_sam.segmentation.runner import (
    decode_binary_mask,
    encode_binary_mask,
)


MODELS = ("sam1", "sam2", "sam3")
STRATA = (
    "no_overlap__low_mask_area",
    "no_overlap__high_mask_area",
    "overlap__low_mask_area",
    "overlap__high_mask_area",
)


def classify_tile_mapping(
    template_score: float,
    *,
    exact_pixels: bool,
    min_template_score: float,
) -> str:
    if template_score < min_template_score:
        return "low_template_score"
    if not exact_pixels:
        return "pixel_mismatch"
    return "matched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map SAMRS SOTA tiles back to iSAID source images and evaluate the "
            "SAM1 pseudo masks against independently annotated human masks."
        )
    )
    parser.add_argument(
        "--isaid-root",
        type=Path,
        default=ROOT / "datasets" / "isaid" / "raw",
    )
    parser.add_argument(
        "--samrs-prepared-root",
        type=Path,
        default=(
            STUDY_ROOT
            / "data"
            / "prepared"
            / "samrs_sota_small_vehicle"
        ),
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=STUDY_ROOT / "results",
    )
    parser.add_argument("--bbox-match-iou", type=float, default=0.50)
    parser.add_argument("--template-scale", type=float, default=0.25)
    parser.add_argument("--min-template-score", type=float, default=0.995)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def unique_object_sensitivity(
    model_metrics: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    metric_columns = ("iou", "dice", "precision", "recall", "boundary_iou")
    mapping = matches[
        ["instance_id", "human_object_key", "source_scene_id"]
    ].drop_duplicates()
    if mapping["instance_id"].duplicated().any():
        raise ValueError("Instance maps to more than one human object")
    augmented = model_metrics.merge(
        mapping,
        on=["instance_id", "source_scene_id"],
        how="inner",
        validate="many_to_one",
    )
    if len(augmented) != len(model_metrics):
        raise ValueError("Some model metric rows have no human-object mapping")

    object_metrics = (
        augmented.groupby(
            [
                "model_version",
                "reference_type",
                "human_object_key",
                "source_scene_id",
            ],
            as_index=False,
            sort=True,
        )[list(metric_columns)]
        .mean()
    )
    rows: list[dict[str, object]] = []
    for model in MODELS:
        selected = object_metrics[object_metrics["model_version"] == model]
        pivot = selected.pivot(
            index=["human_object_key", "source_scene_id"],
            columns="reference_type",
            values="iou",
        ).reset_index()
        if not {"human", "pseudo_sam1"}.issubset(pivot.columns):
            raise ValueError(f"{model} unique-object sensitivity is not dual-reference")
        pivot["iou_inflation"] = (
            pivot["pseudo_sam1"].astype(float) - pivot["human"].astype(float)
        )
        interval = clustered_bootstrap_mean(
            {
                str(scene_id): group["iou_inflation"].astype(float).tolist()
                for scene_id, group in pivot.groupby(
                    "source_scene_id",
                    sort=True,
                )
            },
            bootstrap_samples=bootstrap_samples,
            confidence_level=0.95,
            seed=bootstrap_seed,
        )
        row: dict[str, object] = {
            "model": model,
            "unique_human_objects": int(pivot["human_object_key"].nunique()),
            "source_scenes": int(pivot["source_scene_id"].nunique()),
            "mean_iou_inflation": interval.estimate,
            "iou_inflation_ci_lower": interval.lower,
            "iou_inflation_ci_upper": interval.upper,
            "bootstrap_samples": interval.bootstrap_samples,
        }
        for reference_type in ("human", "pseudo_sam1"):
            reference_rows = selected[
                selected["reference_type"] == reference_type
            ]
            for metric in metric_columns:
                row[f"{reference_type}_mean_{metric}"] = float(
                    reference_rows[metric].mean()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def load_isaid_small_vehicle_annotations(
    isaid_root: Path,
) -> tuple[
    dict[str, Path],
    dict[str, list[dict[str, Any]]],
    list[Path],
]:
    image_paths: dict[str, Path] = {}
    annotations_by_file: dict[str, list[dict[str, Any]]] = {}
    input_paths: list[Path] = []
    for split in ("train", "val"):
        annotation_path = (
            isaid_root / split / "Annotations" / f"iSAID_{split}.json"
        )
        input_paths.append(annotation_path)
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {
            int(row["id"]): normalize_category_name(str(row["name"]))
            for row in payload["categories"]
        }
        file_by_image_id = {
            int(row["id"]): str(row["file_name"])
            for row in payload["images"]
        }
        for file_name in file_by_image_id.values():
            image_path = isaid_root / split / "images" / file_name
            if file_name in image_paths:
                raise ValueError(f"Duplicate iSAID source image: {file_name}")
            image_paths[file_name] = image_path
        for annotation in payload["annotations"]:
            category_name = normalize_category_name(
                str(
                    annotation.get("category_name")
                    or categories[int(annotation["category_id"])]
                )
            )
            if category_name != "small-vehicle":
                continue
            file_name = file_by_image_id[int(annotation["image_id"])]
            annotations_by_file.setdefault(file_name, []).append(annotation)
    return image_paths, annotations_by_file, input_paths


def locate_tile_in_source(
    source: np.ndarray,
    tile: np.ndarray,
    *,
    scale: float,
) -> tuple[int, int, float, bool]:
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

    source_height, source_width = source_gray.shape
    tile_height, tile_width = tile_gray.shape
    if source_height < tile_height or source_width < tile_width:
        if source_height >= tile_height and source_width < tile_width:
            result = cv2.matchTemplate(
                source_gray,
                tile_gray[:, :source_width],
                cv2.TM_CCOEFF_NORMED,
            )
            _, score, _, location = cv2.minMaxLoc(result)
            tile_x, tile_y = 0, int(location[1])
        elif source_height < tile_height and source_width >= tile_width:
            result = cv2.matchTemplate(
                source_gray,
                tile_gray[:source_height, :],
                cv2.TM_CCOEFF_NORMED,
            )
            _, score, _, location = cv2.minMaxLoc(result)
            tile_x, tile_y = int(location[0]), 0
        else:
            result = cv2.matchTemplate(
                tile_gray,
                source_gray,
                cv2.TM_CCOEFF_NORMED,
            )
            _, score, _, location = cv2.minMaxLoc(result)
            tile_x, tile_y = -int(location[0]), -int(location[1])

        source_x1 = max(0, tile_x)
        source_y1 = max(0, tile_y)
        source_x2 = min(source_width, tile_x + tile_width)
        source_y2 = min(source_height, tile_y + tile_height)
        tile_x1 = source_x1 - tile_x
        tile_y1 = source_y1 - tile_y
        tile_x2 = source_x2 - tile_x
        tile_y2 = source_y2 - tile_y
        exact_pixels = np.array_equal(
            source[source_y1:source_y2, source_x1:source_x2],
            tile[tile_y1:tile_y2, tile_x1:tile_x2],
        )
        return tile_x, tile_y, float(score), bool(exact_pixels)

    source_small = cv2.resize(
        source_gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    tile_small = cv2.resize(
        tile_gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    coarse_result = cv2.matchTemplate(
        source_small,
        tile_small,
        cv2.TM_CCOEFF_NORMED,
    )
    _, _, _, coarse_location = cv2.minMaxLoc(coarse_result)
    approximate_x = int(round(coarse_location[0] / scale))
    approximate_y = int(round(coarse_location[1] / scale))

    margin = max(8, int(round(4.0 / scale)))
    search_x1 = max(0, approximate_x - margin)
    search_y1 = max(0, approximate_y - margin)
    search_x2 = min(
        source.shape[1],
        approximate_x + tile.shape[1] + margin,
    )
    search_y2 = min(
        source.shape[0],
        approximate_y + tile.shape[0] + margin,
    )
    search_region = source_gray[search_y1:search_y2, search_x1:search_x2]
    result = cv2.matchTemplate(
        search_region,
        tile_gray,
        cv2.TM_CCOEFF_NORMED,
    )
    _, score, _, location = cv2.minMaxLoc(result)
    tile_x = search_x1 + location[0]
    tile_y = search_y1 + location[1]
    exact_pixels = np.array_equal(
        source[tile_y : tile_y + tile.shape[0], tile_x : tile_x + tile.shape[1]],
        tile,
    )
    return tile_x, tile_y, float(score), bool(exact_pixels)


def human_mask_in_tile(
    annotation: dict[str, Any],
    *,
    tile_x: int,
    tile_y: int,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    mask = np.zeros((tile_height, tile_width), dtype=np.uint8)
    for polygon in annotation.get("segmentation", []):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        points[:, 0] -= tile_x
        points[:, 1] -= tile_y
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], color=1)
    return mask.astype(bool)


def match_annotations(
    samrs_annotations: list[dict[str, Any]],
    human_annotations: list[dict[str, Any]],
    human_tile_boxes: list[list[float]],
    *,
    threshold: float,
) -> list[tuple[int, int, float]]:
    if not samrs_annotations or not human_annotations:
        return []
    similarities = np.asarray(
        [
            [
                bbox_iou_xywh(
                    [float(value) for value in samrs_annotation["bbox"]],
                    human_box,
                )
                for human_box in human_tile_boxes
            ]
            for samrs_annotation in samrs_annotations
        ],
        dtype=np.float64,
    )
    samrs_indices, human_indices = linear_sum_assignment(1.0 - similarities)
    return [
        (int(samrs_index), int(human_index), float(similarities[samrs_index, human_index]))
        for samrs_index, human_index in zip(
            samrs_indices,
            human_indices,
            strict=True,
        )
        if similarities[samrs_index, human_index] >= threshold
    ]


def summarize_metric_rows(
    rows: list[InstanceMetricRow],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for model in MODELS:
        model_rows = [row for row in rows if row.model_version == model]
        for reference_type in ("human", "pseudo_sam1"):
            reference_rows = [
                row
                for row in model_rows
                if row.reference_type == reference_type
            ]
            for stratum in ("overall", *STRATA):
                selected = (
                    reference_rows
                    if stratum == "overall"
                    else [row for row in reference_rows if row.stratum == stratum]
                )
                if not selected:
                    continue
                aggregate = aggregate_instance_metrics(selected)
                output.append(
                    {
                        "model": model,
                        "reference_type": reference_type,
                        "stratum": stratum,
                        **aggregate.to_dict(),
                    }
                )
    return output


def main() -> None:
    args = parse_args()
    output_root = args.study_root / "analysis" / "shared_human_reference_audit"
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"{manifest_path} exists. Use --force for an intentional rerun."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    samrs_coco_path = (
        args.samrs_prepared_root / "test" / "_annotations.coco.json"
    )
    samrs_metadata_path = args.samrs_prepared_root / "test" / "metadata.csv"
    samrs_images_root = args.samrs_prepared_root / "test" / "images"
    samrs_coco = COCO(str(samrs_coco_path))
    metadata = pd.read_csv(samrs_metadata_path)
    metadata_by_file = metadata.set_index("file_name").to_dict("index")
    (
        isaid_image_paths,
        human_annotations_by_file,
        isaid_annotation_paths,
    ) = load_isaid_small_vehicle_annotations(args.isaid_root)

    predictions_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    prediction_paths: list[Path] = []
    for model in MODELS:
        prediction_path = (
            args.study_root
            / "predictions"
            / "samrs_sota_small_vehicle"
            / model
            / "gt_bbox"
            / "predictions.jsonl"
        )
        prediction_paths.append(prediction_path)
        predictions_by_model[model] = {
            str(row["instance_id"]): row
            for row in read_jsonl(prediction_path)
        }

    match_rows: list[dict[str, object]] = []
    reference_quality_rows: list[dict[str, object]] = []
    model_metric_rows: list[InstanceMetricRow] = []
    human_reference_rows: list[dict[str, object]] = []
    image_mapping_rows: list[dict[str, object]] = []

    for image_record in samrs_coco.loadImgs(sorted(samrs_coco.getImgIds())):
        image_id = int(image_record["id"])
        file_name = str(image_record["file_name"])
        source_scene_id = str(metadata_by_file[file_name]["source_scene_id"])
        source_file_name = f"{source_scene_id}.png"
        source_path = isaid_image_paths.get(source_file_name)
        if source_path is None:
            image_mapping_rows.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "source_file_name": source_file_name,
                    "status": "missing_isaid_source",
                }
            )
            continue
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        tile_path = samrs_images_root / file_name
        tile = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
        if source is None or tile is None:
            raise ValueError(f"Could not read {source_path} or {tile_path}")
        tile_x, tile_y, template_score, exact_pixels = locate_tile_in_source(
            source,
            tile,
            scale=args.template_scale,
        )
        mapping_status = classify_tile_mapping(
            template_score,
            exact_pixels=exact_pixels,
            min_template_score=args.min_template_score,
        )
        image_mapping_rows.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "source_file_name": source_file_name,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "template_score": template_score,
                "exact_pixels": exact_pixels,
                "status": mapping_status,
            }
        )
        if mapping_status != "matched":
            continue

        human_annotations: list[dict[str, Any]] = []
        human_tile_boxes: list[list[float]] = []
        for annotation in human_annotations_by_file.get(source_file_name, []):
            clipped_bbox = clip_bbox_to_tile(
                [float(value) for value in annotation["bbox"]],
                tile_x,
                tile_y,
                tile.shape[1],
            )
            if clipped_bbox is None:
                continue
            mask = human_mask_in_tile(
                annotation,
                tile_x=tile_x,
                tile_y=tile_y,
                tile_width=tile.shape[1],
                tile_height=tile.shape[0],
            )
            if int(mask.sum()) < 8:
                continue
            copied = dict(annotation)
            copied["_tile_mask"] = mask
            human_annotations.append(copied)
            human_tile_boxes.append(clipped_bbox)

        samrs_annotations = sorted(
            samrs_coco.loadAnns(samrs_coco.getAnnIds(imgIds=[image_id])),
            key=lambda row: int(row["id"]),
        )
        matches = match_annotations(
            samrs_annotations,
            human_annotations,
            human_tile_boxes,
            threshold=args.bbox_match_iou,
        )
        for samrs_index, human_index, bbox_iou in matches:
            samrs_annotation = samrs_annotations[samrs_index]
            human_annotation = human_annotations[human_index]
            instance_id = (
                f"samrs_sota_small_vehicle:{image_id}:{int(samrs_annotation['id'])}"
            )
            human_mask = np.asarray(
                human_annotation["_tile_mask"],
                dtype=bool,
            )
            pseudo_mask = samrs_coco.annToMask(samrs_annotation).astype(bool)
            quality = binary_mask_metrics(pseudo_mask, human_mask)
            stratum = str(metadata_by_file[file_name]["stratum"])
            match_rows.append(
                {
                    "instance_id": instance_id,
                    "image_id": image_id,
                    "file_name": file_name,
                    "source_scene_id": source_scene_id,
                    "stratum": stratum,
                    "samrs_annotation_id": int(samrs_annotation["id"]),
                    "isaid_annotation_id": int(human_annotation["id"]),
                    "human_object_key": (
                        f"{source_file_name}:{int(human_annotation['id'])}"
                    ),
                    "bbox_iou": bbox_iou,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "template_score": template_score,
                    "exact_pixels": exact_pixels,
                }
            )
            reference_quality_rows.append(
                {
                    "instance_id": instance_id,
                    "source_scene_id": source_scene_id,
                    "stratum": stratum,
                    **asdict(quality),
                }
            )
            human_reference_rows.append(
                {
                    "instance_id": instance_id,
                    "source_scene_id": source_scene_id,
                    "stratum": stratum,
                    "isaid_annotation_id": int(human_annotation["id"]),
                    "bbox_match_iou": bbox_iou,
                    "mask_rle": encode_binary_mask(human_mask),
                }
            )
            for model in MODELS:
                prediction_row = predictions_by_model[model][instance_id]
                prediction_mask = decode_binary_mask(
                    prediction_row["predicted_mask_rle"]
                )
                model_metric_rows.extend(
                    evaluate_prediction_references(
                        run_id=str(prediction_row["run_id"]),
                        model_id=str(prediction_row["model_id"]),
                        model_version=model,
                        prompt_type=str(prediction_row["prompt_type"]),
                        image_id=str(prediction_row["image_id"]),
                        instance_id=instance_id,
                        source_scene_id=source_scene_id,
                        stratum=stratum,
                        prediction=prediction_mask,
                        references={
                            "human": human_mask,
                            "pseudo_sam1": pseudo_mask,
                        },
                    )
                )

    if not match_rows:
        raise ValueError("No SAMRS-to-iSAID small-vehicle instances were matched")

    mappings = pd.DataFrame(image_mapping_rows)
    matches = pd.DataFrame(match_rows)
    reference_quality = pd.DataFrame(reference_quality_rows)
    model_metrics = pd.DataFrame([row.to_dict() for row in model_metric_rows])
    model_summary = pd.DataFrame(summarize_metric_rows(model_metric_rows))
    unique_object_results = unique_object_sensitivity(
        model_metrics,
        matches,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    inflation_rows = reference_inflation_rows(model_metric_rows)
    inflation = pd.DataFrame(inflation_rows)
    inflation_intervals = []
    for model in MODELS:
        model_rows = [
            row
            for row in inflation_rows
            if str(row["run_id"]).endswith(f"{model}-gt-bbox")
        ]
        if not model_rows:
            model_rows = [
                row
                for row in inflation_rows
                if model in str(row["run_id"])
            ]
        interval = clustered_inflation_interval(
            model_rows,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=0.95,
            seed=args.bootstrap_seed,
        )
        inflation_intervals.append({"model": model, **asdict(interval)})

    ranking = compare_model_rankings(model_metric_rows)
    reference_summary_rows: list[dict[str, object]] = []
    for stratum in ("overall", *STRATA):
        selected = (
            reference_quality
            if stratum == "overall"
            else reference_quality[reference_quality["stratum"] == stratum]
        )
        if selected.empty:
            continue
        reference_summary_rows.append(
            {
                "stratum": stratum,
                "count": int(len(selected)),
                "source_scenes": int(selected["source_scene_id"].nunique()),
                **{
                    f"mean_{metric}": float(selected[metric].mean())
                    for metric in (
                        "iou",
                        "dice",
                        "precision",
                        "recall",
                        "boundary_iou",
                    )
                },
            }
        )

    output_paths = {
        "image_mappings": output_root / "image_mappings.csv",
        "instance_matches": output_root / "instance_matches.csv",
        "human_references": output_root / "human_references.jsonl",
        "reference_quality": output_root / "reference_quality_instance.csv",
        "reference_summary": output_root / "reference_quality_summary.csv",
        "model_metrics": output_root / "model_dual_reference_metrics.csv",
        "model_summary": output_root / "model_dual_reference_summary.csv",
        "unique_object_sensitivity": (
            output_root / "unique_human_object_sensitivity.csv"
        ),
        "reference_inflation": output_root / "model_reference_inflation.csv",
        "inflation_intervals": output_root / "model_reference_inflation_ci.json",
        "ranking": output_root / "ranking_comparison.json",
    }
    mappings.to_csv(output_paths["image_mappings"], index=False)
    matches.to_csv(output_paths["instance_matches"], index=False)
    with output_paths["human_references"].open("w", encoding="utf-8") as handle:
        for row in human_reference_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    reference_quality.to_csv(output_paths["reference_quality"], index=False)
    pd.DataFrame(reference_summary_rows).to_csv(
        output_paths["reference_summary"],
        index=False,
    )
    model_metrics.to_csv(output_paths["model_metrics"], index=False)
    model_summary.to_csv(output_paths["model_summary"], index=False)
    unique_object_results.to_csv(
        output_paths["unique_object_sensitivity"],
        index=False,
    )
    inflation.to_csv(output_paths["reference_inflation"], index=False)
    write_json(output_paths["inflation_intervals"], inflation_intervals)
    write_json(output_paths["ranking"], asdict(ranking))

    output_hashes = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in output_paths.items()
    }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "parameters": {
            "bbox_match_iou": args.bbox_match_iou,
            "template_scale": args.template_scale,
            "min_template_score": args.min_template_score,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "counts": {
            "samrs_test_images": len(samrs_coco.getImgIds()),
            "mapped_images": int((mappings["status"] == "matched").sum()),
            "pixel_exact_images": int(
                (
                    mappings["status"].eq("matched")
                    & mappings["exact_pixels"].fillna(False).astype(bool)
                ).sum()
            ),
            "samrs_instances": len(samrs_coco.getAnnIds()),
            "matched_instances": len(matches),
            "unique_human_objects": int(
                matches["human_object_key"].nunique()
            ),
            "matched_source_scenes": int(matches["source_scene_id"].nunique()),
        },
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in [
                samrs_coco_path,
                samrs_metadata_path,
                *isaid_annotation_paths,
                *prediction_paths,
            ]
        ],
        "outputs": output_hashes,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(output_root)


if __name__ == "__main__":
    main()
