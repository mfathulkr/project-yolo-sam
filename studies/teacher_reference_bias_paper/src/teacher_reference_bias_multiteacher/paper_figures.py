from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools.coco import COCO


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from yolo_sam.segmentation.runner import decode_binary_mask

from .io import read_jsonl
from .paths import (
    BBOX_SOURCES,
    MODELS,
    REPO_ROOT,
    REFERENCES,
    STRATA,
    ExperimentSource,
    prediction_path,
    reference_path,
)


MODEL_COLORS = {"sam1": "#0072B2", "sam2": "#009E73", "sam3": "#D55E00"}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def reference_masks(
    source: ExperimentSource,
    reference_type: str,
    coco: COCO,
) -> dict[str, np.ndarray]:
    if reference_type in {"human", "published_samrs_reference"}:
        return {
            f"{source.dataset_id}:{int(annotation['image_id'])}:{int(annotation['id'])}": coco.annToMask(annotation).astype(bool)
            for annotation in coco.loadAnns(coco.getAnnIds())
        }
    return {
        str(row["instance_id"]): decode_binary_mask(row["mask_rle"])
        for row in read_jsonl(reference_path(source, reference_type))
    }


def representative_image_records(source: ExperimentSource) -> list[dict[str, object]]:
    """Select four qualitative images without looking at model/reference scores."""

    metadata = pd.read_csv(source.prepared_root / "test" / "metadata.csv")
    records: list[dict[str, object]] = []
    used_scenes: set[str] = set()
    for stratum in STRATA[1:]:
        candidates = metadata[metadata["stratum"] == stratum].copy()
        if candidates.empty:
            raise ValueError(f"Nitel figür için boş stratum: {stratum}")
        median = float(candidates["mask_area_ratio"].median())
        candidates["distance"] = (candidates["mask_area_ratio"] - median).abs()
        candidates = candidates.sort_values(["distance", "file_name"])
        unused = candidates[~candidates["source_scene_id"].isin(used_scenes)]
        if unused.empty:
            raise ValueError(
                f"Nitel figür için farklı kaynak sahne bulunamadı: {stratum}"
            )
        row = unused.iloc[0]
        used_scenes.add(str(row["source_scene_id"]))
        records.append(
            {
                "stratum": stratum,
                "canonical_image_id": f"{source.dataset_id}:{int(row['image_id'])}",
                "coco_image_id": int(row["image_id"]),
                "file_name": str(row["file_name"]),
                "source_scene_id": str(row["source_scene_id"]),
                "mask_area_ratio": float(row["mask_area_ratio"]),
                "stratum_mask_area_ratio_median": median,
                "selection_method": (
                    "model_and_reference_independent_stratum_median_mask_area"
                ),
            }
        )
    return records


def _reference_overlay(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    output = image.astype(np.float32).copy()
    color = np.asarray([0, 114, 178], dtype=np.float32)
    output[reference] = 0.40 * output[reference] + 0.60 * color
    return np.clip(output, 0, 255).astype(np.uint8)


def _error_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    output = image.astype(np.float32).copy()
    for mask, color in (
        (prediction & reference, np.asarray([0, 158, 115], dtype=np.float32)),
        (prediction & ~reference, np.asarray([230, 159, 0], dtype=np.float32)),
        (~prediction & reference, np.asarray([204, 121, 167], dtype=np.float32)),
    ):
        output[mask] = 0.35 * output[mask] + 0.65 * color
    return np.clip(output, 0, 255).astype(np.uint8)


def qualitative_figure(
    *,
    source: ExperimentSource,
    reference_type: str,
    metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    configure_style()
    coco = COCO(str(source.coco_path))
    predictions = {
        model: {
            str(row["instance_id"]): row
            for row in read_jsonl(prediction_path(source, model, "gt_bbox"))
        }
        for model in MODELS
    }
    references = reference_masks(source, reference_type, coco)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.loadAnns(coco.getAnnIds()):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(
            annotation
        )

    selected_images = representative_image_records(source)
    figure, axes = plt.subplots(4, 5, figsize=(11.2, 9.1), squeeze=False)
    titles = (
        "Input + all GT bbox",
        f"{REFERENCES[reference_type].display_name} union",
        "SAM1",
        "SAM2",
        "SAM3",
    )
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title)

    for row_index, selected_image in enumerate(selected_images):
        stratum = str(selected_image["stratum"])
        canonical_image_id = str(selected_image["canonical_image_id"])
        image_id = int(selected_image["coco_image_id"])
        annotations = sorted(
            annotations_by_image[image_id], key=lambda item: int(item["id"])
        )
        instance_ids = [
            f"{source.dataset_id}:{image_id}:{int(annotation['id'])}"
            for annotation in annotations
        ]
        missing = [
            instance_id
            for instance_id in instance_ids
            if instance_id not in references
            or any(instance_id not in predictions[model] for model in MODELS)
        ]
        if missing:
            raise ValueError(f"Nitel figürde eksik instance: {missing[:5]}")
        image_record = coco.loadImgs([image_id])[0]
        image = np.asarray(
            Image.open(source.images_root / image_record["file_name"]).convert("RGB")
        )
        reference = np.zeros(image.shape[:2], dtype=bool)
        for instance_id in instance_ids:
            reference |= references[instance_id]

        axes[row_index, 0].imshow(image)
        for annotation in annotations:
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            axes[row_index, 0].add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    fill=False,
                    edgecolor="#00FF66",
                    linewidth=1.3,
                )
            )
        axes[row_index, 1].imshow(_reference_overlay(image, reference))
        for column_index, model in enumerate(MODELS, 2):
            prediction = np.zeros(image.shape[:2], dtype=bool)
            for instance_id in instance_ids:
                prediction |= decode_binary_mask(
                    predictions[model][instance_id]["predicted_mask_rle"]
                )
            axes[row_index, column_index].imshow(
                _error_overlay(image, prediction, reference)
            )
            union = np.logical_or(prediction, reference).sum()
            iou = (
                float(np.logical_and(prediction, reference).sum() / union)
                if union
                else 0.0
            )
            axes[row_index, column_index].text(
                0.02,
                0.97,
                f"Union IoU {iou:.3f}",
                transform=axes[row_index, column_index].transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
            )
        axes[row_index, 0].set_ylabel(stratum.replace("__", "\n").replace("_", " "))
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    figure.text(
        0.5,
        0.006,
        "Yeşil: TP | Turuncu: FP | Pembe: FN. Her satırdaki bütün hedef instance'lar dahildir.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def qualitative_selection_records(
    source: ExperimentSource,
) -> list[dict[str, object]]:
    """Record the exact images and target-instance coverage shown in figures."""

    coco = COCO(str(source.coco_path))
    images = {int(row["id"]): row for row in coco.loadImgs(coco.getImgIds())}
    annotation_counts = {
        image_id: len(coco.getAnnIds(imgIds=[image_id])) for image_id in images
    }
    records: list[dict[str, object]] = []
    selected_images = representative_image_records(source)
    for reference_type in source.reference_types:
        for selection in selected_images:
            stratum = str(selection["stratum"])
            canonical_image_id = str(selection["canonical_image_id"])
            image_id = int(selection["coco_image_id"])
            instance_count = int(annotation_counts[image_id])
            if instance_count < 1:
                raise ValueError(
                    f"Nitel figür hedef instance içermiyor: {reference_type}/{image_id}"
                )
            records.append(
                {
                    "reference_type": reference_type,
                    "stratum": stratum,
                    "canonical_image_id": canonical_image_id,
                    "coco_image_id": image_id,
                    "file_name": str(images[image_id]["file_name"]),
                    "source_scene_id": str(selection["source_scene_id"]),
                    "mask_area_ratio": float(selection["mask_area_ratio"]),
                    "stratum_mask_area_ratio_median": float(
                        selection["stratum_mask_area_ratio_median"]
                    ),
                    "selection_method": str(selection["selection_method"]),
                    "target_instance_count": instance_count,
                    "prompt_count_per_model": instance_count,
                    "display_scope": "all_target_instances",
                }
            )
    return records


def model_reference_matrix_figure(
    source: ExperimentSource,
    aggregates: pd.DataFrame,
    output_path: Path,
) -> Path:
    configure_style()
    overall = aggregates[aggregates["stratum"] == "overall"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), squeeze=False)
    for column, bbox_source in enumerate(BBOX_SOURCES):
        pivot = (
            overall[overall["bbox_source"] == bbox_source]
            .pivot(index="model", columns="reference_type", values="mean_iou")
            .loc[list(MODELS), list(source.reference_types)]
        )
        axis = axes[0, column]
        image = axis.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1)
        for y in range(pivot.shape[0]):
            for x in range(pivot.shape[1]):
                value = float(pivot.iloc[y, x])
                axis.text(
                    x,
                    y,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.30 or value > 0.82 else "black",
                    fontweight="bold",
                )
        axis.set_xticks(
            range(len(source.reference_types)),
            [REFERENCES[name].display_name.replace(" referansı", "") for name in source.reference_types],
            rotation=20,
            ha="right",
        )
        axis.set_yticks(range(3), [model.upper() for model in MODELS])
        axis.set_title("GT bbox" if bbox_source == "gt_bbox" else "YOLO bbox")
        axis.set_xlabel("Değerlendirme referansı")
        axis.set_ylabel("Değerlendirilen model")
    figure.suptitle(f"{source.experiment_id}: model–referans Overall Avg IoU")
    figure.subplots_adjust(left=0.07, right=0.91, bottom=0.22, top=0.86, wspace=0.30)
    color_axis = figure.add_axes((0.93, 0.20, 0.015, 0.62))
    figure.colorbar(image, cax=color_axis, label="Avg IoU")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return output_path


def reference_effect_figure(
    source: ExperimentSource,
    effects: pd.DataFrame,
    output_path: Path,
) -> Path:
    configure_style()
    comparisons = source.reference_types[1:]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), sharey=True)
    offsets = np.linspace(-0.24, 0.24, len(comparisons))
    for axis, bbox_source in zip(axes, BBOX_SOURCES, strict=True):
        selected = effects[effects["bbox_source"] == bbox_source]
        x = np.arange(3)
        for reference_type, offset, color in zip(
            comparisons,
            offsets,
            (MODEL_COLORS["sam1"], MODEL_COLORS["sam2"], MODEL_COLORS["sam3"]),
            strict=True,
        ):
            rows = (
                selected[selected["comparison_reference"] == reference_type]
                .set_index("model")
                .loc[list(MODELS)]
            )
            values = rows["delta_iou"].to_numpy(float)
            errors = np.vstack(
                (
                    values - rows["delta_ci_lower"].to_numpy(float),
                    rows["delta_ci_upper"].to_numpy(float) - values,
                )
            )
            axis.errorbar(
                x + offset,
                values,
                yerr=errors,
                fmt="o",
                capsize=3,
                label=REFERENCES[reference_type].display_name,
                color=color,
            )
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_xticks(x, [model.upper() for model in MODELS])
        axis.set_title("GT bbox" if bbox_source == "gt_bbox" else "YOLO bbox")
        axis.set_ylabel("Referans − temel referans Avg IoU farkı")
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=7, loc="best")
    figure.suptitle(
        f"{source.experiment_id}: referans kaynaklı skor değişimi (%95 sahne-kümeli GA)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_figure_manifest(
    source: ExperimentSource,
    outputs: list[Path],
    inputs: list[Path],
    qualitative_selection: list[dict[str, object]],
) -> Path:
    import hashlib

    def row(path: Path) -> dict[str, object]:
        return {
            "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    path = source.figures_root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "status": "completed",
                "experiment_id": source.experiment_id,
                "qualitative_scope": "all_target_instances_in_selected_images",
                "qualitative_selection": qualitative_selection,
                "inputs": [row(item) for item in sorted(set(inputs))],
                "outputs": [row(item) for item in sorted(set(outputs))],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
