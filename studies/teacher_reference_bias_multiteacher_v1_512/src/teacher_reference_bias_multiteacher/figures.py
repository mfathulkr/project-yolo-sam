from __future__ import annotations

import hashlib
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

from yolo_sam.segmentation.runner import decode_binary_mask

from .io import read_jsonl
from .paths import DATASETS, MODELS, STRATA, prediction_path, reference_path


MODEL_COLORS = {"sam1": "#0072B2", "sam2": "#009E73", "sam3": "#D55E00"}
REFERENCE_LABELS = {
    "human": "İnsan",
    "pseudo_sam1": "SAM1 pseudo",
    "pseudo_sam2": "SAM2 pseudo",
    "pseudo_sam3": "SAM3 pseudo",
}


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


def _select_representative_images(
    metrics: pd.DataFrame,
    *,
    dataset_id: str,
    reference_type: str,
) -> list[str]:
    selected = metrics[
        (metrics["dataset_id"] == dataset_id)
        & (metrics["reference_type"] == reference_type)
        & (metrics["bbox_source"] == "gt_bbox")
    ]
    coverage = selected.groupby("instance_id")["model"].nunique()
    if set(coverage.astype(int)) != {3}:
        raise ValueError(f"Eksik model kapsamı: {dataset_id}/{reference_type}")
    per_image = (
        selected.groupby(
            ["image_id", "source_scene_id", "stratum"],
            as_index=False,
            sort=True,
        )["iou"]
        .mean()
        .rename(columns={"iou": "mean_model_iou"})
    )
    image_ids: list[str] = []
    used_scenes: set[str] = set()
    for stratum in STRATA[1:]:
        candidates = per_image[per_image["stratum"] == stratum].copy()
        median = float(candidates["mean_model_iou"].median())
        candidates["distance"] = (candidates["mean_model_iou"] - median).abs()
        candidates = candidates.sort_values(["distance", "image_id"])
        unused = candidates[~candidates["source_scene_id"].isin(used_scenes)]
        row = (unused if not unused.empty else candidates).iloc[0]
        image_ids.append(str(row["image_id"]))
        used_scenes.add(str(row["source_scene_id"]))
    return image_ids


def qualitative_figure(
    *,
    dataset_id: str,
    reference_type: str,
    metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    if reference_type not in {"pseudo_sam2", "pseudo_sam3"}:
        raise ValueError("Bu uzantıda yalnız SAM2/SAM3 nitel figürü üretilir")
    configure_style()
    source = DATASETS[dataset_id]
    coco = COCO(str(source.coco_path))
    predictions = {
        model: {
            str(row["instance_id"]): row
            for row in read_jsonl(prediction_path(source, model, "gt_bbox"))
        }
        for model in MODELS
    }
    teacher = reference_type.removeprefix("pseudo_")
    references = {
        str(row["instance_id"]): row
        for row in read_jsonl(reference_path(dataset_id, teacher))
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation_id in coco.getAnnIds():
        annotation = coco.loadAnns([annotation_id])[0]
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(
            annotation
        )

    image_ids = _select_representative_images(
        metrics,
        dataset_id=dataset_id,
        reference_type=reference_type,
    )
    figure, axes = plt.subplots(4, 5, figsize=(11.2, 9.1), squeeze=False)
    titles = (
        "Input + all GT bbox",
        f"{teacher.upper()} reference union",
        "SAM1",
        "SAM2",
        "SAM3",
    )
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title)

    for row_index, (stratum, canonical_image_id) in enumerate(
        zip(STRATA[1:], image_ids, strict=True)
    ):
        image_id = int(canonical_image_id.rsplit(":", 1)[-1])
        annotations = sorted(
            annotations_by_image[image_id], key=lambda item: int(item["id"])
        )
        instance_ids = [
            f"{dataset_id}:{image_id}:{int(annotation['id'])}"
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
            reference |= decode_binary_mask(references[instance_id]["mask_rle"])

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
                else 1.0
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


def model_reference_matrix_figure(
    aggregates: pd.DataFrame,
    output_path: Path,
) -> Path:
    configure_style()
    overall = aggregates[aggregates["stratum"] == "overall"]
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), squeeze=False)
    for row, dataset_id in enumerate(DATASETS):
        for column, bbox_source in enumerate(("gt_bbox", "yolo_bbox")):
            selected = overall[
                (overall["dataset_id"] == dataset_id)
                & (overall["bbox_source"] == bbox_source)
            ]
            pivot = selected.pivot_table(
                index="model",
                columns="reference_type",
                values="mean_iou",
                aggfunc="mean",
            ).loc[list(MODELS), ["human", "pseudo_sam1", "pseudo_sam2", "pseudo_sam3"]]
            axis = axes[row, column]
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
                        color="white" if value < 0.30 or value > 0.78 else "black",
                        fontweight="bold",
                    )
            axis.set_xticks(range(4), ["Human", "SAM1", "SAM2", "SAM3"])
            axis.set_yticks(range(3), ["SAM1", "SAM2", "SAM3"])
            axis.set_title(f"{dataset_id.replace('_', ' ')} · {bbox_source.replace('_', ' ')}")
            axis.set_xlabel("Evaluation reference")
            axis.set_ylabel("Evaluated model")
    figure.suptitle("Model–reference IoU matrix", fontsize=13, fontweight="bold")
    figure.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.08,
        top=0.90,
        wspace=0.28,
        hspace=0.34,
    )
    color_axis = figure.add_axes((0.91, 0.18, 0.022, 0.64))
    figure.colorbar(image, cax=color_axis, label="Mean instance IoU")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return output_path


def reference_effect_figure(effects: pd.DataFrame, output_path: Path) -> Path:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharey=True)
    reference_order = ("pseudo_sam1", "pseudo_sam2", "pseudo_sam3")
    offsets = (-0.24, 0.0, 0.24)
    for row, dataset_id in enumerate(DATASETS):
        for column, bbox_source in enumerate(("gt_bbox", "yolo_bbox")):
            axis = axes[row, column]
            selected = effects[
                (effects["dataset_id"] == dataset_id)
                & (effects["bbox_source"] == bbox_source)
            ]
            x = np.arange(3)
            for reference_type, offset, color in zip(
                reference_order,
                offsets,
                (MODEL_COLORS["sam1"], MODEL_COLORS["sam2"], MODEL_COLORS["sam3"]),
                strict=True,
            ):
                rows = selected[selected["pseudo_reference"] == reference_type].set_index("model").loc[list(MODELS)]
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
                    label=REFERENCE_LABELS[reference_type],
                    color=color,
                )
            axis.axhline(0, color="#444444", linewidth=0.8)
            axis.set_xticks(x, ["SAM1", "SAM2", "SAM3"])
            axis.set_title(f"{dataset_id.replace('_', ' ')} · {bbox_source.replace('_', ' ')}")
            axis.set_ylabel("Pseudo − human mean IoU")
            axis.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(ncol=3, fontsize=8, loc="upper left")
    figure.suptitle("Reference-induced score change (scene-clustered 95% CI)", fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_figure_manifest(paths: list[Path], output_path: Path) -> None:
    files = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": f"{output_path.parent.name}/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    output_path.write_text(
        json.dumps(
            {"schema_version": 1, "files": files},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
