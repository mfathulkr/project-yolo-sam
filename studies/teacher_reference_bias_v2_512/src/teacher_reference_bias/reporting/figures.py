from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from pycocotools.coco import COCO

from teacher_reference_bias.reporting.analysis import MODEL_ORDER, STRATUM_ORDER
from yolo_sam.segmentation.runner import decode_binary_mask


MODEL_COLORS = {
    "sam1": "#0072B2",
    "sam2": "#009E73",
    "sam3": "#D55E00",
}
MODEL_LABELS = {"sam1": "SAM1", "sam2": "SAM2", "sam3": "SAM3"}
STRATUM_LABELS = {
    "no_overlap__low_mask_area": "No Overlap × Low Mask",
    "no_overlap__high_mask_area": "No Overlap × High Mask",
    "overlap__low_mask_area": "Overlap × Low Mask",
    "overlap__high_mask_area": "Overlap × High Mask",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(figure: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def gt_bbox_reference_comparison(
    aggregates: pd.DataFrame,
    output_path: Path,
) -> Path:
    configure_style()
    panels = [
        ("isaid_plane", "human", "iSAID · Human Reference"),
        ("isaid_plane", "pseudo_sam1", "iSAID · SAM1 Pseudo Reference"),
        ("samrs_sota_plane", "pseudo_sam1", "SAMRS SOTA · SAM1 Pseudo Reference"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharey=True)
    for axis, (dataset_id, reference_type, title) in zip(
        axes,
        panels,
        strict=True,
    ):
        selected = aggregates[
            (aggregates["dataset_id"] == dataset_id)
            & (aggregates["bbox_source"] == "gt_bbox")
            & (aggregates["reference_type"] == reference_type)
            & (aggregates["stratum"] == "overall")
        ].set_index("model")
        x_values = np.arange(len(MODEL_ORDER))
        means = np.asarray(
            [float(selected.loc[model, "mean_iou"]) for model in MODEL_ORDER]
        )
        lower = np.asarray(
            [float(selected.loc[model, "iou_ci_lower"]) for model in MODEL_ORDER]
        )
        upper = np.asarray(
            [float(selected.loc[model, "iou_ci_upper"]) for model in MODEL_ORDER]
        )
        axis.bar(
            x_values,
            means,
            color=[MODEL_COLORS[model] for model in MODEL_ORDER],
            width=0.68,
        )
        axis.errorbar(
            x_values,
            means,
            yerr=np.vstack([means - lower, upper - means]),
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=1,
        )
        for x_value, mean in zip(x_values, means, strict=True):
            axis.text(
                x_value,
                min(0.98, mean + 0.045),
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_title(title)
        axis.set_xticks(x_values, [MODEL_LABELS[model] for model in MODEL_ORDER])
        axis.set_ylim(0.0, 1.08)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Instance-level Mean IoU")
    figure.suptitle(
        "Aynı GT bbox, farklı referans kaynağı: ölçülen sıralama ve skor değişimi",
        fontsize=12,
        y=1.03,
    )
    figure.tight_layout()
    return _save_figure(figure, output_path)


def reference_inflation_figure(
    inflation: pd.DataFrame,
    output_path: Path,
) -> Path:
    configure_style()
    selected = inflation[
        (inflation["dataset_id"] == "isaid_plane")
        & (inflation["bbox_source"] == "gt_bbox")
        & (inflation["stratum"] == "overall")
    ].set_index("model")
    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    x_values = np.arange(len(MODEL_ORDER))
    means = np.asarray(
        [float(selected.loc[model, "mean_iou_inflation"]) for model in MODEL_ORDER]
    )
    lower = np.asarray(
        [float(selected.loc[model, "iou_inflation_ci_lower"]) for model in MODEL_ORDER]
    )
    upper = np.asarray(
        [float(selected.loc[model, "iou_inflation_ci_upper"]) for model in MODEL_ORDER]
    )
    axis.bar(
        x_values,
        means,
        color=[MODEL_COLORS[model] for model in MODEL_ORDER],
        width=0.62,
    )
    axis.errorbar(
        x_values,
        means,
        yerr=np.vstack([means - lower, upper - means]),
        fmt="none",
        ecolor="#222222",
        capsize=4,
    )
    for x_value, mean, low, high in zip(
        x_values,
        means,
        lower,
        upper,
        strict=True,
    ):
        axis.text(
            x_value,
            high + 0.012,
            f"+{mean:.3f}\n[{low:.3f}, {high:.3f}]",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.axhline(0.0, color="#222222", linewidth=0.8)
    axis.set_xticks(x_values, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axis.set_ylabel("Pseudo IoU − Human IoU")
    axis.set_ylim(0.0, max(0.48, float(upper.max()) + 0.07))
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.set_title(
        "iSAID kontrollü deneyinde SAM1 pseudo referansının yarattığı skor enflasyonu"
    )
    figure.tight_layout()
    return _save_figure(figure, output_path)


def shared_human_reference_figure(
    summary: pd.DataFrame,
    inflation_intervals: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    configure_style()
    selected = summary[summary["stratum"] == "overall"].copy()
    selected["model"] = selected["model"].astype(str)
    scores = selected.pivot(
        index="model",
        columns="reference_type",
        values="mean_iou",
    )
    intervals = {
        str(row["model"]): row
        for row in inflation_intervals
    }
    figure, axis = plt.subplots(figsize=(7.8, 4.2))
    x_values = np.arange(len(MODEL_ORDER), dtype=float)
    width = 0.34
    human = np.asarray(
        [float(scores.loc[model, "human"]) for model in MODEL_ORDER]
    )
    pseudo = np.asarray(
        [float(scores.loc[model, "pseudo_sam1"]) for model in MODEL_ORDER]
    )
    axis.bar(
        x_values - width / 2,
        human,
        width=width,
        color="#777777",
        label="Bağımsız iSAID insan referansı",
    )
    axis.bar(
        x_values + width / 2,
        pseudo,
        width=width,
        color=[MODEL_COLORS[model] for model in MODEL_ORDER],
        edgecolor="#222222",
        linewidth=0.6,
        hatch="//",
    )
    for index, model in enumerate(MODEL_ORDER):
        interval = intervals[model]
        estimate = float(interval["estimate"])
        lower = float(interval["lower"])
        upper = float(interval["upper"])
        axis.text(
            x_values[index],
            min(1.07, pseudo[index] + 0.055),
            f"+{estimate:.3f}\n%95 GA [{lower:.3f}, {upper:.3f}]",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        axis.plot(
            [x_values[index] - width / 2, x_values[index] + width / 2],
            [human[index], pseudo[index]],
            color="#222222",
            linewidth=0.8,
            alpha=0.7,
        )
    axis.set_xticks(
        x_values,
        [MODEL_LABELS[model] for model in MODEL_ORDER],
    )
    axis.set_ylim(0.0, 1.14)
    axis.set_ylabel("Instance düzeyinde ortalama IoU")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(
        handles=[
            Patch(
                facecolor="#777777",
                label="Bağımsız iSAID insan referansı",
            ),
            Patch(
                facecolor="#FFFFFF",
                edgecolor="#222222",
                hatch="//",
                label="SAM1 üretimli SAMRS referansı",
            ),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
    )
    axis.set_title(
        "Aynı tahminler, iki referans: üretici-referans skor artışı"
    )
    figure.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    return _save_figure(figure, output_path)


def strata_heatmap(
    aggregates: pd.DataFrame,
    output_path: Path,
) -> Path:
    configure_style()
    conditions = [
        ("isaid_plane", "human", "iSAID · Human"),
        ("isaid_plane", "pseudo_sam1", "iSAID · SAM1 Pseudo"),
        ("samrs_sota_plane", "pseudo_sam1", "SAMRS · SAM1 Pseudo"),
    ]
    strata = list(STRATUM_ORDER[1:])
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    image = None
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        matrix = np.full((len(conditions), len(strata)), np.nan)
        for row_index, (dataset_id, reference_type, _) in enumerate(conditions):
            selected = aggregates[
                (aggregates["dataset_id"] == dataset_id)
                & (aggregates["model"] == model)
                & (aggregates["bbox_source"] == "gt_bbox")
                & (aggregates["reference_type"] == reference_type)
            ].set_index("stratum")
            for column_index, stratum in enumerate(strata):
                if stratum in selected.index:
                    matrix[row_index, column_index] = float(
                        selected.loc[stratum, "mean_iou"]
                    )
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="black" if 0.25 < value < 0.85 else "white",
                        fontsize=8,
                        fontweight="bold",
                    )
        axis.set_title(MODEL_LABELS[model])
        axis.set_xticks(
            np.arange(len(strata)),
            [
                "No Ov.\nLow",
                "No Ov.\nHigh",
                "Overlap\nLow",
                "Overlap\nHigh",
            ],
        )
    axes[0].set_yticks(
        np.arange(len(conditions)),
        [condition[2] for condition in conditions],
    )
    figure.subplots_adjust(left=0.13, right=0.88, top=0.84, bottom=0.16, wspace=0.13)
    if image is not None:
        colorbar_axis = figure.add_axes((0.91, 0.16, 0.012, 0.68))
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Ortalama IoU")
    figure.suptitle(
        "GT-bbox sonuçlarının overlap × mask area katmanlarına göre dağılımı",
        fontsize=12,
    )
    return _save_figure(figure, output_path)


def detector_seed_figure(
    detector_metrics: pd.DataFrame,
    output_path: Path,
) -> Path | None:
    if detector_metrics.empty:
        return None
    configure_style()
    metrics = ("bbox_AP50", "bbox_AP75", "bbox_AP90", "bbox_AP50_95")
    labels = ("AP50", "AP75", "AP90", "AP50–95")
    datasets = sorted(detector_metrics["dataset_id"].unique())
    figure, axis = plt.subplots(figsize=(7.4, 4.0))
    x_values = np.arange(len(metrics))
    width = 0.34
    colors = ("#0072B2", "#D55E00")
    for dataset_index, dataset_id in enumerate(datasets):
        selected = detector_metrics[detector_metrics["dataset_id"] == dataset_id]
        means = np.asarray([float(selected[metric].mean()) for metric in metrics])
        stds = np.asarray([float(selected[metric].std(ddof=1)) for metric in metrics])
        offsets = x_values + (dataset_index - (len(datasets) - 1) / 2) * width
        axis.bar(
            offsets,
            means,
            width=width,
            yerr=stds if len(selected) > 1 else None,
            capsize=3,
            color=colors[dataset_index % len(colors)],
            label=dataset_id,
        )
    axis.set_xticks(x_values, labels)
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("COCO bbox AP")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False)
    axis.set_title("YOLO detector sonuçları: sabit seed 42")
    figure.tight_layout()
    return _save_figure(figure, output_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _expanded_square_crop(
    bbox_xywh: list[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox_xywh
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    side = min(512.0, max(160.0, 4.0 * max(width, height)))
    x1 = max(0, int(round(center_x - side / 2.0)))
    y1 = max(0, int(round(center_y - side / 2.0)))
    x2 = min(image_width, int(round(center_x + side / 2.0)))
    y2 = min(image_height, int(round(center_y + side / 2.0)))
    if x2 - x1 < side and x1 == 0:
        x2 = min(image_width, int(round(side)))
    if y2 - y1 < side and y1 == 0:
        y2 = min(image_height, int(round(side)))
    if x2 - x1 < side and x2 == image_width:
        x1 = max(0, image_width - int(round(side)))
    if y2 - y1 < side and y2 == image_height:
        y1 = max(0, image_height - int(round(side)))
    return x1, y1, x2, y2


def _reference_overlay(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    output = image.astype(np.float32).copy()
    color = np.asarray([0, 174, 239], dtype=np.float32)
    output[reference] = 0.45 * output[reference] + 0.55 * color
    return np.clip(output, 0, 255).astype(np.uint8)


def _error_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    output = image.astype(np.float32).copy()
    true_positive = prediction & reference
    false_positive = prediction & ~reference
    false_negative = ~prediction & reference
    colors = (
        (true_positive, np.asarray([0, 158, 115], dtype=np.float32)),
        (false_positive, np.asarray([230, 159, 0], dtype=np.float32)),
        (false_negative, np.asarray([204, 121, 167], dtype=np.float32)),
    )
    for mask, color in colors:
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
        & (metrics["model"].isin(MODEL_ORDER))
        & (metrics["bbox_source"] == "gt_bbox")
        & (metrics["reference_type"] == reference_type)
    ]
    model_counts = selected.groupby("instance_id")["model"].nunique()
    if set(model_counts.astype(int)) != {len(MODEL_ORDER)}:
        raise ValueError(
            f"Incomplete GT-bbox model coverage for {dataset_id}/{reference_type}"
        )
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
    used_scene_ids: set[str] = set()
    for stratum in STRATUM_ORDER[1:]:
        stratum_rows = per_image[per_image["stratum"] == stratum].copy()
        if stratum_rows.empty:
            raise ValueError(f"No representative candidate for {dataset_id}/{stratum}")
        median = float(stratum_rows["mean_model_iou"].median())
        stratum_rows["distance_to_median"] = (
            stratum_rows["mean_model_iou"].astype(float) - median
        ).abs()
        ordered = stratum_rows.sort_values(
            ["distance_to_median", "image_id"]
        )
        unused = ordered[
            ~ordered["source_scene_id"].astype(str).isin(used_scene_ids)
        ]
        row = (unused if not unused.empty else ordered).iloc[0]
        image_ids.append(str(row["image_id"]))
        used_scene_ids.add(str(row["source_scene_id"]))
    return image_ids


def qualitative_gt_bbox_figure(
    *,
    study_root: Path,
    prepared_root: Path,
    dataset_id: str,
    reference_type: str,
    canonical_metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    configure_style()
    coco = COCO(str(prepared_root / "test" / "_annotations.coco.json"))
    images_root = prepared_root / "test" / "images"
    prediction_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODEL_ORDER:
        path = (
            study_root
            / "predictions"
            / dataset_id
            / model
            / "gt_bbox"
            / "predictions.jsonl"
        )
        prediction_by_model[model] = {
            str(row["instance_id"]): row for row in _read_jsonl(path)
        }
    pseudo_reference_path = (
        study_root
        / "references"
        / dataset_id
        / "sam1_gt_bbox_pseudo.jsonl"
    )
    pseudo_reference_by_instance = (
        {
            str(row["instance_id"]): row
            for row in _read_jsonl(pseudo_reference_path)
        }
        if reference_type == "pseudo_sam1"
        and pseudo_reference_path.is_file()
        else {}
    )

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation_id in coco.getAnnIds():
        annotation = coco.loadAnns([annotation_id])[0]
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(
            annotation
        )

    image_ids = _select_representative_images(
        canonical_metrics,
        dataset_id=dataset_id,
        reference_type=reference_type,
    )
    figure, axes = plt.subplots(
        len(image_ids),
        5,
        figsize=(11.2, 9.1),
        squeeze=False,
    )
    column_titles = ("Input + all GT bbox", "Reference union", "SAM1", "SAM2", "SAM3")
    for axis, title in zip(axes[0], column_titles, strict=True):
        axis.set_title(title, fontsize=10)

    for row_index, (stratum, canonical_image_id) in enumerate(
        zip(STRATUM_ORDER[1:], image_ids, strict=True)
    ):
        image_id = int(canonical_image_id.rsplit(":", 1)[-1])
        annotations = sorted(
            annotations_by_image[image_id],
            key=lambda annotation: int(annotation["id"]),
        )
        instance_ids = [
            f"{dataset_id}:{image_id}:{int(annotation['id'])}"
            for annotation in annotations
        ]
        for model in MODEL_ORDER:
            missing = [
                instance_id
                for instance_id in instance_ids
                if instance_id not in prediction_by_model[model]
            ]
            if missing:
                raise ValueError(
                    f"Missing {model} GT-bbox predictions for "
                    f"{dataset_id}/image={image_id}: {missing[:5]}"
                )
        image_record = coco.loadImgs([image_id])[0]
        image = np.asarray(
            Image.open(images_root / str(image_record["file_name"])).convert("RGB")
        )
        if pseudo_reference_by_instance:
            missing_references = [
                instance_id
                for instance_id in instance_ids
                if instance_id not in pseudo_reference_by_instance
            ]
            if missing_references:
                raise ValueError(
                    f"Missing pseudo references for {dataset_id}/image={image_id}: "
                    f"{missing_references[:5]}"
                )
            reference = np.zeros(image.shape[:2], dtype=bool)
            for instance_id in instance_ids:
                reference |= decode_binary_mask(
                    pseudo_reference_by_instance[instance_id]["mask_rle"]
                )
        else:
            reference = np.zeros(image.shape[:2], dtype=bool)
            for annotation in annotations:
                reference |= coco.annToMask(annotation).astype(bool)

        axes[row_index, 0].imshow(image)
        for annotation in annotations:
            bbox_x, bbox_y, bbox_width, bbox_height = [
                float(value) for value in annotation["bbox"]
            ]
            axes[row_index, 0].add_patch(
                Rectangle(
                    (bbox_x, bbox_y),
                    bbox_width,
                    bbox_height,
                    fill=False,
                    edgecolor="#00FF66",
                    linewidth=1.5,
                )
            )
        axes[row_index, 1].imshow(_reference_overlay(image, reference))
        for column_index, model in enumerate(MODEL_ORDER, start=2):
            prediction = np.zeros(image.shape[:2], dtype=bool)
            for instance_id in instance_ids:
                prediction |= decode_binary_mask(
                    prediction_by_model[model][instance_id]["predicted_mask_rle"]
                )
            axes[row_index, column_index].imshow(
                _error_overlay(image, prediction, reference)
            )
            union = np.logical_or(prediction, reference).sum()
            union_iou = (
                float(np.logical_and(prediction, reference).sum() / union)
                if union
                else 1.0
            )
            axes[row_index, column_index].text(
                0.02,
                0.97,
                f"Union IoU {union_iou:.3f}",
                transform=axes[row_index, column_index].transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 2, "edgecolor": "none"},
            )
        axes[row_index, 0].set_ylabel(
            f"{STRATUM_LABELS[stratum]}\n{len(instance_ids)} instances",
            fontsize=8,
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    figure.suptitle(
        f"{dataset_id}: tüm GT-bbox istemlerinin birleşik görünümü · "
        "TP yeşil, FP turuncu, FN pembe",
        fontsize=12,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.7, w_pad=0.25)
    return _save_figure(figure, output_path)
