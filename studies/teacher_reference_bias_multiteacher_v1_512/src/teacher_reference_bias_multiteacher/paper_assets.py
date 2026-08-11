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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from pycocotools.coco import COCO

from yolo_sam.segmentation.runner import decode_binary_mask

from .figures import (
    MODEL_COLORS,
    configure_style,
    model_reference_matrix_figure,
    reference_effect_figure,
)
from .io import read_jsonl
from .paths import DATASETS, MODELS, STRATA, prediction_path, reference_path


PAPER_ROOT = Path(__file__).resolve().parents[2] / "paper"
TABLE_ROOT = PAPER_ROOT / "assets" / "tables"
FIGURE_ROOT = PAPER_ROOT / "assets" / "figures"

REFERENCE_COLORS = {
    "human": "#CC79A7",
    "pseudo_sam1": MODEL_COLORS["sam1"],
    "pseudo_sam2": MODEL_COLORS["sam2"],
    "pseudo_sam3": MODEL_COLORS["sam3"],
}
REFERENCE_LABELS = {
    "human": "Human",
    "pseudo_sam1": "SAM1 pseudo",
    "pseudo_sam2": "SAM2 pseudo",
    "pseudo_sam3": "SAM3 pseudo",
}
DATASET_LABELS = {
    "isaid_plane": "iSAID Plane",
    "isaid_small_vehicle": "iSAID Small Vehicle",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_table(frame: pd.DataFrame, stem: str) -> list[Path]:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_ROOT / f"{stem}.csv"
    tex_path = TABLE_ROOT / f"{stem}.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(
        frame.to_latex(index=False, escape=True, float_format="%.3f"),
        encoding="utf-8",
    )
    return [csv_path, tex_path]


def _save_figure(figure: plt.Figure, stem: str) -> list[Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("png", "pdf"):
        path = FIGURE_ROOT / f"{stem}.{suffix}"
        figure.savefig(path, dpi=260, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def build_tables(
    aggregates: pd.DataFrame,
    effects: pd.DataFrame,
    advantages: pd.DataFrame,
    agreements: pd.DataFrame,
    empty_stats: pd.DataFrame,
    detector: pd.DataFrame,
) -> list[Path]:
    outputs: list[Path] = []
    overall = aggregates[aggregates["stratum"] == "overall"].copy()

    design_rows = []
    for dataset_id, source in DATASETS.items():
        sample = overall[overall["dataset_id"] == dataset_id].iloc[0]
        design_rows.append(
            {
                "Dataset / target": DATASET_LABELS[dataset_id],
                "Images": 512,
                "Source scenes": int(sample["source_scene_count"]),
                "Instances": int(sample["instance_count"]),
                "Images per stratum": 128,
                "References": "Human, SAM1, SAM2, SAM3",
                "Candidate models": "SAM1, SAM2, SAM3",
                "Prompts": "GT bbox, YOLO bbox",
            }
        )
    outputs += _write_table(pd.DataFrame(design_rows), "table_1_experimental_design")

    matrix_rows: list[dict[str, Any]] = []
    for (dataset_id, bbox_source, model), group in overall.groupby(
        ["dataset_id", "bbox_source", "model"], sort=True
    ):
        values = group.set_index("reference_type")["mean_iou"]
        matrix_rows.append(
            {
                "Dataset": DATASET_LABELS[dataset_id],
                "Prompt": bbox_source.replace("_", " ").upper(),
                "Model": model.upper(),
                "Human": values["human"],
                "SAM1 pseudo": values["pseudo_sam1"],
                "SAM2 pseudo": values["pseudo_sam2"],
                "SAM3 pseudo": values["pseudo_sam3"],
            }
        )
    outputs += _write_table(pd.DataFrame(matrix_rows), "table_2_model_reference_iou")

    own_effects = effects[
        effects.apply(
            lambda row: row["pseudo_reference"] == f"pseudo_{row['model']}", axis=1
        )
    ].copy()
    own_effects["Dataset"] = own_effects["dataset_id"].map(DATASET_LABELS)
    own_effects["Prompt"] = own_effects["bbox_source"].str.replace("_", " ").str.upper()
    own_effects["Model / reference"] = own_effects["model"].str.upper()
    own_effects["Identity control"] = own_effects["bbox_source"].map(
        {"gt_bbox": "Yes (tautological)", "yolo_bbox": "No"}
    )
    own_effects["95% CI"] = own_effects.apply(
        lambda row: f"[{row['delta_ci_lower']:.3f}, {row['delta_ci_upper']:.3f}]",
        axis=1,
    )
    own_effects_table = own_effects[
        [
            "Dataset",
            "Prompt",
            "Model / reference",
            "instance_count",
            "human_mean_iou",
            "pseudo_mean_iou",
            "delta_iou",
            "95% CI",
            "Identity control",
        ]
    ].rename(
        columns={
            "instance_count": "N",
            "human_mean_iou": "Human IoU",
            "pseudo_mean_iou": "Own-pseudo IoU",
            "delta_iou": "Delta IoU",
        }
    )
    outputs += _write_table(own_effects_table, "table_3_self_reference_effect")

    advantage_table = advantages.copy()
    advantage_table["Dataset"] = advantage_table["dataset_id"].map(DATASET_LABELS)
    advantage_table["Prompt"] = advantage_table["bbox_source"].str.replace("_", " ").str.upper()
    advantage_table["Teacher"] = advantage_table["teacher"].str.upper()
    advantage_table = advantage_table[
        [
            "Dataset",
            "Prompt",
            "Teacher",
            "teacher_score",
            "other_models_mean",
            "teacher_advantage",
            "identity_control",
        ]
    ].rename(
        columns={
            "teacher_score": "Teacher IoU",
            "other_models_mean": "Other-model mean",
            "teacher_advantage": "Teacher advantage",
            "identity_control": "Identity control",
        }
    )
    outputs += _write_table(advantage_table, "table_4_teacher_advantage")

    human_agreement = agreements[agreements["reference_a"] == "human"].copy()
    human_agreement["Dataset"] = human_agreement["dataset_id"].map(DATASET_LABELS)
    human_agreement["Teacher"] = (
        human_agreement["reference_b"].str.removeprefix("pseudo_").str.upper()
    )
    integrity = human_agreement.merge(
        empty_stats,
        left_on=["dataset_id", "Teacher"],
        right_on=["dataset_id", "teacher_label"],
        how="left",
        validate="one_to_one",
    )
    integrity["Dataset"] = integrity["dataset_id"].map(DATASET_LABELS)
    integrity = integrity[
        [
            "Dataset",
            "Teacher",
            "instance_count_x",
            "mean_instance_iou",
            "empty_masks",
            "empty_rate",
        ]
    ].rename(
        columns={
            "instance_count_x": "N",
            "mean_instance_iou": "Human agreement IoU",
            "empty_masks": "Empty masks",
            "empty_rate": "Empty rate",
        }
    )
    outputs += _write_table(integrity, "table_5_reference_integrity")

    detector_table = detector.copy()
    detector_table["Dataset"] = detector_table["dataset_id"].map(DATASET_LABELS)
    detector_table = detector_table[
        [
            "Dataset",
            "seed_ids",
            "fixed_confidence_threshold_mean",
            "bbox_AP50_mean",
            "bbox_AP75_mean",
            "bbox_AP90_mean",
            "bbox_AP50_95_mean",
            "precision_at_bbox_iou50_mean",
            "recall_at_bbox_iou50_mean",
        ]
    ].rename(
        columns={
            "seed_ids": "Seed",
            "fixed_confidence_threshold_mean": "Confidence threshold",
            "bbox_AP50_mean": "bbox AP50",
            "bbox_AP75_mean": "bbox AP75",
            "bbox_AP90_mean": "bbox AP90",
            "bbox_AP50_95_mean": "bbox AP50-95",
            "precision_at_bbox_iou50_mean": "Precision@bboxIoU50",
            "recall_at_bbox_iou50_mean": "Recall@bboxIoU50",
        }
    )
    outputs += _write_table(detector_table, "table_6_detector_control")

    stratified = aggregates[
        (aggregates["bbox_source"] == "yolo_bbox")
        & aggregates.apply(
            lambda row: row["reference_type"] in {"human", f"pseudo_{row['model']}"},
            axis=1,
        )
    ].copy()
    pivot = stratified.pivot_table(
        index=["dataset_id", "model", "stratum", "instance_count"],
        columns="reference_type",
        values="mean_iou",
        aggfunc="first",
    ).reset_index()
    rows = []
    for row in pivot.to_dict("records"):
        own_column = f"pseudo_{row['model']}"
        rows.append(
            {
                "Dataset": DATASET_LABELS[row["dataset_id"]],
                "Stratum": row["stratum"],
                "Model / reference": row["model"].upper(),
                "N": int(row["instance_count"]),
                "Human IoU": row["human"],
                "Own-pseudo IoU": row[own_column],
                "Delta IoU": row[own_column] - row["human"],
            }
        )
    outputs += _write_table(pd.DataFrame(rows), "table_s1_stratified_self_reference_effect")
    return outputs


def study_design_figure() -> list[Path]:
    configure_style()
    figure, axis = plt.subplots(figsize=(12.0, 4.5))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5)
    axis.axis("off")

    boxes = (
        (0.25, 0.65, 2.45, 3.7, "Fixed samples\n\n2 targets × 512 images\n4 strata × 128 images\nHuman masks and boxes", "#E8F1F8"),
        (3.25, 0.65, 2.45, 3.7, "Evaluation references\n\nHuman\nSAM1 pseudo\nSAM2 pseudo\nSAM3 pseudo\n(GT-box prompted)", "#F5EAF1"),
        (6.25, 0.65, 2.45, 3.7, "Candidate predictions\n\nSAM1, SAM2, SAM3\n×\nGT bbox, YOLO bbox\nPredictions held fixed", "#E8F4ED"),
        (9.25, 0.65, 2.45, 3.7, "Paired evaluation\n\nInstance-level metrics\nScene-clustered 95% CI\nReference × model matrix\nRank audit\nEmpty-mask audit", "#FFF2D9"),
    )
    for x, y, width, height, text, color in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.03,rounding_size=0.06",
                facecolor=color,
                edgecolor="#333333",
                linewidth=1.0,
            )
        )
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)
    for start, end in ((2.72, 3.22), (5.72, 6.22), (8.72, 9.22)):
        axis.add_patch(
            FancyArrowPatch(
                (start, 2.5),
                (end, 2.5),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color="#333333",
            )
        )
    axis.text(
        6,
        0.18,
        "Controlled variable: the evaluation reference. Images, instances, prompts, predictions, and metrics are unchanged.",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
    return _save_figure(figure, "figure_1_study_design")


def reference_integrity_figure(
    agreements: pd.DataFrame,
    empty_stats: pd.DataFrame,
) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    teacher_order = ["SAM1", "SAM2", "SAM3"]
    x = np.arange(3)
    width = 0.34
    for offset, dataset_id in zip((-width / 2, width / 2), DATASETS, strict=True):
        selected = agreements[
            (agreements["dataset_id"] == dataset_id)
            & (agreements["reference_a"] == "human")
        ].copy()
        values = (
            selected.assign(
                teacher=selected["reference_b"].str.removeprefix("pseudo_").str.upper()
            )
            .set_index("teacher")
            .loc[teacher_order, "mean_instance_iou"]
            .to_numpy(float)
        )
        axes[0].bar(x + offset, values, width, label=DATASET_LABELS[dataset_id])
        selected_empty = (
            empty_stats[empty_stats["dataset_id"] == dataset_id]
            .set_index("teacher_label")
            .loc[teacher_order, "empty_rate"]
            .to_numpy(float)
        )
        axes[1].bar(x + offset, selected_empty, width, label=DATASET_LABELS[dataset_id])
    axes[0].set_title("Agreement with human reference")
    axes[0].set_ylabel("Mean instance IoU")
    axes[1].set_title("Empty pseudo-reference masks")
    axes[1].set_ylabel("Empty-mask rate")
    for axis in axes:
        axis.set_xticks(x, teacher_order)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    return _save_figure(figure, "figure_4_reference_integrity")


def stratified_effect_figure(aggregates: pd.DataFrame) -> list[Path]:
    configure_style()
    selected = aggregates[
        (aggregates["bbox_source"] == "yolo_bbox")
        & aggregates.apply(
            lambda row: row["reference_type"] in {"human", f"pseudo_{row['model']}"},
            axis=1,
        )
    ].copy()
    pivot = selected.pivot_table(
        index=["dataset_id", "stratum", "model"],
        columns="reference_type",
        values="mean_iou",
        aggfunc="first",
    ).reset_index()
    values = []
    row_labels = []
    for dataset_id in DATASETS:
        for stratum in STRATA:
            row = []
            for model in MODELS:
                record = pivot[
                    (pivot["dataset_id"] == dataset_id)
                    & (pivot["stratum"] == stratum)
                    & (pivot["model"] == model)
                ].iloc[0]
                row.append(record[f"pseudo_{model}"] - record["human"])
            values.append(row)
            short = stratum.replace("no_overlap", "no overlap").replace("__", " / ").replace("_", " ")
            row_labels.append(f"{DATASET_LABELS[dataset_id]} · {short}")
    matrix = np.asarray(values, dtype=float)
    figure, axis = plt.subplots(figsize=(8.2, 6.7))
    image = axis.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(0.35, float(matrix.max())))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"+{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 0.24 else "black",
                fontsize=8,
            )
    axis.set_xticks(range(3), [model.upper() for model in MODELS])
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_xlabel("Evaluated model and matching pseudo reference")
    axis.set_title("Own-pseudo minus human IoU under YOLO-box prompting")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Delta mean instance IoU")
    figure.tight_layout()
    return _save_figure(figure, "figure_5_stratified_self_reference_effect")


def _sam1_reference_path(dataset_id: str) -> Path:
    source = DATASETS[dataset_id]
    return (
        source.canonical_study
        / "results"
        / "references"
        / dataset_id
        / "sam1_gt_bbox_pseudo.jsonl"
    )


def _reference_records(dataset_id: str) -> dict[str, dict[str, Any]]:
    paths = {
        "pseudo_sam1": _sam1_reference_path(dataset_id),
        "pseudo_sam2": reference_path(dataset_id, "sam2"),
        "pseudo_sam3": reference_path(dataset_id, "sam3"),
    }
    return {
        reference_type: {
            str(row["instance_id"]): row for row in read_jsonl(path)
        }
        for reference_type, path in paths.items()
    }


def _select_reference_examples(metrics: pd.DataFrame, dataset_id: str) -> list[str]:
    selected = metrics[
        (metrics["dataset_id"] == dataset_id)
        & (metrics["reference_type"] == "human")
        & (metrics["bbox_source"] == "gt_bbox")
    ]
    per_image = (
        selected.groupby(["image_id", "source_scene_id", "stratum"], as_index=False)["iou"]
        .mean()
        .rename(columns={"iou": "mean_reference_agreement"})
    )
    image_ids: list[str] = []
    used_scenes: set[str] = set()
    for stratum in STRATA[1:]:
        candidates = per_image[per_image["stratum"] == stratum].copy()
        median = float(candidates["mean_reference_agreement"].median())
        candidates["distance"] = (candidates["mean_reference_agreement"] - median).abs()
        candidates = candidates.sort_values(["distance", "image_id"])
        unused = candidates[~candidates["source_scene_id"].isin(used_scenes)]
        row = (unused if not unused.empty else candidates).iloc[0]
        image_ids.append(str(row["image_id"]))
        used_scenes.add(str(row["source_scene_id"]))
    return image_ids


def _overlay(image: np.ndarray, mask: np.ndarray, color: str) -> np.ndarray:
    rgb = np.asarray(plt.matplotlib.colors.to_rgb(color), dtype=np.float32) * 255
    output = image.astype(np.float32).copy()
    output[mask] = output[mask] * 0.38 + rgb * 0.62
    return np.clip(output, 0, 255).astype(np.uint8)


def qualitative_reference_figure(
    metrics: pd.DataFrame,
    dataset_id: str,
) -> list[Path]:
    configure_style()
    source = DATASETS[dataset_id]
    coco = COCO(str(source.coco_path))
    references = _reference_records(dataset_id)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.loadAnns(coco.getAnnIds()):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    image_ids = _select_reference_examples(metrics, dataset_id)
    figure, axes = plt.subplots(4, 5, figsize=(11.4, 9.1), squeeze=False)
    titles = ("Input + all GT boxes", "Human", "SAM1 pseudo", "SAM2 pseudo", "SAM3 pseudo")
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title)
    for row_index, (stratum, canonical_image_id) in enumerate(zip(STRATA[1:], image_ids, strict=True)):
        image_id = int(canonical_image_id.rsplit(":", 1)[-1])
        annotations = sorted(annotations_by_image[image_id], key=lambda row: int(row["id"]))
        image_record = coco.loadImgs([image_id])[0]
        image = np.asarray(Image.open(source.images_root / image_record["file_name"]).convert("RGB"))
        axes[row_index, 0].imshow(image)
        for annotation in annotations:
            x, y, width, height = map(float, annotation["bbox"])
            axes[row_index, 0].add_patch(
                Rectangle((x, y), width, height, fill=False, edgecolor="#00FF66", linewidth=1.2)
            )
        human = np.zeros(image.shape[:2], dtype=bool)
        instance_ids = []
        for annotation in annotations:
            human |= coco.annToMask(annotation).astype(bool)
            instance_ids.append(f"{dataset_id}:{image_id}:{int(annotation['id'])}")
        axes[row_index, 1].imshow(_overlay(image, human, REFERENCE_COLORS["human"]))
        for column, reference_type in enumerate(("pseudo_sam1", "pseudo_sam2", "pseudo_sam3"), 2):
            mask = np.zeros(image.shape[:2], dtype=bool)
            for instance_id in instance_ids:
                row = references[reference_type].get(instance_id)
                if row is None:
                    raise ValueError(f"Eksik pseudo referans: {reference_type}/{instance_id}")
                mask |= decode_binary_mask(row["mask_rle"])
            axes[row_index, column].imshow(_overlay(image, mask, REFERENCE_COLORS[reference_type]))
            union = np.logical_or(mask, human).sum()
            union_iou = float(np.logical_and(mask, human).sum() / union) if union else 1.0
            axes[row_index, column].text(
                0.02,
                0.97,
                f"Union IoU {union_iou:.3f}",
                transform=axes[row_index, column].transAxes,
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
    figure.suptitle(f"{DATASET_LABELS[dataset_id]} reference masks (all target instances)", fontweight="bold")
    figure.text(
        0.5,
        0.007,
        "Deterministic median-difficulty examples; each row includes every target instance in the image.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.97))
    return _save_figure(figure, f"figure_q_{dataset_id}_reference_examples")


def build_figures(
    aggregates: pd.DataFrame,
    effects: pd.DataFrame,
    agreements: pd.DataFrame,
    empty_stats: pd.DataFrame,
    metrics: pd.DataFrame,
) -> list[Path]:
    outputs = study_design_figure()
    for stem, builder, frame in (
        ("figure_2_model_reference_iou_matrix", model_reference_matrix_figure, aggregates),
        ("figure_3_reference_effect_with_ci", reference_effect_figure, effects),
    ):
        for suffix in ("png", "pdf"):
            path = FIGURE_ROOT / f"{stem}.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            builder(frame, path)
            outputs.append(path)
    outputs += reference_integrity_figure(agreements, empty_stats)
    outputs += stratified_effect_figure(aggregates)
    for dataset_id in DATASETS:
        outputs += qualitative_reference_figure(metrics, dataset_id)
    return outputs


def write_manifest(table_paths: list[Path], figure_paths: list[Path]) -> Path:
    manifest_path = PAPER_ROOT / "assets" / "manifest.json"
    records = []
    for kind, paths in (("table", table_paths), ("figure", figure_paths)):
        for path in paths:
            records.append(
                {
                    "kind": kind,
                    "path": str(path.relative_to(PAPER_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "files": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path
