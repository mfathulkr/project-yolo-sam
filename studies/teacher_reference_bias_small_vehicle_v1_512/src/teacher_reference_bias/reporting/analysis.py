from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy.stats import kendalltau, spearmanr, wilcoxon

from yolo_sam.evaluation.statistics import clustered_bootstrap_mean


MODEL_ORDER = ("sam1", "sam2", "sam3")
METRIC_COLUMNS = (
    "iou",
    "dice",
    "precision",
    "recall",
    "boundary_iou",
)
SUCCESS_THRESHOLDS = (0.50, 0.75, 0.90)
STRATUM_ORDER = (
    "overall",
    "no_overlap__low_mask_area",
    "no_overlap__high_mask_area",
    "overlap__low_mask_area",
    "overlap__high_mask_area",
)
DETECTOR_METRIC_COLUMNS = (
    "images",
    "ground_truth_instances",
    "detections_for_ap",
    "fixed_confidence_threshold",
    "bbox_AP50",
    "bbox_AP75",
    "bbox_AP90",
    "bbox_AP50_95",
    "precision_at_bbox_iou50",
    "recall_at_bbox_iou50",
    "true_positive_at_bbox_iou50",
    "false_positive_at_bbox_iou50",
    "false_negative_at_bbox_iou50",
    "precision_at_bbox_iou75",
    "recall_at_bbox_iou75",
    "true_positive_at_bbox_iou75",
    "false_positive_at_bbox_iou75",
    "false_negative_at_bbox_iou75",
    "precision_at_bbox_iou90",
    "recall_at_bbox_iou90",
    "true_positive_at_bbox_iou90",
    "false_positive_at_bbox_iou90",
    "false_negative_at_bbox_iou90",
    "dataset_id",
    "seed",
    "split",
    "ap_confidence_floor",
    "confidence_threshold_source_split",
    "confidence_threshold_selection_method",
)
DETECTOR_AGGREGATE_METRICS = (
    "fixed_confidence_threshold",
    "bbox_AP50",
    "bbox_AP75",
    "bbox_AP90",
    "bbox_AP50_95",
    "precision_at_bbox_iou50",
    "recall_at_bbox_iou50",
    "precision_at_bbox_iou75",
    "recall_at_bbox_iou75",
    "precision_at_bbox_iou90",
    "recall_at_bbox_iou90",
)
SEGMENTATION_SEED_METRICS = (
    "mean_iou",
    "mean_dice",
    "mean_precision",
    "mean_recall",
    "mean_boundary_iou",
    "success_at_iou_50",
    "success_at_iou_75",
    "success_at_iou_90",
)
PREDICTION_STATUS_AUDIT_COLUMNS = (
    "dataset_id",
    "model",
    "bbox_source",
    "detector_seed",
    "row_kind",
    "prediction_file",
    "total_rows",
    "unique_instance_ids",
    "duplicate_instance_ids",
    "ok",
    "empty_mask",
    "missing_bbox",
    "inference_error",
    "zero_area_masks",
    "nonzero_area_masks",
    "nonempty_masks_without_prompt_overlap",
    "status_area_mismatches",
)
TRAINING_HEALTH_COLUMNS = (
    "dataset_id",
    "seed",
    "results_file",
    "epochs_completed",
    "final_epoch",
    "nonfinite_cells",
    "nonfinite_validation_loss_cells",
    "final_core_metrics_finite",
    "final_precision",
    "final_recall",
    "final_ap50",
    "final_ap50_95",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=np.float64)
    if values.size == 0:
        return []
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty(values.size, dtype=np.float64)
    running_max = 0.0
    count = values.size
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (count - rank))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def _seed_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"seed_(\d+)(?:_dual_reference)?", part)
        if match:
            return int(match.group(1))
    return None


def _model_from_path(path: Path) -> str:
    for model in MODEL_ORDER:
        if model in path.parts:
            return model
    raise ValueError(f"Cannot infer model from {path}")


def _dataset_from_path(path: Path, evaluation_root: Path) -> str:
    relative = path.relative_to(evaluation_root)
    if not relative.parts:
        raise ValueError(f"Cannot infer dataset from {path}")
    return relative.parts[0]


def _bbox_source_from_path(path: Path) -> str:
    if "gt_bbox" in path.parts or "gt_bbox_dual_reference" in path.parts:
        return "gt_bbox"
    if "yolo_bbox" in path.parts:
        return "yolo_bbox"
    raise ValueError(f"Cannot infer bbox source from {path}")


def _preferred_metric_files(evaluation_root: Path) -> list[Path]:
    candidates = sorted(evaluation_root.glob("**/metrics_instance.csv"))
    preferred: dict[tuple[str, str, str, int | None], Path] = {}
    for path in candidates:
        try:
            key = (
                _dataset_from_path(path, evaluation_root),
                _model_from_path(path),
                _bbox_source_from_path(path),
                _seed_from_path(path),
            )
        except ValueError:
            continue
        current = preferred.get(key)
        path_is_dual = "dual_reference" in str(path.parent)
        current_is_dual = current is not None and "dual_reference" in str(current.parent)
        if current is None or (path_is_dual and not current_is_dual):
            preferred[key] = path
    return sorted(preferred.values())


def collect_canonical_metrics(
    evaluation_root: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    metric_files = _preferred_metric_files(evaluation_root)
    if not metric_files:
        raise FileNotFoundError(f"No metrics_instance.csv under {evaluation_root}")
    frames: list[pd.DataFrame] = []
    for path in metric_files:
        frame = pd.read_csv(path)
        required = {
            "instance_id",
            "source_scene_id",
            "reference_type",
            "stratum",
            *METRIC_COLUMNS,
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame.insert(0, "dataset_id", _dataset_from_path(path, evaluation_root))
        frame.insert(1, "model", _model_from_path(path))
        frame.insert(2, "bbox_source", _bbox_source_from_path(path))
        frame.insert(3, "detector_seed", _seed_from_path(path))
        frame["metric_source"] = str(path)
        frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True)
    key_columns = [
        "dataset_id",
        "model",
        "bbox_source",
        "detector_seed",
        "reference_type",
        "instance_id",
    ]
    duplicate = metrics.duplicated(key_columns, keep=False)
    if duplicate.any():
        example = metrics.loc[duplicate, key_columns].head(5).to_dict("records")
        raise ValueError(f"Duplicate canonical metric keys: {example}")
    metrics["detector_seed"] = metrics["detector_seed"].astype("Int64")
    return metrics, metric_files


def _stratum_groups(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "overall", frame
    for stratum in STRATUM_ORDER[1:]:
        selected = frame[frame["stratum"] == stratum]
        if not selected.empty:
            yield stratum, selected


def aggregate_metrics(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    group_columns = [
        "dataset_id",
        "model",
        "bbox_source",
        "detector_seed",
        "reference_type",
    ]
    rows: list[dict[str, object]] = []
    for key, frame in metrics.groupby(group_columns, dropna=False, sort=True):
        for stratum, selected in _stratum_groups(frame):
            values_by_scene = {
                str(scene_id): group["iou"].astype(float).tolist()
                for scene_id, group in selected.groupby("source_scene_id", sort=True)
            }
            interval = clustered_bootstrap_mean(
                values_by_scene,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            )
            row = dict(zip(group_columns, key, strict=True))
            row.update(
                {
                    "stratum": stratum,
                    "instance_count": int(len(selected)),
                    "source_scene_count": int(selected["source_scene_id"].nunique()),
                    **{
                        f"mean_{metric}": float(selected[metric].mean())
                        for metric in METRIC_COLUMNS
                    },
                    "iou_ci_lower": interval.lower,
                    "iou_ci_upper": interval.upper,
                    "confidence_level": confidence_level,
                    "bootstrap_samples": bootstrap_samples,
                }
            )
            for threshold in SUCCESS_THRESHOLDS:
                row[f"success_at_iou_{int(threshold * 100)}"] = float(
                    (selected["iou"] >= threshold).mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_delta_interval(
    paired: pd.DataFrame,
    left_column: str,
    right_column: str,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> tuple[float, float, float]:
    values_by_scene = {
        str(scene_id): (
            group[left_column].astype(float) - group[right_column].astype(float)
        ).tolist()
        for scene_id, group in paired.groupby("source_scene_id", sort=True)
    }
    interval = clustered_bootstrap_mean(
        values_by_scene,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
    )
    return interval.estimate, interval.lower, interval.upper


def paired_model_comparisons(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    group_columns = [
        "dataset_id",
        "bbox_source",
        "detector_seed",
        "reference_type",
    ]
    rows: list[dict[str, object]] = []
    for key, frame in metrics.groupby(group_columns, dropna=False, sort=True):
        for stratum, selected in _stratum_groups(frame):
            models = [model for model in MODEL_ORDER if model in set(selected["model"])]
            for left_index, model_a in enumerate(models):
                for model_b in models[left_index + 1 :]:
                    left = selected[selected["model"] == model_a][
                        ["instance_id", "source_scene_id", "iou"]
                    ].rename(columns={"iou": "iou_a"})
                    right = selected[selected["model"] == model_b][
                        ["instance_id", "source_scene_id", "iou"]
                    ].rename(
                        columns={
                            "source_scene_id": "source_scene_id_b",
                            "iou": "iou_b",
                        }
                    )
                    paired = left.merge(right, on="instance_id", how="inner")
                    if paired.empty:
                        continue
                    if not (
                        paired["source_scene_id"] == paired["source_scene_id_b"]
                    ).all():
                        raise ValueError("Source scene mismatch in paired model comparison")
                    estimate, lower, upper = _paired_delta_interval(
                        paired,
                        "iou_a",
                        "iou_b",
                        bootstrap_samples=bootstrap_samples,
                        confidence_level=confidence_level,
                        bootstrap_seed=bootstrap_seed,
                    )
                    scene_differences = (
                        paired.assign(
                            iou_difference=(
                                paired["iou_a"].astype(float)
                                - paired["iou_b"].astype(float)
                            )
                        )
                        .groupby("source_scene_id", sort=True)[
                            "iou_difference"
                        ]
                        .mean()
                        .to_numpy(dtype=float)
                    )
                    p_value = (
                        1.0
                        if np.allclose(scene_differences, 0.0)
                        else float(
                            wilcoxon(
                                scene_differences,
                                alternative="two-sided",
                                zero_method="pratt",
                            ).pvalue
                        )
                    )
                    row = dict(zip(group_columns, key, strict=True))
                    row.update(
                        {
                            "stratum": stratum,
                            "model_a": model_a,
                            "model_b": model_b,
                            "paired_instances": int(len(paired)),
                            "source_scenes": int(
                                paired["source_scene_id"].nunique()
                            ),
                            "mean_iou_delta_a_minus_b": estimate,
                            "iou_delta_ci_lower": lower,
                            "iou_delta_ci_upper": upper,
                            "wilcoxon_p": p_value,
                            "wilcoxon_unit": "source_scene_mean",
                            "wilcoxon_observations": int(
                                len(scene_differences)
                            ),
                        }
                    )
                    rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    family_columns = [
        "dataset_id",
        "bbox_source",
        "detector_seed",
        "reference_type",
        "stratum",
    ]
    result["wilcoxon_p_holm"] = np.nan
    for _, indices in result.groupby(
        family_columns,
        dropna=False,
        sort=True,
    ).groups.items():
        index_list = list(indices)
        result.loc[index_list, "wilcoxon_p_holm"] = holm_adjust(
            result.loc[index_list, "wilcoxon_p"].tolist()
        )
    result["significant_at_0_05"] = result["wilcoxon_p_holm"] < 0.05
    return result


def reference_inflation(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    group_columns = ["dataset_id", "model", "bbox_source", "detector_seed"]
    rows: list[dict[str, object]] = []
    for key, frame in metrics.groupby(group_columns, dropna=False, sort=True):
        if not {"human", "pseudo_sam1"}.issubset(set(frame["reference_type"])):
            continue
        for stratum, selected in _stratum_groups(frame):
            human = selected[selected["reference_type"] == "human"][
                ["instance_id", "source_scene_id", *METRIC_COLUMNS]
            ]
            pseudo = selected[selected["reference_type"] == "pseudo_sam1"][
                ["instance_id", "source_scene_id", *METRIC_COLUMNS]
            ].rename(
                columns={
                    "source_scene_id": "source_scene_id_pseudo",
                    **{
                        metric: f"{metric}_pseudo"
                        for metric in METRIC_COLUMNS
                    },
                }
            )
            paired = human.merge(pseudo, on="instance_id", how="inner")
            if paired.empty:
                continue
            if not (
                paired["source_scene_id"] == paired["source_scene_id_pseudo"]
            ).all():
                raise ValueError("Source scene mismatch in reference pairing")
            estimate, lower, upper = _paired_delta_interval(
                paired,
                "iou_pseudo",
                "iou",
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                bootstrap_seed=bootstrap_seed,
            )
            row = dict(zip(group_columns, key, strict=True))
            row.update(
                {
                    "stratum": stratum,
                    "paired_instances": int(len(paired)),
                    "source_scenes": int(paired["source_scene_id"].nunique()),
                    "human_mean_iou": float(paired["iou"].mean()),
                    "pseudo_mean_iou": float(paired["iou_pseudo"].mean()),
                    "mean_iou_inflation": estimate,
                    "iou_inflation_ci_lower": lower,
                    "iou_inflation_ci_upper": upper,
                }
            )
            for metric in METRIC_COLUMNS[1:]:
                row[f"mean_{metric}_inflation"] = float(
                    (
                        paired[f"{metric}_pseudo"].astype(float)
                        - paired[metric].astype(float)
                    ).mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def ranking_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["dataset_id", "bbox_source", "detector_seed"]
    rows: list[dict[str, object]] = []
    for key, frame in metrics.groupby(group_columns, dropna=False, sort=True):
        if not {"human", "pseudo_sam1"}.issubset(set(frame["reference_type"])):
            continue
        for stratum, selected in _stratum_groups(frame):
            score_table = (
                selected.groupby(["model", "reference_type"], sort=True)["iou"]
                .mean()
                .unstack()
            )
            models = [model for model in MODEL_ORDER if model in score_table.index]
            if len(models) < 2:
                continue
            score_table = score_table.loc[models]
            if score_table[["human", "pseudo_sam1"]].isna().any().any():
                continue
            human_order = sorted(
                models,
                key=lambda model: (-float(score_table.loc[model, "human"]), model),
            )
            pseudo_order = sorted(
                models,
                key=lambda model: (
                    -float(score_table.loc[model, "pseudo_sam1"]),
                    model,
                ),
            )
            human_rank = {model: rank for rank, model in enumerate(human_order, 1)}
            pseudo_rank = {model: rank for rank, model in enumerate(pseudo_order, 1)}
            correlation = float(
                spearmanr(
                    [human_rank[model] for model in models],
                    [pseudo_rank[model] for model in models],
                ).statistic
            )
            kendall = float(
                kendalltau(
                    [human_rank[model] for model in models],
                    [pseudo_rank[model] for model in models],
                ).statistic
            )
            row = dict(zip(group_columns, key, strict=True))
            row.update(
                {
                    "stratum": stratum,
                    "human_order": ">".join(human_order),
                    "pseudo_order": ">".join(pseudo_order),
                    "spearman_rank_correlation": correlation,
                    "kendall_tau": kendall,
                    "models_changing_rank": sum(
                        human_rank[model] != pseudo_rank[model]
                        for model in models
                    ),
                }
            )
            if "sam1" in models:
                alternatives = [model for model in models if model != "sam1"]
                row["sam1_teacher_advantage_human"] = float(
                    score_table.loc["sam1", "human"]
                    - score_table.loc[alternatives, "human"].mean()
                )
                row["sam1_teacher_advantage_pseudo"] = float(
                    score_table.loc["sam1", "pseudo_sam1"]
                    - score_table.loc[alternatives, "pseudo_sam1"].mean()
                )
                row["sam1_teacher_advantage_change"] = float(
                    row["sam1_teacher_advantage_pseudo"]
                    - row["sam1_teacher_advantage_human"]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def collect_detector_metrics(detector_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(detector_root.glob("*/seed_*/evaluation/test/metrics.json"))
    rows: list[dict[str, object]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload)
    if not rows:
        return pd.DataFrame(columns=DETECTOR_METRIC_COLUMNS), files
    return pd.DataFrame(rows), files


def collect_training_health(
    detector_root: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(detector_root.glob("*/seed_*/train/results.csv"))
    rows: list[dict[str, object]] = []
    core_columns = {
        "final_precision": "metrics/precision(B)",
        "final_recall": "metrics/recall(B)",
        "final_ap50": "metrics/mAP50(B)",
        "final_ap50_95": "metrics/mAP50-95(B)",
    }
    for path in files:
        relative = path.relative_to(detector_root)
        if len(relative.parts) != 4:
            raise ValueError(f"Unexpected detector training path: {path}")
        seed_match = re.fullmatch(r"seed_(\d+)", relative.parts[1])
        if seed_match is None:
            raise ValueError(f"Cannot parse detector seed from {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Detector training results are empty: {path}")
        missing = {"epoch", *core_columns.values()} - set(frame.columns)
        if missing:
            raise ValueError(
                f"Detector training results are missing columns: "
                f"{path}: {sorted(missing)}"
            )
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        nonfinite = ~np.isfinite(numeric.to_numpy(dtype=float))
        validation_loss_columns = [
            column
            for column in numeric.columns
            if column.startswith("val/") and column.endswith("_loss")
        ]
        validation_nonfinite = (
            int(
                (
                    ~np.isfinite(
                        numeric[validation_loss_columns].to_numpy(dtype=float)
                    )
                ).sum()
            )
            if validation_loss_columns
            else 0
        )
        final = numeric.iloc[-1]
        final_metrics = {
            output_name: float(final[source_name])
            for output_name, source_name in core_columns.items()
        }
        rows.append(
            {
                "dataset_id": relative.parts[0],
                "seed": int(seed_match.group(1)),
                "results_file": str(path),
                "epochs_completed": int(len(frame)),
                "final_epoch": int(final["epoch"]),
                "nonfinite_cells": int(nonfinite.sum()),
                "nonfinite_validation_loss_cells": validation_nonfinite,
                "final_core_metrics_finite": bool(
                    all(np.isfinite(value) for value in final_metrics.values())
                ),
                **final_metrics,
            }
        )
    return pd.DataFrame(rows, columns=TRAINING_HEALTH_COLUMNS), files


def collect_prediction_status_audit(
    prediction_root: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(prediction_root.glob("*/*/*/**/predictions.jsonl"))
    files.extend(
        sorted(
            prediction_root.glob(
                "*/*/yolo_bbox/seed_*/unmatched_detector_predictions.jsonl"
            )
        )
    )
    files = sorted(set(files))
    rows: list[dict[str, object]] = []
    valid_statuses = {"ok", "empty_mask", "missing_bbox", "inference_error"}
    for path in files:
        relative = path.relative_to(prediction_root)
        if len(relative.parts) < 4:
            raise ValueError(f"Unexpected prediction path: {path}")
        dataset_id, model, bbox_source = relative.parts[:3]
        seed = None
        if bbox_source == "yolo_bbox":
            seed_match = re.fullmatch(r"seed_(\d+)", relative.parts[3])
            if seed_match is None:
                raise ValueError(f"Cannot parse detector seed from {path}")
            seed = int(seed_match.group(1))
        row_kind = (
            "unmatched_detector"
            if path.name == "unmatched_detector_predictions.jsonl"
            else "matched_ground_truth"
        )
        status_counts = {status: 0 for status in valid_statuses}
        instance_ids: list[str] = []
        zero_area_masks = 0
        nonzero_area_masks = 0
        nonempty_masks_without_prompt_overlap = 0
        status_area_mismatches = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                status = str(record.get("status", ""))
                if status not in valid_statuses:
                    raise ValueError(
                        f"Unknown status {status!r} in {path}:{line_number}"
                    )
                status_counts[status] += 1
                instance_ids.append(str(record["instance_id"]))
                encoded = record.get("predicted_mask_rle")
                if not isinstance(encoded, dict):
                    raise ValueError(
                        f"Missing mask RLE in {path}:{line_number}"
                    )
                area = float(mask_utils.area(encoded))
                if area > 0:
                    nonzero_area_masks += 1
                    input_bbox = record.get("input_bbox")
                    if (
                        not isinstance(input_bbox, list)
                        or len(input_bbox) != 4
                    ):
                        raise ValueError(
                            f"Missing input bbox in {path}:{line_number}"
                        )
                    x, y, width, height = (
                        float(value) for value in input_bbox
                    )
                    decoded = mask_utils.decode(encoded).astype(bool)
                    x1 = max(0, int(np.floor(x)))
                    y1 = max(0, int(np.floor(y)))
                    x2 = min(decoded.shape[1], int(np.ceil(x + width)))
                    y2 = min(decoded.shape[0], int(np.ceil(y + height)))
                    if (
                        x2 <= x1
                        or y2 <= y1
                        or not decoded[y1:y2, x1:x2].any()
                    ):
                        nonempty_masks_without_prompt_overlap += 1
                else:
                    zero_area_masks += 1
                expected_nonzero = status == "ok"
                if expected_nonzero != (area > 0):
                    status_area_mismatches += 1

        unique_instances = len(set(instance_ids))
        rows.append(
            {
                "dataset_id": dataset_id,
                "model": model,
                "bbox_source": bbox_source,
                "detector_seed": seed,
                "row_kind": row_kind,
                "prediction_file": str(path),
                "total_rows": len(instance_ids),
                "unique_instance_ids": unique_instances,
                "duplicate_instance_ids": len(instance_ids) - unique_instances,
                **status_counts,
                "zero_area_masks": zero_area_masks,
                "nonzero_area_masks": nonzero_area_masks,
                "nonempty_masks_without_prompt_overlap": (
                    nonempty_masks_without_prompt_overlap
                ),
                "status_area_mismatches": status_area_mismatches,
            }
        )
    return pd.DataFrame(rows, columns=PREDICTION_STATUS_AUDIT_COLUMNS), files


def detector_seed_summary(detector_metrics: pd.DataFrame) -> pd.DataFrame:
    if detector_metrics.empty:
        columns = ["dataset_id", "seed_count", "seed_ids"]
        for metric in DETECTOR_AGGREGATE_METRICS:
            columns.extend((f"{metric}_mean", f"{metric}_std"))
        return pd.DataFrame(columns=columns)
    metric_columns = [
        metric
        for metric in DETECTOR_AGGREGATE_METRICS
        if metric in detector_metrics.columns
    ]
    rows: list[dict[str, object]] = []
    for dataset_id, frame in detector_metrics.groupby("dataset_id", sort=True):
        row: dict[str, object] = {
            "dataset_id": dataset_id,
            "seed_count": int(frame["seed"].nunique()),
            "seed_ids": ",".join(
                str(seed) for seed in sorted(frame["seed"].astype(int).unique())
            ),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def segmentation_seed_summary(aggregates: pd.DataFrame) -> pd.DataFrame:
    empty_columns = ["dataset_id", "model", "reference_type", "seed_count"]
    for metric in SEGMENTATION_SEED_METRICS:
        empty_columns.extend((f"{metric}_seed_mean", f"{metric}_seed_std"))
    if aggregates.empty:
        return pd.DataFrame(columns=empty_columns)
    selected = aggregates[
        (aggregates["bbox_source"] == "yolo_bbox")
        & (aggregates["stratum"] == "overall")
    ]
    if selected.empty:
        return pd.DataFrame(columns=empty_columns)
    group_columns = ["dataset_id", "model", "reference_type"]
    metric_columns = list(SEGMENTATION_SEED_METRICS)
    rows: list[dict[str, object]] = []
    for key, frame in selected.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, key, strict=True))
        row["seed_count"] = int(frame["detector_seed"].nunique())
        for metric in metric_columns:
            row[f"{metric}_seed_mean"] = float(frame[metric].mean())
            row[f"{metric}_seed_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def write_analysis_manifest(
    path: Path,
    *,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    parameters: dict[str, object],
) -> None:
    payload = {
        "schema_version": 1,
        "status": "completed",
        "parameters": parameters,
        "inputs": [
            {"path": str(input_path), "sha256": sha256_file(input_path)}
            for input_path in sorted(set(inputs))
        ],
        "outputs": [
            {"path": str(output_path), "sha256": sha256_file(output_path)}
            for output_path in sorted(set(outputs))
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
