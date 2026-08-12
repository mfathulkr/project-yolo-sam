from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from yolo_sam.evaluation.statistics import clustered_bootstrap_mean

from .paths import (
    BBOX_SOURCES,
    DATASETS_BY_DATASET_ID,
    MODELS,
    REFERENCES,
    STRATA,
)


METRICS = ("iou", "dice", "precision", "recall")
SUCCESS_THRESHOLDS = (0.50, 0.75, 0.90)


def stratum_groups(frame: pd.DataFrame):
    yield "overall", frame
    for stratum in STRATA[1:]:
        yield stratum, frame[frame["stratum"] == stratum]


def aggregate_metrics(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
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
        for stratum, selected in stratum_groups(frame):
            if selected.empty:
                raise ValueError(f"Boş stratum: {key}/{stratum}")
            interval = clustered_bootstrap_mean(
                {
                    str(scene): group["iou"].astype(float).tolist()
                    for scene, group in selected.groupby("source_scene_id", sort=True)
                },
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
                        for metric in METRICS
                    },
                    "mean_boundary_iou": float("nan"),
                    "iou_ci_lower": interval.lower,
                    "iou_ci_upper": interval.upper,
                    "confidence_level": confidence_level,
                    "bootstrap_samples": bootstrap_samples,
                    **{
                        f"success_at_iou_{int(threshold * 100)}": float(
                            (selected["iou"] >= threshold).mean()
                        )
                        for threshold in SUCCESS_THRESHOLDS
                    },
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def paired_reference_effects(
    metrics: pd.DataFrame,
    *,
    baseline_reference: str,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    keys = ["dataset_id", "model", "bbox_source", "detector_seed"]
    rows: list[dict[str, object]] = []
    for key, frame in metrics.groupby(keys, dropna=False, sort=True):
        baseline = frame[frame["reference_type"] == baseline_reference]
        if baseline.empty:
            raise ValueError(f"Temel referans satırı eksik: {key}/{baseline_reference}")
        for reference_type in sorted(set(frame["reference_type"]) - {baseline_reference}):
            candidate = frame[frame["reference_type"] == reference_type]
            paired = baseline[["instance_id", "source_scene_id", "iou"]].merge(
                candidate[["instance_id", "iou"]],
                on="instance_id",
                how="inner",
                validate="one_to_one",
                suffixes=("_baseline", "_candidate"),
            )
            if len(paired) != len(baseline):
                raise ValueError(f"Eksik eşleşmiş referans satırı: {key}/{reference_type}")
            paired["delta"] = paired["iou_candidate"] - paired["iou_baseline"]
            interval = clustered_bootstrap_mean(
                {
                    str(scene): group["delta"].astype(float).tolist()
                    for scene, group in paired.groupby("source_scene_id", sort=True)
                },
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            )
            row = dict(zip(keys, key, strict=True))
            row.update(
                {
                    "baseline_reference": baseline_reference,
                    "comparison_reference": reference_type,
                    "instance_count": len(paired),
                    "baseline_mean_iou": float(paired["iou_baseline"].mean()),
                    "comparison_mean_iou": float(paired["iou_candidate"].mean()),
                    "delta_iou": interval.estimate,
                    "delta_ci_lower": interval.lower,
                    "delta_ci_upper": interval.upper,
                    "confidence_level": confidence_level,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def ranking_table(
    aggregates: pd.DataFrame,
    *,
    reference_order: tuple[str, ...],
    baseline_reference: str,
) -> pd.DataFrame:
    overall = aggregates[aggregates["stratum"] == "overall"]
    rows: list[dict[str, object]] = []
    for (dataset_id, bbox_source), frame in overall.groupby(
        ["dataset_id", "bbox_source"], sort=True
    ):
        baseline_scores = (
            frame[frame["reference_type"] == baseline_reference]
            .set_index("model")["mean_iou"]
            .to_dict()
        )
        baseline_order = tuple(
            sorted(baseline_scores, key=lambda model: (-baseline_scores[model], model))
        )
        baseline_rank = {model: rank for rank, model in enumerate(baseline_order, 1)}
        for reference_type in reference_order:
            selected = frame[frame["reference_type"] == reference_type]
            scores = selected.set_index("model")["mean_iou"].to_dict()
            order = tuple(sorted(scores, key=lambda model: (-scores[model], model)))
            rank = {model: index for index, model in enumerate(order, 1)}
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "bbox_source": bbox_source,
                    "reference_type": reference_type,
                    "baseline_reference": baseline_reference,
                    "ranking": " > ".join(model.upper() for model in order),
                    "rank_changes_vs_baseline": sum(
                        rank[model] != baseline_rank[model] for model in order
                    ),
                    "top_model": order[0].upper(),
                    "top_mean_iou": scores[order[0]],
                }
            )
    return pd.DataFrame(rows)


def teacher_advantage_table(aggregates: pd.DataFrame) -> pd.DataFrame:
    overall = aggregates[aggregates["stratum"] == "overall"]
    rows: list[dict[str, object]] = []
    for (dataset_id, bbox_source, reference_type), frame in overall.groupby(
        ["dataset_id", "bbox_source", "reference_type"], sort=True
    ):
        teacher = REFERENCES[str(reference_type)].teacher
        if teacher is None:
            continue
        scores = frame.set_index("model")["mean_iou"]
        alternatives = [model for model in scores.index if model != teacher]
        rows.append(
            {
                "dataset_id": dataset_id,
                "bbox_source": bbox_source,
                "reference_type": reference_type,
                "teacher": teacher,
                "teacher_score": float(scores.loc[teacher]),
                "other_models_mean": float(scores.loc[alternatives].mean()),
                "teacher_advantage": float(
                    scores.loc[teacher] - scores.loc[alternatives].mean()
                ),
                "identity_control": (
                    bbox_source == "gt_bbox"
                    and reference_type != "published_samrs_reference"
                ),
            }
        )
    return pd.DataFrame(rows)


def reference_agreement_table(
    aggregates: pd.DataFrame,
    *,
    reference_order: tuple[str, ...],
) -> pd.DataFrame:
    overall = aggregates[
        (aggregates["stratum"] == "overall")
        & (aggregates["bbox_source"] == "gt_bbox")
    ]
    rows: list[dict[str, object]] = []
    for dataset_id, frame in overall.groupby("dataset_id", sort=True):
        for left, right in combinations(reference_order, 2):
            left_teacher = REFERENCES[left].teacher
            right_teacher = REFERENCES[right].teacher
            if left_teacher is not None and left != "published_samrs_reference":
                model, measured_reference = left_teacher, right
            elif right_teacher is not None and right != "published_samrs_reference":
                model, measured_reference = right_teacher, left
            else:
                raise ValueError(f"Referans anlaşması ölçülemiyor: {left}/{right}")
            row = frame[
                (frame["model"] == model)
                & (frame["reference_type"] == measured_reference)
            ]
            if len(row) != 1:
                raise ValueError(
                    f"Referans anlaşma satırı eksik: {dataset_id}/{left}/{right}"
                )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "reference_a": left,
                    "reference_b": right,
                    "mean_instance_iou": float(row.iloc[0]["mean_iou"]),
                    "instance_count": int(row.iloc[0]["instance_count"]),
                }
            )
    return pd.DataFrame(rows)


def validate_metric_cube(metrics: pd.DataFrame) -> None:
    key = [
        "dataset_id",
        "model",
        "bbox_source",
        "detector_seed",
        "reference_type",
        "instance_id",
    ]
    if metrics.duplicated(key).any():
        raise ValueError("Birleşik metrik küpünde yinelenen anahtar var")
    for dataset_id, selected in metrics.groupby("dataset_id", sort=True):
        source = DATASETS_BY_DATASET_ID[str(dataset_id)]
        expected = source.instance_count * len(MODELS) * len(BBOX_SOURCES) * len(
            source.reference_types
        )
        if len(selected) != expected:
            raise ValueError(
                f"{dataset_id}: {expected} yerine {len(selected)} metrik satırı var"
            )
        if set(selected["reference_type"]) != set(source.reference_types):
            raise ValueError(f"{dataset_id}: referans kümesi protokolle uyuşmuyor")
        counts = selected.groupby(
            ["model", "bbox_source", "reference_type"], dropna=False
        )["instance_id"].nunique()
        if set(counts.astype(int)) != {source.instance_count}:
            raise ValueError(f"{dataset_id}: eksik instance koşulu var")

        identity_reference = {
            "sam1": (
                "pseudo_sam1"
                if source.dataset_family == "isaid"
                else "reproduced_pseudo_sam1"
            ),
            "sam2": "pseudo_sam2",
            "sam3": "pseudo_sam3",
        }
        for model, reference_type in identity_reference.items():
            identity = selected[
                (selected["model"] == model)
                & (selected["bbox_source"] == "gt_bbox")
                & (selected["reference_type"] == reference_type)
            ]
            reference_pixels = (
                identity["true_positive_pixels"].astype(int)
                + identity["false_negative_pixels"].astype(int)
            )
            expected_values = np.where(reference_pixels > 0, 1.0, 0.0)
            for metric in METRICS:
                if not np.allclose(identity[metric].astype(float), expected_values):
                    raise ValueError(
                        f"{dataset_id}/{model}: coverage-aware identity kontrolü "
                        f"uyuşmuyor ({metric})"
                    )
