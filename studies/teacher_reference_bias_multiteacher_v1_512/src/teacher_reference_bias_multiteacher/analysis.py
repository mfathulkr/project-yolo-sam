from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from yolo_sam.evaluation.statistics import clustered_bootstrap_mean

from .paths import DATASETS, REFERENCE_TYPES, STRATA


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
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    keys = ["dataset_id", "model", "bbox_source", "detector_seed"]
    rows: list[dict[str, object]] = []
    for key, frame in metrics.groupby(keys, dropna=False, sort=True):
        human = frame[frame["reference_type"] == "human"]
        if human.empty:
            continue
        for reference_type in REFERENCE_TYPES[1:]:
            pseudo = frame[frame["reference_type"] == reference_type]
            paired = human[["instance_id", "source_scene_id", "iou"]].merge(
                pseudo[["instance_id", "iou"]],
                on="instance_id",
                how="inner",
                validate="one_to_one",
                suffixes=("_human", "_pseudo"),
            )
            if len(paired) != len(human):
                raise ValueError(f"Eksik paired referans satırı: {key}/{reference_type}")
            paired["delta"] = paired["iou_pseudo"] - paired["iou_human"]
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
                    "pseudo_reference": reference_type,
                    "instance_count": len(paired),
                    "human_mean_iou": float(paired["iou_human"].mean()),
                    "pseudo_mean_iou": float(paired["iou_pseudo"].mean()),
                    "delta_iou": interval.estimate,
                    "delta_ci_lower": interval.lower,
                    "delta_ci_upper": interval.upper,
                    "confidence_level": confidence_level,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def ranking_table(aggregates: pd.DataFrame) -> pd.DataFrame:
    overall = aggregates[aggregates["stratum"] == "overall"]
    rows: list[dict[str, object]] = []
    for (dataset_id, bbox_source), frame in overall.groupby(
        ["dataset_id", "bbox_source"], sort=True
    ):
        human_order: tuple[str, ...] | None = None
        for reference_type in REFERENCE_TYPES:
            selected = frame[frame["reference_type"] == reference_type]
            scores = selected.groupby("model")["mean_iou"].mean().to_dict()
            order = tuple(sorted(scores, key=lambda model: (-scores[model], model)))
            if reference_type == "human":
                human_order = order
            if human_order is None:
                raise ValueError("Human sıralaması pseudo sıralamadan önce gelmeli")
            human_rank = {model: rank for rank, model in enumerate(human_order, 1)}
            rank = {model: rank for rank, model in enumerate(order, 1)}
            changes = sum(rank[model] != human_rank[model] for model in order)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "bbox_source": bbox_source,
                    "reference_type": reference_type,
                    "ranking": " > ".join(model.upper() for model in order),
                    "rank_changes_vs_human": changes,
                    "top_model": order[0].upper(),
                    "top_mean_iou": scores[order[0]],
                }
            )
    return pd.DataFrame(rows)


def teacher_advantage_table(aggregates: pd.DataFrame) -> pd.DataFrame:
    overall = aggregates[aggregates["stratum"] == "overall"]
    rows: list[dict[str, object]] = []
    for dataset_id in DATASETS:
        for bbox_source in ("gt_bbox", "yolo_bbox"):
            for teacher in ("sam1", "sam2", "sam3"):
                reference_type = f"pseudo_{teacher}"
                selected = overall[
                    (overall["dataset_id"] == dataset_id)
                    & (overall["bbox_source"] == bbox_source)
                    & (overall["reference_type"] == reference_type)
                ]
                scores = selected.groupby("model")["mean_iou"].mean()
                alternatives = [model for model in scores.index if model != teacher]
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "bbox_source": bbox_source,
                        "teacher": teacher,
                        "teacher_score": float(scores.loc[teacher]),
                        "other_models_mean": float(scores.loc[alternatives].mean()),
                        "teacher_advantage": float(
                            scores.loc[teacher] - scores.loc[alternatives].mean()
                        ),
                        "identity_control": bbox_source == "gt_bbox",
                    }
                )
    return pd.DataFrame(rows)


def reference_agreement_table(aggregates: pd.DataFrame) -> pd.DataFrame:
    overall = aggregates[
        (aggregates["stratum"] == "overall")
        & (aggregates["bbox_source"] == "gt_bbox")
    ]
    rows: list[dict[str, object]] = []
    reference_to_model = {
        "pseudo_sam1": "sam1",
        "pseudo_sam2": "sam2",
        "pseudo_sam3": "sam3",
    }
    for dataset_id in DATASETS:
        selected = overall[overall["dataset_id"] == dataset_id]
        for left, right in combinations(REFERENCE_TYPES, 2):
            if left == "human":
                model = reference_to_model[right]
                row = selected[
                    (selected["model"] == model)
                    & (selected["reference_type"] == "human")
                ]
            else:
                model = reference_to_model[left]
                row = selected[
                    (selected["model"] == model)
                    & (selected["reference_type"] == right)
                ]
            if len(row) != 1:
                raise ValueError(f"Referans agreement satırı eksik: {dataset_id}/{left}/{right}")
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
    for dataset_id, source in DATASETS.items():
        selected = metrics[metrics["dataset_id"] == dataset_id]
        expected = source.teacher_instance_count * 3 * 2 * 4
        if len(selected) != expected:
            raise ValueError(
                f"{dataset_id}: {expected} yerine {len(selected)} metrik satırı var"
            )
        for (model, bbox_source), condition in selected.groupby(
            ["model", "bbox_source"], sort=True
        ):
            counts = condition.groupby("reference_type")["instance_id"].nunique()
            if set(counts.index) != set(REFERENCE_TYPES):
                raise ValueError(f"Eksik referans: {dataset_id}/{model}/{bbox_source}")
            if set(counts.astype(int)) != {source.teacher_instance_count}:
                raise ValueError(f"Eksik instance: {dataset_id}/{model}/{bbox_source}")

    identity = metrics[
        (metrics["bbox_source"] == "gt_bbox")
        & (
            metrics.apply(
                lambda row: row["reference_type"] == f"pseudo_{row['model']}",
                axis=1,
            )
        )
    ]
    if not np.allclose(identity[["iou", "dice", "precision", "recall"]], 1.0):
        raise ValueError("Teacher self-reference GT-bbox özdeşlik kontrolü 1.0 değil")
