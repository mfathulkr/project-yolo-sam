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


def paired_teacher_affinity_contrasts(
    metrics: pd.DataFrame,
    *,
    baseline_reference: str,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    """Measure producer-specific affinity, beyond generic pseudo-label easiness.

    ``self_vs_cross`` compares one model against its own pseudo reference with the
    same model against the other two teachers' pseudo references. ``relative_did``
    is a paired difference-in-differences: the producer model's advantage over the
    other models on its own reference minus that advantage on the baseline
    reference. Both contrasts are computed per instance and bootstrapped by source
    scene, so a generic improvement shared by every pseudo reference does not by
    itself count as teacher affinity.
    """

    keys = ["dataset_id", "bbox_source", "detector_seed"]
    rows: list[dict[str, object]] = []
    for key, frame in metrics.groupby(keys, dropna=False, sort=True):
        available_references = set(frame["reference_type"].astype(str))
        pseudo_references = tuple(
            reference_type
            for reference_type in sorted(available_references)
            if REFERENCES[reference_type].teacher is not None
            and reference_type != "published_samrs_reference"
        )
        if len(pseudo_references) != len(MODELS):
            raise ValueError(
                f"Üç üretici pseudo referansı bekleniyordu: {key}/{pseudo_references}"
            )
        own_reference = {
            model: next(
                reference_type
                for reference_type in pseudo_references
                if REFERENCES[reference_type].teacher == model
            )
            for model in MODELS
        }

        for stratum, selected in stratum_groups(frame):
            pivot = selected.pivot(
                index=["instance_id", "source_scene_id"],
                columns=["model", "reference_type"],
                values="iou",
            )
            required_columns = {
                (model, reference_type)
                for model in MODELS
                for reference_type in (baseline_reference, *pseudo_references)
            }
            missing = required_columns - set(pivot.columns)
            if missing or pivot[list(required_columns)].isna().any().any():
                raise ValueError(
                    f"Teacher-affinity eşleşmesi eksik: {key}/{stratum}/{missing}"
                )

            for model in MODELS:
                own = own_reference[model]
                cross = tuple(
                    reference_type
                    for reference_type in pseudo_references
                    if REFERENCES[reference_type].teacher != model
                )
                other_models = tuple(candidate for candidate in MODELS if candidate != model)
                self_vs_cross = (
                    pivot[(model, own)]
                    - pivot[[(model, reference_type) for reference_type in cross]].mean(
                        axis=1
                    )
                )
                own_relative_advantage = (
                    pivot[(model, own)]
                    - pivot[[(candidate, own) for candidate in other_models]].mean(axis=1)
                )
                baseline_relative_advantage = (
                    pivot[(model, baseline_reference)]
                    - pivot[
                        [
                            (candidate, baseline_reference)
                            for candidate in other_models
                        ]
                    ].mean(axis=1)
                )
                relative_did = own_relative_advantage - baseline_relative_advantage

                def interval(values: pd.Series):
                    values_frame = values.rename("contrast").reset_index()
                    return clustered_bootstrap_mean(
                        {
                            str(scene): group["contrast"].astype(float).tolist()
                            for scene, group in values_frame.groupby(
                                "source_scene_id", sort=True
                            )
                        },
                        bootstrap_samples=bootstrap_samples,
                        confidence_level=confidence_level,
                        seed=bootstrap_seed,
                    )

                self_interval = interval(self_vs_cross)
                did_interval = interval(relative_did)
                row = dict(zip(keys, key, strict=True))
                row.update(
                    {
                        "stratum": stratum,
                        "model": model,
                        "baseline_reference": baseline_reference,
                        "own_reference": own,
                        "cross_references": "+".join(cross),
                        "instance_count": int(len(pivot)),
                        "source_scene_count": int(
                            pivot.index.get_level_values("source_scene_id").nunique()
                        ),
                        "self_vs_cross_iou": self_interval.estimate,
                        "self_vs_cross_ci_lower": self_interval.lower,
                        "self_vs_cross_ci_upper": self_interval.upper,
                        "relative_advantage_did": did_interval.estimate,
                        "relative_advantage_did_ci_lower": did_interval.lower,
                        "relative_advantage_did_ci_upper": did_interval.upper,
                        "confidence_level": confidence_level,
                        "bootstrap_samples": bootstrap_samples,
                        "identity_control": bbox_source_is_identity_control(
                            str(key[1])
                        ),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def bbox_source_is_identity_control(bbox_source: str) -> bool:
    return bbox_source == "gt_bbox"


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
