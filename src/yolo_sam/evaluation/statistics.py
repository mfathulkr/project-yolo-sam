from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import kendalltau

from yolo_sam.evaluation.instance_metrics import InstanceMetricRow


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    bootstrap_samples: int
    clusters: int
    observations: int


@dataclass(frozen=True)
class RankingComparison:
    human_order: tuple[str, ...]
    pseudo_order: tuple[str, ...]
    spearman_correlation: float
    kendall_tau: float
    rank_changes: int


def clustered_bootstrap_mean(
    values_by_cluster: dict[str, list[float]],
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapInterval:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if not values_by_cluster:
        raise ValueError("At least one cluster is required")
    if any(not values for values in values_by_cluster.values()):
        raise ValueError("Clusters cannot be empty")

    cluster_ids = sorted(values_by_cluster)
    all_values = np.asarray(
        [
            value
            for cluster_id in cluster_ids
            for value in values_by_cluster[cluster_id]
        ],
        dtype=np.float64,
    )
    cluster_sums = np.asarray(
        [
            np.asarray(values_by_cluster[cluster_id], dtype=np.float64).sum()
            for cluster_id in cluster_ids
        ],
        dtype=np.float64,
    )
    cluster_counts = np.asarray(
        [len(values_by_cluster[cluster_id]) for cluster_id in cluster_ids],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
    cluster_count = len(cluster_ids)
    max_sampled_indices_per_chunk = 2_000_000
    chunk_size = max(
        1,
        min(
            bootstrap_samples,
            max_sampled_indices_per_chunk // cluster_count,
        ),
    )
    for start in range(0, bootstrap_samples, chunk_size):
        stop = min(start + chunk_size, bootstrap_samples)
        sampled_indices = rng.integers(
            0,
            cluster_count,
            size=(stop - start, cluster_count),
        )
        sampled_sums = cluster_sums[sampled_indices].sum(axis=1)
        sampled_counts = cluster_counts[sampled_indices].sum(axis=1)
        bootstrap_means[start:stop] = sampled_sums / sampled_counts

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return BootstrapInterval(
        estimate=float(all_values.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        clusters=len(cluster_ids),
        observations=len(all_values),
    )


def clustered_inflation_interval(
    inflation_rows: Iterable[dict[str, object]],
    metric_key: str = "iou_inflation",
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapInterval:
    values_by_scene: dict[str, list[float]] = defaultdict(list)
    for row in inflation_rows:
        values_by_scene[str(row["source_scene_id"])].append(float(row[metric_key]))
    return clustered_bootstrap_mean(
        dict(values_by_scene),
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )


def _average_by_model(
    rows: Iterable[InstanceMetricRow],
    reference_type: str,
    metric: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.reference_type == reference_type:
            grouped[row.model_id].append(float(getattr(row, metric)))
    if not grouped:
        raise ValueError(f"No rows found for reference type {reference_type!r}")
    return {
        model_id: float(np.mean(values))
        for model_id, values in grouped.items()
    }


def compare_model_rankings(
    rows: Iterable[InstanceMetricRow],
    human_reference_type: str = "human",
    pseudo_reference_type: str = "pseudo_sam1",
    metric: str = "iou",
) -> RankingComparison:
    materialized = list(rows)
    human_scores = _average_by_model(materialized, human_reference_type, metric)
    pseudo_scores = _average_by_model(materialized, pseudo_reference_type, metric)
    common_models = sorted(set(human_scores) & set(pseudo_scores))
    if len(common_models) < 2:
        raise ValueError("At least two common models are required for ranking comparison")

    human_order = tuple(
        sorted(common_models, key=lambda model: (-human_scores[model], model))
    )
    pseudo_order = tuple(
        sorted(common_models, key=lambda model: (-pseudo_scores[model], model))
    )
    human_rank = {model: rank for rank, model in enumerate(human_order, start=1)}
    pseudo_rank = {model: rank for rank, model in enumerate(pseudo_order, start=1)}
    human_vector = np.asarray([human_rank[model] for model in common_models], dtype=np.float64)
    pseudo_vector = np.asarray([pseudo_rank[model] for model in common_models], dtype=np.float64)
    correlation = float(np.corrcoef(human_vector, pseudo_vector)[0, 1])
    kendall = float(kendalltau(human_vector, pseudo_vector).statistic)
    rank_changes = sum(human_rank[model] != pseudo_rank[model] for model in common_models)
    return RankingComparison(
        human_order=human_order,
        pseudo_order=pseudo_order,
        spearman_correlation=correlation,
        kendall_tau=kendall,
        rank_changes=rank_changes,
    )
