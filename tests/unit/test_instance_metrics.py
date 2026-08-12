from __future__ import annotations

import unittest

import numpy as np

from yolo_sam.evaluation.instance_metrics import (
    InstanceMetricRow,
    aggregate_instance_metrics,
    binary_mask_metrics,
    evaluate_prediction_references,
    reference_inflation_rows,
)
from yolo_sam.evaluation.lazy_references import LazyMaskReferences
from yolo_sam.evaluation.statistics import (
    clustered_bootstrap_mean,
    clustered_inflation_interval,
    compare_model_rankings,
)


class InstanceMetricsTest(unittest.TestCase):
    def test_lazy_reference_store_decodes_only_requested_mask(self) -> None:
        decoded: list[str] = []

        def decode(value: str) -> np.ndarray:
            decoded.append(value)
            return np.asarray([[value == "positive"]], dtype=bool)

        references = LazyMaskReferences(
            {"instance-a": "positive", "instance-b": "negative"},
            decode,
        )

        self.assertEqual(len(references), 2)
        self.assertEqual(set(references), {"instance-a", "instance-b"})
        self.assertEqual(decoded, [])
        self.assertTrue(references.mask("instance-a")[0, 0])
        self.assertEqual(decoded, ["positive"])

    def test_lazy_reference_store_rejects_non_image_mask(self) -> None:
        references = LazyMaskReferences(
            {"instance": "encoded"},
            lambda _: np.zeros((1, 1, 1), dtype=bool),
        )
        with self.assertRaisesRegex(ValueError, "must be 2-D"):
            references.mask("instance")

    def test_binary_metrics_use_pixel_confusion_counts(self) -> None:
        reference = np.zeros((4, 4), dtype=bool)
        prediction = np.zeros((4, 4), dtype=bool)
        reference[0:2, 0:2] = True
        prediction[0:2, 1:3] = True

        metrics = binary_mask_metrics(prediction, reference)

        self.assertEqual(metrics.true_positive_pixels, 2)
        self.assertEqual(metrics.false_positive_pixels, 2)
        self.assertEqual(metrics.false_negative_pixels, 2)
        self.assertAlmostEqual(metrics.iou, 2 / 6)
        self.assertAlmostEqual(metrics.dice, 0.5)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)

    def test_empty_prediction_against_nonempty_reference_is_zero(self) -> None:
        prediction = np.zeros((3, 3), dtype=bool)
        reference = np.zeros((3, 3), dtype=bool)
        reference[1, 1] = True
        metrics = binary_mask_metrics(prediction, reference)
        self.assertEqual(metrics.iou, 0.0)
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)

    def test_empty_pseudo_reference_for_known_instance_is_not_success(self) -> None:
        prediction = np.zeros((3, 3), dtype=bool)
        reference = np.zeros_like(prediction)
        metrics = binary_mask_metrics(
            prediction,
            reference,
            known_positive_instance=True,
        )
        self.assertEqual(metrics.iou, 0.0)
        self.assertEqual(metrics.dice, 0.0)
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.boundary_iou, 0.0)

    def test_aggregate_success_rates_are_not_called_map(self) -> None:
        rows = [
            self._row("a", "human", 0.95),
            self._row("b", "human", 0.70),
            self._row("c", "human", 0.10),
        ]
        aggregate = aggregate_instance_metrics(rows)
        self.assertAlmostEqual(aggregate.success_at_iou_50, 2 / 3)
        self.assertAlmostEqual(aggregate.success_at_iou_75, 1 / 3)
        self.assertAlmostEqual(aggregate.success_at_iou_90, 1 / 3)

    def test_reference_inflation_is_paired_per_prediction(self) -> None:
        rows = [
            self._row("a", "human", 0.4),
            self._row("a", "pseudo_sam1", 0.9),
        ]
        inflation = reference_inflation_rows(rows)
        self.assertEqual(len(inflation), 1)
        self.assertAlmostEqual(float(inflation[0]["iou_inflation"]), 0.5)

    def test_one_prediction_is_evaluated_against_two_references(self) -> None:
        prediction = np.zeros((4, 4), dtype=bool)
        prediction[1:3, 1:3] = True
        human = prediction.copy()
        pseudo = np.zeros_like(prediction)
        pseudo[1, 1] = True
        rows = evaluate_prediction_references(
            run_id="run",
            model_id="sam1",
            model_version="sam1",
            prompt_type="gt_bbox",
            image_id="image",
            instance_id="instance",
            source_scene_id="scene",
            stratum="no_overlap_low",
            prediction=prediction,
            references={"human": human, "pseudo_sam1": pseudo},
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].iou, 1.0)
        self.assertEqual(rows[1].iou, 0.25)

    def test_clustered_bootstrap_and_ranking_comparison(self) -> None:
        rows = [
            self._row_for_model("sam1", "a", "human", 0.5),
            self._row_for_model("sam1", "a", "pseudo_sam1", 0.9),
            self._row_for_model("sam2", "a", "human", 0.8),
            self._row_for_model("sam2", "a", "pseudo_sam1", 0.7),
            self._row_for_model("sam1", "b", "human", 0.4),
            self._row_for_model("sam1", "b", "pseudo_sam1", 0.8),
            self._row_for_model("sam2", "b", "human", 0.7),
            self._row_for_model("sam2", "b", "pseudo_sam1", 0.6),
        ]
        inflation = reference_inflation_rows(rows)
        interval = clustered_inflation_interval(inflation, bootstrap_samples=100, seed=7)
        ranking = compare_model_rankings(rows)
        self.assertEqual(interval.clusters, 1)
        self.assertEqual(ranking.human_order, ("sam2", "sam1"))
        self.assertEqual(ranking.pseudo_order, ("sam1", "sam2"))
        self.assertAlmostEqual(ranking.spearman_correlation, -1.0)
        self.assertAlmostEqual(ranking.kendall_tau, -1.0)
        self.assertEqual(ranking.rank_changes, 2)

    def test_vectorized_cluster_bootstrap_matches_explicit_resampling(
        self,
    ) -> None:
        values_by_cluster = {
            "scene-a": [0.1, 0.3, 0.5],
            "scene-b": [0.7],
            "scene-c": [0.2, 0.4],
        }
        bootstrap_samples = 250
        seed = 17
        interval = clustered_bootstrap_mean(
            values_by_cluster,
            bootstrap_samples=bootstrap_samples,
            confidence_level=0.95,
            seed=seed,
        )

        cluster_ids = sorted(values_by_cluster)
        rng = np.random.default_rng(seed)
        explicit_means = []
        for _ in range(bootstrap_samples):
            sampled_ids = rng.choice(
                cluster_ids,
                size=len(cluster_ids),
                replace=True,
            )
            sampled_values = [
                value
                for cluster_id in sampled_ids
                for value in values_by_cluster[str(cluster_id)]
            ]
            explicit_means.append(float(np.mean(sampled_values)))
        expected_lower, expected_upper = np.quantile(
            np.asarray(explicit_means),
            [0.025, 0.975],
        )

        self.assertAlmostEqual(interval.estimate, 2.2 / 6.0)
        self.assertAlmostEqual(interval.lower, expected_lower)
        self.assertAlmostEqual(interval.upper, expected_upper)
        self.assertEqual(interval.clusters, 3)
        self.assertEqual(interval.observations, 6)

    @staticmethod
    def _row(instance_id: str, reference_type: str, value: float) -> InstanceMetricRow:
        return InstanceMetricRow(
            run_id="run",
            model_id="sam1",
            model_version="sam1",
            prompt_type="gt_bbox",
            image_id="image",
            instance_id=instance_id,
            source_scene_id="scene",
            reference_type=reference_type,
            stratum="overall",
            iou=value,
            dice=value,
            precision=value,
            recall=value,
            boundary_iou=value,
            true_positive_pixels=1,
            false_positive_pixels=0,
            false_negative_pixels=0,
        )

    @staticmethod
    def _row_for_model(
        model_id: str,
        instance_id: str,
        reference_type: str,
        value: float,
    ) -> InstanceMetricRow:
        row = InstanceMetricsTest._row(instance_id, reference_type, value)
        return InstanceMetricRow(
            **{
                **row.to_dict(),
                "model_id": model_id,
                "model_version": model_id,
            }
        )


if __name__ == "__main__":
    unittest.main()
