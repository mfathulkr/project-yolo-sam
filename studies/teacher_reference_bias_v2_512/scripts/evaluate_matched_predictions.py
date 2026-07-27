from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pycocotools.coco import COCO

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.evaluation.instance_metrics import (
    InstanceMetricRow,
    aggregate_instance_metrics,
    binary_mask_metrics,
    evaluate_prediction_references,
    reference_inflation_rows,
)
from yolo_sam.evaluation.statistics import clustered_inflation_interval
from yolo_sam.runtime.manifest import (
    finish_run_manifest,
    new_run_manifest,
    write_run_manifest,
)
from yolo_sam.segmentation.runner import decode_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate canonical instance predictions against human and optional pseudo references."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument(
        "--coco-reference-type",
        choices=("human", "pseudo_sam1"),
        default="human",
        help="Provenance of the masks stored in the COCO file.",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--pseudo-references", type=Path, default=None)
    parser.add_argument("--unmatched-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--config-hash", default="not-provided")
    parser.add_argument("--run-id")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def human_references(
    coco: COCO,
    dataset_id: str,
) -> dict[str, np.ndarray]:
    references = {}
    for annotation_id in coco.getAnnIds():
        annotation = coco.loadAnns([annotation_id])[0]
        instance_id = (
            f"{dataset_id}:{int(annotation['image_id'])}:{int(annotation['id'])}"
        )
        references[instance_id] = coco.annToMask(annotation).astype(bool)
    return references


def pseudo_references(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    references = {}
    for row in read_jsonl(path):
        instance_id = str(row["instance_id"])
        if instance_id in references:
            raise ValueError(f"Duplicate pseudo reference: {instance_id}")
        references[instance_id] = decode_binary_mask(row["mask_rle"])
    return references


def validate_instance_sets(
    predictions: list[dict[str, Any]],
    coco_reference: dict[str, np.ndarray],
    pseudo: dict[str, np.ndarray],
) -> None:
    prediction_ids = [str(row["instance_id"]) for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Prediction file contains duplicate instance IDs")
    prediction_set = set(prediction_ids)
    reference_set = set(coco_reference)
    if prediction_set != reference_set:
        raise ValueError(
            "Prediction and COCO reference instance sets differ: "
            f"missing_predictions={len(reference_set - prediction_set)}, "
            f"unexpected_predictions={len(prediction_set - reference_set)}"
        )
    if pseudo and prediction_set != set(pseudo):
        raise ValueError(
            "Prediction and pseudo reference instance sets differ: "
            f"missing_pseudo={len(prediction_set - set(pseudo))}, "
            f"unexpected_pseudo={len(set(pseudo) - prediction_set)}"
        )


def evaluate_rows(
    predictions: list[dict[str, Any]],
    coco_reference: dict[str, np.ndarray],
    coco_reference_type: str,
    pseudo: dict[str, np.ndarray],
) -> list[InstanceMetricRow]:
    rows: list[InstanceMetricRow] = []
    for prediction in predictions:
        instance_id = str(prediction["instance_id"])
        references = {coco_reference_type: coco_reference[instance_id]}
        if pseudo:
            references["pseudo_sam1"] = pseudo[instance_id]
        prediction_mask = decode_binary_mask(prediction["predicted_mask_rle"])
        rows.extend(
            evaluate_prediction_references(
                run_id=str(prediction["run_id"]),
                model_id=str(prediction["model_id"]),
                model_version=str(prediction["model_version"]),
                prompt_type=str(prediction["prompt_type"]),
                image_id=str(prediction["image_id"]),
                instance_id=instance_id,
                source_scene_id=str(prediction["source_scene_id"]),
                stratum=str(prediction["stratum"]),
                prediction=prediction_mask,
                references=references,
            )
        )
    return rows


def summary_rows(metrics: list[InstanceMetricRow]) -> list[dict[str, object]]:
    output = []
    reference_types = sorted({row.reference_type for row in metrics})
    for reference_type in reference_types:
        reference_rows = [
            row for row in metrics if row.reference_type == reference_type
        ]
        groups = {"overall": reference_rows}
        groups.update(
            {
                stratum: [row for row in reference_rows if row.stratum == stratum]
                for stratum in sorted({row.stratum for row in reference_rows})
            }
        )
        for stratum, rows in groups.items():
            aggregate = aggregate_instance_metrics(rows)
            output.append(
                {
                    "reference_type": reference_type,
                    "stratum": stratum,
                    **aggregate.to_dict(),
                }
            )
    return output


def image_union_rows(
    predictions: list[dict[str, Any]],
    coco_reference: dict[str, np.ndarray],
    coco_reference_type: str,
    pseudo: dict[str, np.ndarray],
    unmatched_predictions: list[dict[str, Any]] | None = None,
) -> list[dict[str, object]]:
    grouped_predictions: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        grouped_predictions.setdefault(str(prediction["image_id"]), []).append(prediction)
    unmatched_by_image: dict[str, list[dict[str, Any]]] = {}
    for prediction in unmatched_predictions or []:
        unmatched_by_image.setdefault(str(prediction["image_id"]), []).append(prediction)

    rows = []
    for image_id, image_predictions in sorted(grouped_predictions.items()):
        prediction_masks = [
            decode_binary_mask(row["predicted_mask_rle"])
            for row in image_predictions
        ]
        prediction_masks.extend(
            decode_binary_mask(row["predicted_mask_rle"])
            for row in unmatched_by_image.get(image_id, [])
        )
        prediction_union = np.logical_or.reduce(prediction_masks)
        coco_reference_union = np.logical_or.reduce(
            [coco_reference[str(row["instance_id"])] for row in image_predictions]
        )
        references = {coco_reference_type: coco_reference_union}
        if pseudo:
            references["pseudo_sam1"] = np.logical_or.reduce(
                [pseudo[str(row["instance_id"])] for row in image_predictions]
            )
        exemplar = image_predictions[0]
        for reference_type, reference in references.items():
            metrics = binary_mask_metrics(prediction_union, reference)
            rows.append(
                {
                    "run_id": exemplar["run_id"],
                    "model_id": exemplar["model_id"],
                    "prompt_type": exemplar["prompt_type"],
                    "image_id": image_id,
                    "source_scene_id": exemplar["source_scene_id"],
                    "stratum": exemplar["stratum"],
                    "reference_type": reference_type,
                    **metrics.__dict__,
                }
            )
    return rows


def run_evaluation(args: argparse.Namespace) -> dict[str, str]:
    predictions = read_jsonl(args.predictions)
    coco = COCO(str(args.coco))
    coco_reference = human_references(coco, args.dataset_id)
    pseudo = pseudo_references(args.pseudo_references)
    if pseudo and args.coco_reference_type == "pseudo_sam1":
        raise ValueError(
            "The COCO reference and optional secondary reference cannot both be pseudo_sam1"
        )
    unmatched = (
        read_jsonl(args.unmatched_predictions)
        if args.unmatched_predictions is not None
        else []
    )
    validate_instance_sets(predictions, coco_reference, pseudo)
    metrics = evaluate_rows(
        predictions,
        coco_reference,
        args.coco_reference_type,
        pseudo,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row.to_dict() for row in metrics]).to_csv(
        args.output_dir / "metrics_instance.csv",
        index=False,
    )
    pd.DataFrame(summary_rows(metrics)).to_csv(
        args.output_dir / "summary_instance.csv",
        index=False,
    )
    pd.DataFrame(
        image_union_rows(
            predictions,
            coco_reference,
            args.coco_reference_type,
            pseudo,
            unmatched,
        )
    ).to_csv(
        args.output_dir / "metrics_image_union.csv",
        index=False,
    )

    outputs = {
        "metrics_instance": str(
            args.output_dir / "metrics_instance.csv"
        ),
        "summary_instance": str(
            args.output_dir / "summary_instance.csv"
        ),
        "metrics_image_union": str(
            args.output_dir / "metrics_image_union.csv"
        ),
    }
    if pseudo:
        inflation = reference_inflation_rows(metrics)
        pd.DataFrame(inflation).to_csv(
            args.output_dir / "reference_inflation_instance.csv",
            index=False,
        )
        interval = clustered_inflation_interval(
            inflation,
            metric_key="iou_inflation",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        (args.output_dir / "reference_inflation_bootstrap.json").write_text(
            json.dumps(interval.__dict__, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs["reference_inflation_instance"] = str(
            args.output_dir / "reference_inflation_instance.csv"
        )
        outputs["reference_inflation_bootstrap"] = str(
            args.output_dir / "reference_inflation_bootstrap.json"
        )
    return outputs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    inputs = {
        "coco": str(args.coco.resolve()),
        "predictions": str(args.predictions.resolve()),
    }
    if args.pseudo_references is not None:
        inputs["pseudo_references"] = str(
            args.pseudo_references.resolve()
        )
    if args.unmatched_predictions is not None:
        inputs["unmatched_predictions"] = str(
            args.unmatched_predictions.resolve()
        )
    manifest = new_run_manifest(
        project_root=ROOT,
        run_id=(
            args.run_id
            or f"{args.dataset_id}-matched-prediction-evaluation"
        ),
        stage="matched_prediction_evaluation",
        config_hash=args.config_hash,
        inputs=inputs,
        parameters={
            "dataset_id": args.dataset_id,
            "coco_reference_type": args.coco_reference_type,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "has_pseudo_reference": args.pseudo_references is not None,
            "has_unmatched_predictions": (
                args.unmatched_predictions is not None
            ),
        },
    )
    write_run_manifest(manifest_path, manifest)
    try:
        manifest["outputs"] = run_evaluation(args)
    except Exception as exc:
        finish_run_manifest(manifest, status="failed", error=str(exc))
        write_run_manifest(manifest_path, manifest)
        raise
    finish_run_manifest(manifest, status="completed")
    write_run_manifest(manifest_path, manifest)
    print(args.output_dir)


if __name__ == "__main__":
    main()
