from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import yaml
from docx import Document


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
REPORT_ROOT = STUDY_ROOT / "reports" / "full_metrics"
REPORTS = (
    "isaid_plane_human",
    "isaid_plane_pseudo_sam1",
    "samrs_sota_plane",
)
EXPECTED_REPORT_METADATA = {
    "isaid_plane_human": {
        "dataset_id": "isaid_plane",
        "reference_type": "human",
    },
    "isaid_plane_pseudo_sam1": {
        "dataset_id": "isaid_plane",
        "reference_type": "pseudo_sam1",
    },
    "samrs_sota_plane": {
        "dataset_id": "samrs_sota_plane",
        "reference_type": "pseudo_sam1",
    },
}
EXPECTED_SEGMENTATION_COLUMNS = (
    "Pipeline",
    "Images",
    "Avg IoU",
    "Avg Dice",
    "Avg Precision",
    "Avg Recall",
    "IoU ≥ 0.50",
    "IoU ≥ 0.75",
    "IoU ≥ 0.90",
)
EXPECTED_DETECTOR_COLUMNS = (
    "Detector",
    "Images",
    "BBox mAP50",
    "BBox mAP75",
    "BBox mAP90",
    "BBox mAP50-95",
    "BBox Precision@0.50",
    "BBox Recall@0.50",
    "BBox Precision@0.75",
    "BBox Recall@0.75",
    "BBox Precision@0.90",
    "BBox Recall@0.90",
)
EXPECTED_DATASET_INSTANCES = {
    "isaid_plane": 5_447,
    "samrs_sota_plane": 3_713,
}
PROTOCOL = yaml.safe_load(
    (STUDY_ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8")
)
EXPECTED_SEEDS = {int(seed) for seed in PROTOCOL["detector_seeds"]}


def metric_mean(value: object) -> float:
    mean = float(str(value).split("±", maxsplit=1)[0].strip())
    if not math.isfinite(mean) or not 0.0 <= mean <= 1.0:
        raise AssertionError(f"Metric is outside [0, 1]: {value!r}")
    return mean


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def validate_fingerprint(path: Path, fingerprint: dict[str, object]) -> None:
    if not path.is_file():
        raise AssertionError(f"Missing fingerprinted file: {path}")
    if path.stat().st_size != int(fingerprint["bytes"]):
        raise AssertionError(f"{path}: byte-size mismatch")
    if sha256_file(path) != str(fingerprint["sha256"]):
        raise AssertionError(f"{path}: SHA-256 mismatch")


def validate_run_manifest(path: Path, payload: dict[str, object]) -> None:
    stage = payload.get("stage")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        raise AssertionError(f"{path}: invalid parameters")
    if stage == "gt_bbox_segmentation":
        if parameters.get("prompt_type") != "gt_bbox":
            raise AssertionError(f"{path}: wrong GT-bbox prompt lineage")
        segmenter_config = parameters.get("model_config", {})
    elif stage == "yolo_bbox_segmentation":
        if int(parameters.get("seed", -1)) not in EXPECTED_SEEDS:
            raise AssertionError(f"{path}: non-canonical detector seed")
        segmenter_config = parameters.get("segmenter", {})
    else:
        segmenter_config = {}
    if parameters.get("model") == "sam3":
        if not isinstance(segmenter_config, dict):
            raise AssertionError(f"{path}: missing SAM3 configuration")
        if segmenter_config.get("inference_interface") != "sam3_tracker_pvs":
            raise AssertionError(f"{path}: SAM3 is not tracker PVS")
        if float(segmenter_config.get("mask_threshold", 1.0)) != 0.0:
            raise AssertionError(f"{path}: unsafe SAM3 mask threshold")
        if int(segmenter_config.get("box_batch_size", 0)) <= 0:
            raise AssertionError(f"{path}: missing SAM3 box batch size")

    for section in (
        "input_file_fingerprints_at_finish",
        "output_file_fingerprints",
    ):
        if section not in payload:
            raise AssertionError(f"{path}: missing {section}")
        fingerprints = payload[section]
        if not isinstance(fingerprints, dict):
            raise AssertionError(f"{path}: invalid {section}")
        for fingerprint_name, fingerprint in fingerprints.items():
            if not isinstance(fingerprint, dict) or "path" not in fingerprint:
                raise AssertionError(f"{path}: malformed {section} entry")
            fingerprint_path = resolve_manifest_path(str(fingerprint["path"]))
            validate_fingerprint(fingerprint_path, fingerprint)


def validate_analysis_manifest() -> None:
    path = STUDY_ROOT / "results" / "analysis" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise AssertionError(f"{path}: analysis is not completed")
    for section in ("inputs", "outputs"):
        for fingerprint in payload.get(section, []):
            source = resolve_manifest_path(str(fingerprint["path"]))
            if not source.is_file() or sha256_file(source) != fingerprint["sha256"]:
                raise AssertionError(f"{path}: stale {section} fingerprint for {source}")


def validate_segmentation_tables(report_root: Path) -> None:
    table_paths = sorted(
        path
        for path in (report_root / "tables").glob("*.csv")
        if path.name not in {
            "detector_summary.csv",
            "reference_sensitivity.csv",
        }
    )
    if len(table_paths) != 5:
        raise AssertionError(
            f"{report_root.name}: expected 5 segmentation tables, "
            f"found {len(table_paths)}"
        )
    reference_type = str(
        EXPECTED_REPORT_METADATA[report_root.name]["reference_type"]
    )
    expected_stems = {
        f"{reference_type}__overall",
        f"{reference_type}__no_overlap__low_mask_area",
        f"{reference_type}__no_overlap__high_mask_area",
        f"{reference_type}__overlap__low_mask_area",
        f"{reference_type}__overlap__high_mask_area",
    }
    actual_stems = {path.stem for path in table_paths}
    if actual_stems != expected_stems:
        raise AssertionError(
            f"{report_root.name}: expected table stems {sorted(expected_stems)}, "
            f"found {sorted(actual_stems)}"
        )
    for path in table_paths:
        frame = pd.read_csv(path)
        if tuple(frame.columns) != EXPECTED_SEGMENTATION_COLUMNS:
            raise AssertionError(
                f"{path}: unexpected columns {tuple(frame.columns)}"
            )
        if len(frame) != 6:
            raise AssertionError(f"{path}: expected 6 pipelines, found {len(frame)}")
        expected_images = 512 if path.stem.endswith("__overall") else 128
        image_counts = set(frame["Images"].astype(int))
        if image_counts != {expected_images}:
            raise AssertionError(
                f"{path}: expected Images={expected_images}, got {image_counts}"
            )
        if set(frame["Pipeline"]) != {
            "SAM1 GT bbox",
            "SAM1 YOLO bbox",
            "SAM2 GT bbox",
            "SAM2 YOLO bbox",
            "SAM3 GT bbox",
            "SAM3 YOLO bbox",
        }:
            raise AssertionError(f"{path}: pipeline set is incomplete")
        for column in EXPECTED_SEGMENTATION_COLUMNS[2:]:
            for value in frame[column]:
                metric_mean(value)


def validate_detector_table(report_root: Path) -> None:
    path = report_root / "tables" / "detector_summary.csv"
    frame = pd.read_csv(path)
    if tuple(frame.columns) != EXPECTED_DETECTOR_COLUMNS:
        raise AssertionError(f"{path}: unexpected columns {tuple(frame.columns)}")
    if len(frame) != 1 or int(frame.iloc[0]["Images"]) != 512:
        raise AssertionError(f"{path}: expected one 512-image detector row")
    expected_label = (
        f"(seed {next(iter(EXPECTED_SEEDS))})"
        if len(EXPECTED_SEEDS) == 1
        else f"({len(EXPECTED_SEEDS)} seed)"
    )
    if expected_label not in str(frame.iloc[0]["Detector"]):
        raise AssertionError(f"{path}: detector summary has the wrong seed scope")
    for column in EXPECTED_DETECTOR_COLUMNS[2:]:
        metric_mean(frame.iloc[0][column])


def validate_manifest(report_root: Path) -> None:
    path = report_root / "report_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["study_id"] != STUDY_ROOT.name:
        raise AssertionError(f"{path}: wrong study_id")
    expected = EXPECTED_REPORT_METADATA[report_root.name]
    if manifest["dataset_id"] != expected["dataset_id"]:
        raise AssertionError(f"{path}: wrong dataset_id")
    if manifest["reference_sections"] != [expected["reference_type"]]:
        raise AssertionError(f"{path}: wrong reference section")
    if manifest["report_format"] != "legacy_samrs_full_metric_colored":
        raise AssertionError(f"{path}: wrong report format")
    for file_name, expected_hash in {
        **manifest["inputs"],
        **manifest["outputs"],
    }.items():
        file_path = Path(file_name)
        if not file_path.is_absolute():
            file_path = REPO_ROOT / file_path
        if not file_path.is_file():
            raise AssertionError(f"{path}: missing manifest file {file_path}")
        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            raise AssertionError(
                f"{path}: hash mismatch for {file_path}"
            )


def validate_rendered_documents(report_root: Path) -> None:
    stem = report_root.name
    markdown_path = report_root / f"{stem}_full_metric_document.md"
    docx_path = report_root / f"{stem}_full_metric_document_colored.docx"
    pdf_path = report_root / f"{stem}_full_metric_document_colored.pdf"
    for path in (markdown_path, docx_path, pdf_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty report: {path}")

    with zipfile.ZipFile(docx_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise AssertionError(
                f"{docx_path}: corrupt DOCX member {corrupt_member}"
            )
        media_members = [
            name
            for name in archive.namelist()
            if name.startswith("word/media/")
        ]
        if len(media_members) != 4:
            raise AssertionError(
                f"{docx_path}: expected 4 embedded qualitative images, "
                f"found {len(media_members)}"
            )
    expected_docx_tables = 6 + int(
        (report_root / "tables" / "reference_sensitivity.csv").is_file()
    )
    actual_docx_tables = len(Document(docx_path).tables)
    if actual_docx_tables != expected_docx_tables:
        raise AssertionError(
            f"{docx_path}: expected {expected_docx_tables} tables, "
            f"found {actual_docx_tables}"
        )

    pdf_info = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    page_line = next(
        (
            line
            for line in pdf_info.splitlines()
            if line.startswith("Pages:")
        ),
        "",
    )
    if not page_line:
        raise AssertionError(f"{pdf_path}: pdfinfo did not report a page count")
    page_count = int(page_line.split(":", maxsplit=1)[1].strip())
    if page_count < 12:
        raise AssertionError(
            f"{pdf_path}: expected at least 12 readable pages, found {page_count}"
        )

    completed = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    text = completed.stdout
    for required in (
        "Overall",
        "No Overlap",
        "Overlap",
        "Avg IoU",
        "BBox mAP50",
        "SAM1 GT bbox",
        "SAM3 YOLO bbox",
        "512",
        "128",
    ):
        if required not in text:
            raise AssertionError(f"{pdf_path}: missing PDF text {required!r}")
    for forbidden in (
        "mAP proxy",
        "mAP50 proxy",
        "Pred/GT Area",
        "Boundary IoU",
        "RemoteSAM",
        "RingMo",
    ):
        if forbidden in text:
            raise AssertionError(
                f"{pdf_path}: forbidden legacy metric/pipeline {forbidden!r}"
            )


def completed_manifest_count(pattern: str, expected: int) -> None:
    paths = sorted((STUDY_ROOT / "results").glob(pattern))
    paths = [
        path
        for path in paths
        if all(
            int(match.group(1)) in EXPECTED_SEEDS
            for part in path.parts
            if (match := re.match(r"seed_(\d+)(?:_|$)", part)) is not None
        )
    ]
    if len(paths) != expected:
        raise AssertionError(
            f"{pattern}: expected {expected} manifests, found {len(paths)}"
        )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise AssertionError(
                f"{path}: expected completed, found {payload.get('status')!r}"
            )
        validate_run_manifest(path, payload)


def validate_prepared_test_sets() -> None:
    prepared_root = STUDY_ROOT / "data" / "prepared"
    segmenter_audit = json.loads(
        (
            STUDY_ROOT
            / "results"
            / "audits"
            / "segmenter_provenance.json"
        ).read_text(encoding="utf-8")
    )
    if segmenter_audit.get("status") != "pass":
        raise AssertionError("Segmenter provenance audit did not pass")
    if {
        str(row["model"])
        for row in segmenter_audit.get("models", [])
        if bool(row.get("passed"))
    } != {"sam1", "sam2", "sam3"}:
        raise AssertionError("Segmenter checkpoint provenance is incomplete")

    for dataset_id, expected_instances in EXPECTED_DATASET_INSTANCES.items():
        raw_audit = json.loads(
            (
                STUDY_ROOT
                / "results"
                / "dataset_audits"
                / dataset_id
                / "audit.json"
            ).read_text(encoding="utf-8")
        )
        prepared_audit = json.loads(
            (
                STUDY_ROOT
                / "results"
                / "audits"
                / f"{dataset_id}_prepared_dataset.json"
            ).read_text(encoding="utf-8")
        )
        content_manifest = json.loads(
            (prepared_root / dataset_id / "content_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if not bool(raw_audit.get("passed")):
            raise AssertionError(f"{dataset_id}: raw provenance audit did not pass")
        if not bool(prepared_audit.get("passed")):
            raise AssertionError(f"{dataset_id}: prepared-data audit did not pass")
        if (
            content_manifest.get("status") != "completed"
            or content_manifest.get("scope") != "full_prepared_dataset"
        ):
            raise AssertionError(
                f"{dataset_id}: prepared content manifest is incomplete"
            )

        test_root = prepared_root / dataset_id / "test"
        metadata = pd.read_csv(test_root / "metadata.csv")
        if len(metadata) != 512:
            raise AssertionError(
                f"{dataset_id}: expected 512 test images, found {len(metadata)}"
            )
        stratum_counts = metadata["stratum"].value_counts().to_dict()
        expected_counts = {
            "no_overlap__low_mask_area": 128,
            "no_overlap__high_mask_area": 128,
            "overlap__low_mask_area": 128,
            "overlap__high_mask_area": 128,
        }
        if stratum_counts != expected_counts:
            raise AssertionError(
                f"{dataset_id}: unexpected test strata {stratum_counts}"
            )
        annotations = json.loads(
            (test_root / "_annotations.coco.json").read_text(encoding="utf-8")
        )
        if len(annotations["images"]) != 512:
            raise AssertionError(
                f"{dataset_id}: COCO image count is not 512"
            )
        if len(annotations["annotations"]) != expected_instances:
            raise AssertionError(
                f"{dataset_id}: expected {expected_instances} instances, "
                f"found {len(annotations['annotations'])}"
            )
        object_count_by_image: dict[int, int] = {}
        area_sum_by_image: dict[int, float] = {}
        image_area_by_id = {
            int(image["id"]): int(image["width"]) * int(image["height"])
            for image in annotations["images"]
        }
        for annotation in annotations["annotations"]:
            image_id = int(annotation["image_id"])
            object_count_by_image[image_id] = object_count_by_image.get(image_id, 0) + 1
            area_sum_by_image[image_id] = area_sum_by_image.get(image_id, 0.0) + float(
                annotation["area"]
            )
        for _, row in metadata.iterrows():
            image_id = int(row["image_id"])
            expected_count = object_count_by_image.get(image_id, 0)
            expected_area = area_sum_by_image.get(image_id, 0.0)
            expected_ratio = expected_area / float(image_area_by_id[image_id])
            if int(row["num_objects"]) != expected_count:
                raise AssertionError(f"{dataset_id}/{image_id}: object-count mismatch")
            if not math.isclose(
                float(row["mask_area_pixels"]),
                expected_area,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise AssertionError(f"{dataset_id}/{image_id}: mask-area mismatch")
            if not math.isclose(
                float(row["mask_area_ratio"]),
                expected_ratio,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AssertionError(f"{dataset_id}/{image_id}: mask-area ratio mismatch")
        annotated_image_ids = {
            int(annotation["image_id"])
            for annotation in annotations["annotations"]
        }
        coco_image_ids = {
            int(image["id"]) for image in annotations["images"]
        }
        if annotated_image_ids != coco_image_ids:
            raise AssertionError(
                f"{dataset_id}: test set contains an image without a plane instance"
            )


def validate_analysis_outputs() -> None:
    validate_analysis_manifest()
    analysis_root = STUDY_ROOT / "results" / "analysis"

    canonical = pd.read_csv(
        analysis_root / "canonical_instance_metrics.csv",
        dtype={"detector_seed": "Int64"},
    )
    conditions_per_reference = 3 * (1 + len(EXPECTED_SEEDS))
    expected_canonical_rows = (
        2 * conditions_per_reference * EXPECTED_DATASET_INSTANCES["isaid_plane"]
        + conditions_per_reference
        * EXPECTED_DATASET_INSTANCES["samrs_sota_plane"]
    )
    if len(canonical) != expected_canonical_rows:
        raise AssertionError(
            f"Expected {expected_canonical_rows} canonical rows, "
            f"found {len(canonical)}"
        )
    canonical_keys = [
        "dataset_id",
        "model",
        "bbox_source",
        "detector_seed",
        "reference_type",
        "instance_id",
    ]
    if canonical.duplicated(canonical_keys).any():
        raise AssertionError("Canonical instance metrics contain duplicate keys")
    expected_reference_rows = {
        ("isaid_plane", "human"): (
            conditions_per_reference * EXPECTED_DATASET_INSTANCES["isaid_plane"]
        ),
        ("isaid_plane", "pseudo_sam1"): (
            conditions_per_reference * EXPECTED_DATASET_INSTANCES["isaid_plane"]
        ),
        ("samrs_sota_plane", "pseudo_sam1"): (
            conditions_per_reference
            * EXPECTED_DATASET_INSTANCES["samrs_sota_plane"]
        ),
    }
    actual_reference_rows = {
        (str(dataset_id), str(reference_type)): int(len(frame))
        for (dataset_id, reference_type), frame in canonical.groupby(
            ["dataset_id", "reference_type"],
            sort=True,
        )
    }
    if actual_reference_rows != expected_reference_rows:
        raise AssertionError(
            "Unexpected canonical dataset/reference row counts: "
            f"{actual_reference_rows}"
        )
    for column in ("iou", "dice", "precision", "recall"):
        values = pd.to_numeric(canonical[column], errors="coerce")
        if (
            values.isna().any()
            or (values < 0.0).any()
            or (values > 1.0).any()
        ):
            raise AssertionError(
                f"Canonical metric {column} is not finite within [0, 1]"
            )

    training = pd.read_csv(analysis_root / "training_health_audit.csv")
    expected_detector_runs = 2 * len(EXPECTED_SEEDS)
    if len(training) != expected_detector_runs:
        raise AssertionError(
            f"Expected {expected_detector_runs} detector trainings, found {len(training)}"
        )
    for dataset_id, frame in training.groupby("dataset_id"):
        if set(frame["seed"].astype(int)) != EXPECTED_SEEDS:
            raise AssertionError(
                f"{dataset_id}: detector training seeds are incomplete"
            )
        for row in frame.itertuples(index=False):
            args_path = Path(str(row.results_file)).parent / "args.yaml"
            args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
            expected_training_args = {
                "epochs": 100,
                "batch": 12,
                "imgsz": 1024,
                "patience": 30,
                "deterministic": True,
                "seed": int(row.seed),
            }
            actual_training_args = {
                key: args.get(key) for key in expected_training_args
            }
            if actual_training_args != expected_training_args:
                raise AssertionError(
                    f"{dataset_id}/seed_{row.seed}: unexpected training args "
                    f"{actual_training_args}"
                )
            if Path(str(args.get("model", ""))).name != "yolo26x.pt":
                raise AssertionError(
                    f"{dataset_id}/seed_{row.seed}: wrong detector base model"
                )
    final_finite = training["final_core_metrics_finite"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    if not final_finite.all():
        raise AssertionError("At least one detector has non-finite final metrics")

    detectors = pd.read_csv(
        analysis_root / "detector_metrics_by_seed.csv"
    )
    if len(detectors) != expected_detector_runs:
        raise AssertionError(
            f"Expected {expected_detector_runs} detector test rows, found {len(detectors)}"
        )
    if set(detectors["images"].astype(int)) != {512}:
        raise AssertionError("Detector test rows are not all 512 images")
    for dataset_id, frame in detectors.groupby("dataset_id"):
        if set(frame["seed"].astype(int)) != EXPECTED_SEEDS:
            raise AssertionError(
                f"{dataset_id}: detector test seeds are incomplete"
            )
        if set(frame["confidence_threshold_source_split"]) != {"validation"}:
            raise AssertionError(
                f"{dataset_id}: test confidence threshold is not validation-selected"
            )
        if set(frame["confidence_threshold_selection_method"]) != {"max_f1"}:
            raise AssertionError(
                f"{dataset_id}: unexpected confidence selection method"
            )
        for seed in EXPECTED_SEEDS:
            detector_root = (
                STUDY_ROOT
                / "results"
                / "detectors"
                / str(dataset_id)
                / f"seed_{seed}"
                / "evaluation"
            )
            test_metrics = json.loads(
                (detector_root / "test" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            selection = json.loads(
                (
                    detector_root
                    / "validation"
                    / "selected_confidence_threshold.json"
                ).read_text(encoding="utf-8")
            )
            expected_threshold = float(
                selection["selected_confidence_threshold"]
            )
            if not math.isclose(
                float(test_metrics["fixed_confidence_threshold"]),
                expected_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AssertionError(
                    f"{dataset_id}/seed_{seed}: test threshold differs from validation"
                )
            for model in ("sam1", "sam2", "sam3"):
                prediction_manifest = json.loads(
                    (
                        STUDY_ROOT
                        / "results"
                        / "predictions"
                        / str(dataset_id)
                        / model
                        / "yolo_bbox"
                        / f"seed_{seed}"
                        / "manifest.json"
                    ).read_text(encoding="utf-8")
                )
                actual_threshold = float(
                    prediction_manifest["parameters"][
                        "selected_confidence_threshold"
                    ]
                )
                if not math.isclose(
                    actual_threshold,
                    expected_threshold,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise AssertionError(
                        f"{dataset_id}/{model}/seed_{seed}: inference threshold "
                        "differs from validation"
                    )

    predictions = pd.read_csv(
        analysis_root / "prediction_status_audit.csv"
    )
    matched = predictions[predictions["row_kind"] == "matched_ground_truth"]
    unmatched = predictions[
        predictions["row_kind"] == "unmatched_detector"
    ]
    expected_matched_files = 2 * conditions_per_reference
    if len(matched) != expected_matched_files:
        raise AssertionError(
            f"Expected {expected_matched_files} matched prediction files, found {len(matched)}"
        )
    expected_unmatched_files = 2 * 3 * len(EXPECTED_SEEDS)
    if len(unmatched) != expected_unmatched_files:
        raise AssertionError(
            f"Expected {expected_unmatched_files} unmatched-detector files, found {len(unmatched)}"
        )
    for dataset_id, expected_instances in EXPECTED_DATASET_INSTANCES.items():
        frame = matched[matched["dataset_id"] == dataset_id]
        if len(frame) != conditions_per_reference:
            raise AssertionError(
                f"{dataset_id}: expected {conditions_per_reference} matched conditions, found {len(frame)}"
            )
        for column in ("total_rows", "unique_instance_ids"):
            if set(frame[column].astype(int)) != {expected_instances}:
                raise AssertionError(
                    f"{dataset_id}: {column} does not equal "
                    f"{expected_instances}"
                )
    for frame_name, frame in (("matched", matched), ("unmatched", unmatched)):
        for column in (
            "duplicate_instance_ids",
            "inference_error",
            "status_area_mismatches",
        ):
            if int(frame[column].astype(int).sum()) != 0:
                raise AssertionError(
                    f"{frame_name}: non-zero {column}"
                )
        # A segmenter may select a neighboring object despite a valid bbox prompt.
        # Keep that mask unchanged so the metric records the real model failure.
        outside_prompt = frame[
            "nonempty_masks_without_prompt_overlap"
        ].astype(int)
        if (outside_prompt < 0).any() or (
            outside_prompt > frame["nonzero_area_masks"].astype(int)
        ).any():
            raise AssertionError(
                f"{frame_name}: invalid nonempty_masks_without_prompt_overlap"
            )

    aggregates = pd.read_csv(analysis_root / "aggregate_metrics.csv")
    expected_aggregates = 3 * 5 * conditions_per_reference
    if len(aggregates) != expected_aggregates:
        raise AssertionError(
            f"Expected {expected_aggregates} aggregate rows, found {len(aggregates)}"
        )
    for dataset_id, expected_instances in EXPECTED_DATASET_INSTANCES.items():
        test_root = STUDY_ROOT / "data" / "prepared" / dataset_id / "test"
        metadata = pd.read_csv(test_root / "metadata.csv")
        annotations = json.loads(
            (test_root / "_annotations.coco.json").read_text(encoding="utf-8")
        )
        stratum_by_image_id = {
            int(row["image_id"]): str(row["stratum"])
            for _, row in metadata.iterrows()
        }
        expected_by_stratum = {
            stratum: 0
            for stratum in (
                "no_overlap__low_mask_area",
                "no_overlap__high_mask_area",
                "overlap__low_mask_area",
                "overlap__high_mask_area",
            )
        }
        for annotation in annotations["annotations"]:
            expected_by_stratum[
                stratum_by_image_id[int(annotation["image_id"])]
            ] += 1
        expected_by_stratum["overall"] = expected_instances
        for stratum, expected_count in expected_by_stratum.items():
            frame = aggregates[
                (aggregates["dataset_id"] == dataset_id)
                & (aggregates["stratum"] == stratum)
            ]
            actual_counts = set(frame["instance_count"].astype(int))
            if actual_counts != {expected_count}:
                raise AssertionError(
                    f"{dataset_id}/{stratum}: expected aggregate instance "
                    f"count {expected_count}, found {actual_counts}"
                )

    completed_manifest_count(
        "detectors/*/seed_*/manifest.json", expected_detector_runs
    )
    completed_manifest_count(
        "detectors/*/seed_*/evaluation/test/manifest.json",
        expected_detector_runs,
    )
    completed_manifest_count("predictions/*/*/gt_bbox/manifest.json", 6)
    completed_manifest_count(
        "predictions/*/*/yolo_bbox/seed_*/manifest.json",
        expected_unmatched_files,
    )
    completed_manifest_count(
        "evaluation/**/manifest.json", 6 * (1 + len(EXPECTED_SEEDS))
    )


def main() -> None:
    validate_prepared_test_sets()
    validate_analysis_outputs()
    for report_id in REPORTS:
        report_root = REPORT_ROOT / report_id
        validate_segmentation_tables(report_root)
        validate_detector_table(report_root)
        validate_rendered_documents(report_root)
        validate_manifest(report_root)
        print(f"PASS {report_id}")


if __name__ == "__main__":
    main()
