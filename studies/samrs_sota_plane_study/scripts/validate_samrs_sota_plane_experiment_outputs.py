from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.config import load_config, resolve_path


PIPELINES = {
    "sam3_text": "sam3_text_output_dir",
    "yolo_sam3": "yolo_sam3_output_dir",
    "gt_box_sam3": "gt_box_sam3_output_dir",
    "sam3_hybrid_yolo": "sam3_hybrid_yolo_output_dir",
    "remotesam_text": "remotesam_text_output_dir",
    "yolo_ringmo_sam": "yolo_ringmo_sam_output_dir",
    "gt_box_ringmo_sam": "gt_box_ringmo_sam_output_dir",
    "yolo_sam1": "sam1_yolo_output_dir",
    "gt_box_sam1": "sam1_gt_box_output_dir",
    "yolo_sam2": "yolo_sam2_output_dir",
    "gt_box_sam2": "gt_box_sam2_output_dir",
}

STRATA = [
    "no_overlap__low_mask_area",
    "no_overlap__high_mask_area",
    "overlap__low_mask_area",
    "overlap__high_mask_area",
]

REPORT_FILES = [
    "samrs_sota_plane_full_metric_document.md",
    "samrs_sota_plane_full_metric_document_colored.docx",
    "samrs_sota_plane_full_metric_document_colored.pdf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final SAMRS SOTA plane experiment artifacts.")
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "configs" / "yolo26x.yaml")
    parser.add_argument("--expected-per-stratum", type=int, default=128)
    parser.add_argument("--allow-smaller-strata", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=STUDY_ROOT / "reports")
    return parser.parse_args()


def fail(message: str) -> None:
    raise AssertionError(message)


def expected_pipeline_count(config: dict) -> int:
    paths = config.get("paths", {})
    return sum(1 for output_key in PIPELINES.values() if output_key in paths)


def validate_dataset(config: dict, expected_per_stratum: int, allow_smaller_strata: bool) -> list[str]:
    notes: list[str] = []
    prepared_eval = resolve_path(config["paths"]["prepared_dataset_dir"]) / config["dataset"]["eval_split"]
    if not prepared_eval.exists():
        fail(f"Missing prepared eval split: {prepared_eval}")
    metadata_path = prepared_eval / "metadata.csv"
    if not metadata_path.exists():
        fail(f"Missing eval metadata: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if metadata.empty:
        fail("Eval metadata is empty")
    notes.append(f"Eval images: {len(metadata)}")

    stratum_counts = metadata["stratum"].value_counts().sort_index()
    missing = [stratum for stratum in STRATA if int(stratum_counts.get(stratum, 0)) == 0]
    if missing:
        fail(f"Missing eval strata: {missing}; available counts: {stratum_counts.to_dict()}")

    for stratum in STRATA:
        count = int(stratum_counts.get(stratum, 0))
        if allow_smaller_strata:
            if count > expected_per_stratum:
                fail(f"{stratum}: expected at most {expected_per_stratum}, found {count}")
        elif count != expected_per_stratum:
            fail(f"{stratum}: expected {expected_per_stratum}, found {count}")
    notes.append(f"Strata counts: {stratum_counts.to_dict()}")

    if not (prepared_eval / "_annotations.coco.json").exists():
        fail(f"Missing eval COCO annotations: {prepared_eval / '_annotations.coco.json'}")
    return notes


def validate_pipeline_outputs(config: dict) -> list[str]:
    notes: list[str] = []
    prepared_eval = resolve_path(config["paths"]["prepared_dataset_dir"]) / config["dataset"]["eval_split"]
    metadata = pd.read_csv(prepared_eval / "metadata.csv")
    expected_images = len(metadata)
    expected_ringmo_class_ids = [int(value) for value in config.get("ringmo_sam", {}).get("class_ids", [])]

    for pipeline, output_key in PIPELINES.items():
        if output_key not in config.get("paths", {}):
            continue
        output_dir = resolve_path(config["paths"][output_key])
        mask_dir = output_dir / "masks"
        raw_dir = output_dir / "raw"
        if not mask_dir.exists():
            fail(f"{pipeline}: missing masks dir {mask_dir}")
        mask_count = len(list(mask_dir.glob("*.png")))
        if mask_count != expected_images:
            fail(f"{pipeline}: expected {expected_images} masks, found {mask_count}")
        if raw_dir.exists():
            raw_paths = list(raw_dir.glob("*.json"))
            raw_count = len(raw_paths)
            if raw_count != expected_images:
                fail(f"{pipeline}: expected {expected_images} raw JSON files, found {raw_count}")
            if "ringmo" in pipeline and expected_ringmo_class_ids:
                for raw_path in raw_paths:
                    payload = json.loads(raw_path.read_text(encoding="utf-8"))
                    class_ids = [int(value) for value in payload.get("class_ids", [])]
                    if class_ids != expected_ringmo_class_ids:
                        fail(
                            f"{pipeline}: {raw_path.name} class_ids {class_ids} "
                            f"!= config {expected_ringmo_class_ids}"
                        )
            notes.append(f"{pipeline}: {mask_count} masks and {raw_count} raw JSON files")
        else:
            notes.append(f"{pipeline}: {mask_count} masks")
    return notes


def validate_metrics(config: dict, report_dir: Path) -> list[str]:
    notes: list[str] = []
    metrics_dir = resolve_path(config["paths"]["sam3_triplet_metrics_dir"])
    per_image_path = metrics_dir / "per_image_stratified_metrics.csv"
    if not per_image_path.exists():
        fail(f"Missing per-image metric CSV: {per_image_path}")
    per_image = pd.read_csv(per_image_path)

    prepared_eval = resolve_path(config["paths"]["prepared_dataset_dir"]) / config["dataset"]["eval_split"]
    expected_images = len(pd.read_csv(prepared_eval / "metadata.csv"))
    expected_rows = expected_images * expected_pipeline_count(config)
    if len(per_image) != expected_rows:
        fail(f"Expected {expected_rows} per-image metric rows, found {len(per_image)}")
    notes.append(f"Per-image metrics: {len(per_image)} rows")

    for filename in ["summary_overall_stratified.csv", "summary_by_stratum.csv", "pairwise_iou_by_stratum.csv"]:
        path = metrics_dir / filename
        if not path.exists():
            fail(f"Missing metric summary: {path}")
        notes.append(f"Metric summary present: {path}")

    tables_dir = report_dir / "tables" / "full_metric_document"
    for filename in ["per_image_metrics_selected_pipelines.csv", "summary_all_tables_selected_pipelines.csv"]:
        path = tables_dir / filename
        if not path.exists():
            fail(f"Missing report table CSV: {path}")
        notes.append(f"Report table present: {path}")
    return notes


def validate_detector_and_report(report_dir: Path) -> list[str]:
    notes: list[str] = []
    detector_path = (
        STUDY_ROOT
        / "results"
        / "pipelines"
        / "detector_metrics"
        / "yolo_detector_eval_metrics.csv"
    )
    if not detector_path.exists():
        fail(f"Missing YOLO detector metrics: {detector_path}")
    detector = pd.read_csv(detector_path)
    required_detector_columns = {
        "bbox_mAP50",
        "bbox_mAP75",
        "bbox_mAP90",
        "bbox_mAP50_95",
        "precision_at_iou50",
        "recall_at_iou50",
    }
    missing_columns = sorted(required_detector_columns - set(detector.columns))
    if missing_columns:
        fail(f"Detector metrics missing columns: {missing_columns}")
    notes.append(f"Detector metrics present: {detector_path}")

    for filename in REPORT_FILES:
        path = report_dir / filename
        if not path.exists():
            fail(f"Missing report artifact: {path}")
        if path.stat().st_size == 0:
            fail(f"Empty report artifact: {path}")
        notes.append(f"Report artifact present: {path}")

    md_path = report_dir / "samrs_sota_plane_full_metric_document.md"
    text = md_path.read_text(encoding="utf-8")
    for phrase in ["pseudo-mask", "SAM1", "mAP50 proxy", "YOLO Detector BBox Metrics"]:
        if phrase not in text:
            fail(f"Report markdown missing phrase: {phrase}")
    return notes


def validate_visualizations(config: dict) -> list[str]:
    notes: list[str] = []
    vis_dir = resolve_path(config["paths"]["sam3_triplet_visualizations_dir"])
    selected_path = vis_dir / "selected_qualitative_samples.csv"
    if not selected_path.exists():
        fail(f"Missing selected qualitative CSV: {selected_path}")
    selected = pd.read_csv(selected_path)
    if selected.empty:
        fail(f"Selected qualitative CSV is empty: {selected_path}")

    for _, row in selected.iterrows():
        image_path = vis_dir / f"{row['stratum']}__{Path(row['file_name']).stem}.png"
        if not image_path.exists():
            fail(f"Missing visualization: {image_path}")
        with Image.open(image_path) as image:
            if image.width <= 0 or image.height <= 0:
                fail(f"Invalid visualization size for {image_path}: {image.size}")
    notes.append(f"Qualitative visualizations present: {len(selected)}")
    return notes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(report_dir: Path, notes: list[str]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    qa_path = report_dir / "QA_MANIFEST.md"
    qa_path.write_text("# SAMRS SOTA Plane QA Manifest\n\nAll checks passed.\n\n" + "\n".join(f"- {note}" for note in notes) + "\n", encoding="utf-8")

    artifact_paths: list[Path] = []
    artifact_paths.extend(report_dir.glob("samrs_sota_plane_full_metric_document*"))
    artifact_paths.extend((report_dir / "tables" / "full_metric_document").glob("*.csv"))
    artifact_paths.extend((report_dir / "tables" / "full_metric_document" / "raw_summaries").glob("*.csv"))
    artifact_paths.append(qa_path)

    rows: list[dict[str, object]] = []
    for path in sorted({path for path in artifact_paths if path.exists()}):
        rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(report_dir / "ARTIFACT_MANIFEST.csv", index=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    notes: list[str] = []
    notes.extend(validate_dataset(config, args.expected_per_stratum, args.allow_smaller_strata))
    notes.extend(validate_pipeline_outputs(config))
    notes.extend(validate_metrics(config, args.report_dir))
    notes.extend(validate_detector_and_report(args.report_dir))
    notes.extend(validate_visualizations(config))
    write_manifest(args.report_dir, notes)
    print("\n".join(notes))


if __name__ == "__main__":
    main()
