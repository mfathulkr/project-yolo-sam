#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from PIL import Image
from pycocotools import mask as mask_utils


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(STUDY_ROOT / "src"))

from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    BBOX_SOURCES,
    DATASETS,
    MODELS,
    REFERENCE_TYPES,
    STRATA,
    evaluation_path,
    reference_path,
)
from teacher_reference_bias_multiteacher.io import (  # noqa: E402
    validate_hash_manifest,
)


REPORTS = (
    ("isaid_plane_pseudo_sam2", "pseudo_sam2"),
    ("isaid_plane_pseudo_sam3", "pseudo_sam3"),
    ("isaid_small_vehicle_pseudo_sam2", "pseudo_sam2"),
    ("isaid_small_vehicle_pseudo_sam3", "pseudo_sam3"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rle_signature(rle: dict[str, object]) -> tuple[tuple[int, int], str]:
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    size = tuple(int(value) for value in rle["size"])
    if len(size) != 2:
        raise AssertionError(f"Invalid RLE size: {size}")
    return (size[0], size[1]), str(counts)


def validate_reference_files() -> None:
    for dataset_id, source in DATASETS.items():
        for teacher in ("sam2", "sam3"):
            path = reference_path(dataset_id, teacher)
            frame = pd.read_json(path, lines=True, dtype=False)
            if len(frame) != source.teacher_instance_count:
                raise AssertionError(f"{path}: wrong row count")
            if frame["instance_id"].nunique() != len(frame):
                raise AssertionError(f"{path}: duplicate instance id")
            if set(frame["reference_type"]) != {f"pseudo_{teacher}"}:
                raise AssertionError(f"{path}: wrong reference type")
            if set(frame["teacher_model"]) != {teacher}:
                raise AssertionError(f"{path}: wrong teacher model")
            if set(frame["teacher_prompt_type"]) != {"gt_bbox"}:
                raise AssertionError(f"{path}: wrong prompt type")
            if not set(frame["teacher_prediction_status"]).issubset(
                {"ok", "empty_mask"}
            ):
                raise AssertionError(f"{path}: unexpected status")

            decoded_areas = []
            for row in frame.to_dict(orient="records"):
                rle = dict(row["mask_rle"])
                if isinstance(rle.get("counts"), str):
                    rle["counts"] = rle["counts"].encode("ascii")
                decoded_areas.append(int(mask_utils.area(rle)))
            decoded_areas = np.asarray(decoded_areas, dtype=np.int64)
            declared_areas = frame["reference_mask_pixels"].astype(int).to_numpy()
            declared_empty = frame["reference_is_empty"].astype(bool).to_numpy()
            if not np.array_equal(decoded_areas, declared_areas):
                raise AssertionError(f"{path}: decoded and declared mask areas differ")
            if not np.array_equal(decoded_areas == 0, declared_empty):
                raise AssertionError(f"{path}: empty flag does not match RLE area")
            expected_status = np.where(decoded_areas == 0, "empty_mask", "ok")
            if not np.array_equal(
                frame["teacher_prediction_status"].astype(str).to_numpy(),
                expected_status,
            ):
                raise AssertionError(f"{path}: teacher status does not match RLE area")

            reference_manifest_path = path.with_suffix(".manifest.json")
            manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                raise AssertionError(f"{reference_manifest_path}: incomplete")
            if manifest.get("output_sha256") != sha256_file(path):
                raise AssertionError(f"{reference_manifest_path}: output hash mismatch")
            if manifest.get("instance_count") != len(frame):
                raise AssertionError(f"{reference_manifest_path}: wrong instance count")
            if manifest.get("empty_reference_count") != int((decoded_areas == 0).sum()):
                raise AssertionError(f"{reference_manifest_path}: wrong empty count")
            if manifest.get("known_positive_empty_reference_policy") != "score_zero":
                raise AssertionError(f"{reference_manifest_path}: unsafe empty policy")

            source_predictions = REPO_ROOT / manifest["source_predictions"]
            source_manifest_path = REPO_ROOT / manifest["source_prediction_manifest"]
            if sha256_file(source_predictions) != manifest["source_predictions_sha256"]:
                raise AssertionError(f"{reference_manifest_path}: source hash mismatch")
            if sha256_file(source_manifest_path) != manifest["source_prediction_manifest_sha256"]:
                raise AssertionError(f"{reference_manifest_path}: source manifest hash mismatch")
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            if source_manifest.get("status") != "completed":
                raise AssertionError(f"{source_manifest_path}: incomplete source")
            if source_manifest.get("stage") != "gt_bbox_segmentation":
                raise AssertionError(f"{source_manifest_path}: wrong source stage")
            if source_manifest.get("config_hash") != manifest.get("source_prediction_config_hash"):
                raise AssertionError(f"{reference_manifest_path}: config lineage mismatch")
            if source_manifest.get("run_id") != manifest.get("source_prediction_run_id"):
                raise AssertionError(f"{reference_manifest_path}: run lineage mismatch")
            source_parameters = source_manifest.get("parameters", {})
            if source_parameters.get("model") != teacher:
                raise AssertionError(f"{source_manifest_path}: wrong source teacher")
            if source_parameters.get("prompt_type") != "gt_bbox":
                raise AssertionError(f"{source_manifest_path}: wrong source prompt")

            source_frame = pd.read_json(source_predictions, lines=True, dtype=False)
            source_by_instance = {
                str(row["instance_id"]): row
                for row in source_frame.to_dict(orient="records")
            }
            if len(source_by_instance) != len(source_frame):
                raise AssertionError(f"{source_predictions}: duplicate instance id")
            if set(source_by_instance) != set(frame["instance_id"].astype(str)):
                raise AssertionError(
                    f"{reference_manifest_path}: source/reference instance sets differ"
                )
            identity_mismatches = []
            for row in frame.to_dict(orient="records"):
                source_row = source_by_instance[str(row["instance_id"])]
                if (
                    rle_signature(dict(row["mask_rle"]))
                    != rle_signature(dict(source_row["predicted_mask_rle"]))
                    or str(row["teacher_prediction_status"])
                    != str(source_row["status"])
                ):
                    identity_mismatches.append(str(row["instance_id"]))
            if identity_mismatches:
                raise AssertionError(
                    f"{reference_manifest_path}: {len(identity_mismatches)} pseudo masks "
                    f"are not identical to their declared source; examples="
                    f"{identity_mismatches[:5]}"
                )
            if teacher == "sam3":
                source_model_config = source_parameters.get("model_config", {})
                if source_model_config.get("inference_interface") != "sam3_tracker_pvs":
                    raise AssertionError(f"{source_manifest_path}: SAM3 source is not PVS")
                if float(source_model_config.get("mask_threshold", 1.0)) != 0.0:
                    raise AssertionError(f"{source_manifest_path}: unsafe SAM3 threshold")
                if int(source_model_config.get("box_batch_size", 0)) <= 0:
                    raise AssertionError(f"{source_manifest_path}: missing SAM3 batch size")


def validate_analysis() -> None:
    root = STUDY_ROOT / "results" / "analysis"
    aggregate = pd.read_csv(root / "aggregate_metrics.csv")
    effects = pd.read_csv(root / "paired_reference_effects.csv")
    advantages = pd.read_csv(root / "teacher_advantage.csv")
    ranking = pd.read_csv(root / "ranking_by_reference.csv")
    agreement = pd.read_csv(root / "reference_agreement.csv")
    metrics = pd.read_csv(root / "canonical_instance_metrics.csv")
    expected_aggregate = (
        len(DATASETS)
        * len(MODELS)
        * len(BBOX_SOURCES)
        * len(REFERENCE_TYPES)
        * len(STRATA)
    )
    if len(aggregate) != expected_aggregate:
        raise AssertionError(f"aggregate rows: {len(aggregate)} != {expected_aggregate}")
    if len(effects) != 36 or len(advantages) != 12 or len(ranking) != 16 or len(agreement) != 12:
        raise AssertionError("analysis summary shape mismatch")
    expected_metric_rows = sum(
        source.teacher_instance_count for source in DATASETS.values()
    ) * len(MODELS) * len(BBOX_SOURCES) * len(REFERENCE_TYPES)
    if len(metrics) != expected_metric_rows:
        raise AssertionError(f"metric cube rows: {len(metrics)} != {expected_metric_rows}")
    key_columns = [
        "dataset_id",
        "instance_id",
        "model",
        "bbox_source",
        "reference_type",
    ]
    if metrics.duplicated(key_columns).any():
        raise AssertionError("duplicate rows in canonical metric cube")
    metric_columns = ["iou", "dice", "precision", "recall"]
    if not np.isfinite(metrics[metric_columns].to_numpy(float)).all():
        raise AssertionError("non-finite segmentation metric")
    if ((metrics[metric_columns] < 0) | (metrics[metric_columns] > 1)).any().any():
        raise AssertionError("segmentation metric outside [0, 1]")

    for dataset_id in DATASETS:
        for model in MODELS:
            identity = metrics[
                (metrics["dataset_id"] == dataset_id)
                & (metrics["model"] == model)
                & (metrics["bbox_source"] == "gt_bbox")
                & (metrics["reference_type"] == f"pseudo_{model}")
            ]
            if identity.empty:
                raise AssertionError(f"missing identity control: {dataset_id}/{model}")
            reference_pixels = (
                identity["true_positive_pixels"].astype(int)
                + identity["false_negative_pixels"].astype(int)
            )
            expected = np.where(reference_pixels > 0, 1.0, 0.0)
            for metric in metric_columns:
                if not np.allclose(identity[metric].astype(float), expected):
                    raise AssertionError(
                        "coverage-aware identity control failed: "
                        f"{dataset_id}/{model}/{metric}"
                    )

    # Ranking and effect direction are measured outcomes. Validation checks
    # numerical integrity without forcing the study hypothesis to be true.
    if not np.isfinite(advantages["teacher_advantage"].to_numpy(float)).all():
        raise AssertionError("non-finite teacher advantage")
    effect_columns = [
        "human_mean_iou",
        "pseudo_mean_iou",
        "delta_iou",
        "delta_ci_lower",
        "delta_ci_upper",
    ]
    if not np.isfinite(effects[effect_columns].to_numpy(float)).all():
        raise AssertionError("non-finite paired reference effect")


def validate_evaluation_manifests() -> None:
    validated = 0
    metric_columns = ["iou", "dice", "precision", "recall"]
    for dataset_id, source in DATASETS.items():
        for model in MODELS:
            for bbox_source in BBOX_SOURCES:
                metrics_path = evaluation_path(dataset_id, model, bbox_source)
                manifest_path = metrics_path.with_name("manifest.json")
                manifest = validate_hash_manifest(
                    manifest_path,
                    repository_root=REPO_ROOT,
                )
                if int(manifest.get("schema_version", 0)) < 3:
                    raise AssertionError(f"{manifest_path}: outdated schema")
                expected = {
                    "dataset_id": dataset_id,
                    "model": model,
                    "bbox_source": bbox_source,
                    "metric_schema_version": 2,
                    "primary_granularity": "instance",
                    "instance_weighting": "equal",
                    "known_positive_empty_reference_policy": "score_zero",
                }
                for key, value in expected.items():
                    if manifest.get(key) != value:
                        raise AssertionError(
                            f"{manifest_path}: {key}={manifest.get(key)!r} != {value!r}"
                        )
                expected_seed = 42 if bbox_source == "yolo_bbox" else None
                if manifest.get("detector_seed") != expected_seed:
                    raise AssertionError(f"{manifest_path}: wrong detector seed")
                if manifest.get("output") != metrics_path.relative_to(REPO_ROOT).as_posix():
                    raise AssertionError(f"{manifest_path}: wrong output path")

                frame = pd.read_csv(metrics_path)
                expected_rows = source.teacher_instance_count * 2
                if len(frame) != expected_rows:
                    raise AssertionError(f"{metrics_path}: wrong metric row count")
                if set(frame["reference_type"]) != {"pseudo_sam2", "pseudo_sam3"}:
                    raise AssertionError(f"{metrics_path}: wrong reference types")
                if frame.duplicated(["instance_id", "reference_type"]).any():
                    raise AssertionError(f"{metrics_path}: duplicate metric keys")
                empty = frame["reference_is_empty"].astype(bool)
                if empty.any() and not np.allclose(
                    frame.loc[empty, metric_columns].astype(float),
                    0.0,
                ):
                    raise AssertionError(
                        f"{metrics_path}: empty known-positive reference did not score zero"
                    )
                validated += 1
    expected_count = len(DATASETS) * len(MODELS) * len(BBOX_SOURCES)
    if validated != expected_count:
        raise AssertionError(
            f"evaluation manifest count: {validated} != {expected_count}"
        )


def validate_manifest(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "files" in payload:
        output_root = path.parent.parent
        for row in payload["files"]:
            output = output_root / row["path"]
            if output.stat().st_size != int(row["bytes"]):
                raise AssertionError(f"{output}: byte-size mismatch")
            if sha256_file(output) != row["sha256"]:
                raise AssertionError(f"{output}: hash mismatch")
        input_root = (
            REPO_ROOT
            if payload.get("input_path_root") == "repository"
            else path.parents[2]
        )
        for row in payload.get("inputs", []):
            source = input_root / row["path"]
            if source.stat().st_size != int(row["bytes"]):
                raise AssertionError(f"{source}: input byte-size mismatch")
            if sha256_file(source) != row["sha256"]:
                raise AssertionError(f"{source}: input hash mismatch")
        return
    validate_hash_manifest(path, repository_root=REPO_ROOT)


def pdf_page_count(path: Path) -> int:
    output = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)$", output, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f"{path}: missing PDF page count")
    return int(match.group(1))


def validate_pdf_render(path: Path, minimum_pages: int) -> None:
    pages = pdf_page_count(path)
    if pages < minimum_pages:
        raise AssertionError(f"{path}: only {pages} pages")
    with tempfile.TemporaryDirectory(prefix="multiteacher_pdf_") as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "45", str(path), str(prefix)],
            check=True,
            capture_output=True,
        )
        rendered = sorted(Path(tmp).glob("page-*.png"))
        if len(rendered) != pages:
            raise AssertionError(f"{path}: rendered page count mismatch")
        for rendered_page in rendered:
            gray = np.asarray(Image.open(rendered_page).convert("L"))
            nonwhite_fraction = float((gray < 248).mean())
            if nonwhite_fraction < 0.002:
                raise AssertionError(f"{path}: blank-looking page {rendered_page.name}")


def validate_full_metric_reports() -> None:
    root = STUDY_ROOT / "reports" / "full_metrics"
    expected_columns = (
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
    for stem, reference_type in REPORTS:
        report_root = root / stem
        validate_manifest(report_root / "report_manifest.json")
        table_paths = sorted(
            path
            for path in (report_root / "tables").glob("*.csv")
            if path.name not in {"detector_summary.csv", "reference_sensitivity.csv"}
        )
        if len(table_paths) != 5:
            raise AssertionError(f"{stem}: expected five segmentation tables")
        for path in table_paths:
            frame = pd.read_csv(path)
            if tuple(frame.columns) != expected_columns or len(frame) != 6:
                raise AssertionError(f"{path}: wrong table schema")
            expected_images = 512 if path.stem.endswith("__overall") else 128
            if set(frame["Images"].astype(int)) != {expected_images}:
                raise AssertionError(f"{path}: wrong image count")
            if not path.stem.startswith(reference_type + "__"):
                raise AssertionError(f"{path}: wrong reference section")

        markdown = report_root / f"{stem}_full_metric_document.md"
        docx = report_root / f"{stem}_full_metric_document_colored.docx"
        pdf = report_root / f"{stem}_full_metric_document_colored.pdf"
        text = markdown.read_text(encoding="utf-8")
        for required in ("Overall", "No Overlap", "Overlap", "Avg IoU", "BBox mAP50", "512", "128"):
            if required not in text:
                raise AssertionError(f"{markdown}: missing {required}")
        for forbidden in ("mAP proxy", "Boundary IoU", "RemoteSAM", "RingMo", "32 görüntü"):
            if forbidden in text:
                raise AssertionError(f"{markdown}: forbidden legacy content {forbidden}")
        with zipfile.ZipFile(docx) as archive:
            if archive.testzip() is not None:
                raise AssertionError(f"{docx}: corrupt DOCX")
            media = [name for name in archive.namelist() if name.startswith("word/media/")]
            if len(media) != 4:
                raise AssertionError(f"{docx}: expected four qualitative images")
        if len(Document(docx).tables) != 7:
            raise AssertionError(f"{docx}: expected seven tables")
        pdf_text = subprocess.run(
            ["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True
        ).stdout
        for required in ("Overall", "Avg IoU", "BBox mAP50", "SAM1 GT bbox", "SAM3 YOLO bbox"):
            if required not in pdf_text:
                raise AssertionError(f"{pdf}: missing text {required}")
        validate_pdf_render(pdf, minimum_pages=12)


def validate_comparison_report() -> None:
    root = STUDY_ROOT / "reports" / "teacher_comparison"
    validate_manifest(root / "manifest.json")
    markdown = root / "sam_teacher_pseudo_reference_comparison.md"
    docx = root / "sam_teacher_pseudo_reference_comparison_colored.docx"
    pdf = root / "sam_teacher_pseudo_reference_comparison_colored.pdf"
    text = markdown.read_text(encoding="utf-8")
    for required in (
        "SAM1/SAM2/SAM3",
        "Referanslar Arası Anlaşma",
        "Boş Maske Denetimi",
        "Bilinen insan GT instance",
        "eğitimde yararsız olduğunu kanıtlamaz",
    ):
        if required not in text:
            raise AssertionError(f"{markdown}: missing {required}")
    for forbidden in ("5.345/12.051", "%44,4", "44.4%"):
        if forbidden in text:
            raise AssertionError(f"{markdown}: stale SAM3 result {forbidden}")
    with zipfile.ZipFile(docx) as archive:
        if archive.testzip() is not None:
            raise AssertionError(f"{docx}: corrupt DOCX")
    validate_pdf_render(pdf, minimum_pages=8)


def validate_paper_assets() -> None:
    manifest = STUDY_ROOT / "paper" / "assets" / "manifest.json"
    validate_manifest(manifest)
    figure_root = STUDY_ROOT / "paper" / "assets" / "figures"
    for path in figure_root.glob("*.png"):
        image = np.asarray(Image.open(path).convert("RGB"))
        if image.shape[0] < 500 or image.shape[1] < 700:
            raise AssertionError(f"{path}: figure resolution too small")
        if float(image.std()) < 5:
            raise AssertionError(f"{path}: visually blank figure")
    table_root = STUDY_ROOT / "paper" / "assets" / "tables"
    if len(list(table_root.glob("*.csv"))) != 7 or len(list(table_root.glob("*.tex"))) != 7:
        raise AssertionError("paper table package is incomplete")

    main_tex = STUDY_ROOT / "paper" / "overleaf" / "main.tex"
    bib = STUDY_ROOT / "paper" / "overleaf" / "ref.bib"
    tex_text = main_tex.read_text(encoding="utf-8")
    bib_text = bib.read_text(encoding="utf-8")
    for required in (
        r"\documentclass{elektr}",
        r"\input{elksty.tex}",
        r"\section*{Introduction}",
        r"\section*{Literature Review}",
        r"\section*{Materials and Methods}",
        r"\section*{Results}",
        r"\section*{Discussion}",
        r"\section*{Conclusion}",
        r"\bibliographystyle{elsarticle-num}",
    ):
        if required not in tex_text:
            raise AssertionError(f"{main_tex}: missing {required}")
    if tex_text.count("{") != tex_text.count("}"):
        raise AssertionError(f"{main_tex}: unbalanced braces")
    citation_text = "\n".join(
        re.sub(r"^\s*%\s?", "", line) for line in tex_text.splitlines()
    )
    citation_keys = set()
    for block in re.findall(r"\\cite\{([^}]+)\}", citation_text):
        citation_keys.update(key.strip() for key in block.split(","))
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    missing = citation_keys - bib_keys
    if missing:
        raise AssertionError(f"missing BibTeX keys: {sorted(missing)}")
    if len(bib_keys) != len(re.findall(r"@\w+\{", bib_text)):
        raise AssertionError("duplicate BibTeX keys")


def validate_reproducibility_documentation() -> None:
    guide = STUDY_ROOT / "docs" / "REPRODUCIBILITY_FIELD_GUIDE.md"
    method = STUDY_ROOT / "docs" / "METHOD.md"
    paper_plan = STUDY_ROOT / "docs" / "PAPER_STRUCTURE.md"
    asset_plan = STUDY_ROOT / "docs" / "PAPER_ASSET_PLAN.md"
    main_tex = STUDY_ROOT / "paper" / "overleaf" / "main.tex"
    handoff = REPO_ROOT / "docs" / "summary" / "TEACHER_REFERENCE_BIAS_HANDOFF.md"

    def normalized_text(path: Path) -> str:
        return " ".join(path.read_text(encoding="utf-8").split())

    guide_text = normalized_text(guide)
    required_guide_terms = (
        "Promptable Visual Segmentation",
        "Promptable Concept Segmentation",
        "Sam3TrackerModel",
        "multimask_output=False",
        "mask_threshold=0,0",
        "0,28115004301071167",
        "0,2740148901939392",
        "Known-positive",
        "instance-macro",
        "source scene",
        "SHA-256",
        "Kırmızı Bayraklar",
    )
    for required in required_guide_terms:
        if required not in guide_text:
            raise AssertionError(f"{guide}: missing reproducibility term {required}")

    method_text = normalized_text(method)
    for required in (
        "REPRODUCIBILITY_FIELD_GUIDE.md",
        "Promptable Visual Segmentation",
        "Promptable Concept Segmentation",
        "multimask_output=False",
    ):
        if required not in method_text:
            raise AssertionError(f"{method}: missing reproducibility term {required}")

    paper_plan_text = normalized_text(paper_plan)
    asset_plan_text = normalized_text(asset_plan)
    tex_text = normalized_text(main_tex)
    for path, text, required in (
        (paper_plan, paper_plan_text, "REPRODUCIBILITY_FIELD_GUIDE.md"),
        (asset_plan, asset_plan_text, "Table S2 - Reproducibility contract"),
        (main_tex, tex_text, "Promptable Visual Segmentation"),
        (main_tex, tex_text, "known-positive"),
    ):
        if required not in text:
            raise AssertionError(f"{path}: missing reproducibility term {required}")

    handoff_text = normalized_text(handoff)
    for required in (
        "Promptable Visual Segmentation",
        "Promptable Concept Segmentation",
        "0,28115004301071167",
        "0,2740148901939392",
        "REPRODUCIBILITY_FIELD_GUIDE.md",
        "PAPER_STRUCTURE.md",
        "234 passed, 18 warnings, 2 subtests passed",
    ):
        if required not in handoff_text:
            raise AssertionError(f"{handoff}: missing handoff term {required}")


def main() -> None:
    validate_reference_files()
    validate_evaluation_manifests()
    validate_analysis()
    validate_manifest(STUDY_ROOT / "results" / "analysis" / "manifest.json")
    validate_manifest(STUDY_ROOT / "results" / "figures" / "manifest.json")
    for path in (STUDY_ROOT / "reports" / "full_metrics").glob("*/report_manifest.json"):
        validate_manifest(path)
    validate_full_metric_reports()
    validate_comparison_report()
    validate_paper_assets()
    validate_reproducibility_documentation()
    print(
        "PASS: references, analysis, reports, rendered PDFs, paper assets, "
        "Overleaf sources, and reproducibility documentation"
    )


if __name__ == "__main__":
    main()
