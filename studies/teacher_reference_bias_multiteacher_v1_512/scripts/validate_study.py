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
    reference_path,
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

    overall = aggregate[aggregate["stratum"] == "overall"]
    for dataset_id in DATASETS:
        for model in MODELS:
            identity = overall[
                (overall["dataset_id"] == dataset_id)
                & (overall["model"] == model)
                & (overall["bbox_source"] == "gt_bbox")
                & (overall["reference_type"] == f"pseudo_{model}")
            ].iloc[0]
            if float(identity["mean_iou"]) != 1.0:
                raise AssertionError(f"identity control failed: {dataset_id}/{model}")
            yolo = overall[
                (overall["dataset_id"] == dataset_id)
                & (overall["bbox_source"] == "yolo_bbox")
                & (overall["reference_type"] == f"pseudo_{model}")
            ].sort_values("mean_iou", ascending=False)
            if str(yolo.iloc[0]["model"]) != model:
                raise AssertionError(f"own teacher is not top: {dataset_id}/{model}")
    if not (advantages["teacher_advantage"] > 0).all():
        raise AssertionError("teacher advantage must be positive in all controls")
    own_yolo = effects[
        (effects["bbox_source"] == "yolo_bbox")
        & effects.apply(
            lambda row: row["pseudo_reference"] == f"pseudo_{row['model']}", axis=1
        )
    ]
    if not (own_yolo["delta_ci_lower"] > 0).all():
        raise AssertionError("YOLO own-reference 95% CI crosses zero")


def validate_manifest(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "files" in payload:
        root = path.parent.parent
        for row in payload["files"]:
            output = root / row["path"]
            if output.stat().st_size != int(row["bytes"]):
                raise AssertionError(f"{output}: byte-size mismatch")
            if sha256_file(output) != row["sha256"]:
                raise AssertionError(f"{output}: hash mismatch")
        return
    for group in ("inputs", "outputs"):
        for name, expected in payload[group].items():
            relative = Path(name)
            if relative.is_absolute():
                raise AssertionError(f"{path}: absolute path is not portable: {name}")
            output = REPO_ROOT / relative
            if not output.is_file() or sha256_file(output) != expected:
                raise AssertionError(f"{path}: hash mismatch for {output}")


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
    markdown = root / "sam_teacher_pseudo_reference_comparison.md"
    docx = root / "sam_teacher_pseudo_reference_comparison_colored.docx"
    pdf = root / "sam_teacher_pseudo_reference_comparison_colored.pdf"
    text = markdown.read_text(encoding="utf-8")
    for required in (
        "SAM1/SAM2/SAM3",
        "Referanslar Arası Anlaşma",
        "Boş Maske Denetimi",
        "5.345/12.051",
        "eğitimde yararsız olduğunu kanıtlamaz",
    ):
        if required not in text:
            raise AssertionError(f"{markdown}: missing {required}")
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


def main() -> None:
    validate_reference_files()
    validate_analysis()
    validate_manifest(STUDY_ROOT / "results" / "analysis" / "manifest.json")
    validate_manifest(STUDY_ROOT / "results" / "figures" / "manifest.json")
    for path in (STUDY_ROOT / "reports" / "full_metrics").glob("*/report_manifest.json"):
        validate_manifest(path)
    validate_full_metric_reports()
    validate_comparison_report()
    validate_paper_assets()
    print("PASS: references, analysis, reports, rendered PDFs, paper assets, and Overleaf sources")


if __name__ == "__main__":
    main()
