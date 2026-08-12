from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from pycocotools.coco import COCO


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.analysis import (  # noqa: E402
    validate_metric_cube,
)
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    BBOX_SOURCES,
    DATASETS,
    MODELS,
    REPO_ROOT as PATHS_REPO_ROOT,
    prediction_path,
    reference_path,
)
from teacher_reference_bias.config import load_dataset_study_config  # noqa: E402
from yolo_sam.runtime.manifest import (  # noqa: E402
    validate_completed_run_manifest,
)


EXPECTED_REFERENCE_TYPES = {
    "isaid_plane": ("human", "pseudo_sam1", "pseudo_sam2", "pseudo_sam3"),
    "isaid_small_vehicle": (
        "human",
        "pseudo_sam1",
        "pseudo_sam2",
        "pseudo_sam3",
    ),
    "samrs_plane": (
        "published_samrs_reference",
        "reproduced_pseudo_sam1",
        "pseudo_sam2",
        "pseudo_sam3",
    ),
    "samrs_small_vehicle": (
        "published_samrs_reference",
        "reproduced_pseudo_sam1",
        "pseudo_sam2",
        "pseudo_sam3",
    ),
}

REPORT_STRATA = (
    "overall",
    "no_overlap__low_mask_area",
    "no_overlap__high_mask_area",
    "overlap__low_mask_area",
    "overlap__high_mask_area",
)
REPORT_METRICS = {
    "Avg IoU": "mean_iou",
    "Avg Dice": "mean_dice",
    "Avg Precision": "mean_precision",
    "Avg Recall": "mean_recall",
    "IoU ≥ 0.50": "success_at_iou_50",
    "IoU ≥ 0.75": "success_at_iou_75",
    "IoU ≥ 0.90": "success_at_iou_90",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, str]] = []

    def pass_(self, name: str, detail: str = "") -> None:
        self.checks.append({"name": name, "status": "PASS", "detail": detail})

    def fail(self, name: str, detail: str) -> None:
        self.checks.append({"name": name, "status": "FAIL", "detail": detail})

    def run(self, name: str, function: Callable[[], str | None]) -> None:
        try:
            detail = function() or ""
        except Exception as exc:  # noqa: BLE001 - audit must collect all failures
            self.fail(name, f"{type(exc).__name__}: {exc}")
        else:
            self.pass_(name, detail)

    @property
    def failures(self) -> list[dict[str, str]]:
        return [row for row in self.checks if row["status"] == "FAIL"]


def assert_file(path: Path, minimum_bytes: int = 1) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size < minimum_bytes:
        raise ValueError(f"Dosya beklenenden küçük: {path} ({path.stat().st_size})")


def validate_prepared(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    experiment_root = STUDY_ROOT / "experiments" / experiment_id
    dataset = load_dataset_study_config(experiment_root / "config.yaml")
    master = load_dataset_study_config(experiment_root / "master_config.yaml")
    if dataset.master_prepared_root != master.prepared_root:
        raise ValueError("Matched config ile master config aynı kaynak havuzunu göstermiyor")
    if dataset.master_prepared_root is None:
        raise ValueError("master_prepared_root tanımlı değil")
    master_manifest_path = dataset.master_prepared_root / "content_manifest.json"
    assert_file(master_manifest_path)
    provenance_path = source.prepared_root / "master_provenance.json"
    assert_file(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    recorded_root = Path(str(provenance["master_prepared_root"]))
    if not recorded_root.is_absolute():
        recorded_root = REPO_ROOT / recorded_root
    if recorded_root.resolve() != dataset.master_prepared_root.resolve():
        raise ValueError("master_provenance canonical master kökünü göstermiyor")
    recorded_manifest = Path(str(provenance["master_content_manifest"]))
    if not recorded_manifest.is_absolute():
        recorded_manifest = REPO_ROOT / recorded_manifest
    if recorded_manifest.resolve() != master_manifest_path.resolve():
        raise ValueError(
            "master_provenance canonical master content manifest yolunu göstermiyor"
        )
    expected_master_hash = str(provenance["master_content_manifest_sha256"])
    actual_master_hash = sha256_file(master_manifest_path)
    if expected_master_hash != actual_master_hash:
        raise ValueError(
            "master_provenance content manifest SHA-256 uyuşmuyor: "
            f"{expected_master_hash} != {actual_master_hash}"
        )
    master_manifest = json.loads(master_manifest_path.read_text(encoding="utf-8"))
    if tuple(master_manifest.get("splits", ())) != (
        "train",
        "validation",
        "test_pool",
        "test",
    ):
        raise ValueError("Master content manifest test_pool dahil dört split'i kapsamıyor")
    coco = COCO(str(source.coco_path))
    image_count = len(coco.getImgIds())
    instance_count = len(coco.getAnnIds())
    if image_count != 512 or instance_count != source.instance_count:
        raise ValueError(
            f"images/instances={image_count}/{instance_count}, "
            f"expected=512/{source.instance_count}"
        )
    metadata = pd.read_csv(source.prepared_root / "test" / "metadata.csv")
    counts = metadata["stratum"].value_counts().to_dict()
    if set(counts.values()) != {128} or len(counts) != 4:
        raise ValueError(f"Tabaka dağılımı 4×128 değil: {counts}")
    scenes = int(metadata["source_scene_id"].nunique())
    for required in (
        source.prepared_root / "content_manifest.json",
        source.prepared_root / "detector_training_content_manifest.json",
        source.prepared_root / "data.yaml",
    ):
        assert_file(required)
    data_yaml = (source.prepared_root / "data.yaml").read_text(encoding="utf-8")
    if "path:" in data_yaml:
        raise ValueError("data.yaml mutlak/özel path alanı içeriyor")
    return f"512 görüntü, {instance_count} instance, {scenes} kaynak sahne"


def manifest_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in manifest_strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in manifest_strings(child)]
    return []


def validate_run_manifests(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    manifests: list[Path] = []
    for path in sorted(source.results_root.glob("**/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("input_file_fingerprints_at_finish"), dict):
            continue
        if payload.get("path_base") != "repository_root":
            raise ValueError(f"Run manifest repository-relative değil: {path}")
        strings = manifest_strings(
            {
                "inputs": payload.get("inputs"),
                "outputs": payload.get("outputs"),
                "input_fingerprints": payload.get(
                    "input_file_fingerprints_at_finish"
                ),
                "output_fingerprints": payload.get("output_file_fingerprints"),
            }
        )
        absolute = [value for value in strings if value.startswith("/")]
        if absolute:
            raise ValueError(f"Run manifest mutlak yol içeriyor: {path}: {absolute[:2]}")
        validate_completed_run_manifest(path)
        manifests.append(path)
    if len(manifests) != 9:
        raise ValueError(f"9 yerine {len(manifests)} strict run manifest bulundu")

    legacy_studies = (
        "teacher_reference_bias_v2_512",
        "teacher_reference_bias_small_vehicle_v1_512",
    )
    legacy_tokens = tuple(f"studies/{name}" for name in legacy_studies)
    companion_paths = sorted(
        list(source.results_root.glob("predictions/**/effective_config.input.json"))
        + list(
            source.results_root.glob(
                "predictions/**/segmenter_provenance.input.json"
            )
        )
        + list(source.results_root.glob("detector/*/train/args.yaml"))
    )
    for path in companion_paths:
        content = path.read_text(encoding="utf-8")
        hits = [token for token in legacy_tokens if token in content]
        if hits:
            raise ValueError(f"Taşınmamış companion yol kaydı: {path}: {hits}")
    return f"{len(manifests)} strict ve taşınabilir run manifest"


def validate_predictions(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    for model in MODELS:
        for bbox_source in BBOX_SOURCES:
            path = prediction_path(source, model, bbox_source)
            assert_file(path, 100)
            with path.open(encoding="utf-8") as handle:
                count = sum(1 for line in handle if line.strip())
            if count != source.instance_count:
                raise ValueError(
                    f"{model}/{bbox_source}: {count} != {source.instance_count}"
                )
            manifest = json.loads(
                (path.parent / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("status") != "completed":
                raise ValueError(f"Tamamlanmamış prediction manifest: {path.parent}")
    return f"{len(MODELS) * len(BBOX_SOURCES)} prediction kümesi tam"


def validate_references(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    if tuple(source.reference_types) != EXPECTED_REFERENCE_TYPES[experiment_id]:
        raise ValueError(f"Referans sözleşmesi yanlış: {source.reference_types}")
    for reference_type in source.reference_types:
        path = reference_path(source, reference_type)
        assert_file(path, 100)
        if reference_type not in {"human", "published_samrs_reference"}:
            with path.open(encoding="utf-8") as handle:
                count = sum(1 for line in handle if line.strip())
            if count != source.instance_count:
                raise ValueError(f"{reference_type}: {count} referans satırı")
    empty_stats = pd.read_csv(source.analysis_root / "reference_empty_stats.csv")
    if set(empty_stats["reference_type"]) != set(source.reference_types):
        raise ValueError("Boş maske denetimi bütün referansları kapsamıyor")
    return ", ".join(source.reference_types)


def validate_analysis(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    cube_path = source.analysis_root / "canonical_instance_metrics.csv"
    metrics = pd.read_csv(cube_path)
    metrics["detector_seed"] = metrics["detector_seed"].astype("Int64")
    validate_metric_cube(metrics)
    expected_rows = source.instance_count * 3 * 2 * 4
    if len(metrics) != expected_rows:
        raise ValueError(f"Metric cube {len(metrics)} != {expected_rows}")
    for filename in (
        "aggregate_metrics.csv",
        "paired_reference_effects.csv",
        "ranking_by_reference.csv",
        "teacher_advantage.csv",
        "reference_agreement.csv",
        "reference_empty_stats.csv",
        "detector_summary.csv",
        "manifest.json",
        "metric_cube_manifest.json",
    ):
        assert_file(source.analysis_root / filename)
    return f"{len(metrics)} instance-metric satırı"


def validate_manifest(path: Path, *, verify_hashes: bool = False) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"Manifest tamamlanmamış: {path}")
    records: list[dict[str, object]] = []
    for key in ("inputs", "outputs"):
        value = payload.get(key, [])
        if isinstance(value, dict):
            records.extend(
                {"path": item_path, "sha256": item_hash}
                for item_path, item_hash in value.items()
            )
        elif isinstance(value, list):
            records.extend(value)
    checked = 0
    for record in records:
        item_path = str(record["path"])
        if item_path.startswith("/") or item_path.startswith("home/"):
            raise ValueError(f"Portable olmayan manifest yolu: {item_path}")
        item = PATHS_REPO_ROOT / item_path
        assert_file(item)
        if verify_hashes and record.get("sha256"):
            if sha256_file(item) != str(record["sha256"]):
                raise ValueError(f"Manifest hash uyuşmazlığı: {item}")
        checked += 1
    return f"{checked} portable dependency/output"


def pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"PDF page count okunamadı: {path}")


def validate_full_reports(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    aggregate = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
    detector = pd.read_csv(source.analysis_root / "detector_summary.csv").iloc[0]
    for reference_type in source.reference_types:
        root = source.reports_root / "full_metrics" / reference_type
        slug = f"{source.experiment_id}_{reference_type}"
        markdown = root / f"{slug}_full_metric_document.md"
        docx = root / f"{slug}_full_metric_document_colored.docx"
        pdf = root / f"{slug}_full_metric_document_colored.pdf"
        for path in (markdown, docx, pdf):
            assert_file(path, 1000)
        if pdf_pages(pdf) < 13:
            raise ValueError(f"Full-metric PDF sayfa sayısı düşük: {pdf}")
        with zipfile.ZipFile(docx) as archive:
            if "word/document.xml" not in archive.namelist():
                raise ValueError(f"Geçersiz DOCX: {docx}")
        content = markdown.read_text(encoding="utf-8")
        for token in (
            "Overall",
            "No Overlap × Low Mask Area",
            "No Overlap × High Mask Area",
            "Overlap × Low Mask Area",
            "Overlap × High Mask Area",
            "Avg Precision",
            "Avg Recall",
            "BBox mAP50-95",
            "bütün GT",
        ):
            if token not in content:
                raise ValueError(f"Rapor metni eksik ({token}): {markdown}")
        validate_manifest(root / "report_manifest.json", verify_hashes=True)
        for stratum in REPORT_STRATA:
            table = pd.read_csv(
                root / "tables" / f"{reference_type}__{stratum}.csv"
            )
            if len(table) != len(MODELS) * len(BBOX_SOURCES):
                raise ValueError(f"Rapor tablosu 6 pipeline içermiyor: {root}/{stratum}")
            for _, report_row in table.iterrows():
                model_label, bbox_label = str(report_row["Pipeline"]).split(" ", 1)
                model = model_label.lower()
                bbox_source = "gt_bbox" if bbox_label == "GT bbox" else "yolo_bbox"
                selected = aggregate[
                    (aggregate["reference_type"] == reference_type)
                    & (aggregate["stratum"] == stratum)
                    & (aggregate["model"] == model)
                    & (aggregate["bbox_source"] == bbox_source)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Aggregate eşleşmesi tekil değil: {experiment_id}/"
                        f"{reference_type}/{stratum}/{model}/{bbox_source}"
                    )
                expected = selected.iloc[0]
                expected_images = 512 if stratum == "overall" else 128
                if int(report_row["Images"]) != expected_images:
                    raise ValueError(f"Rapor görüntü sayısı yanlış: {root}/{stratum}")
                for report_column, aggregate_column in REPORT_METRICS.items():
                    if abs(
                        float(report_row[report_column])
                        - round(float(expected[aggregate_column]), 3)
                    ) > 1e-9:
                        raise ValueError(
                            f"Rapor metriği aggregate ile uyuşmuyor: {root}/"
                            f"{stratum}/{report_row['Pipeline']}/{report_column}"
                        )
        detector_table = pd.read_csv(root / "tables" / "detector_summary.csv").iloc[0]
        detector_columns = {
            "BBox mAP50": "bbox_AP50_mean",
            "BBox mAP75": "bbox_AP75_mean",
            "BBox mAP90": "bbox_AP90_mean",
            "BBox mAP50-95": "bbox_AP50_95_mean",
            "BBox Precision@0.50": "precision_at_bbox_iou50_mean",
            "BBox Recall@0.50": "recall_at_bbox_iou50_mean",
            "BBox Precision@0.75": "precision_at_bbox_iou75_mean",
            "BBox Recall@0.75": "recall_at_bbox_iou75_mean",
            "BBox Precision@0.90": "precision_at_bbox_iou90_mean",
            "BBox Recall@0.90": "recall_at_bbox_iou90_mean",
        }
        for report_column, analysis_column in detector_columns.items():
            if abs(
                float(detector_table[report_column])
                - round(float(detector[analysis_column]), 3)
            ) > 1e-9:
                raise ValueError(
                    f"Detector rapor metriği yanlış: {root}/{report_column}"
                )
    return f"{len(source.reference_types)} full-metric MD/DOCX/PDF"


def validate_cross_report(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    root = source.reports_root / "cross_analysis"
    slug = f"{source.experiment_id}_cross_reference_analysis"
    paths = (
        root / f"{slug}.md",
        root / f"{slug}_colored.docx",
        root / f"{slug}_colored.pdf",
    )
    for path in paths:
        assert_file(path, 1000)
    if pdf_pages(paths[2]) < 15:
        raise ValueError("Cross-analysis PDF beklenen kapsamdan kısa")
    content = paths[0].read_text(encoding="utf-8")
    if content.count("Avg IoU") < 10:
        raise ValueError("Cross-analysis 5×2 matris kapsamı eksik")
    validate_manifest(root / "report_manifest.json", verify_hashes=True)
    return f"{pdf_pages(paths[2])} sayfa"


def validate_figures(experiment_id: str) -> str:
    source = DATASETS[experiment_id]
    expected = [
        source.figures_root / f"{reference}_gt_bbox_qualitative.png"
        for reference in source.reference_types
    ] + [
        source.figures_root / "model_reference_iou_matrix.png",
        source.figures_root / "reference_effect_with_ci.png",
    ]
    for path in expected:
        assert_file(path, 50_000)
    manifest = json.loads(
        (source.figures_root / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("qualitative_scope") != "all_target_instances_in_selected_images":
        raise ValueError("Nitel figür bütün hedef instance kapsamını ilan etmiyor")
    validate_manifest(source.figures_root / "manifest.json", verify_hashes=True)
    return "4 nitel + 2 analiz figürü"


def validate_paper_outputs() -> str:
    main_root = STUDY_ROOT / "analysis"
    for name in (
        "main_cross_analysis.md",
        "main_cross_analysis_colored.docx",
        "main_cross_analysis_colored.pdf",
        "report_manifest.json",
    ):
        assert_file(main_root / name, 100 if name.endswith(".json") else 1000)
    validate_manifest(main_root / "report_manifest.json", verify_hashes=True)
    assets = STUDY_ROOT / "paper_writing" / "assets"
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("scope") != "four_experiments_no_cross_dataset_pooling":
        raise ValueError("Paper asset scope dört deneyi ayırmıyor")
    validate_manifest(assets / "manifest.json", verify_hashes=True)
    figures = list((assets / "figures").glob("*.pdf"))
    tables = list((assets / "tables").glob("*.tex"))
    if len(figures) != 5 or len(tables) != 6:
        raise ValueError(f"Paper assets eksik: {len(figures)} figür/{len(tables)} tablo")
    for required in (
        STUDY_ROOT / "paper_writing" / "overleaf" / "main.tex",
        STUDY_ROOT / "paper_writing" / "overleaf" / "ref.bib",
        STUDY_ROOT / "paper_writing" / "overleaf" / "README.md",
        STUDY_ROOT / "paper_writing" / "PAPER_STRUCTURE.md",
        STUDY_ROOT / "literature_review" / "LITERATURE_REVIEW.md",
        STUDY_ROOT / "literature_review" / "SEARCH_AUDIT.md",
    ):
        assert_file(required, 500)
    return "5 figür, 6 tablo, main report ve Overleaf iskeleti"


def validate_active_paths() -> str:
    retired_paths = [
        REPO_ROOT / "studies" / "teacher_reference_bias_v1",
        STUDY_ROOT / "archives",
        *(source.root / "archives" for source in DATASETS.values()),
    ]
    for retired in retired_paths:
        if retired.exists():
            raise ValueError(f"Emekliye ayrılmış kopya hâlâ mevcut: {retired}")
    forbidden = (
        "teacher_reference_bias_v2_512",
        "teacher_reference_bias_small_vehicle_v1_512",
        "teacher_reference_bias_multiteacher_v1_512",
        "/home/ssyzai/",
    )
    scanned = 0
    for root in (
        STUDY_ROOT / "src",
        STUDY_ROOT / "scripts",
        STUDY_ROOT / "configs",
        STUDY_ROOT / "experiments",
        STUDY_ROOT / "docs",
        STUDY_ROOT / "paper_writing",
        STUDY_ROOT / "literature_review",
    ):
        for path in root.rglob("*"):
            if not path.is_file() or "archives" in path.parts or "results" in path.parts:
                continue
            if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".tex", ".bib", ".json"}:
                continue
            if path.name in {
                "MIGRATION_MANIFEST.json",
                "RUN_MANIFEST_MIGRATION_AUDIT.json",
                "QA_REPORT.json",
                "QA_REPORT.md",
                "validate_paper_study.py",
            }:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            hits = [token for token in forbidden if token in text]
            if hits:
                raise ValueError(f"Eski/mutlak yol {hits} dosyada bulundu: {path}")
            scanned += 1
    return f"{scanned} aktif metin/kod dosyası"


def write_report(audit: Audit) -> None:
    payload = {
        "schema_version": 1,
        "status": "failed" if audit.failures else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": audit.checks,
        "pass_count": len(audit.checks) - len(audit.failures),
        "fail_count": len(audit.failures),
    }
    json_path = STUDY_ROOT / "docs" / "QA_REPORT.json"
    md_path = STUDY_ROOT / "docs" / "QA_REPORT.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# QA Report",
        "",
        f"Durum: **{payload['status']}**",
        "",
        f"PASS: {payload['pass_count']} | FAIL: {payload['fail_count']}",
        "",
        "| Kontrol | Durum | Ayrıntı |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['name']} | {row['status']} | {row['detail'].replace('|', '/')} |"
        for row in audit.checks
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


def main() -> None:
    audit = Audit()
    if REPO_ROOT != PATHS_REPO_ROOT:
        audit.fail("repo_root_contract", f"{REPO_ROOT} != {PATHS_REPO_ROOT}")
    else:
        audit.pass_("repo_root_contract", str(REPO_ROOT))
    for experiment_id in DATASETS:
        audit.run(f"{experiment_id}:prepared", lambda name=experiment_id: validate_prepared(name))
        audit.run(f"{experiment_id}:predictions", lambda name=experiment_id: validate_predictions(name))
        audit.run(f"{experiment_id}:run_manifests", lambda name=experiment_id: validate_run_manifests(name))
        audit.run(f"{experiment_id}:references", lambda name=experiment_id: validate_references(name))
        audit.run(f"{experiment_id}:analysis", lambda name=experiment_id: validate_analysis(name))
        audit.run(f"{experiment_id}:figures", lambda name=experiment_id: validate_figures(name))
        audit.run(f"{experiment_id}:full_reports", lambda name=experiment_id: validate_full_reports(name))
        audit.run(f"{experiment_id}:cross_report", lambda name=experiment_id: validate_cross_report(name))
    audit.run("paper_outputs", validate_paper_outputs)
    audit.run("active_paths", validate_active_paths)
    write_report(audit)
    for row in audit.checks:
        print(f"{row['status']} {row['name']}: {row['detail']}")
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
