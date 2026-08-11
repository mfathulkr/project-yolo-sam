from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
CANONICAL_REPORT_SOURCE = (
    REPO_ROOT
    / "studies"
    / "teacher_reference_bias_small_vehicle_v1_512"
    / "src"
)
for source_root in (
    STUDY_ROOT / "src",
    REPO_ROOT / "src",
    CANONICAL_REPORT_SOURCE,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias.reporting.full_metric_document import (  # noqa: E402
    ReferenceSection,
    ReportSpec,
    write_report,
)
from teacher_reference_bias_multiteacher.paths import DATASETS  # noqa: E402
from teacher_reference_bias_multiteacher.io import (  # noqa: E402
    relativize_path_hash_manifest,
)


ANALYSIS_ROOT = STUDY_ROOT / "results" / "analysis"
FIGURES_ROOT = STUDY_ROOT / "results" / "figures"
REPORTS_ROOT = STUDY_ROOT / "reports" / "full_metrics"


def combined_detector_summary() -> Path:
    frames = []
    for dataset_id, source in DATASETS.items():
        path = source.canonical_analysis_root / "detector_seed_summary.csv"
        frame = pd.read_csv(path)
        frames.append(frame[frame["dataset_id"] == dataset_id])
    output = ANALYSIS_ROOT / "detector_seed_summary.csv"
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    return output


def metric_value(
    aggregates: pd.DataFrame,
    *,
    dataset_id: str,
    reference_type: str,
    model: str,
    bbox_source: str,
    metric: str = "mean_iou",
) -> float:
    row = aggregates[
        (aggregates["dataset_id"] == dataset_id)
        & (aggregates["reference_type"] == reference_type)
        & (aggregates["model"] == model)
        & (aggregates["bbox_source"] == bbox_source)
        & (aggregates["stratum"] == "overall")
    ]
    if len(row) != 1:
        raise ValueError(
            f"Eksik aggregate: {dataset_id}/{reference_type}/{model}/{bbox_source}"
        )
    return float(row.iloc[0][metric])


def reference_comparison_table(
    aggregates: pd.DataFrame,
    *,
    dataset_id: str,
    reference_type: str,
) -> pd.DataFrame:
    rows = []
    for model in ("sam1", "sam2", "sam3"):
        for bbox_source in ("gt_bbox", "yolo_bbox"):
            human = metric_value(
                aggregates,
                dataset_id=dataset_id,
                reference_type="human",
                model=model,
                bbox_source=bbox_source,
            )
            pseudo = metric_value(
                aggregates,
                dataset_id=dataset_id,
                reference_type=reference_type,
                model=model,
                bbox_source=bbox_source,
            )
            rows.append(
                {
                    "Model": model.upper(),
                    "BBox": "GT bbox" if bbox_source == "gt_bbox" else "YOLO bbox",
                    "Human IoU": f"{human:.3f}",
                    "Reference IoU": f"{pseudo:.3f}",
                    "IoU Farkı": f"{pseudo - human:+.3f}",
                }
            )
    return pd.DataFrame(rows)


def ranking(
    aggregates: pd.DataFrame,
    *,
    dataset_id: str,
    reference_type: str,
    bbox_source: str,
) -> str:
    values = {
        model: metric_value(
            aggregates,
            dataset_id=dataset_id,
            reference_type=reference_type,
            model=model,
            bbox_source=bbox_source,
        )
        for model in ("sam1", "sam2", "sam3")
    }
    return " > ".join(
        model.upper() for model in sorted(values, key=lambda item: (-values[item], item))
    )


def empty_reference_stats(dataset_id: str, teacher: str) -> tuple[int, float]:
    source = DATASETS[dataset_id]
    prediction_file = (
        source.predictions_root / teacher / "gt_bbox" / "predictions.jsonl"
    )
    statuses = pd.read_json(prediction_file, lines=True, typ="frame")[["status"]]
    empty = int((statuses["status"] == "empty_mask").sum())
    return empty, empty / len(statuses)


def common_scope(dataset_id: str, target_label: str) -> tuple[str, ...]:
    return (
        f"Veri seti iSAID, hedef sınıf {target_label} ve giriş çözünürlüğü 1024×1024 pikseldir.",
        "Test kümesi 512 görüntüdür. Dört Overlap × Mask Area grubunun her birinde tam 128 görüntü vardır.",
        "No Overlap, görüntüdeki hiçbir iki insan GT bbox'un kesişmemesi; Overlap ise en az bir bbox çiftinin IoU değerinin 0,001 veya üstünde olmasıdır.",
        "Low/High Mask Area ayrımı insan çizimli hedef maskelerinin görüntüdeki toplam alanına göre, testten önce dondurulmuş veri setine özgü eşikle yapılmıştır. Referans değişse bile stratum üyeliği değiştirilmemiştir.",
        "SAM1, SAM2 ve SAM3 aynı 512 görüntüde hem GT bbox hem seed 42 YOLO bbox istemiyle çalıştırılmıştır.",
        "Yeni inference yapılmamıştır. İnsan, SAM1, SAM2 ve SAM3 değerlendirmelerinde aynı dondurulmuş model tahminleri kullanılmış; yalnız karşılaştırılan referans maske değiştirilmiştir.",
        "SAM2/SAM3 pseudo referansları ilgili öğretmenin insan GT bbox istemiyle ürettiği instance maskeleridir. İnsan kutusunun kullanılması nedeniyle bu referanslar insan lokalizasyonundan tamamen bağımsız değildir.",
        "Maske metrikleri instance-level hesaplanır; her hedef örnek eşit ağırlıktadır. Büyük nesneler küçük nesneleri piksel sayısıyla perdelemez.",
        "YOLO detector sonuçları referans maskeden bağımsızdır ve bütün pseudo raporlarda aynı gerçek bbox mAP değerleri tekrar kullanılır.",
    )


def report_spec(
    aggregates: pd.DataFrame,
    *,
    dataset_id: str,
    teacher: str,
) -> ReportSpec:
    source = DATASETS[dataset_id]
    reference_type = f"pseudo_{teacher}"
    teacher_gt = metric_value(
        aggregates,
        dataset_id=dataset_id,
        reference_type=reference_type,
        model=teacher,
        bbox_source="gt_bbox",
    )
    teacher_yolo = metric_value(
        aggregates,
        dataset_id=dataset_id,
        reference_type=reference_type,
        model=teacher,
        bbox_source="yolo_bbox",
    )
    human_teacher_gt = metric_value(
        aggregates,
        dataset_id=dataset_id,
        reference_type="human",
        model=teacher,
        bbox_source="gt_bbox",
    )
    empty_count, empty_rate = empty_reference_stats(dataset_id, teacher)
    slug = f"{dataset_id}_pseudo_{teacher}"
    return ReportSpec(
        study_id=STUDY_ROOT.name,
        dataset_id=dataset_id,
        slug=slug,
        title=f"{dataset_id.replace('_', ' ').title()} {teacher.upper()} Pseudo Reference Full Metric Document",
        dataset_label=dataset_id.replace("_", " ").title(),
        reference_sections=(
            ReferenceSection(
                reference_type=reference_type,
                title=f"{teacher.upper()} Pseudo Referansı",
                note=(
                    f"Değerlendirme {teacher.upper()} modelinin insan GT bbox "
                    "istemiyle ürettiği dondurulmuş instance maskelerine karşı yapılmıştır."
                ),
            ),
        ),
        qualitative_image=(
            FIGURES_ROOT / f"{dataset_id}_pseudo_{teacher}_gt_bbox_qualitative.png"
        ),
        scope_bullets=common_scope(dataset_id, source.target_label),
        context_bullets=(
            "Bu belge bağımsız ground truth performansı değil, değerlendirme referansının model skorunu nasıl değiştirdiğini gösteren kontrollü bir referans duyarlılığı deneyidir.",
            f"Referans öğretmeni {teacher.upper()}, referans istemi insan GT bbox ve örnek sayısı {source.teacher_instance_count:,}.".replace(",", "."),
            f"Referans kümesinde {empty_count:,} boş maske vardır ({empty_rate:.1%}).".replace(",", "."),
            "Bir öğretmenin GT-bbox tahminini aynı tahmin maskesine karşı ölçen diagonal hücre özdeşlik gereği 1,000 olur; bu sonuç model başarısı olarak yorumlanamaz.",
            "Detector mAP değerleri bbox ölçümüdür. Avg IoU, Dice, Precision, Recall ve IoU eşik oranları piksel maskesi ölçümüdür; IoU eşik oranları mAP değildir.",
        ),
        discussion_bullets=(
            f"{teacher.upper()} GT-bbox diagonal kontrolü {teacher_gt:.3f} değerindedir. Bu tam skor beklenen özdeşlik kontrolüdür.",
            f"Aynı {teacher.upper()} modeli insan referansında GT-bbox Avg IoU {human_teacher_gt:.3f} verir; referansın kendi çıktısına çevrilmesi görünürde {teacher_gt - human_teacher_gt:+.3f} artış üretir.",
            f"YOLO bbox kullanıldığında {teacher.upper()} modeli kendi pseudo referansında {teacher_yolo:.3f} Avg IoU verir; bbox değişmesine rağmen aynı model ailesine ait referans avantajı sürmektedir.",
            f"İnsan referansı GT-bbox sıralaması {ranking(aggregates, dataset_id=dataset_id, reference_type='human', bbox_source='gt_bbox')}; bu pseudo referans sıralaması {ranking(aggregates, dataset_id=dataset_id, reference_type=reference_type, bbox_source='gt_bbox')} biçimindedir.",
            f"{empty_count:,} boş öğretmen maskesi, özellikle boş tahmin–boş referans eşleşmelerinde öz-skoru yükseltebilir. Bu nedenle boş referans oranı skorlarla birlikte raporlanmalıdır.".replace(",", "."),
            "Sonuç, pseudo etiketlerin kullanılamaz olduğunu değil; pseudo etiket üreticisiyle aday modelin aynı veya yakın aileden olduğu değerlendirmelerde bağımsız insan referansı olmadan model seçimi yapılamayacağını gösterir.",
        ),
        target_label=source.target_label,
    )


def main() -> None:
    aggregates_path = ANALYSIS_ROOT / "aggregate_metrics.csv"
    aggregates = pd.read_csv(aggregates_path)
    detector_path = combined_detector_summary()
    for dataset_id in DATASETS:
        for teacher in ("sam2", "sam3"):
            reference_type = f"pseudo_{teacher}"
            spec = report_spec(
                aggregates,
                dataset_id=dataset_id,
                teacher=teacher,
            )
            output = write_report(
                spec=spec,
                output_dir=REPORTS_ROOT / spec.slug,
                aggregates_path=aggregates_path,
                detector_summary_path=detector_path,
                comparison_table=reference_comparison_table(
                    aggregates,
                    dataset_id=dataset_id,
                    reference_type=reference_type,
                ),
                comparison_note=(
                    "Görüntü, instance, bbox istemi ve model tahmini aynıdır; "
                    f"yalnız değerlendirme referansı insan maskesinden {teacher.upper()} "
                    "pseudo maskesine değişmiştir."
                ),
                extra_input_paths=(
                    ANALYSIS_ROOT / "paired_reference_effects.csv",
                ),
            )
            relativize_path_hash_manifest(
                REPORTS_ROOT / spec.slug / "report_manifest.json",
                REPO_ROOT,
            )
            print(output["pdf"])


if __name__ == "__main__":
    main()
