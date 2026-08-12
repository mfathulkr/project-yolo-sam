from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias.reporting.full_metric_document import (  # noqa: E402
    ReferenceSection,
    ReportSpec,
    write_report,
)
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    DATASETS,
    MODELS,
    REFERENCES,
    ExperimentSource,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dört deney için legacy full-metric belgelerini üret."
    )
    parser.add_argument("--experiment", choices=tuple(DATASETS))
    parser.add_argument("--reference")
    return parser.parse_args()


def metric_value(
    aggregates: pd.DataFrame,
    *,
    source: ExperimentSource,
    reference_type: str,
    model: str,
    bbox_source: str,
) -> float:
    selected = aggregates[
        (aggregates["dataset_id"] == source.dataset_id)
        & (aggregates["reference_type"] == reference_type)
        & (aggregates["model"] == model)
        & (aggregates["bbox_source"] == bbox_source)
        & (aggregates["stratum"] == "overall")
    ]
    if len(selected) != 1:
        raise ValueError(
            "Beklenen tek aggregate satırı bulunamadı: "
            f"{source.experiment_id}/{reference_type}/{model}/{bbox_source}"
        )
    return float(selected.iloc[0]["mean_iou"])


def ranking(
    aggregates: pd.DataFrame,
    *,
    source: ExperimentSource,
    reference_type: str,
    bbox_source: str,
) -> str:
    values = {
        model: metric_value(
            aggregates,
            source=source,
            reference_type=reference_type,
            model=model,
            bbox_source=bbox_source,
        )
        for model in MODELS
    }
    return " > ".join(
        model.upper()
        for model in sorted(values, key=lambda name: (-values[name], name))
    )


def comparison_table(
    aggregates: pd.DataFrame,
    *,
    source: ExperimentSource,
    reference_type: str,
) -> pd.DataFrame | None:
    baseline = source.reference_types[0]
    if reference_type == baseline:
        return None
    rows: list[dict[str, str]] = []
    for model in MODELS:
        for bbox_source in ("gt_bbox", "yolo_bbox"):
            base = metric_value(
                aggregates,
                source=source,
                reference_type=baseline,
                model=model,
                bbox_source=bbox_source,
            )
            score = metric_value(
                aggregates,
                source=source,
                reference_type=reference_type,
                model=model,
                bbox_source=bbox_source,
            )
            rows.append(
                {
                    "Model": model.upper(),
                    "BBox": "GT bbox" if bbox_source == "gt_bbox" else "YOLO bbox",
                    "Temel Referans IoU": f"{base:.3f}",
                    "Reference IoU": f"{score:.3f}",
                    "IoU Farkı": f"{score - base:+.3f}",
                }
            )
    return pd.DataFrame(rows)


def common_scope(source: ExperimentSource) -> tuple[str, ...]:
    family = "iSAID" if source.dataset_family == "isaid" else "SAMRS SOTA"
    return (
        f"Veri kaynağı {family}, hedef sınıf {source.target_label} ve model giriş çözünürlüğü 1024×1024 pikseldir.",
        "Test kümesi 512 görüntüdür. Dört Overlap × Mask Area grubunun her birinde tam 128 görüntü vardır.",
        "No Overlap, görüntüdeki hedef GT bbox çiftlerinin kesişmemesi; Overlap ise en az bir hedef bbox çiftinin IoU değerinin 0,001 veya üstünde olmasıdır.",
        "Low/High Mask Area ayrımı yayımlanmış temel instance maskelerinin görüntü içindeki toplam alan oranına göre, testten önce dondurulmuş veri setine özgü eşikle yapılmıştır. Referans türü değişse bile stratum üyeliği değişmez.",
        "SAM1, SAM2 ve SAM3 tahminleri aynı 512 görüntüde hem GT bbox hem de seed 42 ile eğitilmiş YOLO bbox istemiyle bir kez üretilmiş ve bütün referanslara karşı değişmeden yeniden değerlendirilmiştir.",
        "SAM3 bbox koşulu Sam3Tracker PVS arayüzüyle, multimask_output=False ve mask_threshold=0.0 ayarlarıyla çalıştırılmıştır. PCS kavram örneği arayüzü kullanılmamıştır.",
        "Maske metrikleri instance-level hesaplanır; her hedef örnek eşit ağırlıktadır. Büyük nesneler küçük nesneleri piksel sayısıyla perdelemez.",
        "YOLO detector tablosu referans maskeden bağımsızdır; bütün referans raporlarında aynı dondurulmuş bbox sonuçları kullanılır.",
        "Nitel görsellerde seçilen görüntüdeki bütün hedef instance'lar modele ayrı GT bbox istemleri olarak verilir ve maskeler yalnız gösterim amacıyla birleştirilir.",
    )


def context_bullets(
    source: ExperimentSource,
    reference_type: str,
    empty_stats: pd.DataFrame,
) -> tuple[str, ...]:
    reference = REFERENCES[reference_type]
    selected = empty_stats[empty_stats["reference_type"] == reference_type]
    if len(selected) != 1:
        raise ValueError(f"Boş referans istatistiği eksik: {reference_type}")
    empty_count = int(selected.iloc[0]["empty_count"])
    empty_rate = float(selected.iloc[0]["empty_rate"])
    bullets = [
        f"Bu belgede değerlendirme referansı {reference.display_name} ve değerlendirilen instance sayısı {source.instance_count:,}.".replace(",", "."),
        f"Referans kümesinde {empty_count:,} boş maske vardır ({empty_rate:.2%}). Bilinen pozitif nesnede boş pseudo maske başarı sayılmaz ve 0 puanlanır.".replace(",", "."),
        "Detector mAP değerleri bbox ölçümüdür. Avg IoU, Dice, Precision, Recall ve IoU eşik oranları piksel maskesi ölçümüdür; IoU eşik oranları mAP değildir.",
    ]
    if reference.is_independent_human:
        bullets.append(
            "Bu iSAID insan anotasyonu bağımsız kontrol referansıdır ve model kalitesine ilişkin ana bilimsel yorum bu referansa dayanır."
        )
    elif reference.is_published_samrs:
        bullets.extend(
            (
                "Bu etiketler insan ground truth değildir. SAMRS veri seti tarafından yayımlanmış, SAM tabanlı otomatik üretim hattından gelen referanslardır.",
                "Yayımlanmış SAMRS etiketi ile bu çalışmada güncel SAM1 checkpoint'i kullanılarak yeniden üretilen referans birbirine çok yakın olabilir; fakat aynı dosya veya bağımsız insan anotasyonu değildir.",
            )
        )
    else:
        bullets.extend(
            (
                f"Bu pseudo referans {reference.teacher.upper()} modeline insan/yayımlanmış GT bbox verilerek instance başına üretilmiştir; lokalizasyon kutusu referans veri setinden gelir.",
                "GT-bbox diagonal hücre aynı dondurulmuş tahmin ile kendi pseudo referansını karşılaştıran özdeşlik/kapsama kontrolüdür. Bu hücre bağımsız segmentasyon başarısı değildir.",
            )
        )
    return tuple(bullets)


def discussion_bullets(
    aggregates: pd.DataFrame,
    *,
    source: ExperimentSource,
    reference_type: str,
) -> tuple[str, ...]:
    baseline = source.reference_types[0]
    reference = REFERENCES[reference_type]
    reference_rank_gt = ranking(
        aggregates,
        source=source,
        reference_type=reference_type,
        bbox_source="gt_bbox",
    )
    reference_rank_yolo = ranking(
        aggregates,
        source=source,
        reference_type=reference_type,
        bbox_source="yolo_bbox",
    )
    bullets = [
        f"Bu referansta Overall GT-bbox sıralaması {reference_rank_gt}; YOLO-bbox sıralaması {reference_rank_yolo} biçimindedir.",
        "GT bbox ile YOLO bbox arasındaki fark, segmenterden önceki detection hatasının uçtan uca sisteme etkisini gösterir.",
    ]
    if reference_type == baseline:
        if reference.is_independent_human:
            bullets.append(
                "Bu insan referansı model ailelerinden bağımsız olduğu için modeller arası kalite karşılaştırmasının güvenilir kontrolüdür."
            )
        else:
            bullets.append(
                "Bu yayımlanmış SAMRS referansı bağımsız insan kontrolü değildir; sonuçlar model başarısından çok SAM-türevi referansla uyumu da içerir."
            )
    else:
        teacher = str(reference.teacher)
        base_score = metric_value(
            aggregates,
            source=source,
            reference_type=baseline,
            model=teacher,
            bbox_source="yolo_bbox",
        )
        own_score = metric_value(
            aggregates,
            source=source,
            reference_type=reference_type,
            model=teacher,
            bbox_source="yolo_bbox",
        )
        bullets.extend(
            (
                f"{teacher.upper()} modeli YOLO bbox koşulunda temel referansa karşı {base_score:.3f}, kendi öğretmen ailesinin referansına karşı {own_score:.3f} Avg IoU verir; görünür fark {own_score - base_score:+.3f}'tür.",
                "Kendi pseudo referansında yüksek skor, modelin gerçek dünyada daha doğru olduğunu tek başına kanıtlamaz; aynı model ailesinin benzer sınır ve hata tercihlerini ödüllendiren teacher-reference affinity etkisini gösterebilir.",
            )
        )
    bullets.append(
        "Ana sonuç yalnız Overall tablosuna dayandırılmamalıdır; aynı yönün dört Overlap × Mask Area tabakasında korunup korunmadığı deney içi çapraz analiz belgesinde ayrıca gösterilir."
    )
    return tuple(bullets)


def report_spec(
    source: ExperimentSource,
    aggregates: pd.DataFrame,
    empty_stats: pd.DataFrame,
    reference_type: str,
) -> ReportSpec:
    reference = REFERENCES[reference_type]
    slug = f"{source.experiment_id}_{reference_type}"
    return ReportSpec(
        study_id=STUDY_ROOT.name,
        dataset_id=source.dataset_id,
        slug=slug,
        title=(
            f"{source.experiment_id.replace('_', ' ').title()} - "
            f"{reference.display_name} Full Metric Document"
        ),
        dataset_label=source.experiment_id.replace("_", " ").title(),
        reference_sections=(
            ReferenceSection(
                reference_type=reference_type,
                title=reference.display_name,
                note=(
                    f"Bütün SAM1/2/3 tahminleri değişmeden tutulmuş ve "
                    f"{reference.display_name} ile değerlendirilmiştir."
                ),
            ),
        ),
        qualitative_image=(
            source.figures_root / f"{reference_type}_gt_bbox_qualitative.png"
        ),
        scope_bullets=common_scope(source),
        context_bullets=context_bullets(source, reference_type, empty_stats),
        discussion_bullets=discussion_bullets(
            aggregates,
            source=source,
            reference_type=reference_type,
        ),
        target_label=source.target_label,
    )


def archive_existing_report(source: ExperimentSource, reference_type: str) -> None:
    output_dir = source.reports_root / "full_metrics" / reference_type
    if not output_dir.exists():
        return
    archive_dir = (
        source.root
        / "archives"
        / "pre_unification"
        / "reports"
        / reference_type
    )
    if archive_dir.exists():
        shutil.rmtree(output_dir)
        return
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.rename(archive_dir)


def write_one(source: ExperimentSource, reference_type: str) -> Path:
    aggregates_path = source.analysis_root / "aggregate_metrics.csv"
    detector_path = source.analysis_root / "detector_summary.csv"
    aggregates = pd.read_csv(aggregates_path)
    empty_stats = pd.read_csv(source.analysis_root / "reference_empty_stats.csv")
    spec = report_spec(source, aggregates, empty_stats, reference_type)
    table = comparison_table(
        aggregates,
        source=source,
        reference_type=reference_type,
    )
    baseline = source.reference_types[0]
    archive_existing_report(source, reference_type)
    output = write_report(
        spec=spec,
        output_dir=source.reports_root / "full_metrics" / reference_type,
        aggregates_path=aggregates_path,
        detector_summary_path=detector_path,
        comparison_table=table,
        comparison_note=(
            None
            if table is None
            else "Görüntü, instance, bbox istemi ve model tahmini aynıdır; "
            f"yalnız değerlendirme referansı {REFERENCES[baseline].display_name} "
            f"yerine {REFERENCES[reference_type].display_name} olarak değiştirilmiştir."
        ),
        extra_input_paths=(
            source.analysis_root / "manifest.json",
            source.analysis_root / "paired_reference_effects.csv",
            source.figures_root / "manifest.json",
        ),
    )
    return output["pdf"]


def main() -> None:
    args = parse_args()
    experiments = (args.experiment,) if args.experiment else tuple(DATASETS)
    for experiment_id in experiments:
        source = DATASETS[experiment_id]
        references = (
            (args.reference,) if args.reference else source.reference_types
        )
        for reference_type in references:
            if reference_type not in source.reference_types:
                raise ValueError(
                    f"{experiment_id} için geçersiz referans: {reference_type}"
                )
            print(write_one(source, reference_type))


if __name__ == "__main__":
    main()
