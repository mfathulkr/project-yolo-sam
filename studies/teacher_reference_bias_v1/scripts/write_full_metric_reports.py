from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
SHARED_SRC = REPO_ROOT / "src"
STUDY_SRC = STUDY_ROOT / "src"
for source_root in (SHARED_SRC, STUDY_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias.reporting.full_metric_document import (  # noqa: E402
    ReferenceSection,
    ReportSpec,
    build_isaid_reference_effect_table,
    build_samrs_shared_reference_table,
    write_report,
)


ANALYSIS_DIR = STUDY_ROOT / "results" / "analysis"
FIGURES_DIR = STUDY_ROOT / "results" / "figures"
REPORTS_DIR = STUDY_ROOT / "reports" / "full_metrics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-reference-bias çalışması için iSAID ve SAMRS SOTA "
            "renkli tam metrik raporlarını üret."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "isaid_plane", "samrs_sota_plane"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=REPORTS_DIR)
    return parser.parse_args()


def isaid_spec() -> ReportSpec:
    return ReportSpec(
        dataset_id="isaid_plane",
        slug="isaid_plane",
        title="iSAID Plane Segmentation Metric Report",
        dataset_label="iSAID Plane",
        reference_sections=(
            ReferenceSection(
                reference_type="human",
                title="İnsan Referansı",
                note=(
                    "Birincil değerlendirme resmi iSAID insan çizimli instance "
                    "maskelerine karşı yapılmıştır."
                ),
            ),
            ReferenceSection(
                reference_type="pseudo_sam1",
                title="Kontrollü SAM1 Pseudo Referansı",
                note=(
                    "Aynı görüntü ve tahminler, SAM1 GT-bbox çıktısından "
                    "dondurulan pseudo maskelere karşı yeniden ölçülmüştür. "
                    "Bu bölüm bağımsız benchmark sonucu değil, referans "
                    "kaynağı yanlılığı kontrolüdür."
                ),
            ),
        ),
        qualitative_image=FIGURES_DIR / "isaid_plane_gt_bbox_qualitative.png",
        scope_bullets=(
            "Veri seti iSAID, hedef sınıf plane ve giriş çözünürlüğü 1024×1024'tür.",
            "Test kümesinde 128 görüntü vardır. Overall tablosu 128 görüntüyü, dört overlap × mask-area tablosunun her biri 32 görüntüyü kapsar.",
            "SAM1, SAM2 ve SAM3 aynı görüntülerde hem resmi GT bbox hem YOLO bbox ile çalıştırılmıştır.",
            "YOLO bbox sonuçları üç ayrı YOLO eğitiminin ortalaması ± standart sapmasıdır.",
            "Birincil referans iSAID'ın insan çizimli maskesidir. Aynı tahminler ayrıca kontrollü SAM1 pseudo maskesine karşı ölçülerek referans yanlılığı gösterilir.",
        ),
        context_bullets=(
            "iSAID maskeleri insanlar tarafından çizildiği için bağımsız değerlendirme referansıdır.",
            "Kontrollü pseudo referans, iSAID görüntülerindeki resmi GT bbox'lar SAM1'e verilerek üretilmiştir; insan maskelerinin yerine geçirilmemiştir.",
            "İnsan ve pseudo tablolarında görüntü, bbox ve model tahmini aynıdır. Yalnız karşılaştırılan referans maske değişir.",
            "SAM1'in kendi ürettiği pseudo maskeye karşı çok yüksek görünmesi beklenen bir teacher-reference bias etkisidir; modelin insan maskesinde kusursuz olduğu anlamına gelmez.",
            "Detector mAP değerleri bbox ölçümüdür. Segmentasyon tablolarındaki IoU, Dice, Precision ve Recall ise piksel maskesi ölçümüdür.",
        ),
        discussion_bullets=(
            "İnsan referansında GT-bbox ortalama IoU değerleri SAM1/SAM2/SAM3 için sırasıyla 0,661/0,650/0,667'dir; üç model birbirine yakındır ve SAM3 küçük farkla en yüksektir.",
            "Kontrollü SAM1 pseudo referansında aynı GT-bbox çıktılarının IoU değerleri 1,000/0,831/0,775 olur. SAM1'in 1,000 sonucu tasarım gereğidir: referansı üreten tahmin yine kendisine karşı ölçülmektedir.",
            "Referansın insan maskesinden SAM1 pseudo maskesine değişmesi GT-bbox IoU değerini SAM1/SAM2/SAM3 için yaklaşık +0,339/+0,181/+0,107 yükseltir. Bu fark görüntü zorluğu değil, teacher-reference affinity etkisidir.",
            "YOLO-bbox insan referansında SAM3 ortalama 0,626 IoU ile en yüksek, SAM1 0,607 ve SAM2 0,596 düzeyindedir. Detector hatası GT-bbox koşuluna göre tüm segmenter skorlarını düşürür.",
            "Recall değerlerinin precision değerlerinden belirgin yüksek olması maskelerin gerçek uçağın çoğunu yakalarken sınır dışına taşma, yani over-segmentation eğilimi taşıdığını gösterir.",
            "IoU eşik geçme oranları COCO mask AP değildir. Gerçek mAP yalnız YOLO detector bbox tablosunda raporlanmıştır.",
        ),
    )


def samrs_spec() -> ReportSpec:
    return ReportSpec(
        dataset_id="samrs_sota_plane",
        slug="samrs_sota_plane",
        title="SAMRS SOTA Plane Segmentation Metric Report",
        dataset_label="SAMRS SOTA Plane",
        reference_sections=(
            ReferenceSection(
                reference_type="pseudo_sam1",
                title="Resmi SAMRS SAM1 Pseudo Referansı",
                note=(
                    "SAMRS SOTA maskeleri SAM1 ViT-H ve özgün detection "
                    "prompt'larıyla üretilmiş pseudo maskelerdir. Sonuçlar "
                    "öncelikle bu resmi referansa karşı verilmiştir."
                ),
            ),
        ),
        qualitative_image=FIGURES_DIR / "samrs_sota_plane_gt_bbox_qualitative.png",
        scope_bullets=(
            "Veri seti SAMRS SOTA-RBB, hedef sınıf plane ve giriş çözünürlüğü 1024×1024'tür.",
            "Test kümesinde 128 görüntü vardır. Overall tablosu 128 görüntüyü, dört overlap × mask-area tablosunun her biri 32 görüntüyü kapsar.",
            "SAM1, SAM2 ve SAM3 aynı görüntülerde hem özgün detection bbox hem YOLO bbox ile çalıştırılmıştır.",
            "YOLO bbox sonuçları üç ayrı YOLO eğitiminin ortalaması ± standart sapmasıdır.",
            "Resmi SAMRS SOTA maskeleri insan ground truth'u değil, SAM1 ViT-H ile üretilmiş pseudo maskelerdir.",
        ),
        context_bullets=(
            "SAMRS SOTA-RBB görüntüleri DOTA v2.0 remote-sensing sahnelerinden gelir.",
            "Yayımlanan segmentasyon maskeleri, mevcut detection bbox'larının SAM1'e prompt olarak verilmesiyle otomatik üretilmiştir.",
            "Bu nedenle SAM1 sonucu aynı model ailesinin ürettiği referans biçimine doğal olarak daha yakındır.",
            "Bağımsız kontrol için eşleşebilen SAMRS test görüntüleri aynı DOTA sahnelerindeki iSAID insan maskelerine de karşılaştırılmıştır.",
            "Detector mAP değerleri bbox ölçümüdür. Segmentasyon tablolarındaki IoU, Dice, Precision ve Recall ise piksel maskesi ölçümüdür.",
        ),
        discussion_bullets=(
            "Resmi SAMRS pseudo referansında GT-bbox ortalama IoU değerleri SAM1/SAM2/SAM3 için 0,997/0,791/0,666'dır. SAM1'in neredeyse kusursuz görünmesi veri setinin onun çıktı stiliyle üretilmesinden kaynaklanır.",
            "YOLO-bbox ortalama IoU değerleri SAM1/SAM2/SAM3 için yaklaşık 0,869/0,707/0,591'dir. Detector hatası eklenince skor düşer; model sırası değişmez.",
            "Ortak insan denetiminde tahmin sabit tutulup yalnız referans değiştirildiğinde SAM1 IoU 0,648 insan referansından 0,998 pseudo referansa yükselir. IoU artışı +0,350 ve %95 güven aralığı [0,313, 0,378]'dir.",
            "Aynı enflasyon SAM2 için +0,225, SAM3 için +0,184'tür. Tüm modeller pseudo referansta yükselir; en büyük artış referansı üreten SAM1'dedir.",
            "Bu bulgu SAMRS'nin pretraining veya weak supervision için değersiz olduğunu göstermez. Ancak aynı teacher ailesinin ürettiği maskeler bağımsız test ground truth'u gibi yorumlanırsa model kalitesi olduğundan yüksek görünür.",
            "Geçerli downstream değerlendirme bağımsız insan etiketli test setinde yapılmalı; pseudo-mask üreticisi, checkpoint, prompt ve insan-denetimli subset açıkça raporlanmalıdır.",
        ),
    )


def main() -> None:
    args = parse_args()
    aggregates_path = ANALYSIS_DIR / "aggregate_metrics.csv"
    detector_summary_path = ANALYSIS_DIR / "detector_seed_summary.csv"
    aggregates = pd.read_csv(aggregates_path)

    targets = (
        ("isaid_plane", isaid_spec()),
        ("samrs_sota_plane", samrs_spec()),
    )
    for dataset_id, spec in targets:
        if args.dataset not in ("all", dataset_id):
            continue
        if dataset_id == "isaid_plane":
            comparison_table = build_isaid_reference_effect_table(aggregates)
            extra_input_paths: tuple[Path, ...] = ()
            comparison_note = (
                "Görüntü, instance, bbox ve tahmin sabittir; yalnız karşılaştırılan "
                "referans maskesi değişir. Pozitif IoU enflasyonu pseudo referansın "
                "model çıktı stiline daha yakın olduğunu gösterir."
            )
        else:
            shared_dir = ANALYSIS_DIR / "shared_human_reference_audit"
            shared_summary_path = shared_dir / "model_dual_reference_summary.csv"
            inflation_ci_path = shared_dir / "model_reference_inflation_ci.json"
            shared_summary = pd.read_csv(shared_summary_path)
            inflation_ci = json.loads(
                inflation_ci_path.read_text(encoding="utf-8")
            )
            extra_input_paths = (shared_summary_path, inflation_ci_path)
            comparison_table = build_samrs_shared_reference_table(
                shared_summary,
                inflation_ci,
            )
            comparison_note = (
                "Tablo, aynı GT-bbox tahminlerini SAMRS pseudo maskesi ve "
                "bağımsız iSAID insan maskesi karşısında ölçer. Yalnız "
                "karşılaştırılan referans değişir; güven aralığı aynı kaynak "
                "görüntüden gelen örneklerin ilişkisini hesaba katar."
            )
        paths = write_report(
            spec=spec,
            output_dir=args.output_root / spec.slug,
            aggregates_path=aggregates_path,
            detector_summary_path=detector_summary_path,
            comparison_table=comparison_table,
            comparison_note=comparison_note,
            extra_input_paths=extra_input_paths,
        )
        for output_type, output_path in paths.items():
            print(f"{dataset_id} {output_type}: {output_path}")


if __name__ == "__main__":
    main()
