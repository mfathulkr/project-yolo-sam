from __future__ import annotations

import argparse
import json
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
    build_isaid_reference_effect_table,
    build_samrs_shared_reference_table,
    write_report,
)


STUDY_ID = STUDY_ROOT.name
ANALYSIS_DIR = STUDY_ROOT / "results" / "analysis"
FIGURES_DIR = STUDY_ROOT / "results" / "figures"
REPORTS_DIR = STUDY_ROOT / "reports" / "full_metrics"
STRATUM_LABELS = {
    "no_overlap__low_mask_area": "No Overlap × Low Mask Area",
    "no_overlap__high_mask_area": "No Overlap × High Mask Area",
    "overlap__low_mask_area": "Overlap × Low Mask Area",
    "overlap__high_mask_area": "Overlap × High Mask Area",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "512 görüntülü matched çalışma için ayrı insan ve pseudo "
            "full metric raporları üret."
        )
    )
    parser.add_argument(
        "--report",
        choices=(
            "all",
            "isaid_plane_human",
            "isaid_plane_pseudo_sam1",
            "samrs_sota_plane",
        ),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=REPORTS_DIR)
    return parser.parse_args()


def _condition_mean(
    aggregates: pd.DataFrame,
    *,
    dataset_id: str,
    reference_type: str,
    model: str,
    bbox_source: str,
    metric: str,
) -> float:
    rows = aggregates[
        (aggregates["dataset_id"] == dataset_id)
        & (aggregates["reference_type"] == reference_type)
        & (aggregates["stratum"] == "overall")
        & (aggregates["model"] == model)
        & (aggregates["bbox_source"] == bbox_source)
    ]
    if rows.empty:
        raise ValueError(
            "Eksik aggregate sonuç: "
            f"{dataset_id}/{reference_type}/{model}/{bbox_source}/{metric}"
        )
    return float(rows[metric].mean())


def _model_values(
    aggregates: pd.DataFrame,
    *,
    dataset_id: str,
    reference_type: str,
    bbox_source: str,
    metric: str = "mean_iou",
) -> dict[str, float]:
    return {
        model: _condition_mean(
            aggregates,
            dataset_id=dataset_id,
            reference_type=reference_type,
            model=model,
            bbox_source=bbox_source,
            metric=metric,
        )
        for model in ("sam1", "sam2", "sam3")
    }


def _triple(values: dict[str, float]) -> str:
    return "/".join(f"{values[model]:.3f}".replace(".", ",") for model in values)


def _best_model(values: dict[str, float]) -> tuple[str, float]:
    model = max(values, key=values.__getitem__)
    return model.upper(), values[model]


def _rank_order(values: dict[str, float]) -> str:
    return " > ".join(
        model.upper()
        for model in sorted(
            values,
            key=lambda model: (-values[model], model),
        )
    )


def _gt_to_yolo_loss(
    gt_values: dict[str, float],
    yolo_values: dict[str, float],
) -> dict[str, float]:
    return {
        model: gt_values[model] - yolo_values[model]
        for model in ("sam1", "sam2", "sam3")
    }


def _stratum_extremes(
    aggregates: pd.DataFrame,
    *,
    dataset_id: str,
    reference_type: str,
    bbox_source: str,
) -> tuple[str, float, str, float]:
    selected = aggregates[
        (aggregates["dataset_id"] == dataset_id)
        & (aggregates["reference_type"] == reference_type)
        & (aggregates["bbox_source"] == bbox_source)
        & (aggregates["stratum"].isin(STRATUM_LABELS))
    ]
    if selected.empty:
        raise ValueError(
            f"Alt grup sonuçları eksik: {dataset_id}/{reference_type}/{bbox_source}"
        )
    means = selected.groupby("stratum", sort=True)["mean_iou"].mean()
    best = str(means.idxmax())
    worst = str(means.idxmin())
    return (
        STRATUM_LABELS[best],
        float(means.loc[best]),
        STRATUM_LABELS[worst],
        float(means.loc[worst]),
    )


def _precision_recall_note(
    aggregates: pd.DataFrame,
    *,
    dataset_id: str,
    reference_type: str,
    model: str,
    bbox_source: str,
) -> str:
    precision = _condition_mean(
        aggregates,
        dataset_id=dataset_id,
        reference_type=reference_type,
        model=model,
        bbox_source=bbox_source,
        metric="mean_precision",
    )
    recall = _condition_mean(
        aggregates,
        dataset_id=dataset_id,
        reference_type=reference_type,
        model=model,
        bbox_source=bbox_source,
        metric="mean_recall",
    )
    if recall > precision:
        interpretation = (
            "Recall daha yüksek olduğu için model nesne piksellerini büyük "
            "ölçüde yakalarken hedef dışına taşan pikseller precision değerini "
            "düşürmektedir."
        )
    elif precision > recall:
        interpretation = (
            "Precision daha yüksek olduğu için model boyadığı bölgelerde daha "
            "temizdir; buna karşılık bazı gerçek nesne piksellerini eksik "
            "bırakmaktadır."
        )
    else:
        interpretation = "Precision ve Recall dengelidir."
    return (
        f"{model.upper()} {bbox_source.replace('_', ' ')} koşulunda Overall "
        f"Precision {precision:.3f}, Recall {recall:.3f} olmuştur. "
        f"{interpretation}"
    )


def _common_scope(
    dataset_text: str,
    *,
    area_reference_text: str,
) -> tuple[str, ...]:
    return (
        dataset_text,
        "Test kümesi 512 görüntüdür. Dört overlap × mask-area grubunun her birinde tam 128 görüntü vardır.",
        "Strata tanımı gereği 512 test görüntüsünün tamamında en az bir uçak vardır; detector tablosu negatif arka plan görüntülerini içeren resmi tam benchmark değil, bu dengeli pozitif test alt kümesindeki gerçek COCO bbox değerlendirmesidir.",
        "No Overlap, görüntüdeki hiçbir iki GT bbox'un kesişmemesi; Overlap ise en az bir bbox çiftinin IoU değerinin 0,001 veya üstünde olmasıdır.",
        "Low/High Mask Area ayrımı, görüntüdeki toplam "
        f"{area_reference_text} uçak maskesi alanının veri seti için testten "
        "önce dondurulan eşiğin altında veya üstünde olmasına göre yapılır.",
        "SAM1, SAM2 ve SAM3 aynı görüntülerde hem GT bbox hem YOLO bbox istemiyle çalıştırılmıştır.",
        "YOLO detector her veri setinde ayrıca eğitilmiştir; SAM1, SAM2 ve SAM3 bu veri setlerinde yeniden eğitilmeden veya ince ayar yapılmadan yalnız bbox istemiyle kullanılmıştır.",
        "Detector protokolü aynı olsa da iSAID eğitim bölümü 1.571, SAMRS eğitim bölümü 2.191 görüntüdür; bu nedenle veri setleri arasındaki detector skoru farkı yalnız referans kaynağına bağlanan kontrollü bir etki değildir.",
        "YOLO bbox sonuçları deney başlamadan önce sabitlenen seed 42 ile eğitilmiş tek YOLO26x detector sonucudur.",
        "Maske metrikleri uçak örneği düzeyinde hesaplanır; büyük nesneler küçük nesnelerin sonucunu piksel sayısıyla baskılamaz.",
    )


def _isaid_reference_comparison(
    aggregates: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    return (
        build_isaid_reference_effect_table(aggregates),
        "Görüntü, uçak örneği, bbox ve model tahmini aynıdır; yalnız "
        "karşılaştırılan referans maske insan etiketinden SAM1 pseudo "
        "etiketine değişir. Bu nedenle fark, referans kaynağına duyarlılığı "
        "doğrudan gösterir.",
    )


def isaid_human_spec(aggregates: pd.DataFrame) -> ReportSpec:
    gt = _model_values(
        aggregates,
        dataset_id="isaid_plane",
        reference_type="human",
        bbox_source="gt_bbox",
    )
    yolo = _model_values(
        aggregates,
        dataset_id="isaid_plane",
        reference_type="human",
        bbox_source="yolo_bbox",
    )
    pseudo_gt = _model_values(
        aggregates,
        dataset_id="isaid_plane",
        reference_type="pseudo_sam1",
        bbox_source="gt_bbox",
    )
    gt_best_model, gt_best_value = _best_model(gt)
    yolo_best_model, yolo_best_value = _best_model(yolo)
    best_stratum, best_stratum_iou, worst_stratum, worst_stratum_iou = (
        _stratum_extremes(
            aggregates,
            dataset_id="isaid_plane",
            reference_type="human",
            bbox_source="gt_bbox",
        )
    )
    return ReportSpec(
        study_id=STUDY_ID,
        dataset_id="isaid_plane",
        slug="isaid_plane_human",
        title="iSAID Plane Human Reference Full Metric Document",
        dataset_label="iSAID Plane",
        reference_sections=(
            ReferenceSection(
                reference_type="human",
                title="İnsan Referansı",
                note=(
                    "Değerlendirme resmi iSAID insan çizimli instance "
                    "maskelerine karşı yapılmıştır."
                ),
            ),
        ),
        qualitative_image=FIGURES_DIR / "isaid_plane_gt_bbox_qualitative.png",
        scope_bullets=_common_scope(
            "Veri seti iSAID, hedef sınıf plane ve model giriş çözünürlüğü 1024×1024 pikseldir.",
            area_reference_text="resmi iSAID insan",
        ),
        context_bullets=(
            "iSAID maskeleri insanlar tarafından çizildiği için bu rapor bağımsız insan ground truth sonucudur.",
            "iSAID veri seti kaynağı: https://captain-whu.github.io/iSAID/",
            "iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images makalesi: https://arxiv.org/abs/1905.12886",
            "Bu rapordaki GT bbox, resmi iSAID insan instance anotasyonunda verilen kutudur.",
            "Bu rapor yalnız SAM1/SAM2/SAM3 ve GT/YOLO bbox koşullarını içerir.",
            "Detector mAP değerleri bbox ölçümüdür. IoU, Dice, Precision ve Recall ise piksel maskesi ölçümüdür.",
            "Beş tam SAM1 pseudo referans tablosu ayrı belgede verilmiştir. Bu belgede yalnız aynı tahminlerin referans değişimine duyarlılığını gösteren kısa karşılaştırma özeti bulunur.",
        ),
        discussion_bullets=(
            "İnsan referansında GT-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla "
            f"{_triple(gt)} olarak ölçülmüştür.",
            "İnsan referansında YOLO-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla "
            f"{_triple(yolo)} olarak ölçülmüştür; bunlar sabit seed 42 detector sonuçlarıdır.",
            f"En yüksek insan-referanslı Overall IoU, GT bbox koşulunda {gt_best_model} için "
            f"{gt_best_value:.3f}; YOLO bbox koşulunda {yolo_best_model} için "
            f"{yolo_best_value:.3f} olmuştur.",
            "GT bbox yerine YOLO bbox kullanıldığında Overall IoU kaybı SAM1/SAM2/SAM3 için sırasıyla "
            f"{_triple(_gt_to_yolo_loss(gt, yolo))} olmuştur.",
            "GT-bbox üç-model ortalamasında en yüksek alt grup "
            f"{best_stratum} ({best_stratum_iou:.3f}), en düşük alt grup "
            f"{worst_stratum} ({worst_stratum_iou:.3f}) olmuştur.",
            _precision_recall_note(
                aggregates,
                dataset_id="isaid_plane",
                reference_type="human",
                model=gt_best_model.lower(),
                bbox_source="gt_bbox",
            ),
            "GT bbox koşulu segmenter sınır kalitesini daha doğrudan, YOLO bbox koşulu ise detection ve segmentation hatalarının birleşik etkisini gösterir.",
            "Overlap ve mask-area alt tabloları aynı toplam test kümesinin dengeli, birbirini dışlayan dört parçasıdır; her tabloda 128 görüntü vardır.",
            "Bu insan etiketli sonuçlar, model kaynaklı pseudo etiket değerlendirmesine karşı birincil bağımsız karşılaştırma noktasıdır.",
            "GT-bbox model sıralaması insan referansında "
            f"{_rank_order(gt)}, aynı tahminlerin SAM1 pseudo referansında "
            f"{_rank_order(pseudo_gt)} biçimindedir.",
            "Bu sıralamalar tablo nokta tahminleridir; birbirine yakın insan-referanslı skorlar tek başına istatistiksel üstünlük iddiası değildir.",
        ),
    )


def isaid_pseudo_spec(aggregates: pd.DataFrame) -> ReportSpec:
    gt = _model_values(
        aggregates,
        dataset_id="isaid_plane",
        reference_type="pseudo_sam1",
        bbox_source="gt_bbox",
    )
    yolo = _model_values(
        aggregates,
        dataset_id="isaid_plane",
        reference_type="pseudo_sam1",
        bbox_source="yolo_bbox",
    )
    human_gt = _model_values(
        aggregates,
        dataset_id="isaid_plane",
        reference_type="human",
        bbox_source="gt_bbox",
    )
    deltas = {
        model: gt[model] - human_gt[model] for model in ("sam1", "sam2", "sam3")
    }
    human_best_model, human_best_value = _best_model(human_gt)
    pseudo_best_model, pseudo_best_value = _best_model(gt)
    best_stratum, best_stratum_iou, worst_stratum, worst_stratum_iou = (
        _stratum_extremes(
            aggregates,
            dataset_id="isaid_plane",
            reference_type="pseudo_sam1",
            bbox_source="gt_bbox",
        )
    )
    return ReportSpec(
        study_id=STUDY_ID,
        dataset_id="isaid_plane",
        slug="isaid_plane_pseudo_sam1",
        title="iSAID Plane SAM1 Pseudo Reference Full Metric Document",
        dataset_label="iSAID Plane",
        reference_sections=(
            ReferenceSection(
                reference_type="pseudo_sam1",
                title="Kontrollü SAM1 Pseudo Referansı",
                note=(
                    "Aynı iSAID görüntülerindeki GT bbox'lar SAM1'e verilmiş "
                    "ve dondurulan çıktılar pseudo referans olarak kullanılmıştır."
                ),
            ),
        ),
        qualitative_image=(
            FIGURES_DIR
            / "isaid_plane_pseudo_sam1_gt_bbox_qualitative.png"
        ),
        scope_bullets=_common_scope(
            "Veri seti iSAID, hedef sınıf plane ve model giriş çözünürlüğü 1024×1024 pikseldir.",
            area_reference_text="resmi iSAID insan",
        ),
        context_bullets=(
            "Bu belge bağımsız benchmark sonucu değildir; referans kaynağı yanlılığını ölçen kontrollü deneydir.",
            "iSAID veri seti kaynağı: https://captain-whu.github.io/iSAID/",
            "iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images makalesi: https://arxiv.org/abs/1905.12886",
            "Görüntüler, bbox istemleri ve model tahminleri insan referansı deneyindekiyle aynıdır. Yalnız değerlendirme referansı SAM1 pseudo maskesidir.",
            "Bu rapordaki GT bbox, resmi iSAID insan instance anotasyonunda verilen kutudur; pseudo maskeden türetilmemiştir.",
            "Referansı üreten SAM1'in kendi pseudo maskelerine yüksek benzerlik göstermesi teacher-reference bias beklentisidir.",
            "Beş tam bağımsız iSAID insan referans tablosu ayrı full metric belgede verilmiştir. Bu belgede yalnız aynı tahminlerin referans değişimine duyarlılığını gösteren kısa karşılaştırma özeti bulunur.",
        ),
        discussion_bullets=(
            "SAM1 pseudo referansında GT-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla "
            f"{_triple(gt)} olarak ölçülmüştür.",
            "Kontrollü pseudo referans doğrudan SAM1 GT-bbox tahmininden dondurulduğu için SAM1 GT-bbox satırı bir kimlik kontrolüdür; bu satır bağımsız segmentasyon başarısı olarak yorumlanmaz.",
            "SAM1 pseudo referansında YOLO-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla "
            f"{_triple(yolo)} olarak ölçülmüştür.",
            "Aynı GT-bbox tahminleri insan yerine SAM1 pseudo referansla ölçüldüğünde IoU değişimi SAM1/SAM2/SAM3 için sırasıyla "
            f"{_triple(deltas)} olmuştur.",
            f"GT-bbox model sıralamasının lideri insan referansında {human_best_model} "
            f"({human_best_value:.3f}), SAM1 pseudo referansında {pseudo_best_model} "
            f"({pseudo_best_value:.3f}) olmuştur.",
            "Tam GT-bbox sıralaması insan referansında "
            f"{_rank_order(human_gt)}, SAM1 pseudo referansında "
            f"{_rank_order(gt)} biçimindedir; görülen değişim pseudo referansın "
            "model seçimini etkileyebileceği riskini gösterir.",
            "Bu sıralamalar tablo nokta tahminleridir; birbirine yakın insan-referanslı skorlar tek başına istatistiksel üstünlük iddiası değildir.",
            "GT bbox yerine YOLO bbox kullanıldığında pseudo-referanslı Overall IoU kaybı SAM1/SAM2/SAM3 için sırasıyla "
            f"{_triple(_gt_to_yolo_loss(gt, yolo))} olmuştur.",
            "GT-bbox üç-model ortalamasında en yüksek alt grup "
            f"{best_stratum} ({best_stratum_iou:.3f}), en düşük alt grup "
            f"{worst_stratum} ({worst_stratum_iou:.3f}) olmuştur.",
            _precision_recall_note(
                aggregates,
                dataset_id="isaid_plane",
                reference_type="pseudo_sam1",
                model=pseudo_best_model.lower(),
                bbox_source="gt_bbox",
            ),
            "SAM1'in kendisinin ürettiği referansa yakınlığı, gerçek insan çizimli sınır doğruluğuyla aynı şey değildir.",
            "Pseudo etiketler eğitim veya ön-etiketleme için kullanılabilir; ancak bağımsız test ground truth'u yerine kullanıldığında sonuç model ailesine yanlı görünebilir.",
        ),
    )


def samrs_spec(aggregates: pd.DataFrame) -> ReportSpec:
    gt = _model_values(
        aggregates,
        dataset_id="samrs_sota_plane",
        reference_type="pseudo_sam1",
        bbox_source="gt_bbox",
    )
    yolo = _model_values(
        aggregates,
        dataset_id="samrs_sota_plane",
        reference_type="pseudo_sam1",
        bbox_source="yolo_bbox",
    )
    sam1_teacher_margin = gt["sam1"] - (gt["sam2"] + gt["sam3"]) / 2.0
    best_stratum, best_stratum_iou, worst_stratum, worst_stratum_iou = (
        _stratum_extremes(
            aggregates,
            dataset_id="samrs_sota_plane",
            reference_type="pseudo_sam1",
            bbox_source="gt_bbox",
        )
    )
    return ReportSpec(
        study_id=STUDY_ID,
        dataset_id="samrs_sota_plane",
        slug="samrs_sota_plane",
        title="SAMRS SOTA Plane Full Metric Document",
        dataset_label="SAMRS SOTA Plane",
        reference_sections=(
            ReferenceSection(
                reference_type="pseudo_sam1",
                title="Resmi SAMRS SAM1 Pseudo Referansı",
                note=(
                    "SAMRS SOTA maskeleri SAM1 ViT-H ve özgün detection "
                    "istemlerinden üretilmiş pseudo maskelerdir."
                ),
            ),
        ),
        qualitative_image=(
            FIGURES_DIR / "samrs_sota_plane_gt_bbox_qualitative.png"
        ),
        scope_bullets=_common_scope(
            "Veri seti SAMRS SOTA-RBB, hedef sınıf plane ve model giriş çözünürlüğü 1024×1024 pikseldir.",
            area_reference_text="SAMRS pseudo",
        ),
        context_bullets=(
            "SAMRS SOTA-RBB görüntüleri DOTA v2.0 remote-sensing sahnelerinden gelir.",
            "SAMRS veri seti ve üretim kodu kaynağı: https://github.com/ViTAE-Transformer/SAMRS",
            "SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model makalesi: https://arxiv.org/abs/2305.02034",
            "Yayımlanan segmentasyon maskeleri, mevcut detection istemleri SAM1 ViT-H modeline verilerek otomatik üretilmiştir.",
            "Bu rapordaki GT bbox, yayımlanan özgün SAMRS detection anotasyonudur; pseudo maskeden yeniden türetilmemiştir.",
            "Bu nedenle raporlanan maske başarısı insan çizimli bağımsız ground truth değil, SAM1 kaynaklı pseudo referansa uyumdur.",
            "Detector mAP değerleri bbox ölçümüdür. IoU, Dice, Precision ve Recall ise piksel maskesi ölçümüdür.",
        ),
        discussion_bullets=(
            "Resmi SAMRS pseudo referansında GT-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla "
            f"{_triple(gt)} olarak ölçülmüştür.",
            "YOLO-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla "
            f"{_triple(yolo)} olarak ölçülmüştür; bunlar sabit seed 42 detector sonuçlarıdır.",
            "GT bbox yerine YOLO bbox kullanıldığında Overall IoU kaybı SAM1/SAM2/SAM3 için sırasıyla "
            f"{_triple(_gt_to_yolo_loss(gt, yolo))} olmuştur.",
            "SAM1'in GT-bbox IoU değeri, SAM2 ve SAM3 ortalamasından "
            f"{sam1_teacher_margin:.3f} daha yüksektir; bu fark resmi referansın SAM1 ile üretilmiş olmasıyla birlikte yorumlanmalıdır.",
            "SAMRS maskeleri ayrı bir resmi üretim hattında SAM1 ViT-H ile oluşturulduğu için buradaki SAM1 satırı kontrollü iSAID kimlik kontrolü kadar birebir değildir; yine de aynı model ailesine çok güçlü yakınlık gösterir.",
            "GT-bbox üç-model ortalamasında en yüksek alt grup "
            f"{best_stratum} ({best_stratum_iou:.3f}), en düşük alt grup "
            f"{worst_stratum} ({worst_stratum_iou:.3f}) olmuştur.",
            _precision_recall_note(
                aggregates,
                dataset_id="samrs_sota_plane",
                reference_type="pseudo_sam1",
                model="sam1",
                bbox_source="gt_bbox",
            ),
            "SAM1'in yüksek skoru, aynı model ailesinin ürettiği pseudo maskelere biçimsel yakınlık içerebilir; tek başına insan etiketli gerçek performans kanıtı değildir.",
            "GT bbox ile YOLO bbox arasındaki fark, otomatik detector hatasının segmentasyon zincirine eklediği kaybı gösterir.",
            "SAMRS pseudo etiketleri eğitim ve ölçeklenebilir ön-etiketleme için yararlı olabilir; nihai model karşılaştırması bağımsız insan etiketli test kümesiyle yapılmalıdır.",
        ),
    )


def _samrs_comparison_if_available(
) -> tuple[pd.DataFrame | None, str | None, tuple[Path, ...]]:
    shared_dir = ANALYSIS_DIR / "shared_human_reference_audit"
    shared_summary_path = shared_dir / "model_dual_reference_summary.csv"
    inflation_ci_path = shared_dir / "model_reference_inflation_ci.json"
    if not shared_summary_path.is_file() or not inflation_ci_path.is_file():
        return None, None, ()
    shared_summary = pd.read_csv(shared_summary_path)
    inflation_ci = json.loads(inflation_ci_path.read_text(encoding="utf-8"))
    return (
        build_samrs_shared_reference_table(shared_summary, inflation_ci),
        "Bu ek tablo yalnız aynı uçak örnekleri hem SAMRS pseudo maskesi hem "
        "bağımsız iSAID insan maskesiyle eşleştirilebildiyse üretilir. Tahmin "
        "sabit, değerlendirme referansı değişkendir.",
        (shared_summary_path, inflation_ci_path),
    )


def main() -> None:
    args = parse_args()
    aggregates_path = ANALYSIS_DIR / "aggregate_metrics.csv"
    detector_summary_path = ANALYSIS_DIR / "detector_seed_summary.csv"
    aggregates = pd.read_csv(aggregates_path)

    specs = {
        "isaid_plane_human": isaid_human_spec(aggregates),
        "isaid_plane_pseudo_sam1": isaid_pseudo_spec(aggregates),
        "samrs_sota_plane": samrs_spec(aggregates),
    }
    for report_id, spec in specs.items():
        if args.report not in ("all", report_id):
            continue
        if report_id.startswith("isaid_plane_"):
            comparison_table, comparison_note = _isaid_reference_comparison(
                aggregates
            )
            extra_input_paths: tuple[Path, ...] = ()
        else:
            (
                comparison_table,
                comparison_note,
                extra_input_paths,
            ) = _samrs_comparison_if_available()
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
            print(f"{report_id} {output_type}: {output_path}")


if __name__ == "__main__":
    main()
