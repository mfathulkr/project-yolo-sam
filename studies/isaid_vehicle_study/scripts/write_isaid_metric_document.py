from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.config import load_config, resolve_path
from yolo_sam.data.coco_masks import load_ground_truth_masks
from yolo_sam.evaluation.metrics import compute_mask_metrics
from yolo_sam.io_utils import ensure_dir, load_binary_mask


PIPELINES = [
    ("sam3_text", "SAM3 text only", "sam3_text_output_dir", "SAM3"),
    ("yolo_sam3", "SAM3 YOLO bbox", "yolo_sam3_output_dir", "SAM3"),
    ("gt_box_sam3", "SAM3 GT bbox", "gt_box_sam3_output_dir", "SAM3"),
    ("sam3_hybrid_yolo", "SAM3 hybrid YOLO bbox", "sam3_hybrid_yolo_output_dir", "SAM3"),
    ("remotesam_text", "RemoteSAM text only", "remotesam_text_output_dir", "RemoteSAM"),
    ("gt_box_ringmo_sam", "RingMo-SAM GT bbox", "gt_box_ringmo_sam_output_dir", "RingMo-SAM"),
    ("yolo_ringmo_sam", "RingMo-SAM YOLO bbox", "yolo_ringmo_sam_output_dir", "RingMo-SAM"),
    ("gt_box_sam2", "SAM2 GT bbox", "gt_box_sam2_output_dir", "SAM2"),
    ("yolo_sam2", "SAM2 YOLO bbox", "yolo_sam2_output_dir", "SAM2"),
]

TABLES = [
    ("overall", "Overall", None),
    ("no_overlap__low_mask_area", "No Overlap / Low Mask Area", "no_overlap__low_mask_area"),
    ("no_overlap__high_mask_area", "No Overlap / High Mask Area", "no_overlap__high_mask_area"),
    ("overlap__low_mask_area", "Overlap / Low Mask Area", "overlap__low_mask_area"),
    ("overlap__high_mask_area", "Overlap / High Mask Area", "overlap__high_mask_area"),
]

IOU_THRESHOLDS = [0.50, 0.75, 0.90]
MAP_PROXY_THRESHOLDS = [round(value / 100.0, 2) for value in range(50, 100, 5)]
PER_IMAGE_METRICS_NAME = "per_image_metrics_selected_pipelines.csv"
LONG_SUMMARY_NAME = "summary_all_tables_selected_pipelines.csv"
RAW_SUMMARIES_DIR_NAME = "raw_summaries"
DOCUMENT_TITLE = "iSAID Vehicle Segmentation Metric Report"
SCOPE_HEADING = "Scope"
METRIC_LOGIC_HEADING = "Metric Logic"
CONTEXT_HEADING = "Dataset and Paper Context"
DETECTOR_HEADING = "YOLO Detector BBox Metrics"
SEGMENTATION_HEADING = "Segmentation Tables"
QUALITATIVE_HEADING = "Qualitative Examples"
DISCUSSION_HEADING = "Discussion"
FINDINGS_HEADING = "Findings and Interpretation"
ARTIFACTS_HEADING = "Generated Artifacts"
SEGMENTATION_TABLES_NOTE = (
    "DOCX/PDF çıktılarında 0.0-1.0 aralığındaki başarı metrikleri kırmızıdan sarıya, "
    "sarıdan yeşile giden renk ölçeğiyle boyanır."
)
QUALITATIVE_NOTE = "Bunlar önceki PDF'te kullanılan aynı dört seçilmiş görsel örnektir."

SCOPE_BULLETS = [
    "Veri seti: iSAID değerlendirme bölümü; `Small_Vehicle` ve `Large_Vehicle` çokgenlerinden birleştirilmiş `vehicle` sınıfı kullanıldı.",
    "Değerlendirme kümesi: 128 pozitif 1024 x 1024 görüntü parçası. Kümeler dört gruba dengelendi: örtüşme yok/var ve düşük/yüksek hedef maske alanı.",
    "Bu rapordan çıkarılan hatlar: GroundingDINO + SAM2, SegEarth-OV3 + SAM3 ve SAM3 hybrid GT bbox.",
    "SAM3 hybrid GT bbox çıkarıldı çünkü GT bbox zaten nesnenin gerçek konumunu veriyor. Kusursuz bbox promptunun üstüne metin eklemek ana benchmark için temiz bir ölçüm değil ve yorumu karıştırıyor.",
    "Raporda kalan hatlar: SAM3 text only, SAM3 YOLO bbox, SAM3 GT bbox, SAM3 hybrid YOLO bbox, RemoteSAM text only, RingMo-SAM GT bbox, RingMo-SAM YOLO bbox, SAM2 GT bbox ve SAM2 YOLO bbox.",
]

METRIC_BULLETS = [
    "Tüm segmentasyon metrikleri her görüntü için birleştirilmiş ikili araç maskesi üzerinde hesaplandı: tahmin edilen araç ön planı, GT araç ön planıyla karşılaştırıldı.",
    "TP (doğru pozitif): modelin `vehicle` dediği ve GT maskesinde de gerçekten `vehicle` olan piksel sayısı.",
    "FP (yanlış pozitif): modelin `vehicle` dediği ama GT'de arka plan olan piksel sayısı. FP yüksekse maske araç dışına taşıyor demektir.",
    "FN (yanlış negatif): GT'de `vehicle` olan ama modelin kaçırdığı piksel sayısı. FN yüksekse model araç piksellerini eksik yakalıyor demektir.",
    "TN (doğru negatif) arka plan-arka plan pikselleridir. IoU, Dice, kesinlik ve duyarlılık formüllerinde kullanılmaz; çünkü arka plan çok büyük olduğu için skoru yapay olarak şişirebilir.",
    "`IoU`, piksel kesişiminin piksel birleşimine oranıdır: `TP / (TP + FP + FN)`.",
    "`Dice`, `2TP / (2TP + FP + FN)` olarak hesaplanır. IoU gibi bir örtüşme skorudur ve aynı maske için genellikle IoU'dan daha yüksek görünür.",
    "Kesinlik, `TP / (TP + FP)` olarak hesaplanır. Düşük kesinlik, tahmin maskesinin araç dışı alanları da fazla kapsadığını gösterir.",
    "Duyarlılık, `TP / (TP + FN)` olarak hesaplanır. Düşük duyarlılık, modelin GT araç piksellerini kaçırdığını gösterir.",
    "`Ortalama` metrikler görüntü seviyesinde ortalamadır: metrik önce her görüntü için ayrı hesaplanır, sonra görüntü skorları ortalanır. Tek bir büyük görüntü son ortalamada ekstra ağırlık almaz.",
    "Buna rağmen tek bir görüntünün içinde hesap hâlâ piksel tabanlıdır. Büyük nesneler veya büyük maske bölgeleri o görüntünün TP/FP/FN sayımlarını domine edebilir. Bu etkiyi görmek için düşük/yüksek maske alanı grupları ayrıca raporlanmıştır.",
    "`Pred/GT Area`, tahmin edilen ön plan piksel sayısının GT ön plan piksel sayısına oranıdır. 1'in üzerindeki değerler aşırı segmentasyona, 1'in altındaki değerler eksik segmentasyona işaret eder.",
    "Segmentasyon `mAP50 proxy`, `mAP75 proxy` ve `mAP90 proxy` değerleri görüntü seviyesinde IoU eşik geçme oranlarıdır: `mAP50 proxy` IoU >= 0.50 olan görüntü oranı, `mAP75 proxy` IoU >= 0.75 olan görüntü oranı, `mAP90 proxy` ise IoU >= 0.90 olan görüntü oranıdır.",
    "Segmentasyon `mAP50-95 proxy`, 0.50, 0.55, ..., 0.95 eşiklerindeki görüntü geçme oranlarının ortalamasıdır. Bu değer hâlâ birleştirilmiş maske proxy metriğidir; COCO nesne örneği AP değeri değildir.",
    "YOLO detector metrikleri bbox metriğidir, maske metriği değildir. Detector tarafındaki IoU, tahmin edilen YOLO bbox ile GT bbox arasındaki kutu örtüşmesini ifade eder.",
    "YOLO detector `BBox mAP50`, `BBox mAP75`, `BBox mAP90` ve `BBox mAP50-95` değerleri ayrıca gerçek COCO bounding-box AP metriği olarak hesaplandı.",
]

DETECTOR_METRIC_NOTE = (
    "Not: Bu tablo YOLO'yu yalnızca detector olarak değerlendirir. Buradaki IoU, tahmin edilen "
    "bbox ile GT bbox arasındaki BBox IoU'dur. Maske IoU değildir; maske kalitesi "
    "segmentasyon tablolarında değerlendirilir."
)

CONTEXT_BULLETS = [
    "[iSAID orijinal makalesi](https://arxiv.org/abs/1905.12886): 2.806 yüksek çözünürlüklü hava görüntüsü, 15 kategori ve 655.451 nesne örneği içerir. Makale, hava görüntülerinde nesne örneği segmentasyonunu zor yapan nedenleri açıkça vurgular: görüntü başına çok sayıda nesne, büyük ölçek farkları ve çok sayıda küçük nesne.",
    "Bu çalışma iSAID'i yalnızca birleştirilmiş araç hedefi üzerinden kullanır: `Small_Vehicle` + `Large_Vehicle`. Bu yüzden tablolardaki skorlar resmi 15 sınıflı iSAID nesne örneği AP skoru değildir; araç maskesine odaklanan daha dar bir stres testidir.",
    "[RemoteSAM](https://arxiv.org/abs/2505.18022), RemoteSAM-270K adlı 270K görüntü-metin-maske referanslı segmentasyon veri setini oluşturur ve iSAID, LoveDA, DOTA, HRRSD gibi ana uzaktan algılama kaynaklarını entegre eder. Bu nedenle burada kutusuz metin tabanlı bir temel karşılaştırma hattı olarak güçlü çıkması beklenebilir.",
    "[RemoteSAM proje sayfası](https://github.com/1e12Leon/RemoteSAM), RemoteSAM-270K veri setini ve yer gözlemi referanslı metin istemleri için geniş semantik/özellik kapsamını ayrıca açıklar.",
    "[RingMo-SAM](https://doi.org/10.1109/TGRS.2023.3332219), optik ve SAR görüntüler için geliştirilmiş çok modlu uzaktan algılama SAM tarzı bir modeldir. Makalede, birden çok açık uzaktan algılama veri setinden toplanmış milyonlarca segmentasyon nesne örneği ile büyük ölçekli bir eğitim kümesi kurulduğu; iSAID, ISPRS Vaihingen, ISPRS Potsdam ve AIR-PolSAR-Seg gibi veri setlerinde değerlendirildiği belirtilir.",
    "RemoteSAM'i yalnızca metin durumunda geçmek değil, YOLO + SAM2'nin RemoteSAM yalnızca metin hattını geçmesi bu çalışmadaki asıl pratik başarı noktasıdır. Bizim YOLO dedektörümüz iSAID araç alanına özel lokalizasyon sağlıyor; SAM2 de iyi kutu verildiğinde maskeyi güçlü biçimde tamamlıyor.",
    "RingMo-SAM bu çalışmada yüksek kesinlik ama düşük duyarlılık gösteriyor. Bu, uzaktan algılama için ince ayar yapılmış olmanın tek başına yeterli olmadığını; özellikle küçük ve yoğun araçlarda modelin temiz ama eksik maske üretmeye meyilli olduğunu gösteriyor.",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the full iSAID vehicle metric document.")
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "configs" / "yolo26x_cpu_eval.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=STUDY_ROOT / "reports" / "isaid_vehicle_full_metric_document.md",
    )
    parser.add_argument(
        "--detector-metrics",
        type=Path,
        default=(
            STUDY_ROOT
            / "results"
            / "pipelines"
            / "detector_metrics"
            / "yolo_detector_eval_metrics.csv"
        ),
    )
    return parser.parse_args()


def metric(value: float) -> str:
    return f"{float(value):.4f}"


def pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def count_rate(count: int, total: int) -> str:
    rate = count / float(total) if total else 0.0
    return f"{count}/{total} ({pct(rate)})"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Satır yok._"
    headers = list(df.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(rows)


def markdown_bullets(bullets: list[str]) -> list[str]:
    return [f"- {bullet}" for bullet in bullets]


def table_proxy_note(summary: pd.DataFrame) -> str:
    counts = sorted({int(value) for value in summary["images"].tolist()})
    sample_text = str(counts[0]) if len(counts) == 1 else "/".join(str(value) for value in counts)
    return (
        f"Not: Bu tablodaki segmentasyon mAP proxy değerleri, her model hattı için {sample_text} görüntü "
        f"üzerinden hesaplanan görüntü seviyesinde IoU eşik geçme oranlarıdır. `mAP50 proxy`, "
        f"`mAP75 proxy` ve `mAP90 proxy`; IoU >= 0.50, 0.75 ve 0.90 olan görüntü oranlarıdır. "
        f"`mAP50-95 proxy`, 0.50 ile 0.95 arasındaki eşikleri ortalar."
    )


def require_masks(mask_dir: Path, image_names: list[str]) -> None:
    missing = [name for name in image_names if not (mask_dir / f"{Path(name).stem}.png").exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{mask_dir} is missing {len(missing)} masks, for example: {preview}")


def build_per_image_metrics(config: dict) -> pd.DataFrame:
    split = config["dataset"]["eval_split"]
    prepared_split_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split
    metadata = pd.read_csv(prepared_split_dir / "metadata.csv")
    gt_masks = load_ground_truth_masks(prepared_split_dir)
    image_names = [str(name) for name in metadata["file_name"].tolist()]

    rows: list[dict[str, object]] = []
    for pipeline, label, output_key, family in PIPELINES:
        if output_key not in config["paths"]:
            raise KeyError(f"Missing path key in config: {output_key}")
        mask_dir = resolve_path(config["paths"][output_key]) / "masks"
        require_masks(mask_dir, image_names)

        for _, metadata_row in metadata.iterrows():
            image_name = str(metadata_row["file_name"])
            gt_mask = gt_masks[image_name]
            pred_mask = load_binary_mask(mask_dir / f"{Path(image_name).stem}.png", gt_mask.shape)
            metric_values = compute_mask_metrics(pred_mask, gt_mask)
            rows.append(
                {
                    "image": image_name,
                    "pipeline": pipeline,
                    "pipeline_label": label,
                    "family": family,
                    "stratum": metadata_row.get("stratum", ""),
                    "overlap_group": metadata_row.get("overlap_group", ""),
                    "area_group": metadata_row.get("area_group", ""),
                    "num_objects": int(metadata_row["num_objects"]),
                    "mask_area_ratio": float(metadata_row["mask_area_ratio"]),
                    "max_pair_bbox_iou": float(metadata_row["max_pair_bbox_iou"]),
                    **metric_values,
                }
            )
    return pd.DataFrame(rows)


def summarize_group(metrics: pd.DataFrame, table_key: str, title: str, stratum: str | None) -> pd.DataFrame:
    frame = metrics if stratum is None else metrics[metrics["stratum"] == stratum]
    rows: list[dict[str, object]] = []
    for pipeline, label, _, family in PIPELINES:
        group = frame[frame["pipeline"] == pipeline]
        total = int(group["image"].nunique())
        if total == 0:
            continue
        row: dict[str, object] = {
            "table_key": table_key,
            "table_title": title,
            "pipeline": pipeline,
            "pipeline_label": label,
            "family": family,
            "images": total,
            "avg_iou": float(group["iou"].mean()),
            "avg_dice": float(group["dice"].mean()),
            "avg_precision": float(group["precision"].mean()),
            "avg_recall": float(group["recall"].mean()),
            "avg_pred_gt_area": float(group["pred_area_ratio"].mean()),
            "zero_iou_count": int((group["iou"] == 0.0).sum()),
        }
        for threshold in IOU_THRESHOLDS:
            suffix = int(threshold * 100)
            count = int((group["iou"] >= threshold).sum())
            row[f"iou_ge_{suffix}_count"] = count
            row[f"iou_ge_{suffix}_rate"] = count / float(total)
        proxy_rates = [float((group["iou"] >= threshold).mean()) for threshold in MAP_PROXY_THRESHOLDS]
        row["map50_proxy"] = row["iou_ge_50_rate"]
        row["map75_proxy"] = row["iou_ge_75_rate"]
        row["map90_proxy"] = row["iou_ge_90_rate"]
        row["map50_95_proxy"] = float(sum(proxy_rates) / len(proxy_rates))
        rows.append(row)
    order = {pipeline: index for index, (pipeline, _, _, _) in enumerate(PIPELINES)}
    summary = pd.DataFrame(rows)
    summary["order"] = summary["pipeline"].map(order)
    return summary.sort_values("order").drop(columns=["order"])


def display_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "Pipeline": row["pipeline_label"],
                "Images": int(row["images"]),
                "Avg IoU": metric(row["avg_iou"]),
                "Avg Dice": metric(row["avg_dice"]),
                "Avg Precision": metric(row["avg_precision"]),
                "Avg Recall": metric(row["avg_recall"]),
                "Pred/GT Area": metric(row["avg_pred_gt_area"]),
                "mAP50 proxy": metric(row["map50_proxy"]),
                "mAP75 proxy": metric(row["map75_proxy"]),
                "mAP90 proxy": metric(row["map90_proxy"]),
                "mAP50-95 proxy": metric(row["map50_95_proxy"]),
            }
        )
    return pd.DataFrame(rows)


def display_detector_metrics(detector_metrics_path: Path) -> pd.DataFrame:
    if not detector_metrics_path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(detector_metrics_path)
    if metrics.empty:
        return pd.DataFrame()
    row = metrics.iloc[0]
    split_label = {"eval": "değerlendirme", "train": "eğitim", "val": "doğrulama"}.get(str(row["split"]), row["split"])
    return pd.DataFrame(
        [
            {
                "Split": split_label,
                "Images": int(row["images"]),
                "Detections": int(row["detections_for_ap"]),
                "AP conf": metric(row["ap_conf_threshold"]),
                "Fixed conf": metric(row["fixed_conf_threshold"]),
                "BBox mAP50": metric(row["bbox_mAP50"]),
                "BBox mAP75": metric(row["bbox_mAP75"]),
                "BBox mAP90": metric(row["bbox_mAP90"]),
                "BBox mAP50-95": metric(row["bbox_mAP50_95"]),
                "BBox Precision@0.50": metric(row["precision_at_iou50"]),
                "BBox Recall@0.50": metric(row["recall_at_iou50"]),
                "BBox Precision@0.75": metric(row["precision_at_iou75"]),
                "BBox Recall@0.75": metric(row["recall_at_iou75"]),
                "BBox Precision@0.90": metric(row["precision_at_iou90"]),
                "BBox Recall@0.90": metric(row["recall_at_iou90"]),
            }
        ]
    )


def row_value(summary: pd.DataFrame, pipeline: str, column: str) -> float | None:
    rows = summary[summary["pipeline"] == pipeline]
    if rows.empty:
        return None
    return float(rows.iloc[0][column])


def difference(summary: pd.DataFrame, left: str, right: str, column: str) -> float | None:
    left_value = row_value(summary, left, column)
    right_value = row_value(summary, right, column)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def format_signed(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.4f}"


def discussion_lines(summaries: dict[str, pd.DataFrame]) -> list[str]:
    overall = summaries["overall"]
    best = overall.loc[overall["avg_iou"].idxmax()]
    lines = [
        f"`Overall` tablosunda en yüksek Avg IoU `{best['pipeline_label']}` hattında `{metric(best['avg_iou'])}` olarak ölçüldü.",
        f"SAM3 hybrid YOLO bbox, SAM3 YOLO bbox hattına göre Avg IoU değerini `{format_signed(difference(overall, 'sam3_hybrid_yolo', 'yolo_sam3', 'avg_iou'))}` değiştirdi.",
        f"Bu düşüş tek başına implementation hatası göstergesi değildir; hybrid prompt, bbox-only sonucunun güvenli bir iyileştirmesi gibi çalışmaz. Text + bbox birlikte verildiğinde SAM3'ün mask generation davranışı değişir.",
        f"Bu deneyde `vehicle` text promptu bbox promptlarıyla birleşince over-segmentation davranışı oluşturmuş görünüyor. Over-segmentation, modelin araç dışındaki yol, bina, gölge veya zemin piksellerini de araç maskesine katmasıdır.",
        f"Bu yüzden SAM3 hybrid YOLO bbox hattında Recall küçük bir miktar artarken (`{format_signed(difference(overall, 'sam3_hybrid_yolo', 'yolo_sam3', 'avg_recall'))}`), Precision belirgin düşüyor (`{format_signed(difference(overall, 'sam3_hybrid_yolo', 'yolo_sam3', 'avg_precision'))}`) ve Pred/GT Area `{format_signed(difference(overall, 'sam3_hybrid_yolo', 'yolo_sam3', 'avg_pred_gt_area'))}` artıyor. Yani model daha fazla araç pikseli yakalayabiliyor, fakat araç dışı pikselleri de maskeye kattığı için Avg IoU düşüyor.",
        f"SAM3 GT bbox ile SAM3 YOLO bbox arasındaki fark `{format_signed(difference(overall, 'gt_box_sam3', 'yolo_sam3', 'avg_iou'))}` Avg IoU. Bu fark, detector localization kalitesini SAM3 mask decoder etkisinden ayırmaya yardım eder.",
        f"SAM2 GT bbox ile SAM2 YOLO bbox arasındaki fark `{format_signed(difference(overall, 'gt_box_sam2', 'yolo_sam2', 'avg_iou'))}` Avg IoU.",
        f"RingMo-SAM GT bbox ile RingMo-SAM YOLO bbox arasındaki fark `{format_signed(difference(overall, 'gt_box_ringmo_sam', 'yolo_ringmo_sam', 'avg_iou'))}` Avg IoU.",
        f"YOLO bbox + SAM2, RemoteSAM text only hattına göre Avg IoU değerini `{format_signed(difference(overall, 'yolo_sam2', 'remotesam_text', 'avg_iou'))}` artırdı. Bu rapordaki temel pratik sonuç budur: hedef araç domaininde eğitilmiş bir detector ve düz SAM2 birleşimi, bu kurulumda remote-sensing text/referring modelini geçebiliyor.",
    ]
    for table_key, title, _ in TABLES[1:]:
        summary = summaries[table_key]
        best_row = summary.loc[summary["avg_iou"].idxmax()]
        lines.append(f"`{title}` grubunda en iyi Avg IoU `{best_row['pipeline_label']}` hattında `{metric(best_row['avg_iou'])}` olarak ölçüldü.")
    lines.extend(
        [
            "Düşük maske alanı grupları küçük araçlar için kritik stres testidir. Bir model yüksek alanlı görüntülerde makul görünüp küçük nesnelerde eksik veya taşan maske üretebilir.",
            "Segmentasyon mAP proxy kolonları, tek başına ortalamaların gizleyebileceği hata tiplerini daha görünür yapar. `mAP50 proxy` kaba ama kullanılabilir maskeleri, `mAP90 proxy` ise çok sıkı ve neredeyse kusursuz maske geçme oranını gösterir.",
            "Bu segmentasyon mAP proxy değerleri görüntü seviyesinde IoU eşik geçme oranlarıdır; COCO instance AP değildir. Ayrı YOLO detector tablosu gerçek BBox COCO AP değerlerini raporlar.",
        ]
    )
    return lines


def write_document(
    output_path: Path,
    metrics: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    detector_display: pd.DataFrame,
    tables_dir: Path,
) -> None:
    lines: list[str] = [
        f"# {DOCUMENT_TITLE}",
        "",
        f"## {SCOPE_HEADING}",
        "",
        *markdown_bullets(SCOPE_BULLETS),
        "",
        f"## {METRIC_LOGIC_HEADING}",
        "",
        *markdown_bullets(METRIC_BULLETS),
        "",
        f"## {CONTEXT_HEADING}",
        "",
        *markdown_bullets(CONTEXT_BULLETS),
        "",
        f"## {DETECTOR_HEADING}",
        "",
    ]
    if detector_display.empty:
        lines.append("_Dedektör metrik CSV'si bulunamadı. Bu tabloyu üretmek için `scripts/evaluate_yolo_detector_coco.py` çalıştırılmalı._")
    else:
        lines.extend([DETECTOR_METRIC_NOTE, ""])
        lines.append(markdown_table(detector_display))

    lines.extend(
        [
            "",
            f"## {SEGMENTATION_HEADING}",
            "",
            SEGMENTATION_TABLES_NOTE,
            "",
        ]
    )
    for table_key, title, _ in TABLES:
        lines.extend(
            [
                f"### {title}",
                "",
                table_proxy_note(summaries[table_key]),
                "",
                markdown_table(display_summary(summaries[table_key])),
                "",
            ]
        )

    lines.extend(
        [
            f"## {QUALITATIVE_HEADING}",
            "",
            QUALITATIVE_NOTE,
            "",
            "### No Overlap / Low Mask Area",
            "",
            "![No overlap low mask area](figures/sample_cases/no_overlap__low_mask_area__P2766_0016_hero.png)",
            "",
            "### No Overlap / High Mask Area",
            "",
            "![No overlap high mask area](figures/sample_cases/no_overlap__high_mask_area__P0199_0002_hero.png)",
            "",
            "### Overlap / Low Mask Area",
            "",
            "![Overlap low mask area](figures/sample_cases/overlap__low_mask_area__P2404_0002_hero.png)",
            "",
            "### Overlap / High Mask Area",
            "",
            "![Overlap high mask area](figures/sample_cases/overlap__high_mask_area__P2781_0005_hero.png)",
            "",
            f"## {DISCUSSION_HEADING}",
            "",
            *markdown_bullets(discussion_lines(summaries)),
            "",
            f"## {ARTIFACTS_HEADING}",
            "",
            f"- Görüntü bazlı metrik CSV'si: `{tables_dir.relative_to(output_path.parent) / PER_IMAGE_METRICS_NAME}`",
            f"- Görünen özet tablo CSV'si: `{tables_dir.relative_to(output_path.parent) / LONG_SUMMARY_NAME}`",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_path = args.output.resolve()
    ensure_dir(output_path.parent)
    tables_dir = ensure_dir(output_path.parent / "tables" / "full_metric_document")
    raw_summaries_dir = ensure_dir(tables_dir / RAW_SUMMARIES_DIR_NAME)

    metrics = build_per_image_metrics(config)
    metrics.to_csv(tables_dir / PER_IMAGE_METRICS_NAME, index=False)

    summaries = {
        table_key: summarize_group(metrics, table_key, title, stratum)
        for table_key, title, stratum in TABLES
    }
    display_frames: list[pd.DataFrame] = []
    for table_key, title, _ in TABLES:
        display_frame = display_summary(summaries[table_key]).copy()
        display_frame.insert(0, "Table", title)
        display_frames.append(display_frame)
    display_long_summary = pd.concat(display_frames, ignore_index=True)
    display_long_summary.to_csv(tables_dir / LONG_SUMMARY_NAME, index=False)
    for table_key, summary in summaries.items():
        summary.to_csv(raw_summaries_dir / f"summary_{table_key}.csv", index=False)

    detector_display = display_detector_metrics(args.detector_metrics)
    write_document(output_path, metrics, summaries, detector_display, tables_dir)
    print(output_path)


if __name__ == "__main__":
    main()
