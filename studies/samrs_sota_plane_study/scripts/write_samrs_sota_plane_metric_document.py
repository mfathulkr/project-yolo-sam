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
    ("gt_box_sam1", "SAM1 GT bbox", "sam1_gt_box_output_dir", "SAM1"),
    ("yolo_sam1", "SAM1 YOLO bbox", "sam1_yolo_output_dir", "SAM1"),
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
DOCUMENT_TITLE = "SAMRS SOTA Plane Segmentation Metric Report"

SCOPE_BULLETS = [
    "Veri seti: SAMRS içindeki SOTA subset; hedef sınıf `plane`.",
    "SOTA, DOTA v2.0 kaynaklı yüksek çözünürlüklü remote sensing patch'lerinden türetilmiştir.",
    "Önemli sınır: SAMRS maskeleri insan tarafından çizilmiş kesin GT değil, SAM ile üretilmiş pseudo-mask etiketleridir. Bu rapor bu yüzden `SAMRS-SOTA pseudo-mask benchmark` olarak okunmalıdır.",
    "Değerlendirme kümesi hedef olarak her stratum için 128 pozitif görüntü seçer: Overall, No Overlap / Low Mask Area, No Overlap / High Mask Area, Overlap / Low Mask Area, Overlap / High Mask Area.",
    "Bu deney iSAID vehicle deneyinin aynı mantığını daha az boxy bir nesneye taşır. Plane nesnesinde kanat, kuyruk ve gövde nedeniyle bbox nesne maskesine tam oturmaz; bbox promptunun maskeye etkisi bu yüzden daha anlamlı test edilir.",
    "SAM1 bu deneyde ek baseline olarak eklendi. Böylece aynı bbox kalitesiyle SAM1, SAM2 ve SAM3 davranışı ayrıca görülebilir.",
]

METRIC_BULLETS = [
    "Segmentasyon metrikleri görüntü seviyesinde birleştirilmiş ikili plane maskesi üzerinde hesaplanır.",
    "TP, modelin plane dediği ve pseudo-GT maskesinde de plane olan piksel sayısıdır.",
    "FP, modelin plane dediği ama pseudo-GT'de arka plan olan piksel sayısıdır. FP artarsa maske nesne dışına taşıyor demektir.",
    "FN, pseudo-GT'de plane olan ama modelin kaçırdığı piksel sayısıdır. FN artarsa model hedef maskeyi eksik yakalıyor demektir.",
    "IoU, `TP / (TP + FP + FN)` oranıdır. Pixel-level maske örtüşmesini ölçer.",
    "Dice, `2TP / (2TP + FP + FN)` oranıdır. IoU'ya benzer bir örtüşme skorudur, genellikle IoU'dan daha yüksek görünür.",
    "Precision, `TP / (TP + FP)` oranıdır. Düşük precision, maske tahmininin hedef dışına fazla taştığını gösterir.",
    "Recall, `TP / (TP + FN)` oranıdır. Düşük recall, hedef plane piksellerinin eksik yakalandığını gösterir.",
    "Ortalama metrikler önce her görüntü için ayrıca hesaplanır, sonra görüntü skorlarının ortalaması alınır. Böylece tek bir görüntü rapor ortalamasında ekstra ağırlık almaz.",
    "Yine de tek görüntü içindeki hesap piksel tabanlıdır. Büyük maskeler o görüntünün TP/FP/FN sayımlarını domine edebilir; bu yüzden low/high mask area tabloları ayrıca tutulur.",
    "Pred/GT Area, tahmin edilen plane piksel sayısının pseudo-GT plane piksel sayısına oranıdır. 1'in üstü over-segmentation, 1'in altı under-segmentation işaretidir.",
    "Segmentasyon mAP50 proxy, mAP75 proxy ve mAP90 proxy değerleri COCO AP değildir. Bunlar görüntü seviyesinde IoU eşik geçme oranlarıdır.",
    "mAP50-95 proxy, 0.50, 0.55, ..., 0.95 eşiklerindeki geçme oranlarının ortalamasıdır.",
    "YOLO detector metrikleri bbox metriğidir. Buradaki IoU, tahmin bbox'u ile pseudo-GT bbox'u arasındaki kutu örtüşmesidir; maske IoU değildir.",
]

CONTEXT_BULLETS = [
    "[SAMRS resmi repo](https://github.com/ViTAE-Transformer/SAMRS), veri setinin SAM ve mevcut remote sensing detection veri setlerinden üretildiğini belirtir.",
    "[SAMRS NeurIPS 2023 makalesi](https://papers.nips.cc/paper_files/paper/2023/file/1be3843e534ee06d3a70c7f62b983b31-Paper-Datasets_and_Benchmarks.pdf), SAMRS'in 105.090 görüntü ve 1.668.241 instance içerdiğini raporlar.",
    "SOTA subset, DOTA v2.0 kaynaklı olduğu için kalabalık sahneler, küçük nesneler ve farklı ölçekler açısından iSAID vehicle deneyine yakın bir stres testi verir.",
    "Plane sınıfı vehicle'a göre daha az boxy olduğu için bbox-only prompt ile maskenin kanat/kuyruk gibi çıkıntıları ne kadar yakaladığı daha net görülür.",
]

DETECTOR_METRIC_NOTE = (
    "Not: Bu tablo YOLO'yu yalnızca detector olarak değerlendirir. Buradaki metrikler bbox metriğidir, "
    "maske metrikleri değildir."
)
SEGMENTATION_TABLES_NOTE = (
    "DOCX/PDF çıktılarında 0.0-1.0 aralığındaki başarı metrikleri kırmızıdan sarıya, "
    "sarıdan yeşile giden renk ölçeğiyle boyanır."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the SAMRS SOTA plane metric document.")
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "configs" / "yolo26x.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=STUDY_ROOT / "reports" / "samrs_sota_plane_full_metric_document.md",
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
        "üzerinden hesaplanan görüntü seviyesinde IoU eşik geçme oranlarıdır."
    )


def require_masks(mask_dir: Path, image_names: list[str]) -> None:
    missing = [name for name in image_names if not (mask_dir / f"{Path(name).stem}.png").exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{mask_dir} is missing {len(missing)} masks, for example: {preview}")


def configured_pipelines(config: dict) -> list[tuple[str, str, str, str]]:
    paths = config.get("paths", {})
    return [pipeline for pipeline in PIPELINES if pipeline[2] in paths]


def build_per_image_metrics(config: dict, pipelines: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    split = config["dataset"]["eval_split"]
    prepared_split_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split
    metadata = pd.read_csv(prepared_split_dir / "metadata.csv")
    gt_masks = load_ground_truth_masks(prepared_split_dir)
    image_names = [str(name) for name in metadata["file_name"].tolist()]

    rows: list[dict[str, object]] = []
    for pipeline, label, output_key, family in pipelines:
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


def summarize_group(
    metrics: pd.DataFrame,
    pipelines: list[tuple[str, str, str, str]],
    table_key: str,
    title: str,
    stratum: str | None,
) -> pd.DataFrame:
    frame = metrics if stratum is None else metrics[metrics["stratum"] == stratum]
    rows: list[dict[str, object]] = []
    for pipeline, label, _, family in pipelines:
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
    order = {pipeline: index for index, (pipeline, _, _, _) in enumerate(pipelines)}
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
        f"SAM2 YOLO bbox ile SAM1 YOLO bbox farkı `{format_signed(difference(overall, 'yolo_sam2', 'yolo_sam1', 'avg_iou'))}` Avg IoU.",
        f"SAM3 YOLO bbox ile SAM2 YOLO bbox farkı `{format_signed(difference(overall, 'yolo_sam3', 'yolo_sam2', 'avg_iou'))}` Avg IoU.",
        f"SAM3 hybrid YOLO bbox, SAM3 YOLO bbox hattına göre Avg IoU değerini `{format_signed(difference(overall, 'sam3_hybrid_yolo', 'yolo_sam3', 'avg_iou'))}` değiştirdi.",
        "Hybrid prompt bbox-only sonucunun garantili iyileştirmesi değildir. Text + bbox birlikte verildiğinde modelin maske üretim davranışı değişebilir.",
        "Plane sınıfında bbox maskeye tam oturmadığı için kanat ve kuyruk gibi çıkıntılar bbox promptunun sınırlarını zorlar. Bu, iSAID vehicle deneyinden farklı ve daha zor bir geometri testidir.",
        "Bu rapordaki GT maskeler SAMRS pseudo-mask olduğu için, sonuçlar insan çizimli nihai ground truth liderlik tablosu gibi sunulmamalı; model davranışı karşılaştırması olarak okunmalıdır.",
        "RingMo-SAM çıktısı bu SOTA plane kurulumunda çok düşük kaldı. RingMo semantic class-map için `class_ids: [5]` kullanımı doğrulandı; düşük skor bu yüzden boş maske id hatası olarak değil, modelin pseudo-mask hedefleriyle zayıf örtüşmesi olarak yorumlanmalıdır.",
    ]
    for table_key, title, _ in TABLES[1:]:
        summary = summaries[table_key]
        if summary.empty:
            continue
        best_row = summary.loc[summary["avg_iou"].idxmax()]
        lines.append(f"`{title}` grubunda en iyi Avg IoU `{best_row['pipeline_label']}` hattında `{metric(best_row['avg_iou'])}` olarak ölçüldü.")
    lines.extend(
        [
            "Low Mask Area tabloları küçük veya dar hedefleri gösterir. Bu kısımda düşen skorlar bbox ile maskenin ince detayları yakalayamadığına işaret edebilir.",
            "Overlap tabloları kalabalık sahnelerde birden fazla plane instance'ının birbirine yakın olduğu durumları ayırır. Burada FP/FN dengesi özellikle önemlidir.",
            "Segmentasyon mAP proxy kolonları görüntü seviyesinde eşik geçme oranıdır; COCO instance AP değildir. YOLO detector tablosundaki BBox mAP ise gerçek COCO bbox AP hesabıdır.",
        ]
    )
    return lines


def write_document(
    output_path: Path,
    summaries: dict[str, pd.DataFrame],
    detector_display: pd.DataFrame,
    tables_dir: Path,
) -> None:
    lines: list[str] = [
        f"# {DOCUMENT_TITLE}",
        "",
        "## Scope",
        "",
        *markdown_bullets(SCOPE_BULLETS),
        "",
        "## Metric Logic",
        "",
        *markdown_bullets(METRIC_BULLETS),
        "",
        "## Dataset Context",
        "",
        *markdown_bullets(CONTEXT_BULLETS),
        "",
        "## YOLO Detector BBox Metrics",
        "",
    ]
    if detector_display.empty:
        lines.append("_Dedektör metrik CSV'si bulunamadı. Bu tabloyu üretmek için `scripts/evaluate_yolo_detector_coco.py` çalıştırılmalı._")
    else:
        lines.extend([DETECTOR_METRIC_NOTE, "", markdown_table(detector_display)])

    lines.extend(["", "## Segmentation Tables", "", SEGMENTATION_TABLES_NOTE, ""])
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

    lines.extend(["## Discussion", "", *markdown_bullets(discussion_lines(summaries)), "", "## Generated Artifacts", ""])
    lines.extend(
        [
            f"- Görüntü bazlı metrik CSV'si: `{tables_dir.relative_to(output_path.parent) / PER_IMAGE_METRICS_NAME}`",
            f"- Özet tablo CSV'si: `{tables_dir.relative_to(output_path.parent) / LONG_SUMMARY_NAME}`",
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

    pipelines = configured_pipelines(config)
    metrics = build_per_image_metrics(config, pipelines)
    metrics.to_csv(tables_dir / PER_IMAGE_METRICS_NAME, index=False)

    summaries = {
        table_key: summarize_group(metrics, pipelines, table_key, title, stratum)
        for table_key, title, stratum in TABLES
    }
    display_frames: list[pd.DataFrame] = []
    for table_key, title, _ in TABLES:
        display_frame = display_summary(summaries[table_key]).copy()
        display_frame.insert(0, "Table", title)
        display_frames.append(display_frame)
    pd.concat(display_frames, ignore_index=True).to_csv(tables_dir / LONG_SUMMARY_NAME, index=False)
    for table_key, summary in summaries.items():
        summary.to_csv(raw_summaries_dir / f"summary_{table_key}.csv", index=False)

    detector_display = display_detector_metrics(args.detector_metrics)
    write_document(output_path, summaries, detector_display, tables_dir)
    print(output_path)


if __name__ == "__main__":
    main()
