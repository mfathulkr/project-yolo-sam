from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3_bbox_study.config import load_config, resolve_path


PIPELINE_ORDER = [
    "sam3_text",
    "remotesam_text",
    "segearth_ov3",
    "yolo_sam3",
    "gt_box_sam3",
    "yolo_sam2",
    "grounded_sam2",
    "yolo_ringmo_sam",
    "gt_box_ringmo_sam",
]
STRATUM_ORDER = [
    "no_overlap__low_mask_area",
    "no_overlap__high_mask_area",
    "overlap__low_mask_area",
    "overlap__high_mask_area",
]
STRATUM_LABELS = {
    "no_overlap__low_mask_area": "No overlap / Low area",
    "no_overlap__high_mask_area": "No overlap / High area",
    "overlap__low_mask_area": "Overlap / Low area",
    "overlap__high_mask_area": "Overlap / High area",
}
STRATUM_FILL_COLORS = {
    "No overlap / Low area": "#F4F8FF",
    "No overlap / High area": "#EEF7F1",
    "Overlap / Low area": "#FFF5E8",
    "Overlap / High area": "#F7F0FF",
}
SHORT_PIPELINE_LABELS = {
    "SAM3 text-only": "SAM3 text",
    "RemoteSAM text": "RemoteSAM",
    "SegEarth-OV3 + SAM3": "SegEarth+SAM3",
    "YOLO + SAM3": "YOLO+SAM3",
    "GT bbox + SAM3": "GT+SAM3",
    "YOLO + SAM2": "YOLO+SAM2",
    "GroundingDINO + SAM2": "GroundDINO+SAM2",
    "YOLO + RingMo-SAM": "YOLO+RingMo",
    "GT bbox + RingMo-SAM": "GT+RingMo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the iSAID presentation as a Linux-viewable PDF.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaid_vehicle_yolo26x_cpu_eval.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "presentation_isaid_vehicle_sam3_sam2_study" / "isaid_vehicle_sam3_sam2_summary.pdf",
    )
    return parser.parse_args()


def metric(value: float) -> str:
    return f"{value:.4f}"


def hms(seconds: float) -> str:
    seconds_int = int(round(seconds))
    hours, rem = divmod(seconds_int, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h {minutes}m {secs}s"


def new_slide(pdf: PdfPages, title: str, subtitle: str | None = None):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    fig.text(0.04, 0.94, title, fontsize=24, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.04, 0.895, subtitle, fontsize=10.5, va="top")
    return fig


def add_bullets(fig, bullets: list[str], x: float, y: float, size: int = 16, line_height: float = 0.065) -> None:
    for index, bullet in enumerate(bullets):
        fig.text(x, y - index * line_height, f"- {bullet}", fontsize=size, va="top")


def add_table(
    fig,
    df: pd.DataFrame,
    rect: list[float],
    font_size: int = 8,
    scale_y: float = 1.2,
    group_column: str | None = None,
) -> None:
    ax = fig.add_axes(rect)
    ax.axis("off")
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=list(df.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, scale_y)
    group_col_index = list(df.columns).index(group_column) if group_column in df.columns else None
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f1f3f5")
        elif group_col_index is not None:
            group_value = str(df.iloc[row - 1, group_col_index])
            cell.set_facecolor(STRATUM_FILL_COLORS.get(group_value, "#ffffff"))
        cell.set_linewidth(0.35)


def add_image(fig, image_path: Path, rect: list[float]) -> None:
    ax = fig.add_axes(rect)
    ax.imshow(Image.open(image_path))
    ax.axis("off")


def save(fig, pdf: PdfPages) -> None:
    pdf.savefig(fig)
    plt.close(fig)


def make_overall_slide(pdf: PdfPages, overall: pd.DataFrame, figures_dir: Path) -> None:
    fig = new_slide(pdf, "Overall Results", "Mean scores over 128 eval images")
    add_image(fig, figures_dir / "overall_metrics.png", [0.04, 0.29, 0.50, 0.50])
    table = overall[["pipeline_label", "mean_iou", "mean_dice", "mean_precision", "mean_recall"]].copy()
    table["pipeline_label"] = table["pipeline_label"].map(lambda label: SHORT_PIPELINE_LABELS.get(label, label))
    for column in ["mean_iou", "mean_dice", "mean_precision", "mean_recall"]:
        table[column] = table[column].map(metric)
    table.columns = ["Pipeline", "IoU", "Dice", "Prec.", "Recall"]
    add_table(fig, table, [0.55, 0.47, 0.43, 0.30], font_size=8.2, scale_y=1.18)
    add_bullets(
        fig,
        [
            "Detection-guided SAM improves over text-only in every stratum.",
            "GT boxes + SAM3 is the upper-bound detection case.",
            "YOLO + SAM2 is slightly ahead of YOLO + SAM3 overall in this run.",
            "RemoteSAM text is the strongest box-free text/referring baseline.",
            "SegEarth-OV3 tests SAM3 semantic/instance fusion without boxes.",
            "GroundingDINO + SAM2 is a zero-shot text-to-box baseline.",
            "RingMo-SAM is a remote-sensing fine-tuned box-constrained baseline.",
        ],
        x=0.58,
        y=0.42,
        size=11,
        line_height=0.043,
    )
    save(fig, pdf)


def make_stratified_slide(pdf: PdfPages, figures_dir: Path) -> None:
    fig = new_slide(pdf, "Stratified View", "2x2 split: bbox overlap yes/no x target mask area low/high")
    add_image(fig, figures_dir / "mean_iou_by_stratum.png", [0.03, 0.23, 0.46, 0.61])
    add_image(fig, figures_dir / "mean_dice_by_stratum.png", [0.51, 0.23, 0.46, 0.61])
    add_bullets(
        fig,
        [
            "Low-mask-area strata are the hardest: small vehicles punish over-segmentation and missed detections.",
            "High-mask-area strata give guided methods more signal and higher IoU/Dice.",
        ],
        x=0.06,
        y=0.15,
        size=12,
        line_height=0.045,
    )
    save(fig, pdf)


def sample_path(row: pd.Series, vis_dir: Path) -> Path:
    if "visualization" in row and isinstance(row["visualization"], str) and row["visualization"]:
        path = ROOT / row["visualization"]
        if path.exists():
            return path
    return vis_dir / f"{row['stratum']}__{Path(row['file_name']).stem}.png"


def make_samples_slide(pdf: PdfPages, stratum: str, selected: pd.DataFrame, vis_dir: Path) -> None:
    title = stratum.replace("__", " / ").replace("_", " ")
    fig = new_slide(
        pdf,
        title,
        "Curated qualitative example. Card shows full context, zoomed GT boxes, and model masks.",
    )
    rows = selected[selected["stratum"] == stratum].sort_values("file_name")
    if len(rows) <= 1:
        for _, row in rows.iterrows():
            add_image(fig, sample_path(row, vis_dir), [0.145, 0.08, 0.71, 0.80])
    else:
        top_positions = [0.50, 0.09]
        for top, (_, row) in zip(top_positions, rows.iterrows(), strict=False):
            add_image(fig, sample_path(row, vis_dir), [0.035, top, 0.93, 0.38])
    save(fig, pdf)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    presentation_dir = args.output.parent
    figures_dir = presentation_dir / "figures"
    metrics_dir = resolve_path(config["paths"]["sam3_triplet_metrics_dir"])
    vis_dir = resolve_path(config["paths"]["sam3_triplet_visualizations_dir"])

    overall = pd.read_csv(metrics_dir / "summary_overall_stratified.csv")
    overall["pipeline"] = pd.Categorical(overall["pipeline"], PIPELINE_ORDER, ordered=True)
    overall = overall.sort_values("pipeline")

    by_stratum = pd.read_csv(metrics_dir / "summary_by_stratum.csv")
    by_stratum["stratum"] = pd.Categorical(by_stratum["stratum"], STRATUM_ORDER, ordered=True)
    by_stratum["pipeline"] = pd.Categorical(by_stratum["pipeline"], PIPELINE_ORDER, ordered=True)
    by_stratum = by_stratum.sort_values(["stratum", "pipeline"])

    curated_dir = ROOT / "results" / "isaid_vehicle_visualizations_sam3_triplet_curated"
    curated_csv = curated_dir / "selected_curated_samples.csv"
    selected = pd.read_csv(curated_csv if curated_csv.exists() else vis_dir / "selected_qualitative_samples.csv")
    metadata = pd.read_csv(resolve_path(config["paths"]["prepared_dataset_dir"]) / config["dataset"]["eval_split"] / "metadata.csv")
    area_threshold = float(metadata["mask_area_ratio"].median())
    yolo = pd.read_csv(resolve_path(config["paths"]["yolo_train_run_dir"]) / "train" / "results.csv")
    yolo.columns = [column.strip() for column in yolo.columns]
    best = yolo.loc[yolo["metrics/mAP50-95(B)"].idxmax()]
    last = yolo.iloc[-1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        fig = new_slide(pdf, "iSAID Vehicle: Text SAM vs Detection-Guided SAM")
        add_bullets(
            fig,
            [
                "Goal: compare text-only, detector-guided, oracle-box, zero-shot, and remote-sensing fine-tuned SAM-style variants.",
                "Target: merged vehicle class from iSAID Small_Vehicle + Large_Vehicle.",
                "Eval design: 128 overhead urban tiles, 32 per overlap / mask-area stratum.",
                f"Low/high area split uses eval median mask-area ratio {area_threshold:.6f}.",
                "Final inference was CPU-only; no GPU was used during this reporting run.",
            ],
            x=0.06,
            y=0.78,
            size=17,
            line_height=0.07,
        )
        yolo_table = pd.DataFrame(
            [
                ["Best epoch", int(best["epoch"])],
                ["Stopped epoch", int(last["epoch"])],
                ["Elapsed training", hms(float(last["time"]))],
                ["mAP50 / mAP50-95", f"{metric(float(best['metrics/mAP50(B)']))} / {metric(float(best['metrics/mAP50-95(B)']))}"],
                ["Precision / Recall", f"{metric(float(best['metrics/precision(B)']))} / {metric(float(best['metrics/recall(B)']))}"],
            ],
            columns=["YOLO26x checkpoint", "Value"],
        )
        add_table(fig, yolo_table, [0.08, 0.12, 0.52, 0.28], font_size=12, scale_y=1.4)
        save(fig, pdf)

        make_overall_slide(pdf, overall, figures_dir)
        make_stratified_slide(pdf, figures_dir)

        compact = by_stratum[["stratum", "pipeline_label", "mean_iou", "mean_dice"]].copy()
        compact["stratum"] = compact["stratum"].astype(str).map(STRATUM_LABELS)
        for column in ["mean_iou", "mean_dice"]:
            compact[column] = compact[column].map(metric)
        compact.columns = ["Stratum", "Pipeline", "IoU", "Dice"]
        fig = new_slide(pdf, "Stratified Metrics Table")
        add_table(fig, compact, [0.03, 0.06, 0.94, 0.86], font_size=4.9, scale_y=1.12, group_column="Stratum")
        save(fig, pdf)

        for stratum in STRATUM_ORDER:
            make_samples_slide(pdf, stratum, selected, vis_dir)

        fig = new_slide(pdf, "Takeaways")
        add_bullets(
            fig,
            [
                "YOLO guidance matters: YOLO + SAM3 beats SAM3 text-only in every 2x2 stratum.",
                "RemoteSAM text narrows the box-free gap but remains below YOLO + SAM2 overall.",
                "SegEarth-OV3 increases recall but tends to over-mask low-area scenes.",
                "Detection quality matters: GT bbox + SAM3 remains the best IoU/Dice case overall and by stratum.",
                "SAM2 is competitive with the same YOLO boxes and slightly better overall here.",
                "GroundingDINO + SAM2 tests text-to-box prompting separately from YOLO training.",
                "RingMo-SAM tests a remote-sensing fine-tuned decoder under fixed YOLO/GT boxes.",
                "Text-only SAM3 over-segments small-object scenes, especially low-mask-area tiles.",
                "The selected examples are balanced across overlap/no-overlap and low/high total vehicle mask area.",
            ],
            x=0.07,
            y=0.77,
            size=15,
            line_height=0.06,
        )
        save(fig, pdf)

    print(f"Wrote PDF: {args.output}")


if __name__ == "__main__":
    main()
