from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from .paths import DATASETS, MODELS, REFERENCES, REPO_ROOT, STRATA


REFERENCE_SHORT = {
    "human": "Human",
    "published_samrs_reference": "Published\nSAMRS",
    "reproduced_pseudo_sam1": "Reproduced\nSAM1",
    "pseudo_sam1": "SAM1\npseudo",
    "pseudo_sam2": "SAM2\npseudo",
    "pseudo_sam3": "SAM3\npseudo",
}
EXPERIMENT_SHORT = {
    "isaid_plane": "iSAID Plane",
    "isaid_small_vehicle": "iSAID Small Vehicle",
    "samrs_plane": "SAMRS Plane",
    "samrs_small_vehicle": "SAMRS Small Vehicle",
}
MODEL_COLORS = {"sam1": "#0072B2", "sam2": "#009E73", "sam3": "#D55E00"}
MODEL_MARKERS = {"sam1": "o", "sam2": "s", "sam3": "^"}
JOURNAL_WIDTH_IN = 6.2


def configure_style() -> None:
    plt.rcParams.update(
        {
            # Liberation Serif is the installed metric-compatible Times
            # substitute. The journal embeds the final vector PDF unchanged.
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.7,
            "lines.linewidth": 0.9,
            "lines.markersize": 5.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.03,
        }
    )


def save_figure(figure: plt.Figure, root: Path, stem: str) -> list[Path]:
    outputs: list[Path] = []
    for extension, dpi in (("pdf", 300), ("png", 320)):
        path = root / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def study_design_figure(root: Path) -> list[Path]:
    configure_style()
    figure, axis = plt.subplots(figsize=(JOURNAL_WIDTH_IN, 4.35))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 11.2)
    axis.axis("off")

    box_edge = "#4D4D4D"
    box_fill = "#F5F7F8"
    accent_fill = "#E8F1F7"
    stages = (
        (0.25, 7.15, 3.3, 2.55, "1  Data", "iSAID: human masks\nSAMRS: SAM-derived\nPlane + Small Vehicle\n512 images each"),
        (4.25, 7.15, 3.3, 2.55, "2  Localization", "GT bbox builds references\nYOLO bbox tests models\nSame image and instance\nfor every reference"),
        (8.25, 7.15, 3.3, 2.55, "3  Frozen models", "SAM1 / SAM2 / SAM3\nNo fine-tuning\nPredictions reused for\nevery reference mask"),
        (12.25, 7.15, 3.5, 2.55, "4  Scoring", "Fixed prediction versus\nbaseline + SAM1/2/3\nInstance-macro IoU\nNo test-time changes"),
    )
    for index, (x, y, width, height, title, body) in enumerate(stages):
        axis.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=accent_fill if index in {0, 3} else box_fill,
                edgecolor=box_edge,
                linewidth=0.8,
            )
        )
        axis.text(
            x + 0.18,
            y + height - 0.32,
            title,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=7.25,
        )
        axis.text(
            x + 0.18,
            y + height - 0.88,
            body,
            ha="left",
            va="top",
            fontsize=6.55,
            linespacing=1.22,
        )
    for left, right in zip(stages[:-1], stages[1:], strict=True):
        axis.add_patch(
            FancyArrowPatch(
                (left[0] + left[2] + 0.10, 8.43),
                (right[0] - 0.10, 8.43),
                arrowstyle="-|>",
                mutation_scale=9,
                color=box_edge,
                linewidth=0.8,
            )
        )

    axis.add_patch(
        Rectangle(
            (0.25, 4.65),
            15.5,
            1.25,
            facecolor="#FFF7E6",
            edgecolor="#B07A20",
            linewidth=0.8,
        )
    )
    axis.text(
        8.0,
        5.28,
        "Controlled comparison: image, instance, bbox and prediction stay fixed.\nOnly the evaluation mask changes.",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        linespacing=1.2,
    )

    axis.text(0.25, 3.75, "Primary quantity", ha="left", va="center", fontweight="bold", fontsize=8.5)
    axis.text(
        0.25,
        2.95,
        "Extra IoU on own label",
        ha="left",
        va="center",
        color="#1F4E79",
        fontweight="bold",
        fontsize=9.1,
    )
    axis.text(5.0, 2.95, "=", ha="center", va="center", fontsize=11, fontweight="bold")
    axis.text(6.0, 2.95, "IoU on the model's own label", ha="left", va="center", fontsize=8.2)
    axis.text(11.0, 2.95, "−", ha="center", va="center", fontsize=11, fontweight="bold")
    axis.text(
        11.75,
        2.95,
        "mean IoU on labels from\nthe other two SAM models",
        ha="left",
        va="center",
        fontsize=8.2,
        linespacing=1.12,
    )
    axis.text(
        0.25,
        1.55,
        "Interpretation",
        ha="left",
        va="center",
        fontweight="bold",
        fontsize=8.5,
    )
    axis.text(
        2.9,
        1.55,
        "Positive value: the fixed prediction scores higher with labels made by its own model.",
        ha="left",
        va="center",
        fontsize=7.3,
    )
    return save_figure(figure, root, "figure_1_study_design")


def isaid_own_label_comparison_figure(root: Path) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(JOURNAL_WIDTH_IN, 3.35), sharex=True, sharey=True)
    for panel_index, (axis, experiment_id) in enumerate(
        zip(axes, ("isaid_plane", "isaid_small_vehicle"), strict=True)
    ):
        source = DATASETS[experiment_id]
        aggregates = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
        selected = aggregates[
            (aggregates["stratum"] == "overall")
            & (aggregates["bbox_source"] == "yolo_bbox")
        ]
        pseudo_references = list(source.reference_types[1:])
        y_positions = np.arange(len(MODELS))[::-1]
        for y_position, model in zip(y_positions, MODELS, strict=True):
            model_rows = selected[selected["model"] == model].set_index("reference_type")
            own_reference = f"pseudo_{model}"
            other_references = [value for value in pseudo_references if value != own_reference]
            own_iou = float(model_rows.loc[own_reference, "mean_iou"])
            other_iou = float(model_rows.loc[other_references, "mean_iou"].astype(float).mean())
            axis.plot(
                [other_iou, own_iou],
                [y_position, y_position],
                color="#8A8A8A",
                linewidth=1.2,
                zorder=1,
            )
            axis.scatter(
                other_iou,
                y_position,
                s=32,
                marker="o",
                facecolor="white",
                edgecolor="#666666",
                linewidth=0.9,
                zorder=3,
            )
            axis.scatter(
                own_iou,
                y_position,
                s=38,
                marker="o",
                facecolor="#0072B2",
                edgecolor="#004B73",
                linewidth=0.7,
                zorder=4,
            )
            axis.annotate(
                f"+{own_iou - other_iou:.3f}",
                xy=(own_iou, y_position),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=7.5,
                fontweight="bold",
                color="#1F4E79",
            )
        axis.set_title(f"({chr(97 + panel_index)}) {EXPERIMENT_SHORT[experiment_id]}", loc="left", fontweight="bold")
        axis.set_yticks(y_positions, [model.upper() for model in MODELS])
        axis.set_xlim(0.45, 0.94)
        axis.set_xlabel("Average instance IoU")
        axis.grid(axis="x", color="#D6D6D6", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    legend = (
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#666666", label="Mean on the other two SAM labels"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#0072B2", markeredgecolor="#004B73", label="IoU on the model's own label"),
    )
    figure.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.015))
    figure.text(
        0.5,
        0.015,
        "Number beside the filled marker = extra IoU obtained on the model's own label.",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.90), w_pad=1.5)
    return save_figure(figure, root, "figure_2_isaid_own_label_comparison")


def reference_dependent_winner_figure(root: Path) -> list[Path]:
    configure_style()
    figure, axis = plt.subplots(figsize=(JOURNAL_WIDTH_IN, 3.55))
    axis.set_xlim(-0.55, 4.0)
    axis.set_ylim(-0.6, 4.3)
    axis.axis("off")
    column_labels = ("Baseline reference", "SAM1 label", "SAM2 label", "SAM3 label")
    for column, label in enumerate(column_labels):
        axis.text(column + 0.5, 4.05, label, ha="center", va="center", fontweight="bold", fontsize=7.8)
    baseline_labels = {
        "isaid_plane": "human",
        "isaid_small_vehicle": "human",
        "samrs_plane": "published SAMRS",
        "samrs_small_vehicle": "published SAMRS",
    }
    model_fills = {"sam1": "#DCECF6", "sam2": "#DDEFE8", "sam3": "#F8E4D8"}
    for row_index, (experiment_id, source) in enumerate(DATASETS.items()):
        y = 3.15 - row_index
        aggregates = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
        selected = aggregates[
            (aggregates["stratum"] == "overall")
            & (aggregates["bbox_source"] == "yolo_bbox")
        ]
        reference_columns = (source.reference_types[0], *source.reference_types[1:])
        axis.text(
            -0.12,
            y + 0.35,
            EXPERIMENT_SHORT[experiment_id],
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=7.8,
        )
        axis.text(
            -0.12,
            y + 0.06,
            f"({baseline_labels[experiment_id]} baseline)",
            ha="right",
            va="center",
            fontsize=6.8,
            color="#555555",
        )
        for column, reference_type in enumerate(reference_columns):
            reference_rows = selected[selected["reference_type"] == reference_type]
            winner = reference_rows.loc[reference_rows["mean_iou"].idxmax()]
            model = str(winner["model"])
            score = float(winner["mean_iou"])
            axis.add_patch(
                Rectangle(
                    (column + 0.06, y),
                    0.88,
                    0.72,
                    facecolor=model_fills[model],
                    edgecolor=MODEL_COLORS[model],
                    linewidth=0.8,
                )
            )
            axis.text(
                column + 0.5,
                y + 0.45,
                model.upper(),
                ha="center",
                va="center",
                color="#222222",
                fontweight="bold",
                fontsize=8.2,
            )
            axis.text(
                column + 0.5,
                y + 0.20,
                f"IoU {score:.3f}",
                ha="center",
                va="center",
                color="#333333",
                fontsize=7.1,
            )
    axis.text(
        1.95,
        -0.33,
        "Each cell shows the highest-scoring frozen model for that evaluation reference.",
        ha="center",
        va="center",
        fontsize=7.5,
    )
    return save_figure(figure, root, "figure_3_reference_dependent_model_selection")


def stratified_own_label_figure(root: Path) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(JOURNAL_WIDTH_IN, 4.15), sharex=True, sharey=True)
    stratum_labels = (
        "No overlap / Low area",
        "No overlap / High area",
        "Overlap / Low area",
        "Overlap / High area",
    )
    offsets = {"sam1": 0.18, "sam2": 0.0, "sam3": -0.18}
    for panel_index, (axis, experiment_id) in enumerate(
        zip(axes, ("isaid_plane", "isaid_small_vehicle"), strict=True)
    ):
        source = DATASETS[experiment_id]
        contrasts = pd.read_csv(source.analysis_root / "paired_teacher_affinity_contrasts.csv")
        selected = contrasts[contrasts["bbox_source"] == "yolo_bbox"]
        base_positions = np.arange(4)[::-1]
        for model in MODELS:
            values: list[float] = []
            lowers: list[float] = []
            uppers: list[float] = []
            for stratum in STRATA[1:]:
                row = selected[
                    (selected["model"] == model)
                    & (selected["stratum"] == stratum)
                ].iloc[0]
                values.append(float(row["self_vs_cross_iou"]))
                lowers.append(float(row["self_vs_cross_ci_lower"]))
                uppers.append(float(row["self_vs_cross_ci_upper"]))
            values_array = np.asarray(values)
            positions = base_positions + offsets[model]
            errors = np.vstack((values_array - np.asarray(lowers), np.asarray(uppers) - values_array))
            axis.errorbar(
                values_array,
                positions,
                xerr=errors,
                fmt=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                markerfacecolor=MODEL_COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.5,
                capsize=2,
                elinewidth=0.8,
                label=model.upper(),
                zorder=3,
            )
        axis.axvline(0, color="#444444", linewidth=0.7)
        axis.set_title(f"({chr(97 + panel_index)}) {EXPERIMENT_SHORT[experiment_id]}", loc="left", fontweight="bold")
        axis.set_yticks(base_positions, stratum_labels)
        axis.set_xlim(0, 0.19)
        axis.set_xlabel("Extra IoU on the model's own label")
        axis.grid(axis="x", color="#D6D6D6", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            markerfacecolor=MODEL_COLORS[model],
            linestyle="none",
            label=model.upper(),
        )
        for model in MODELS
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.015))
    figure.text(
        0.5,
        0.01,
        "Markers show the estimate; horizontal bars show the scene-clustered 95% confidence interval.",
        ha="center",
        va="bottom",
        fontsize=7.45,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.91), w_pad=1.2)
    return save_figure(figure, root, "figure_4_stratified_own_label_extra_iou")


def design_table() -> pd.DataFrame:
    rows = []
    for experiment_id, source in DATASETS.items():
        rows.append(
            {
                "Experiment": EXPERIMENT_SHORT[experiment_id],
                "Target": source.target_label,
                "Images": 512,
                "Instances": source.instance_count,
                "Strata": "4 × 128",
                "Baseline reference": REFERENCE_SHORT[source.reference_types[0]].replace("\n", " "),
                "Independent human": "yes" if source.dataset_family == "isaid" else "no",
            }
        )
    return pd.DataFrame(rows)


def control_table() -> pd.DataFrame:
    rows = []
    for experiment_id, source in DATASETS.items():
        aggregates = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
        baseline = source.reference_types[0]
        selected = aggregates[
            (aggregates["stratum"] == "overall")
            & (aggregates["bbox_source"] == "yolo_bbox")
            & (aggregates["reference_type"] == baseline)
        ]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "Experiment": EXPERIMENT_SHORT[experiment_id],
                    "Model": str(row["model"]).upper(),
                    "BBox source": "YOLO bbox",
                    "Baseline reference": REFERENCE_SHORT[baseline].replace("\n", " "),
                    "Avg IoU": float(row["mean_iou"]),
                    "Avg Dice": float(row["mean_dice"]),
                    "Avg Precision": float(row["mean_precision"]),
                    "Avg Recall": float(row["mean_recall"]),
                }
            )
    return pd.DataFrame(rows)


def own_effect_table() -> pd.DataFrame:
    rows = []
    for experiment_id, source in DATASETS.items():
        effects = pd.read_csv(source.analysis_root / "paired_reference_effects.csv")
        own_reference = {
            "sam1": "pseudo_sam1" if source.dataset_family == "isaid" else "reproduced_pseudo_sam1",
            "sam2": "pseudo_sam2",
            "sam3": "pseudo_sam3",
        }
        selected = effects[effects["bbox_source"] == "yolo_bbox"]
        for model in MODELS:
            row = selected[
                (selected["model"] == model)
                & (selected["comparison_reference"] == own_reference[model])
            ].iloc[0]
            rows.append(
                {
                    "Experiment": EXPERIMENT_SHORT[experiment_id],
                    "Model": model.upper(),
                    "Baseline reference": REFERENCE_SHORT[
                        source.reference_types[0]
                    ].replace("\n", " "),
                    "Baseline-reference IoU": float(row["baseline_mean_iou"]),
                    "Own-label IoU": float(row["comparison_mean_iou"]),
                    "Score change": float(row["delta_iou"]),
                }
            )
    return pd.DataFrame(rows)


def direct_affinity_table() -> pd.DataFrame:
    rows = []
    # The main direct-affinity table is restricted to experiments with an
    # independent human control. SAMRS has a SAM1-derived published baseline
    # and is reported separately as reference-integrity evidence.
    for experiment_id in ("isaid_plane", "isaid_small_vehicle"):
        source = DATASETS[experiment_id]
        aggregates = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
        selected = aggregates[
            (aggregates["bbox_source"] == "yolo_bbox")
            & (aggregates["stratum"] == "overall")
        ]
        pseudo_references = list(source.reference_types[1:])
        for model in MODELS:
            model_rows = selected[selected["model"] == model].set_index(
                "reference_type"
            )
            own_reference = f"pseudo_{model}"
            other_references = [
                value for value in pseudo_references if value != own_reference
            ]
            own_iou = float(model_rows.loc[own_reference, "mean_iou"])
            other_iou = float(
                model_rows.loc[other_references, "mean_iou"].astype(float).mean()
            )
            rows.append(
                {
                    "Experiment": EXPERIMENT_SHORT[experiment_id],
                    "Model": model.upper(),
                    "Own-label IoU": own_iou,
                    "Other-SAM-label mean IoU": other_iou,
                    "Extra IoU": own_iou - other_iou,
                }
            )
    return pd.DataFrame(rows)


def samrs_integrity_table() -> pd.DataFrame:
    rows = []
    for experiment_id in ("samrs_plane", "samrs_small_vehicle"):
        source = DATASETS[experiment_id]
        agreement = pd.read_csv(source.analysis_root / "reference_agreement.csv")
        row = agreement[
            (agreement["reference_a"] == "published_samrs_reference")
            & (agreement["reference_b"] == "reproduced_pseudo_sam1")
        ].iloc[0]
        rows.append(
            {
                "Experiment": EXPERIMENT_SHORT[experiment_id],
                "Published vs reproduced SAM1 IoU": float(row["mean_instance_iou"]),
                "Instances": int(row["instance_count"]),
                "Interpretation": "SAM1-like published reference; not human GT",
            }
        )
    return pd.DataFrame(rows)


def detector_table() -> pd.DataFrame:
    frames = []
    for experiment_id, source in DATASETS.items():
        row = pd.read_csv(source.analysis_root / "detector_summary.csv").iloc[0]
        frames.append(
            {
                "Experiment": EXPERIMENT_SHORT[experiment_id],
                "Seed": 42,
                "Evaluation scope": "512 target-positive images",
                "BBox mAP50": float(row["bbox_AP50_mean"]),
                "BBox mAP75": float(row["bbox_AP75_mean"]),
                "BBox mAP90": float(row["bbox_AP90_mean"]),
                "BBox mAP50-95": float(row["bbox_AP50_95_mean"]),
                "Precision@0.50": float(row["precision_at_bbox_iou50_mean"]),
                "Recall@0.50": float(row["recall_at_bbox_iou50_mean"]),
            }
        )
    return pd.DataFrame(frames)


def strata_table() -> pd.DataFrame:
    rows = []
    for experiment_id, source in DATASETS.items():
        aggregates = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
        selected = aggregates[aggregates["bbox_source"] == "yolo_bbox"]
        pseudo_references = [
            reference
            for reference in source.reference_types
            if reference not in {"human", "published_samrs_reference"}
        ]
        for stratum in STRATA[1:]:
            for model in MODELS:
                model_rows = selected[
                    (selected["stratum"] == stratum)
                    & (selected["model"] == model)
                ].set_index("reference_type")
                own_reference = (
                    "reproduced_pseudo_sam1"
                    if source.dataset_family == "samrs" and model == "sam1"
                    else f"pseudo_{model}"
                )
                other_references = [
                    reference
                    for reference in pseudo_references
                    if reference != own_reference
                ]
                own_iou = float(model_rows.loc[own_reference, "mean_iou"])
                other_iou = float(
                    model_rows.loc[other_references, "mean_iou"].astype(float).mean()
                )
                rows.append(
                    {
                        "Experiment": EXPERIMENT_SHORT[experiment_id],
                        "Scene group": stratum.replace("__", " × ").replace("_", " "),
                        "Model": model.upper(),
                        "Own-label IoU": own_iou,
                        "Other-SAM-label mean IoU": other_iou,
                        "Extra IoU": own_iou - other_iou,
                    }
                )
    return pd.DataFrame(rows)


def _latex_escape(value: object) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(source, target)
    return text


def _latex_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.3f}"
    return _latex_escape(value)


def write_table(frame: pd.DataFrame, root: Path, stem: str) -> list[Path]:
    csv_path = root / f"{stem}.csv"
    tex_path = root / f"{stem}.tex"
    frame.to_csv(csv_path, index=False, float_format="%.3f")
    headers = " & ".join(_latex_escape(column) for column in frame.columns)
    rows = [
        " & ".join(_latex_value(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]
    tex_path.write_text(
        "\n".join(
            [
                r"\begin{tabular}{" + "l" * len(frame.columns) + "}",
                r"\toprule",
                headers + r" \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return [csv_path, tex_path]


def generate_assets(study_root: Path) -> dict[str, list[Path]]:
    assets_root = study_root / "paper_writing" / "assets"
    figures_root = assets_root / "figures"
    tables_root = assets_root / "tables"
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    for path in (*figures_root.glob("*"), *tables_root.glob("*")):
        if path.is_file():
            path.unlink()
    figures = [
        *study_design_figure(figures_root),
        *isaid_own_label_comparison_figure(figures_root),
        *reference_dependent_winner_figure(figures_root),
        *stratified_own_label_figure(figures_root),
    ]
    tables = [
        *write_table(design_table(), tables_root, "table_1_experimental_design"),
        *write_table(control_table(), tables_root, "table_2_baseline_reference_results"),
        *write_table(
            direct_affinity_table(),
            tables_root,
            "table_3_direct_teacher_affinity",
        ),
        *write_table(
            own_effect_table(),
            tables_root,
            "table_4_raw_own_reference_effect",
        ),
        *write_table(
            samrs_integrity_table(),
            tables_root,
            "table_5_samrs_reference_integrity",
        ),
        *write_table(detector_table(), tables_root, "table_6_detector_control"),
        *write_table(
            strata_table(),
            tables_root,
            "table_s1_stratified_reference_effect",
        ),
    ]
    inputs = [
        source.analysis_root / filename
        for source in DATASETS.values()
        for filename in (
            "aggregate_metrics.csv",
            "paired_reference_effects.csv",
            "paired_teacher_affinity_contrasts.csv",
            "reference_agreement.csv",
            "detector_summary.csv",
        )
    ]

    def record(path: Path) -> dict[str, object]:
        return {
            "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest_path = assets_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status": "completed",
                "scope": "four_experiments_no_cross_dataset_pooling",
                "inputs": [record(path) for path in inputs],
                "outputs": [record(path) for path in (*figures, *tables)],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"figures": figures, "tables": tables, "manifest": [manifest_path]}
