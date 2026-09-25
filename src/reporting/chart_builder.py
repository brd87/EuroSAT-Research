import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reporting.utils as UTILS
import config


def draw_summary_table(ax, metrics_merged, metrics_merged_ranked):
    model_ids = list(metrics_merged["model_id"])
    keys = UTILS.scalar_keys(metrics_merged)

    ax.axis("off")
    col_labels = ["Model"] + [UTILS.METRIC_LABELS[k] for k in keys]
    cell_text = [[mid] + [UTILS.format_value(metrics_merged[k][i]) for k in keys]
                 for i, mid in enumerate(model_ids)]

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(col_labels))))
    table.scale(1, 1.6)

    for j, key in enumerate(keys):
        best, worst = UTILS.best_worst(metrics_merged_ranked, key)
        for i in range(len(model_ids)):
            cell = table[(i + 1, j + 1)]
            if i in best:
                cell.set_facecolor(UTILS.BEST_CELL)
            elif i in worst:
                cell.set_facecolor(UTILS.WORST_CELL)


def draw_metric_bars(ax, metrics_merged, metrics_merged_ranked, key):
    model_ids = list(metrics_merged["model_id"])
    n_models = len(model_ids)
    values = [0 if v != v else v for v in metrics_merged[key]]  # NaN (e.g. AUC) -> no bar
    best, worst = UTILS.best_worst(metrics_merged_ranked, key)
    colors = [UTILS.BEST_COLOR if i in best else UTILS.WORST_COLOR if i in worst else UTILS.DEFAULT_COLOR
              for i in range(n_models)]
    ax.bar(range(n_models), values, color=colors)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_ids, rotation=30, ha="right", fontsize=8)
    ax.set_title(UTILS.METRIC_LABELS[key], fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def bar_keys(metrics_merged, exclude=UTILS.BAR_EXCLUDED):
    return [k for k in UTILS.scalar_keys(metrics_merged) if k not in exclude]


def summary_table_figure(metrics_merged, metrics_merged_ranked,
                          title="Model Comparison — Summary Metrics"):
    fig, ax = plt.subplots(figsize=(11, 1 + 0.5 * len(metrics_merged["model_id"])))
    draw_summary_table(ax, metrics_merged, metrics_merged_ranked)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


def bar_charts_figure(metrics_merged, metrics_merged_ranked, exclude=UTILS.BAR_EXCLUDED, ncols=3):
    keys = bar_keys(metrics_merged, exclude)
    n = len(keys)
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, keys):
        draw_metric_bars(ax, metrics_merged, metrics_merged_ranked, key)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Per-Metric Comparison (green = best, red = worst)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def confusion_matrix_figures(metrics_merged, models_per_fig=4):
    model_ids = list(metrics_merged["model_id"])
    class_names = UTILS.extract_class_names(metrics_merged["classification_report"][0])

    figs = []
    n_models = len(model_ids)
    for start in range(0, n_models, models_per_fig):
        batch = list(range(start, min(start + models_per_fig, n_models)))
        fig, axes = plt.subplots(1, len(batch), figsize=(4.5 * len(batch), 4.2))
        axes = np.atleast_1d(axes).ravel()
        for ax, i in zip(axes, batch):
            cm = metrics_merged["confusion_matrix"][i]
            cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(class_names, fontsize=7)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.set_title(model_ids[i], fontsize=10, fontweight="bold")
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(c, r, int(cm[r, c]), ha="center", va="center", fontsize=7,
                            color="white" if cm_norm[r, c] > 0.5 else "black")
        fig.tight_layout()
        figs.append(fig)
    return figs


def text_figure(text, figsize=None):
    n_lines = text.count("\n") + 1
    fig, ax = plt.subplots(figsize=figsize or (11, max(4, 0.22 * n_lines)))
    ax.axis("off")
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=9.5, family="monospace",
            transform=ax.transAxes)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig
