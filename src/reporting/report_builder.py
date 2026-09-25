import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import reporting.chart_builder as chart_builder
import reporting.utils as UTILS
import config


def build_pdf(metrics_merged, metrics_merged_ranked, path):
    figs = [
        chart_builder.summary_table_figure(metrics_merged, metrics_merged_ranked),
        chart_builder.bar_charts_figure(metrics_merged, metrics_merged_ranked),
    ]
    figs.extend(chart_builder.confusion_matrix_figures(metrics_merged))
    figs.append(chart_builder.text_figure(conclusions_text(metrics_merged, metrics_merged_ranked)))

    with PdfPages(path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)
    return path


def build_png(metrics_merged, metrics_merged_ranked, path, dpi=150):
    # single-page dashboard: summary table + per-metric bars + verdict,
    # drawn by the same chart_builder functions as the PDF pages
    keys = chart_builder.bar_keys(metrics_merged)
    ncols = 3
    nrows = -(-len(keys) // ncols)
    verdict, _ = _verdict_and_conclusions(metrics_merged, metrics_merged_ranked)
    verdict_lines = textwrap.wrap(verdict, width=140)

    table_height = 1 + 0.4 * len(metrics_merged["model_id"])
    verdict_height = 0.4 + 0.25 * len(verdict_lines)
    fig = plt.figure(figsize=(4.2 * ncols, table_height + 3.2 * nrows + verdict_height + 0.8))
    grid = fig.add_gridspec(nrows + 2, ncols, height_ratios=[table_height] + [3.2] * nrows + [verdict_height])

    ax_table = fig.add_subplot(grid[0, :])
    chart_builder.draw_summary_table(ax_table, metrics_merged, metrics_merged_ranked)
    ax_table.set_title("Summary Metrics (green = best, red = worst)", fontsize=11, fontweight="bold")

    for idx, key in enumerate(keys):
        chart_builder.draw_metric_bars(fig.add_subplot(grid[1 + idx // ncols, idx % ncols]),
                                       metrics_merged, metrics_merged_ranked, key)

    ax_verdict = fig.add_subplot(grid[-1, :])
    ax_verdict.axis("off")
    ax_verdict.text(0.5, 0.5, "\n".join(verdict_lines), ha="center", va="center", fontsize=10, style="italic")

    fig.suptitle("Model Comparison Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def conclusions_text(metrics_merged, metrics_merged_ranked):
    model_ids = metrics_merged["model_id"]
    verdict, conclusions = _verdict_and_conclusions(metrics_merged, metrics_merged_ranked)

    lines = ["AUTOMATED CONCLUSIONS", "=" * 60, ""]
    lines.extend(textwrap.wrap(verdict, width=90))
    lines.append("")

    for i, mid in enumerate(model_ids):
        lines.append(f"[{mid}]")
        strengths = conclusions[mid]["strengths"]
        weaknesses = conclusions[mid]["weaknesses"]
        lines.extend(textwrap.wrap(
            "Strengths:  " + (", ".join(strengths) if strengths else "none stand out"),
            width=90, initial_indent="  ", subsequent_indent="              "))
        lines.extend(textwrap.wrap(
            "Weaknesses: " + (", ".join(weaknesses) if weaknesses else "none stand out"),
            width=90, initial_indent="  ", subsequent_indent="              "))

        acc = metrics_merged["accuracy"][i]
        base = metrics_merged["baseline"][i]
        if (acc - base) < 0.05:
            lines.append(f"  Note: accuracy ({acc:.4f}) is close to the majority-class "
                          f"baseline ({base:.4f}) — the model may not be learning much signal.")
        lines.append("")

    margin = f"{config.DEFAULT_REL_MARGIN:.0%}"
    excluded = ", ".join(UTILS.METRIC_LABELS[k] for k in UTILS.AVERAGE_RANK_EXCLUDED)
    lines.append(f"(Strength = best on a metric by {margin}+ over the runner-up; weakness = worst by {margin}+")
    lines.append(f" under the second-worst. Shown but not counted: {excluded} — it duplicates macro recall.)")
    return "\n".join(lines)


def _add_extremes(conclusions, model_ids, values, ranks, describe):
    # strength = best by a clear margin over the runner-up, weakness = worst by a clear margin
    # under the second-worst. Tied neighbours never pass the margin, so ties produce nothing.
    order = UTILS.leaderboard(ranks)

    best_i, runner_i = order[0], order[1]
    if UTILS.has_clear_margin(values, best_i, runner_i):
        conclusions[model_ids[best_i]]["strengths"].append(describe(best_i))

    worst_i, second_worst_i = order[-1], order[-2]
    if UTILS.has_clear_margin(values, worst_i, second_worst_i):
        conclusions[model_ids[worst_i]]["weaknesses"].append(describe(worst_i))


def _build_conclusions(metrics_merged, metrics_merged_ranked):
    model_ids = metrics_merged["model_id"]
    conclusions = {mid: {"strengths": [], "weaknesses": []} for mid in model_ids}

    for key in UTILS.compared_keys(metrics_merged, metrics_merged_ranked):
        values = metrics_merged[key]
        label = UTILS.METRIC_LABELS[key]
        _add_extremes(conclusions, model_ids, values, metrics_merged_ranked[key],
                      lambda i: f"{label} ({UTILS.format_value(values[i])})")

    reports = metrics_merged["classification_report"]
    cr_rank = metrics_merged_ranked["classification_report"]
    for cls in UTILS.extract_class_names(reports[0]):
        f1_values = [report[cls]["f1-score"] for report in reports]
        _add_extremes(conclusions, model_ids, f1_values, cr_rank[cls]["f1-score"],
                      lambda i: f"class '{cls}' F1")

    return conclusions


def _average_ranks(metrics_merged_ranked, keys, n_models):
    # ranks are aligned with models, so a model's average rank is just the mean of its own cells
    totals = [0] * n_models
    for key in keys:
        for i, rank in enumerate(metrics_merged_ranked[key]):
            totals[i] += rank
    return [total / len(keys) for total in totals]


def _overall_verdict(metrics_merged, metrics_merged_ranked):
    model_ids = metrics_merged["model_id"]
    keys = UTILS.compared_keys(metrics_merged, metrics_merged_ranked)
    averages = _average_ranks(metrics_merged_ranked, keys, len(model_ids))
    standings = UTILS.leaderboard(averages)
    leader_i, runner_i = standings[0], standings[1]

    # the average rank ignores how big the differences are, so the pick has to be backed by
    # metrics where the leader is ahead of the runner-up by a clear margin
    wins = losses = 0
    for key in keys:
        ranks = metrics_merged_ranked[key]
        leader_ahead = ranks[leader_i] < ranks[runner_i]
        better_i, worse_i = (leader_i, runner_i) if leader_ahead else (runner_i, leader_i)
        if UTILS.has_clear_margin(metrics_merged[key], better_i, worse_i):
            if leader_ahead:
                wins += 1
            else:
                losses += 1

    leader, runner = model_ids[leader_i], model_ids[runner_i]
    margin = f"{config.DEFAULT_REL_MARGIN:.0%}"
    if wins > losses:
        verdict = (f"Overall pick: {leader} — best average rank across {len(keys)} compared metrics, "
                   f"and clearly ahead of {runner} (by {margin}+) on {wins} of them vs. {losses} the other way.")
        if averages[runner_i] - averages[leader_i] < config.CLOSE_CALL_THRESHOLD:
            verdict += f" Close call: {runner} trails by only a thin margin in average rank."
    else:
        verdict = (f"No clear winner: {leader} has the best average rank, but is clearly ahead of {runner} "
                   f"(by {margin}+) on {wins} metrics vs. {losses} the other way — too close to call.")

    verdict += " Average rank (0 = best): " + ", ".join(
        f"{model_ids[i]} {averages[i]:.2f}" for i in standings) + "."
    return verdict


def _verdict_and_conclusions(metrics_merged, metrics_merged_ranked):
    # the one place that checks whether there is anything to compare (used by the PDF and the PNG)
    model_ids = metrics_merged["model_id"]
    if len(model_ids) > 1:
        return (_overall_verdict(metrics_merged, metrics_merged_ranked),
                _build_conclusions(metrics_merged, metrics_merged_ranked))
    return (f"Only one model ({model_ids[0]}) was evaluated — nothing to compare it against.",
            {model_ids[0]: {"strengths": [], "weaknesses": []}})