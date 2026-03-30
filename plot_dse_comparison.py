#!/usr/bin/env python3
"""
Generate DSE comparison scatter plots for Ours, NSGA-II, TREAD-M3D, and VAESA.

Usage:
    python plot_dse_comparison.py --dnn resnet18 --T 80
    python plot_dse_comparison.py --dnn resnet50 --T 70 --out my_plot.png

Layout: 2 rows x 2 cols (one per algorithm), single objective per figure.
Each subplot: scatter of all evaluations (green=feasible, red=infeasible)
              + dark navy running-best step line + best value annotation.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Liberation Serif", "Times New Roman"],
    "font.size":       14,
    "axes.titlesize":  14,
    "axes.labelsize":  14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
})
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_ready")


def load_progress(path, bo_phases=("bo", "nsga2", "msa", None)):
    """Load all progress rows from a result directory (multi- or single-file)."""
    files = sorted(glob.glob(os.path.join(path, "progress_*.jsonl")))
    if not files:
        single = os.path.join(path, "progress.jsonl")
        if os.path.exists(single):
            files = [single]
    rows = []
    for f in files:
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return [r for r in rows if r.get("phase") in bo_phases and r.get("metric") is not None]


def find_result_dir(algo, dnn, obj, T_celsius):
    """Locate the result directory for a given experiment.

    Search order:
      1. all_results/turbo_m2/<dnn>_<obj>_<T>c        (turbo_m2 local runs)
      2. all_results/<algo>/<dnn>_<obj>_<T>c           (local single-algo dirs)
      3. all_results/baselines/<algo>_<dnn>_<obj>_<T>c (remote baseline runs)
    """
    candidates = []
    if algo == "turbo_logcei_explorer":
        candidates = [
            os.path.join(RESULTS_ROOT, "turbo", f"{dnn}_{obj}_{T_celsius}c"),
        ]
    else:
        candidates = [
            os.path.join(RESULTS_ROOT, algo, f"{dnn}_{obj}_{T_celsius}c"),
        ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def running_best(rows, T_thresh):
    """Return (iters, running_best_values) — None where no feasible seen yet."""
    best = None
    xs, ys = [], []
    for r in rows:
        m = r.get("metric")
        t = r.get("T_max", 9999)
        xs.append(r["iter"])
        if t <= T_thresh and (best is None or m < best):
            best = m
        ys.append(best)
    return xs, ys


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

ALGO_LABELS = {
    "turbo_m2":              "Ours",
    "turbo_logcei":          "Ours",
    "turbo_logcei_explorer": "Ours",
    "msa":                   "TREAD-M3D",
    "nsga2":                 "NSGA-II",
    "ucb":                   "VAESA",
}

OBJ_LABELS = {
    "edp":   "EDP-Optimal",
    "delay": "Delay-Optimal",
}

OBJ_YLABELS = {
    "edp":   "EDP (cycles·pJ)",
    "delay": "Delay (cycles)",
}


def plot_subplot(ax, rows, T_thresh, obj, algo_label, ylim=(None, None), xlim=None):
    ax.set_yscale("log")
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="gray")
        ax.set_title(algo_label)
        return

    # Filter out non-positive / non-finite metrics before plotting on log scale
    valid_rows = [r for r in rows if r.get("metric") is not None
                  and np.isfinite(r["metric"]) and r["metric"] > 0]

    iters   = [r["iter"] for r in valid_rows]
    metrics = [r["metric"] for r in valid_rows]
    temps   = [r.get("T_max", 9999) for r in valid_rows]
    feasible = [t <= T_thresh for t in temps]

    colors = ["#2ca02c" if f else "#d62728" for f in feasible]
    ax.scatter(iters, metrics, c=colors, s=6, alpha=0.45, linewidths=0, zorder=2,
               rasterized=True)

    xs, ys = running_best(rows, T_thresh)
    valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
    best_val = None
    best_iter = None
    if valid:
        vx, vy = zip(*valid)
        ax.step(vx, vy, where="post", color="#08306b", linewidth=1.8, zorder=3)
        best_idx = int(np.argmin(vy))
        best_val = vy[best_idx]
        best_iter = vx[best_idx]

    if ylim[0] is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        # Enforce minimum 2 decades visible so log scale is visually apparent
        ymin, ymax = ax.get_ylim()
        if ymax / ymin < 100:
            mid = (ymin * ymax) ** 0.5
            ax.set_ylim(mid / 10 - 10, mid * 10)
    if xlim is not None:
        if isinstance(xlim, tuple):
            ax.set_xlim(xlim[0], xlim[1])
        else:
            ax.set_xlim(0, xlim)

    # Yellow star at best feasible point
    if best_val is not None:
        ax.scatter([best_iter], [best_val], marker="*", s=180, color="gold",
                   edgecolors="black", linewidths=0.6, zorder=5)

    ax.set_xlabel("Evaluations")
    ax.set_ylabel(OBJ_YLABELS[obj])
    ax.grid(False)

    # Best value annotation formatted as A×10^B
    if best_val is not None:
        exp = int(np.floor(np.log10(abs(best_val))))
        mantissa = best_val / (10 ** exp)
        best_str = f"Best: {mantissa:.2f}×10$^{{{exp}}}$"
    else:
        best_str = "No feasible"
    ax.set_title(f"{algo_label} - {best_str}")


def _load_and_compute_lims(dnn, T_celsius, algos, objs):
    """Load all data. Returns (all_data, y_lims, x_lims, T_thresh)."""
    T_thresh = T_celsius + 273.15

    all_data = {}
    for algo in algos:
        for obj in objs:
            path = find_result_dir(algo, dnn, obj, T_celsius)
            if path:
                all_data[(algo, obj)] = load_progress(path)
            else:
                all_data[(algo, obj)] = []
                print(f"  [missing] {algo} {dnn} {obj} {T_celsius}c")

    # Let matplotlib autoscale each subplot — no manual y-limits
    y_lims = {(algo, obj): (None, None) for algo in algos for obj in objs}

    # Shared x-limit per objective: max iter seen across all algos, rounded up to
    # nearest 500, with a minimum of 2000 (the intended budget).
    x_lims = {obj: (-100, 2100) for obj in objs}

    return all_data, y_lims, x_lims, T_thresh


def _render_grid(fig, axes_grid, all_data, y_lims, x_lims, T_thresh, algos, objs):
    """Fill a grid of axes (algos × objs)."""
    for row, algo in enumerate(algos):
        for col, obj in enumerate(objs):
            ax = axes_grid[row][col] if len(objs) > 1 else axes_grid[row]
            rows = all_data.get((algo, obj), [])
            plot_subplot(ax, rows, T_thresh, obj, ALGO_LABELS[algo],
                         ylim=y_lims[(algo, obj)], xlim=x_lims[obj])


def make_plot(dnn, T_celsius, out_path):
    """Combined figure: 4 rows × 2 cols (EDP + Delay)."""
    algos = ["turbo_logcei_explorer", "nsga2", "msa", "ucb"]
    objs  = ["edp", "delay"]
    all_data, y_lims, x_lims, T_thresh = _load_and_compute_lims(dnn, T_celsius, algos, objs)

    fig, axes = plt.subplots(len(algos), len(objs), figsize=(10, 4 * len(algos)))
    fig.suptitle(
        f"{dnn.upper()} DSE — Optimisation Metric vs Evaluations "
        f"({T_celsius}°C thermal constraint)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    _render_grid(fig, axes, all_data, y_lims, x_lims, T_thresh, algos, objs)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def make_plot_single_obj(dnn, T_celsius, obj, out_path):
    """Single-objective figure: 2 rows × 2 cols."""
    algos = ["turbo_logcei_explorer", "nsga2", "msa", "ucb"]
    objs  = ["edp", "delay"]  # load both so shared limits are computed correctly
    all_data, y_lims, x_lims, T_thresh = _load_and_compute_lims(dnn, T_celsius, algos, objs)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    for idx, algo in enumerate(algos):
        row, col = divmod(idx, 2)
        rows = all_data.get((algo, obj), [])
        plot_subplot(axes[row][col], rows, T_thresh, obj, ALGO_LABELS[algo],
                     ylim=y_lims[(algo, obj)], xlim=x_lims[obj])

    # Shared legend above all subplots
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=10, label="Infeasible"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=10, label="Feasible"),
        Line2D([0], [0], color="#08306b", linewidth=2, label="Running best"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=11, label="Best feasible"),
    ]
    fig.legend(handles=legend_els, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.06), fontsize=14, frameon=False)

    plt.tight_layout()
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSE comparison scatter plot")
    parser.add_argument("--dnn", required=True,
                        help="DNN name, e.g. resnet18, resnet50, mobilebert, mobilenet_v3")
    parser.add_argument("--T", type=int, required=True,
                        help="Temperature threshold in Celsius, e.g. 80 or 70")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: all_results/plots/<dnn>_dse_comparison_<T>c.png)")
    args = parser.parse_args()

    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    prefix = os.path.join(plots_dir, f"{args.dnn}_dse_comparison_{args.T}c")

    make_plot(args.dnn, args.T, f"{prefix}_combined.png")
    make_plot_single_obj(args.dnn, args.T, "delay", f"{prefix}_delay.png")
    make_plot_single_obj(args.dnn, args.T, "edp",   f"{prefix}_edp.png")
