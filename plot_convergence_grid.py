"""
Convergence grid: N thresholds × 2 objectives (EDP | Delay).
All algorithms overlaid as lines. Single legend on top. No figure title.

Usage
-----
python scripts/plot_convergence_grid.py --dnn mobilevit_s
python scripts/plot_convergence_grid.py --dnn mobilevit_s --thresholds 80 90
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem":   "pdflatex",
    "font.family":     "serif",
    "text.usetex":     True,
    "pgf.rcfonts":     False,
    "font.size":       14,
    "axes.labelsize":  14,
    "axes.titlesize":  14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "font.weight":          "bold",
    "axes.labelweight":     "bold",
    "axes.titleweight":     "bold",
})

import matplotlib.pyplot as plt
import numpy as np

# ── Algorithm registry ────────────────────────────────────────────────────────

# (label, color, linestyle, linewidth, marker, markersize)
ALGO_META = {
    "msa":   (r"TREAD-M3D~\cite{shukla2023tread}", "#636363", "-",  1.5, "D", 3),
    "nsga2": (r"MACO~\cite{zhong2025maco}", "#f0a500", "-",  1.5, "s", 3),
    "ucb":   (r"VAESA-BO~\cite{liu2023deepoheat}", "#9ecae1", "-",  1.5, "^", 3),
    "turbo": ("Ours",       "#1b9e77", "-",  3.0, "o", 6),
}

OBJ_LABELS = {
    "edp":   r"EDP (cycles$\cdot$J)",
    "delay": r"Delay (cycles)",
}

_SCREENING_PHASES = {"screening"}


# ── Data helpers ──────────────────────────────────────────────────────────────

def resolve_progress_path(directory: Path) -> Path | None:
    timestamped = sorted(directory.glob("progress_????????_??????.jsonl"), reverse=True)
    if timestamped:
        return timestamped[0]
    legacy = directory / "progress.jsonl"
    return legacy if legacy.exists() else None


def load_progress(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return [r for r in rows if r.get("phase") not in _SCREENING_PHASES]


def running_best(rows: list[dict], T_thresh_K: float):
    xs, ys = [], []
    cur_best = math.inf
    for i, r in enumerate(rows):
        T = r.get("T_max", math.inf)
        m = r.get("metric", math.inf)
        if T <= T_thresh_K and m < cur_best:
            cur_best = m
        xs.append(i + 1)
        ys.append(cur_best if cur_best < math.inf else float("nan"))
    return np.array(xs), np.array(ys)


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_grid(dnn: str, thresholds: list[int], results_dir: Path, out_path: Path):
    objs = ["edp", "delay"]
    n_rows = len(thresholds)
    n_cols = len(objs)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5, 2.4 * n_rows),
        squeeze=False,
    )

    for row, T_C in enumerate(thresholds):
        T_K = T_C + 273.15
        temp_label = f"{T_C}c"

        for col, obj in enumerate(objs):
            ax = axes[row][col]

            for algo, (label, color, ls, lw, marker, ms) in ALGO_META.items():
                run_dir = results_dir / algo / f"{dnn}_{obj}_{temp_label}"
                path = resolve_progress_path(run_dir)
                if path is None:
                    print(f"  [WARN] No data: {run_dir}", file=sys.stderr)
                    continue
                rows_data = load_progress(path)
                print(f"  Loaded {algo}/{obj}/{T_C}c: {len(rows_data)} evals")
                rb_x, rb_y = running_best(rows_data, T_K)
                ax.step(rb_x, rb_y, where="post",
                        color=color, lw=lw, ls=ls, label=label, zorder=3,
                        marker=marker, markevery=250, markersize=ms,
                        markeredgewidth=1.2)

            ax.set_yscale("log")
            ax.set_ylabel(OBJ_LABELS[obj])
            ax.set_xlabel(r"Evaluations")
            obj_line = "EDP-Optimal" if obj == "edp" else "Delay-Optimal"
            ax.text(0.97, 0.93, rf"{T_C}$^\circ$C threshold" + f"\n{obj_line}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=13, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Single legend above the top row
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.03), frameon=False)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pgf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path.with_suffix('.pgf')} and {out_path.with_suffix('.png')}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dnn", default="mobilevit_s")
    ap.add_argument("--thresholds", nargs="+", type=int, default=[80, 90])
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "plot_ready")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "plots")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    thresholds_str = "_".join(str(t) for t in args.thresholds)
    out_path = args.out_dir / f"{args.dnn}_convergence_{thresholds_str}c"
    make_grid(args.dnn, args.thresholds, args.results_dir, out_path)


if __name__ == "__main__":
    main()
