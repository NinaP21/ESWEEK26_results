"""
Combined DSE figure: 2 rows × 2 cols
  Top row:    running-best convergence (EDP left, Delay right)
  Bottom row: rolling % near thermal threshold (EDP left, Delay right)
All algorithms overlaid as lines in each panel.

Usage examples
--------------
python scripts/plot_dse_combined.py --dnn resnet18 --temp-label 70c --T-threshold 343.15
python scripts/plot_dse_combined.py --dnn mobilebert --temp-label 80c --T-threshold 353.15 --algos msa nsga2 ucb turbo
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family":   "serif",
    "text.usetex":   True,
    "pgf.rcfonts":   False,
    "font.size":     9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Algorithm registry ────────────────────────────────────────────────────────

ALGO_META = {
    "turbo":  ("Ours",       "#1b9e77", "-", 2.0),
    "nsga2":  ("NSGA-II",    "#f0a500", "-", 1.5),
    "ucb":    ("VAESA",      "#9ecae1", "-", 1.5),
    "msa":    ("TREAD-M3D",  "#636363", "-", 1.5),
}

_SCREENING_PHASES = {"screening"}


# ── Data helpers ──────────────────────────────────────────────────────────────

def resolve_progress_path(directory: Path) -> Path | None:
    timestamped = sorted(directory.glob("progress_????????_??????.jsonl"), reverse=True)
    if timestamped:
        return timestamped[0]
    legacy = directory / "progress.jsonl"
    return legacy if legacy.exists() else None


def load_progress(path: Path, last_n: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    bo_rows = [r for r in rows if r.get("phase") not in _SCREENING_PHASES]
    if last_n is not None:
        bo_rows = bo_rows[-last_n:]
    return bo_rows


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


def rolling_pct(T_C: np.ndarray, lo: float, hi: float, window: int = 100):
    in_win = ((T_C >= lo) & (T_C <= hi)).astype(float)
    k = np.ones(min(window, len(in_win))) / min(window, len(in_win))
    return np.convolve(in_win, k, mode="same") * 100


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_combined(algo_data: dict, T_thresh_K: float, dnn: str,
                  out_path: Path, window: int = 100):

    T_thresh_C = T_thresh_K - 273.15
    near_lo    = T_thresh_C - 5
    near_hi    = T_thresh_C + 5

    objs       = ["edp", "delay"]
    obj_labels = {"edp": r"EDP (cycles$\cdot$J)", "delay": r"Delay (cycles)"}

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))

    for col, obj in enumerate(objs):
        ax = axes[col]

        for algo, (label, color, ls, lw) in ALGO_META.items():
            rows = algo_data.get(algo, {}).get(obj)
            if rows is None:
                continue
            rb_x, rb_y = running_best(rows, T_thresh_K)
            ax.step(rb_x, rb_y, where="post",
                    color=color, lw=lw, ls=ls, label=label, zorder=3)

        ax.set_yscale("log")
        ax.set_ylabel(obj_labels[obj])
        ax.set_xlabel(r"Evaluation \#")
        ax.set_title(f"{'EDP' if obj == 'edp' else 'Delay'} — Convergence")
        ax.grid(True, alpha=0.25, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # shared legend above the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.03), frameon=False)

    fig.suptitle(
        rf"{dnn} — {T_thresh_C:.0f}$^\circ$C thermal constraint",
        fontsize=9, y=1.07
    )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pgf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path.with_suffix('.pgf')} and {out_path.with_suffix('.png')}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--algos", nargs="+", default=list(ALGO_META.keys()),
                    help="Algorithms to include (default: all)")
    ap.add_argument("--dnn", default="resnet18")
    ap.add_argument("--temp-label", default="70c")
    ap.add_argument("--T-threshold", type=float, default=343.15,
                    help="Thermal threshold in K (default: 343.15 = 70°C)")
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "plot_ready")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--rolling-window", type=int, default=100)
    ap.add_argument("--last-n", nargs="*", default=[], metavar="ALGO:N")
    args = ap.parse_args()

    last_n_map: dict[str, int] = {}
    for entry in args.last_n:
        if ":" not in entry:
            ap.error(f"--last-n entries must be ALGO:N, got '{entry}'")
        algo, n = entry.split(":", 1)
        last_n_map[algo] = int(n)

    out_dir = args.out_dir or (Path(__file__).parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    algo_data: dict[str, dict[str, list]] = {}
    for algo in args.algos:
        if algo not in ALGO_META:
            print(f"[WARN] Unknown algo '{algo}', skipping", file=sys.stderr)
            continue
        algo_data[algo] = {}
        for obj in ["edp", "delay"]:
            run_dir = args.results_dir / algo / f"{args.dnn}_{obj}_{args.temp_label}"
            path = resolve_progress_path(run_dir)
            if path is None:
                print(f"  [WARN] No progress file in: {run_dir}", file=sys.stderr)
                algo_data[algo][obj] = None
                continue
            last_n = last_n_map.get(algo)
            rows = load_progress(path, last_n=last_n)
            print(f"  Loaded {algo}/{obj}: {len(rows)} evals from {path.name}")
            algo_data[algo][obj] = rows

    out_path = out_dir / f"{args.dnn}_{args.temp_label}_combined"
    plot_combined(algo_data, args.T_threshold, args.dnn, out_path,
                  window=args.rolling_window)


if __name__ == "__main__":
    main()
