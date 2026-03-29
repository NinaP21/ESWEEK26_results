"""
Plot the temperature (T_max in °C) of the running-best feasible solution
over evaluations, for all algorithms overlaid — one panel per objective.

Usage examples
--------------
python scripts/plot_best_temperature.py --dnn mobilevit_s --temp-label 70c --T-threshold 343.15
python scripts/plot_best_temperature.py --dnn resnet18 --temp-label 80c --T-threshold 353.15
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
import numpy as np

ALGO_META = {
    "turbo":    ("Ours (m=4)",   "#1b9e77", "-",  2.0),
    "turbo_m1": ("Ours (m=1)",   "#74c476", "--", 1.5),
    "nsga2":    ("NSGA-II",      "#f0a500", "-",  1.5),
    "ucb":      ("VAESA",        "#9ecae1", "-",  1.5),
    "msa":      ("TREAD-M3D",    "#636363", "-",  1.5),
}

_SCREENING_PHASES = {"screening"}


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


def running_best_temperature(rows: list[dict], T_thresh_K: float):
    """Return (eval_indices, T_max_C of running best feasible solution)."""
    xs, ys = [], []
    cur_best_metric = math.inf
    cur_best_T_C = float("nan")
    for i, r in enumerate(rows):
        T_K = r.get("T_max", math.inf)
        m   = r.get("metric", math.inf)
        if T_K <= T_thresh_K and m < cur_best_metric:
            cur_best_metric = m
            cur_best_T_C    = T_K - 273.15
        xs.append(i + 1)
        ys.append(cur_best_T_C)
    return np.array(xs), np.array(ys)


def plot_best_temperature(algo_data: dict, T_thresh_K: float, dnn: str, out_path: Path):
    T_thresh_C = T_thresh_K - 273.15
    objs       = ["edp", "delay"]
    obj_titles = {"edp": "EDP-Optimal", "delay": "Delay-Optimal"}

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))

    for col, obj in enumerate(objs):
        ax = axes[col]

        for algo, (label, color, ls, lw) in ALGO_META.items():
            rows = algo_data.get(algo, {}).get(obj)
            if rows is None:
                continue
            xs, ys = running_best_temperature(rows, T_thresh_K)
            # mask until first feasible found
            first_valid = next((i for i, v in enumerate(ys) if np.isfinite(v)), None)
            if first_valid is None:
                continue
            ys[:first_valid] = np.nan
            ax.step(xs, ys, where="post", color=color, lw=lw, ls=ls,
                    label=label, zorder=3)

        ax.axhline(T_thresh_C, color="black", lw=1.0, ls="--", alpha=0.6,
                   label=rf"Threshold ({T_thresh_C:.0f}$^\circ$C)")
        ax.set_xlabel(r"Evaluation \#")
        ax.set_ylabel(r"$T_{\max}$ of best solution ($^\circ$C)")
        ax.set_title(obj_titles[obj])
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.04), frameon=False)

    fig.suptitle(
        rf"{dnn} — Temperature of best feasible solution over evaluations",
        fontsize=9, y=1.09
    )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pgf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--algos", nargs="+", default=list(ALGO_META.keys()))
    ap.add_argument("--dnn", default="mobilevit_s")
    ap.add_argument("--temp-label", default="70c")
    ap.add_argument("--T-threshold", type=float, default=343.15)
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "plot_ready")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

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
            rows = load_progress(path)
            print(f"  Loaded {algo}/{obj}: {len(rows)} evals")
            algo_data[algo][obj] = rows

    out_path = out_dir / f"{args.dnn}_{args.temp_label}_best_temperature"
    plot_best_temperature(algo_data, args.T_threshold, args.dnn, out_path)


if __name__ == "__main__":
    main()
