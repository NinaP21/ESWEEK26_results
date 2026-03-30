from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

# =========================
# Matplotlib / LaTeX style
# =========================
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "legend.fontsize": 12,
    "pgf.preamble": r"\usepackage{natbib}",
})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker

# =========================
# Load CSV
# =========================
csv_file = Path(__file__).parent / "dse_results.csv"
df = pd.read_csv(csv_file)

# =========================
# Settings
# =========================
algorithms = ["MSA", "NSGA-II", "UCB", "Explorer"]

# Display labels for x-axis ticks (maps CSV DNN name → displayed label)
# Edit these to control what appears on the x-axis
dnn_labels = {
    "ResNet-18":    "ResNet18",
    "ResNet-50":    "ResNet50",
    "MobileBERT":   "MobileBERT",
    "MobileNet-V3": "MobileNet-V3",
    "MobileViT-S":  "MobileViT",
}

# Colors
c_nsga = "#f0a500"  # amber
c_msa  = "#636363"  # dark gray (TREAD-M3D reference)
c_ucb  = "#9ecae1"  # light blue
c_exp  = "#1b9e77"  # green

color_map = {
    "NSGA-II": c_nsga,
    "MSA": c_msa,
    "UCB": c_ucb,
    "Explorer": c_exp,
}

# =========================
# Helpers
# =========================
def sanitize_filename(text):
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace(".", "p")
    )

def normalize_per_row(df_obj, suffix):
    baseline_col = f"MSA_{suffix}"
    if baseline_col not in df_obj.columns:
        raise ValueError(f"Missing baseline column: {baseline_col}")

    baseline = pd.to_numeric(df_obj[baseline_col], errors="coerce").values
    normalized = {}

    for algo in algorithms:
        col = f"{algo}_{suffix}"
        if col not in df_obj.columns:
            raise ValueError(f"Missing column: {col}")

        vals = pd.to_numeric(df_obj[col], errors="coerce").values

        with np.errstate(divide="ignore", invalid="ignore"):
            norm = vals / baseline

        norm[~np.isfinite(norm)] = np.nan
        normalized[algo] = norm

    return normalized

# =========================
# Plot function
# =========================
def grouped_bar_plot_normalized(df_obj, suffix, ylabel, save_prefix, variance_style="none"):
    """
    variance_style:
      "none"   – no variance shown
      "cv"     – right y-axis with coefficient of variation (%) per algorithm
      "dots"   – scatter individual DNN normalized values over the Average bars
    """
    if df_obj.empty:
        print(f"Skipping {save_prefix}: dataframe is empty")
        return

    dnn_models = df_obj["DNN"].astype(str).tolist()
    x = np.arange(len(dnn_models)) * 1.05
    w = 0.18
    gap = 0.02
    step = w + gap

    data_norm = normalize_per_row(df_obj, suffix)

    avg_norm = {algo: np.nanmean(data_norm[algo]) for algo in algorithms}
    std_norm = {algo: np.nanstd(data_norm[algo]) for algo in algorithms}

    x_avg = x[-1] + 1.05 + 0.25 if len(x) > 0 else 0.0
    x_all = np.append(x, x_avg)
    tick_labels = [dnn_labels.get(d, d) for d in dnn_models] + ["Average"]

    fig, ax = plt.subplots(figsize=(6.2, 2.0))

    for i, algo in enumerate(algorithms):
        shift = (i - 1.5) * step
        vals = data_norm[algo]
        valid = ~np.isnan(vals)

        if np.any(valid):
            ax.bar(
                x[valid] + shift,
                vals[valid],
                width=w,
                color=color_map[algo],
                linewidth=0.4,
                zorder=3,
                label=algo
            )

        if np.any(~valid):
            skipped = [dnn_models[j] for j in np.where(~valid)[0]]
            print(f"Warning: skipped {algo} bars in {save_prefix} for models: {skipped}")

        avg_val = avg_norm[algo]
        if np.isfinite(avg_val):
            ax.bar(
                x_avg + shift,
                avg_val,
                width=w,
                color=color_map[algo],
                linewidth=0.4,
                zorder=3,
                yerr=std_norm[algo],
                error_kw=dict(ecolor="black", elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=4),
            )


    ax.set_xticks(x_all)
    ax.set_xticklabels(tick_labels, ha="center")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_label_coords(-0.08, 0.4)
    ax.axhline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.7)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.grid(True, linestyle=":", alpha=0.45, zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)

    all_valid_vals = [v[~np.isnan(v)] for v in data_norm.values() if np.any(~np.isnan(v))]
    if not all_valid_vals:
        print(f"Skipping {save_prefix}: no valid values to plot")
        plt.close(fig)
        return

    avg_finite = [v for v in avg_norm.values() if np.isfinite(v)]
    ymax = max(
        max(np.max(v) for v in all_valid_vals),
        max(avg_finite) if avg_finite else 0.0,
    )
    ax.set_ylim(0, max(1.2, ymax * 1.12))

    ax.spines["right"].set_visible(False)

    label_map = {
        "NSGA-II": r"MACO~\cite{zhong2025maco}",
        "MSA": r"TREAD-M3D~\cite{shukla2023tread}",
        "UCB": r"VAESA-BO~\cite{liu2023deepoheat}",
        "Explorer": "Ours",
    }

    handles = [
        mpatches.Patch(color=color_map[a], label=label_map[a])
        for a in algorithms
    ]

    if "70C" in save_prefix:
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 1.05),
            frameon=False
        )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    plt.savefig(f"{save_prefix}.pgf", bbox_inches="tight")
    plt.savefig(f"{save_prefix}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Saved: {save_prefix}.pgf and {save_prefix}.png")

# =========================
# Main loop: one plot per objective and threshold
# =========================
print("Available thresholds:", sorted(df["T_threshold_C"].dropna().unique()))
print("Available objectives:", df["Objective"].dropna().unique())

all_thresholds = sorted(df["T_threshold_C"].dropna().unique())
all_objectives = sorted(df["Objective"].dropna().unique())

for threshold in all_thresholds:
    df_thr = df[df["T_threshold_C"] == threshold].copy()
    dnn_order = ["ResNet-18", "ResNet-50", "MobileBERT", "MobileViT-S", "MobileNet-V3"]

    for objective in all_objectives:
        df_obj = df_thr[df_thr["Objective"].astype(str).str.upper() == str(objective).upper()].copy()

        if df_obj.empty:
            continue

        df_obj["DNN"] = pd.Categorical(df_obj["DNN"], categories=dnn_order, ordered=True)
        df_obj = df_obj.sort_values("DNN")

        objective_upper = str(objective).upper()

        if objective_upper == "EDP":
            suffix = "metric"
            ylabel = r"Normalized EDP"
        elif objective_upper == "DELAY":
            suffix = "delay"
            ylabel = r"Normalized Delay"
        else:
            print(f"Skipping unsupported objective: {objective}")
            continue

        out_dir = Path(__file__).parent / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_prefix = str(out_dir / f"normalized_{sanitize_filename(objective_upper)}_{sanitize_filename(threshold)}C")

        grouped_bar_plot_normalized(
            df_obj=df_obj, suffix=suffix, ylabel=ylabel,
            save_prefix=save_prefix, variance_style="none"
        )
        

print("Done.")
