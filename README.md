# ESWEEK 2026 — DSE Results & Plotting Scripts

This repository contains the DSE (Design Space Exploration) experiment results and plotting scripts for the ESWEEK 2026 paper submission.

---

## Repository structure

```
.
├── bar_plots.py                # Normalized bar charts of final DSE results
├── plot_best_temperature.py    # Temperature of best feasible solution over evaluations
├── plot_convergence_grid.py    # Convergence grid across thresholds and objectives
├── plot_dse_combined.py        # Combined convergence plot (EDP and Delay side by side)
├── plot_dse_comparison.py      # Scatter plots of all evaluations per algorithm (2×2 grid)
├── dse_results.csv             # Final DSE results table (all algorithms, DNNs, objectives)
└── plot_ready/
    ├── turbo/                  # Ours — TuRBO logCEI explorer, m=4 (full runs)
    ├── turbo_m1/               # Ours — TuRBO logCEI explorer, m=1
    ├── nsga2/                  # Baseline — NSGA-II
    ├── ucb/                    # Baseline — VAESA (UCB)
    └── msa/                    # Baseline — TREAD-M3D
```

Each subdirectory under `plot_ready/<algo>/` follows the naming convention:

```
<dnn>_<objective>_<temperature>/
    progress_YYYYMMDD_HHMMSS.jsonl
```

**DNNs:** `mobilebert`, `mobilenet_v3`, `mobilevit_s`, `resnet18`, `resnet50`
**Objectives:** `edp`, `delay`
**Temperature thresholds:** `70c`, `80c`, `90c`

---

## Requirements

```bash
pip install matplotlib numpy
```

A LaTeX installation (e.g. `texlive`) is required for PGF/PDF output. PNG output works without it if you change the matplotlib backend.

---

## Scripts

### 1. `bar_plots.py`

Plots normalized bar charts of final DSE results (EDP and Delay) across all DNNs and temperature thresholds. Each algorithm is normalized to the MSA (TREAD-M3D) baseline. Reads from `dse_results.csv` (bundled in the repo).

```bash
python bar_plots.py
```

**Output:** `plots/normalized_{EDP,DELAY}_{70,80,90}C.{png,pgf}`

---

### 2. `plot_best_temperature.py`

Plots the temperature of the running-best feasible solution over evaluations, with one panel per objective (EDP and Delay).

```bash
# Default: mobilevit_s, 70°C threshold
python plot_best_temperature.py

# Specify DNN and threshold
python plot_best_temperature.py --dnn mobilevit_s --temp-label 80c --T-threshold 353.15
python plot_best_temperature.py --dnn resnet18 --temp-label 70c --T-threshold 343.15

# Include only specific algorithms
python plot_best_temperature.py --dnn mobilevit_s --temp-label 80c --algos turbo turbo_m1 nsga2 ucb msa
```

**Note:** The `turbo_m1` data is only available for `mobilevit_s` at `80c`.

**Output:** `plots/<dnn>_<temp>_best_temperature.{png,pgf}`

---

### 3. `plot_convergence_grid.py`

Plots a grid of convergence curves: one row per temperature threshold, two columns for EDP and Delay objectives. All algorithms are overlaid as lines.

```bash
# Default: mobilevit_s, thresholds 80°C and 90°C
python plot_convergence_grid.py

# Specify DNN and thresholds
python plot_convergence_grid.py --dnn resnet18 --thresholds 70 80 90
python plot_convergence_grid.py --dnn mobilebert --thresholds 80 90
```

**Output:** `plots/<dnn>_convergence_<thresholds>c.{png,pgf}`

---

### 4. `plot_dse_comparison.py`

Scatter plots of all DSE evaluations (green=feasible, red=infeasible) with a running-best step line for each algorithm. Produces a 2×2 grid per objective (one subplot per algorithm).

```bash
python plot_dse_comparison.py --dnn mobilevit_s --T 80
python plot_dse_comparison.py --dnn resnet18 --T 70
python plot_dse_comparison.py --dnn mobilebert --T 90
```

**Output:** `plots/<dnn>_dse_comparison_<T>c_{combined.png,delay.pdf,edp.pdf}`

---

### 5. `plot_dse_combined.py`

Plots convergence curves for EDP and Delay side by side, for a single DNN and temperature threshold.

```bash
# Default: resnet18, 70°C threshold
python plot_dse_combined.py

# Specify DNN and threshold
python plot_dse_combined.py --dnn mobilebert --temp-label 80c --T-threshold 353.15
python plot_dse_combined.py --dnn mobilevit_s --temp-label 90c --T-threshold 363.15 --algos turbo nsga2 ucb msa
```

**Output:** `plots/<dnn>_<temp>_combined.{png,pgf}`

---

## Temperature threshold reference

| Label  | Kelvin   | Celsius |
|--------|----------|---------|
| `70c`  | 343.15 K | 70°C    |
| `80c`  | 353.15 K | 80°C    |
| `90c`  | 363.15 K | 90°C    |

---

## Algorithm reference

| Key        | Label        | Description                        |
|------------|--------------|------------------------------------|
| `turbo`    | Ours (m=4)   | TuRBO logCEI explorer, m=4 (full)  |
| `turbo_m1` | Ours (m=1)   | TuRBO logCEI explorer, m=1         |
| `nsga2`    | NSGA-II      | Baseline                           |
| `ucb`      | VAESA        | Baseline                           |
| `msa`      | TREAD-M3D    | Baseline                           |
