# Price with a Smile 😄

*A Production-Style Local Volatility Surface & Δ-Hedging Backtest*

This repository implements and backtests a **Local Volatility (Dupire-type)** pricing and hedging framework against a **Black-Scholes sticky implied-volatility** benchmark using real multi-day option data downloaded using the Polygon API.
It provides a fully reproducible, parallelizable, and modular pipeline for modern quantitative research and for prototyping trading models.

---

## 📘 Project Overview

* **Goal:** Assess whether local volatility surfaces deliver better daily hedging stability and pricing accuracy than traditional sticky-vol models.
* **Scope:** From data ingestion → surface calibration → local-vol extraction → PDE pricing → Δ-hedging simulation → backtest aggregation.
* **Tech Stack:** Python, NumPy, Pandas, Matplotlib, CVXPY, PyArrow, SLURM for parallel job arrays.

---

## ⚙️ Pipeline Components

| File                   | Description                                                             |
| ---------------------- | ----------------------------------------------------------------------- |
| `day_pair_hedge.py`    | Core module: local-vol extraction, pricing engine, and Δ-hedging logic. |
| `make_pairs.py`        | Generates consecutive business-day pairs for backtesting.               |
| `run_pair.py`          | Runs one day-pair (by index) — saves per-pair CSV and JSON summaries.   |
| `aggregate_results.py` | Aggregates pair results → produces summary CSV and plots.               |
| `run_daypairs.sbatch`  | SLURM job-array submission script for parallelized runs.                |
| `artifacts/`           | Output folder containing per-pair results and final summaries.          |
| `ProjectSlides.pptx`   | Presentation slides summarizing results.                                |

---

## 🚀 Quick Start

```bash
# install requirements
python -m pip install -r requirements.txt

# generate day pairs
python make_pairs.py --start 2025-09-23 --end 2025-11-06

# run one pair
python run_pair.py --pairs pairs.txt --idx 0

# aggregate results
python aggregate_results.py --pairs_dir artifacts/pairs \
  --out_csv artifacts/rolling_hedge_pnl.csv
```

**Configuration:**
Global params live at the top of `day_pair_hedge.py`:

```python
SUBSET = dict(max_expiries=8, m_band=(0.95, 1.05))
PDE_GRID = dict(nS=501, nT=600, span=(0.45, 1.75))
```

---

## ☁️ HPC Job Array Example (SOL or similar cluster)

```bash
python make_pairs.py --start 2025-09-23 --end 2025-11-06
wc -l pairs.txt  # N pairs -> 0..N-1

# edit run_daypairs.sbatch
#SBATCH -p htc
#SBATCH --array=0-(N-1)
#SBATCH -t 01:00:00
#SBATCH --mem=4G

# submit
sbatch run_daypairs.sbatch
squeue -u $USER
```

After jobs finish:

```bash
python aggregate_results.py --pairs_dir artifacts/pairs --out_csv artifacts/rolling_hedge_pnl.csv
```

---

## 📊 Outputs

* **Per-pair CSV/JSON:** in `artifacts/pairs/`
* **Aggregated metrics:** `artifacts/rolling_hedge_pnl.csv`
* **Plots:**

  * `artifacts/rolling_pnl_cum.png` — Cumulative mean Δ-hedged P&L
  * `artifacts/rolling_rmse.png` — Daily RMSE across models

---

## 🧩 Results Summary

* Local Volatility surfaces yield smoother Δ-hedged P&L and lower RMSE on select days.
* Black-Scholes sticky vols remain competitive but less stable under smile shifts.
* Framework is modular for extensions (stochastic vol, rough vol, VIX-based calibration).

---

## 📈 Future Extensions

* Local Stochastic volatility calibration
* Γ–Θ hedging extensions
* Monte Carlo-based volatility control

---

## ✍️ Citation

If you use this code or framework in research or teaching, please cite:

> Pereira, J. *Price with a Smile: A Production-style Local Volatility Surface & Backtest* (2025).

---

## 🧠 Author

**Jude Pereira**
PhD Candidate, Theoretical Physics, Arizona State University
[GitHub: @jude2510](https://github.com/jude2510)
