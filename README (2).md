# Evaluate a forecast: AAPL next-day price

Course project for **Big Data Analytics (Arcada), Week 3**.
Question: *can we trust a machine-learning forecast of Apple's next-day closing price more than "tomorrow = today"?*

**Answer in one line:** no. Ridge, XGBoost and CatBoost are 2% to 5% *worse* than the naive forecast on average
(MAE ÷ naive = 1.035, 1.049, 1.022) and beat it in at most one of six walk-forward folds.

## Files

| File | What it is |
|---|---|
| `evaluate_forecast.ipynb` | The complete analysis: data, features, leakage check, walk-forward folds, baselines, models, metrics, four figures |
| `Evaluate_a_Forecast_Report.docx` | The report for the manager (save/export as PDF for submission) |
| `README.md` | This file |

Created when the notebook runs:

```
data/      AAPL_daily_2018-01-01_2026-08-31.csv   cached download (reused on later runs)
figures/   fig1_data_and_folds.png   fig2_forecast_vs_actual.png
           fig3_metric_per_fold.png  fig4_model_comparison.png
outputs/   fold_table.csv  results_per_fold.csv  results_summary.csv  folds_beating_baselines.csv
```

## How to run

1. Python 3.10+ is recommended (developed on 3.13).
2. Install the requirements:
   ```
   pip install numpy pandas matplotlib scikit-learn xgboost catboost yfinance jupyter
   ```
3. Put the notebook in an empty folder, open it in Jupyter or VS Code and choose **Kernel → Restart & Run All**.
   - First run: downloads AAPL from Yahoo Finance (internet needed) and caches it in `data/`.
   - Later runs: use the cached file, so results are identical. Set `REFRESH_DATA = True` in the setup cell to download again.
   - Runtime: a few minutes (18 model fits: 3 models × 6 folds).
4. Insert `figures/fig1_data_and_folds.png` and `figures/fig2_forecast_vs_actual.png` in the two marked spots in the
   Word report if they are not already there (Figures 3 and 4 are already in the report).

## Setup of the experiment

| Item | Choice |
|---|---|
| Data | AAPL daily, 2018-01-02 to end of Aug 2026, `auto_adjust=True` (2,176 days; 2,155 after feature warm-up) |
| Target | tomorrow's log return r(t+1); rebuilt as price P(t)·exp(r̂) before scoring |
| Horizon h | 1 trading day |
| Features (16) | 10 lagged returns, 5- and 20-day volatility, 5- and 20-day momentum, distance to 20-day average, day of week |
| Folds | `TimeSeriesSplit(n_splits=6, test_size=250, gap=1)`, one split shared by every model |
| Scaling | `StandardScaler` for features and standardised target, fitted on each training fold only |
| Baselines | Naive (tomorrow = today); Always up (direction) |
| Models | Ridge(alpha=1), XGBoost (500 trees, lr 0.03, depth 3), CatBoost (500 iterations, lr 0.03, depth 4) |
| Metrics | MAE, MAPE, MASE, MAE ÷ naive, directional accuracy; per fold and mean ± std |
| Seeds | 42 everywhere |

## Results (author's run)

| Model | MAE (USD) | MAPE (%) | MAE ÷ naive | Up/down right (%) |
|---|---|---|---|---|
| Naive | 2.312 ± 0.534 | 1.276 ± 0.171 | 1.000 | n/a |
| Ridge | 2.388 ± 0.524 | 1.322 ± 0.180 | 1.035 ± 0.020 | 50.0 ± 2.2 |
| XGBoost | 2.413 ± 0.511 | 1.340 ± 0.202 | 1.049 ± 0.052 | 51.5 ± 2.2 |
| CatBoost | 2.359 ± 0.525 | 1.305 ± 0.179 | 1.022 ± 0.023 | 50.5 ± 3.1 |
| Always up | n/a | n/a | n/a | 52.9 ± 1.8 |

Your numbers should match to about the third decimal if the cached data are the same. A fresh download can differ
slightly because Yahoo revises adjusted prices.

## Notes

- **No leakage.** Features use trailing windows only (never `rolling(..., center=True)`). A test in the notebook
  recomputes the features on truncated history and asserts they are identical.
- **MASE vs MAE ÷ naive.** MASE divides by the naive error on the *training* fold, and prices were lower then, so it is
  above 1 even for the naive forecast. MAE ÷ naive uses the same test fold and is the fair comparison.
- **Not tuned.** Model settings were fixed in advance and not tuned on the test folds.
