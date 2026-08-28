# Graduate Quantitative Research Notebooks

This directory is a research-facing companion to the monograph and the three reproducible projects. The notebooks are ordered as a coherent laboratory sequence rather than a collection of disconnected demos.

| Lab | Research question | Core methods | Evidence |
|---|---|---|---|
| [01 — Monte Carlo and delta hedging](01_monte_carlo_and_delta_hedging.ipynb) | When do simulation and replication claims survive discretization, misspecification, and costs? | Itô/GBM discretization, confidence intervals, antithetic and control variates, discrete hedging P&L | Deterministic simulation |
| [02 — Yield-curve factor dynamics](02_yield_curve_factor_dynamics.ipynb) | Are three factors enough to describe and forecast the Treasury curve? | PCA, Nelson–Siegel, reconstruction error, recursive VAR, HAC forecast comparison | FRED Treasury panel saved by the project |
| [03 — Portfolio optimization under estimation error](03_portfolio_optimization_under_estimation_error.ipynb) | How much of an “optimal” portfolio is sampling noise? | Covariance conditioning, shrinkage, bootstrap weight dispersion, walk-forward allocation, costs | Kenneth French factor returns saved by the project |
| [04 — Regime risk and backtest integrity](04_regime_risk_and_backtest_integrity.ipynb) | Which conclusions remain after timing, tails, and turnover are audited? | Observable regimes, expected shortfall, volatility targeting, leakage controls, Kupiec coverage | SPY/TLT/GLD/VIX panel saved by the project |
| [05 — Identification and forecasting](05_identification_and_forecasting.ipynb) | Why is prediction not identification, and why is in-sample fit not a forecast? | Omitted-variable bias, IV strength, HAC inference, recursive forecasting, out-of-sample R² | Simulation plus Kenneth French factors |
| [06 — Implied volatility and static arbitrage](06_implied_volatility_and_static_arbitrage.ipynb) | What information is encoded in option prices, and when is a smile internally inconsistent? | Black–Scholes inversion, lognormal mixtures, monotonicity/convexity tests, Breeden–Litzenberger density | Controlled arbitrage-free and contaminated surfaces |

## Research standard

Each notebook contains:

- a precise research question and mathematical setup;
- an explicit information set and reproducible random seed;
- at least one benchmark or null model;
- numerical diagnostics rather than presentation-only charts;
- a deliberately constructed failure mode;
- a model-risk conclusion stating what the evidence does **not** establish;
- exercises that extend the analysis into a serious research project.

The empirical notebooks read the versioned files in `projects/*/output`. Run the corresponding project first only when you want to refresh its data vintage. The provenance files record source, sample, transformations, and fallback status.

## Run the sequence

From the repository root:

```powershell
jupyter lab notebooks
```

To execute and validate the complete curriculum from fresh kernels:

```powershell
python scripts/execute_notebooks.py
```

Pass one or more notebook paths to execute a subset. The runner fails on the first cell error and saves outputs only after a notebook completes.

The notebooks locate the repository root automatically, so they also run when Jupyter starts inside `notebooks/`.
