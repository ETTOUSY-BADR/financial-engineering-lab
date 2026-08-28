# Financial Engineering Lab

An advanced, reproducible laboratory connecting mathematical finance, empirical
economics, market structure, and quantitative research.

The repository is built around one principle: a financial model is credible only
when its assumptions, information set, data lineage, numerical implementation,
uncertainty, costs, and failure modes are all visible.

## The book

The flagship artifact is
[Quantitative Finance: From Mathematical Foundations to Empirical Research](course/quant_finance_book.pdf),
a 203-page professional monograph written in modular LaTeX.

Its 38 chapters and five appendices progress from first principles to independent
research:

1. mathematical analysis, convexity, linear algebra, optimization, probability,
   martingales, Brownian motion, Ito calculus, and changes of measure;
2. intertemporal macroeconomics, monetary policy, state-space models, yield curves,
   affine term structure, and regimes;
3. identification, regression, time series, volatility, extreme value theory,
   causal inference, panels, and machine learning;
4. instruments, microstructure, collateral, counterparty exposure, and model
   governance;
5. stochastic discount factors, CAPM, multifactor pricing, anomalies, and return
   predictability;
6. binomial pricing, Black--Scholes--Merton, Greeks, local and stochastic volatility,
   jumps, exotics, and incomplete-market hedging;
7. curve construction, rate models, credit, securitization, and structured products;
8. portfolio estimation, shrinkage, coherent risk, stress testing, risk parity,
   drawdown, and backtesting;
9. optimization algorithms, Monte Carlo, finite differences, Fourier methods,
   modern machine learning, and reproducible software;
10. source-vetted empirical laboratories, research writing, formula sheets,
    algorithms, and selected solutions.

The master source is [course/quant_finance_book.tex](course/quant_finance_book.tex).
Each discipline has its own numbered folder, and common typography and mathematical
notation live in [course/preamble.tex](course/preamble.tex).

### Build

From PowerShell:

    .\scripts\build_book.ps1

The script runs pdfLaTeX twice so the table of contents, references, and PDF outline
settle. The book uses a 7-by-10-inch monograph trim, readable 12-point typography,
colored theorem structure, linked navigation, worked examples, proofs, research
notes, model-risk warnings, and exercises.

## Graduate research notebooks

The [notebook curriculum](notebooks/README.md) turns the mathematical material into
six executed research laboratories:

1. Monte Carlo convergence, variance reduction, and delta-hedging error;
2. PCA and Nelson--Siegel yield-curve factors with recursive forecast tests;
3. portfolio optimization under covariance error, bootstrap instability, and costs;
4. observable stress regimes, tail risk, VaR coverage, and backtest timing audits;
5. omitted-variable bias, weak instruments, HAC inference, and genuine out-of-sample
   forecasting;
6. implied volatility, lognormal mixtures, static arbitrage, convex quote repair,
   and Breeden--Litzenberger state-price densities.

They are not presentation-only demos. Each notebook states its information set,
derives the model, reports numerical diagnostics, compares a benchmark, constructs a
failure mode, and closes with a model-risk register and research extensions. Tables
and figures are committed as executed outputs. Re-run the complete sequence from
fresh kernels with:

    python scripts/execute_notebooks.py

## Research projects

### Regime-aware cross-asset risk

[projects/regime_risk](projects/regime_risk) studies SPY, TLT, and GLD while using
the VIX only as a lagged observable regime signal. It compares buy-and-hold, a 10%
volatility target, and a cross-asset inverse-volatility allocation after turnover
costs. It reports unconditional and regime risk, VaR/Expected Shortfall, drawdowns,
weights, turnover, and figures. Signals and volatility estimates are shifted before
they affect returns.

### Treasury yield-curve dynamics

[projects/yield_curve_dynamics](projects/yield_curve_dynamics) downloads public FRED
constant-maturity Treasury yields, estimates PCA and Nelson--Siegel factors, and
conducts an expanding-window one-month forecast against a random-walk benchmark.
The report preserves the par-yield interpretation and the December 2021 Treasury
methodology break.

### Robust factor allocation

[projects/factor_allocation](projects/factor_allocation) downloads the Fama--French
five factors and momentum from the Kenneth R. French Data Library. A walk-forward
program shrinks covariance toward its diagonal and solves a constrained
minimum-variance allocation before observing each next month, charging a transparent
turnover cost.

All projects write a provenance manifest. When online data are unavailable,
deterministic synthetic panels keep the software testable and are unmistakably
labeled as non-empirical.

## Quantitative library

[utils/quant_models.py](utils/quant_models.py) contains auditable primitives for:

- simple and log returns, realized and EWMA volatility, drawdown, VaR, Expected
  Shortfall, performance summaries, and Kupiec coverage;
- lagged volatility targeting, covariance shrinkage, PSD projection,
  minimum-variance weights, and Euler risk contributions;
- Nelson--Siegel loadings and weighted curve fitting;
- Black--Scholes calls and puts, Greeks, implied volatility, and Monte Carlo pricing.

The invariant-based test suite covers pricing parity, implied-volatility round trips,
Monte Carlo error, curve-factor recovery, PSD covariance, Euler risk attribution,
lagged signal timing, and risk formulas:

    python -m unittest discover -s tests -v

## Reproduce all projects

Create an environment with Python 3.10 or newer:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    .\scripts\run_all_research.ps1

Generated outputs live inside each project's output folder. Market-data files can be
large; review the provenance file before interpreting any table.

## Data standards

[data/SOURCES.md](data/SOURCES.md) registers primary public sources: FRED/ALFRED,
U.S. Treasury, Federal Reserve Bank of New York, SEC EDGAR, Kenneth French Data
Library, Cboe, ECB, and BIS. Each research dataset should record:

- primary producer and exact endpoint;
- event, publication, ingestion, and decision timestamps;
- instrument identifiers, units, currency, frequency, and adjustment;
- requested and returned sample, missing-value rules, and transformations;
- download time, checksum or immutable raw file, methodology break, and license note.

## Repository map

    course/       modular LaTeX monograph and compiled PDF
    projects/     end-to-end empirical laboratories and generated outputs
    utils/        tested quantitative-finance primitives
    tests/        mathematical and numerical invariants
    data/         source registry and explicitly sourced datasets
    notebooks/    executed graduate research curriculum and model-risk laboratories
    scripts/      reproducible build and research entry points

## Research ethics

This repository is educational research, not investment advice. A risk-neutral
probability is not a physical forecast, synthetic data are not evidence, a selected
backtest is not an untouched test, and a convenient data feed is not an exchange
record. Negative findings and model limitations belong in the result.
