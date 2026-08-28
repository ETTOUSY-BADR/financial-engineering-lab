# Regime-aware cross-asset risk

This laboratory studies equity (SPY), Treasury-duration (TLT), and gold (GLD)
returns while using the VIX index only as an observable regime signal. It implements
two fully timestamped strategies:

1. a 10% SPY volatility target whose date-t exposure uses realized volatility
   estimated only through date t-1;
2. a long-only cross-asset inverse-volatility portfolio whose weights use a lagged
   63-day window.

Both charge two basis points per unit of turnover. Outputs include unconditional and
regime-conditional risk, VaR/Expected Shortfall, drawdowns, weights, turnover,
strategy summaries, and publication-quality figures.

## Research questions

- How do conditional equity, duration, and gold risks change when yesterday's VIX is
  above its trailing 75th percentile?
- Does volatility targeting reduce drawdown after lower average exposure and costs
  are made visible?
- Does inverse-volatility allocation diversify tails, or mainly redistribute
  ordinary volatility?

## Data and timing

The default source is Yahoo Finance adjusted close through yfinance, requested from
2010 onward. The convenience feed is suitable for an open educational prototype,
not a substitute for licensed point-in-time market data. The script stores the
returned panel and a provenance manifest in the output folder.

If download fails, the pipeline creates a deterministic correlated Student-t panel
and labels every run as synthetic. Synthetic results validate code only and must
never be presented as market evidence.

The VIX is not treated as an investable total-return asset. All regime and volatility
forecasts are shifted one observation before they affect a return.

## Run

    python projects/regime_risk/run_research.py

Key outputs are strategy_summary.csv, regime_summary.csv, strategy_wealth.png,
rolling_volatility.png, and provenance.txt.
