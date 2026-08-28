# Regime-aware market risk

This project turns the lab's mathematical foundations into a reproducible empirical study. It downloads adjusted daily prices for a small cross-asset universe, estimates realized volatility and drawdown, and compares historical VaR/CVaR across calm and stressed regimes.

## Research questions

1. How different are equity, bond, gold, and volatility-proxy losses at the 95% and 99% levels?
2. Does a volatility-targeted portfolio reduce drawdown without hiding tail dependence?
3. Which conclusions survive a change in sample period and data source?

## Data provenance

The default source is Yahoo Finance through `yfinance`, using adjusted close prices for `SPY`, `TLT`, `GLD`, and `^VIX` from 2010 onward. The script records the download timestamp and source in `data/market_prices.csv`. If the network is unavailable, it creates a deterministic synthetic geometric-Brownian-motion panel so the educational pipeline remains executable; synthetic output is marked in the report and must never be presented as market evidence.

## Run

```powershell
python projects/regime_risk/run_research.py
```

Outputs are written to `projects/regime_risk/output/`. This is educational research, not investment advice. Historical risk estimates are model- and sample-dependent.
