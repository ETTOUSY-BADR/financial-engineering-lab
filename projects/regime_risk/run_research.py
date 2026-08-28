"""Run a reproducible cross-asset risk study."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.quant_models import drawdown, historical_var_cvar, log_returns, rolling_volatility

OUTPUT = ROOT / "projects" / "regime_risk" / "output"
TICKERS = ["SPY", "TLT", "GLD", "^VIX"]


def load_prices() -> tuple[pd.DataFrame, str]:
    try:
        import yfinance as yf

        prices = yf.download(TICKERS, start="2010-01-01", auto_adjust=True, progress=False)["Close"]
        prices = prices.dropna(how="all").ffill().dropna(how="all")
        if prices.empty:
            raise RuntimeError("empty download")
        return prices, "Yahoo Finance adjusted close"
    except Exception as error:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2010-01-01", periods=2500, freq="B")
        shocks = rng.normal(0.0002, 0.012, (len(dates), len(TICKERS)))
        prices = pd.DataFrame(100 * np.exp(np.cumsum(shocks, axis=0)), index=dates, columns=TICKERS)
        print(f"Network data unavailable ({error}); using labeled synthetic fallback.")
        return prices, "Synthetic GBM fallback (not market data)"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices, source = load_prices()
    returns = prices.apply(log_returns).dropna(how="all")
    summary_rows = []
    for ticker in returns:
        var95, cvar95 = historical_var_cvar(returns[ticker], 0.95)
        var99, cvar99 = historical_var_cvar(returns[ticker], 0.99)
        summary_rows.append({
            "asset": ticker,
            "annualized_volatility": returns[ticker].std() * np.sqrt(252),
            "max_drawdown": drawdown(prices[ticker]).min(),
            "VaR_95_daily_loss": var95,
            "CVaR_95_daily_loss": cvar95,
            "VaR_99_daily_loss": var99,
            "CVaR_99_daily_loss": cvar99,
        })
    summary = pd.DataFrame(summary_rows).set_index("asset")
    prices.to_csv(OUTPUT / "market_prices.csv")
    summary.to_csv(OUTPUT / "risk_summary.csv")
    rolling = returns.apply(rolling_volatility)
    rolling.to_csv(OUTPUT / "rolling_volatility.csv")
    (OUTPUT / "provenance.txt").write_text(f"Source: {source}\nRows: {len(prices)}\nGenerated UTC: {pd.Timestamp.utcnow().isoformat()}\n", encoding="utf-8")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
