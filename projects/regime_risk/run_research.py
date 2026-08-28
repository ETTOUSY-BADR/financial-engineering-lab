"""Run a reproducible, lag-aware cross-asset regime-risk study."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.quant_models import (
    performance_summary,
    rolling_volatility,
    simple_returns,
    volatility_target_returns,
)

OUTPUT = ROOT / "projects" / "regime_risk" / "output"
INVESTABLE = ["SPY", "TLT", "GLD"]
REGIME_SIGNAL = "^VIX"
TICKERS = INVESTABLE + [REGIME_SIGNAL]
START = "2010-01-01"


def synthetic_fallback() -> pd.DataFrame:
    """Correlated heavy-tailed panel for pipeline tests, never empirical claims."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(START, periods=4_200, freq="B")
    correlation = np.array([
        [1.00, -0.20, 0.05],
        [-0.20, 1.00, 0.10],
        [0.05, 0.10, 1.00],
    ])
    scale = np.array([0.011, 0.007, 0.009])
    covariance = np.diag(scale) @ correlation @ np.diag(scale)
    gaussian = rng.multivariate_normal(np.zeros(3), covariance, size=len(dates))
    mixing = np.sqrt(7.0 / rng.chisquare(7.0, size=len(dates)))[:, None]
    shocks = gaussian * mixing + np.array([0.00025, 0.00008, 0.00010])
    assets = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    vix = np.clip(14.0 + 1_100.0 * np.maximum(-shocks[:, 0], 0.0), 9.0, 80.0)
    return pd.DataFrame(np.column_stack([assets, vix]), index=dates, columns=TICKERS)


def load_prices() -> tuple[pd.DataFrame, str]:
    """Download adjusted prices; use an unmistakably labeled fallback."""
    try:
        import yfinance as yf

        raw = yf.download(
            TICKERS,
            start=START,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices = prices.reindex(columns=TICKERS).dropna(how="all").ffill()
        if prices.empty or prices[INVESTABLE].dropna().shape[0] < 500:
            raise RuntimeError("download returned too little usable history")
        return prices, "Yahoo Finance adjusted close via yfinance"
    except Exception as error:
        print(f"Market data unavailable ({error}); using labeled synthetic fallback.")
        return synthetic_fallback(), "Synthetic correlated Student-t fallback (not market evidence)"


def inverse_volatility_strategy(
    returns: pd.DataFrame,
    lookback: int = 63,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Long-only inverse-volatility allocation using only lagged data."""
    forecast = returns.rolling(lookback).std(ddof=1).shift(1) * np.sqrt(252.0)
    inverse = 1.0 / forecast.replace(0.0, np.nan)
    weights = inverse.div(inverse.sum(axis=1), axis=0).fillna(0.0)
    gross = (weights * returns.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).mul(0.5).fillna(0.0)
    cost = turnover * cost_bps * 1e-4
    result = weights.add_prefix("weight_")
    result["turnover"] = turnover
    result["gross_strategy_return"] = gross
    result["cost"] = cost
    result["net_strategy_return"] = gross - cost
    return result


def trailing_regime(vix: pd.Series) -> pd.DataFrame:
    """Observable stress regime from yesterday's VIX and trailing 75th percentile."""
    lagged = vix.shift(1)
    threshold = lagged.rolling(252, min_periods=126).quantile(0.75)
    stress = lagged >= threshold
    return pd.DataFrame({
        "lagged_vix": lagged,
        "trailing_vix_75pct": threshold,
        "stress_regime": stress.astype("Int64"),
    })


def regime_statistics(returns: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    aligned = returns.join(regimes.rename("stress_regime")).dropna()
    for label, group in aligned.groupby("stress_regime"):
        regime_name = "stress" if int(label) == 1 else "calm"
        for asset in returns.columns:
            series = group[asset]
            rows.append({
                "regime": regime_name,
                "asset": asset,
                "observations": len(series),
                "annualized_mean": series.mean() * 252.0,
                "annualized_volatility": series.std(ddof=1) * np.sqrt(252.0),
                "correlation_with_SPY": series.corr(group["SPY"]),
                "daily_1pct_return": series.quantile(0.01),
            })
    return pd.DataFrame(rows).set_index(["regime", "asset"])


def save_figures(
    returns: pd.DataFrame,
    rolling: pd.DataFrame,
    vol_target: pd.DataFrame,
    inverse_vol: pd.DataFrame,
) -> None:
    wealth = pd.DataFrame({
        "SPY buy-and-hold": (1.0 + returns["SPY"].fillna(0.0)).cumprod(),
        "SPY 10% vol target, net": (1.0 + vol_target["net_strategy_return"]).cumprod(),
        "Cross-asset inverse vol, net": (1.0 + inverse_vol["net_strategy_return"]).cumprod(),
    })
    fig, ax = plt.subplots(figsize=(10, 5.6))
    wealth.plot(ax=ax, logy=True, linewidth=1.4)
    ax.set(title="Growth of one dollar (log scale)", ylabel="Wealth", xlabel="")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "strategy_wealth.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    rolling[INVESTABLE].plot(ax=ax, linewidth=1.1)
    ax.set(title="21-day realized volatility", ylabel="Annualized volatility", xlabel="")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "rolling_volatility.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices, source = load_prices()
    returns = simple_returns(prices[INVESTABLE]).dropna(how="all")
    vix = prices[REGIME_SIGNAL].reindex(returns.index).ffill()

    risk_summary = pd.DataFrame({
        asset: performance_summary(returns[asset])
        for asset in returns.columns
    }).T
    regimes = trailing_regime(vix)
    regime_summary = regime_statistics(returns, regimes["stress_regime"])

    vol_target = volatility_target_returns(
        returns["SPY"],
        target_volatility=0.10,
        lookback=21,
        max_leverage=1.5,
        cost_bps=2.0,
    )
    inverse_vol = inverse_volatility_strategy(returns, lookback=63, cost_bps=2.0)
    strategy_summary = pd.DataFrame({
        "SPY_buy_and_hold": performance_summary(returns["SPY"]),
        "SPY_vol_target_net": performance_summary(vol_target["net_strategy_return"]),
        "inverse_vol_net": performance_summary(inverse_vol["net_strategy_return"]),
    }).T
    strategy_summary["average_daily_turnover"] = [
        0.0,
        vol_target["turnover"].mean(),
        inverse_vol["turnover"].mean(),
    ]

    rolling = returns.apply(rolling_volatility)
    prices.to_csv(OUTPUT / "market_prices.csv")
    returns.to_csv(OUTPUT / "simple_returns.csv")
    rolling.to_csv(OUTPUT / "rolling_volatility.csv")
    regimes.to_csv(OUTPUT / "observable_regimes.csv")
    risk_summary.to_csv(OUTPUT / "risk_summary.csv")
    regime_summary.to_csv(OUTPUT / "regime_summary.csv")
    vol_target.to_csv(OUTPUT / "spy_volatility_target.csv")
    inverse_vol.to_csv(OUTPUT / "inverse_volatility_strategy.csv")
    strategy_summary.to_csv(OUTPUT / "strategy_summary.csv")
    save_figures(returns, rolling, vol_target, inverse_vol)

    provenance = [
        f"Source: {source}",
        f"Requested tickers: {', '.join(TICKERS)}",
        f"Requested start: {START}",
        f"Rows: {len(prices)}",
        f"First observation: {prices.index.min()}",
        f"Last observation: {prices.index.max()}",
        f"Generated UTC: {pd.Timestamp.utcnow().isoformat()}",
        "SPY, TLT, and GLD are investable proxies; ^VIX is a signal only.",
        "Regime uses lagged VIX; every strategy volatility input is lagged.",
        "Synthetic fallback is for software validation, not market conclusions.",
    ]
    (OUTPUT / "provenance.txt").write_text("\n".join(provenance) + "\n", encoding="utf-8")
    print("Asset risk summary")
    print(risk_summary.round(4).to_string())
    print("\nStrategy summary")
    print(strategy_summary.round(4).to_string())


if __name__ == "__main__":
    main()
