"""Small, auditable building blocks for quantitative finance experiments."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns after dropping non-positive and missing prices."""
    clean = prices.astype(float).where(prices > 0).dropna()
    return np.log(clean).diff().dropna().rename("log_return")


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sample volatility annualized by the square-root-of-time rule."""
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def rolling_volatility(returns: pd.Series, window: int = 21, periods_per_year: int = 252) -> pd.Series:
    """Realized rolling volatility, requiring a complete window."""
    return returns.rolling(window).std(ddof=1).mul(math.sqrt(periods_per_year)).rename("volatility")


def drawdown(prices: pd.Series) -> pd.Series:
    """Percentage loss from the running maximum."""
    wealth = prices.astype(float).dropna()
    return (wealth / wealth.cummax() - 1.0).rename("drawdown")


def historical_var_cvar(returns: Iterable[float] | pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Return positive-loss historical VaR and CVaR at a confidence level."""
    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not 0 < confidence < 1:
        raise ValueError("returns must be non-empty and confidence must lie in (0, 1)")
    losses = -values
    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    return var, float(tail.mean())


def black_scholes_call(spot: float, strike: float, maturity: float, rate: float, volatility: float) -> float:
    """European call value under Black-Scholes-Merton assumptions."""
    if min(spot, strike, maturity, volatility) <= 0:
        raise ValueError("spot, strike, maturity, and volatility must be positive")
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * math.sqrt(maturity))
    d2 = d1 - volatility * math.sqrt(maturity)
    return float(spot * norm.cdf(d1) - strike * math.exp(-rate * maturity) * norm.cdf(d2))


def monte_carlo_call(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    simulations: int = 100_000,
    seed: int = 7,
) -> tuple[float, float]:
    """Price a European call and report its Monte Carlo standard error."""
    if simulations < 2:
        raise ValueError("simulations must be at least 2")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(simulations)
    terminal = spot * np.exp((rate - 0.5 * volatility**2) * maturity + volatility * math.sqrt(maturity) * shocks)
    discounted_payoffs = math.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0)
    return float(discounted_payoffs.mean()), float(discounted_payoffs.std(ddof=1) / math.sqrt(simulations))
