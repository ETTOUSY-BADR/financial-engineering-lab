"""Auditable building blocks for quantitative-finance research."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2, norm


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns after dropping non-positive and missing prices."""
    clean = prices.astype(float).where(prices > 0).dropna()
    return np.log(clean).diff().dropna().rename("log_return")


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute simple returns after treating non-positive prices as missing."""
    clean = prices.astype(float).where(prices > 0)
    return clean.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sample volatility annualized by the square-root-of-time rule."""
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def rolling_volatility(returns: pd.Series, window: int = 21, periods_per_year: int = 252) -> pd.Series:
    """Realized rolling volatility, requiring a complete window."""
    return returns.rolling(window).std(ddof=1).mul(math.sqrt(periods_per_year)).rename("volatility")


def ewma_volatility(
    returns: pd.Series,
    span: int = 60,
    periods_per_year: int = 252,
) -> pd.Series:
    """Exponentially weighted volatility using observations through each row."""
    if span < 2 or periods_per_year < 1:
        raise ValueError("span must be at least 2 and periods_per_year positive")
    variance = returns.astype(float).ewm(span=span, adjust=False, min_periods=span).var(bias=False)
    return variance.pow(0.5).mul(math.sqrt(periods_per_year)).rename("ewma_volatility")


def drawdown(prices: pd.Series) -> pd.Series:
    """Percentage loss from the running maximum."""
    wealth = prices.astype(float).dropna()
    return (wealth / wealth.cummax() - 1.0).rename("drawdown")


def drawdown_from_returns(returns: pd.Series) -> pd.Series:
    """Drawdown path for simple returns, beginning from unit wealth."""
    clean = returns.astype(float).fillna(0.0)
    wealth = (1.0 + clean).cumprod()
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


def normal_var_es(
    mean_return: float,
    volatility: float,
    confidence: float = 0.99,
) -> tuple[float, float]:
    """Positive-loss Gaussian VaR and Expected Shortfall."""
    if volatility < 0 or not 0 < confidence < 1:
        raise ValueError("volatility must be non-negative and confidence in (0, 1)")
    z = norm.ppf(confidence)
    return (
        float(-mean_return + volatility * z),
        float(-mean_return + volatility * norm.pdf(z) / (1.0 - confidence)),
    )


def kupiec_unconditional_coverage(
    losses: Iterable[float],
    var_forecasts: Iterable[float],
    confidence: float = 0.99,
) -> tuple[float, float, int]:
    """Kupiec likelihood-ratio statistic, p-value, and VaR exception count."""
    loss = np.asarray(list(losses), dtype=float)
    forecast = np.asarray(list(var_forecasts), dtype=float)
    valid = np.isfinite(loss) & np.isfinite(forecast)
    loss, forecast = loss[valid], forecast[valid]
    if loss.size == 0 or loss.size != forecast.size or not 0 < confidence < 1:
        raise ValueError("aligned finite inputs and confidence in (0, 1) are required")
    exceptions = int(np.sum(loss > forecast))
    n = loss.size
    expected = 1.0 - confidence
    observed = float(np.clip(exceptions / n, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
    log_null = (n - exceptions) * math.log(1.0 - expected) + exceptions * math.log(expected)
    log_alt = (n - exceptions) * math.log(1.0 - observed) + exceptions * math.log(observed)
    statistic = max(0.0, -2.0 * (log_null - log_alt))
    return float(statistic), float(chi2.sf(statistic, 1)), exceptions


def performance_summary(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Return a convention-explicit performance and tail-risk summary."""
    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("returns must contain at least one finite value")
    wealth = float(np.prod(1.0 + clean))
    years = len(clean) / periods_per_year
    annual_return = wealth ** (1.0 / years) - 1.0 if wealth > 0 and years > 0 else float("nan")
    annual_vol = float(clean.std(ddof=1) * math.sqrt(periods_per_year)) if len(clean) > 1 else float("nan")
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else float("nan")
    var95, es95 = historical_var_cvar(clean, 0.95)
    var99, es99 = historical_var_cvar(clean, 0.99)
    return pd.Series({
        "observations": len(clean),
        "annualized_geometric_return": annual_return,
        "annualized_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown_from_returns(clean).min()),
        "period_VaR_95_loss": var95,
        "period_ES_95_loss": es95,
        "period_VaR_99_loss": var99,
        "period_ES_99_loss": es99,
        "skewness": float(clean.skew()),
        "excess_kurtosis": float(clean.kurt()),
    })


def volatility_target_returns(
    returns: pd.Series,
    target_volatility: float = 0.10,
    lookback: int = 21,
    max_leverage: float = 1.5,
    cost_bps: float = 2.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Backtest a lagged volatility target with proportional turnover costs."""
    if target_volatility <= 0 or lookback < 2 or max_leverage <= 0 or cost_bps < 0:
        raise ValueError("invalid volatility-target parameters")
    clean = returns.astype(float).sort_index()
    forecast = clean.rolling(lookback).std(ddof=1).mul(math.sqrt(periods_per_year)).shift(1)
    exposure = (target_volatility / forecast).clip(lower=0.0, upper=max_leverage).fillna(0.0)
    turnover = exposure.diff().abs().fillna(exposure.abs())
    gross = exposure * clean.fillna(0.0)
    cost = turnover * cost_bps * 1e-4
    return pd.DataFrame({
        "return": clean,
        "forecast_volatility": forecast,
        "exposure": exposure,
        "turnover": turnover,
        "gross_strategy_return": gross,
        "cost": cost,
        "net_strategy_return": gross - cost,
    })


def diagonal_covariance_shrinkage(
    returns: pd.DataFrame,
    intensity: float = 0.25,
) -> pd.DataFrame:
    """Shrink sample covariance toward its diagonal."""
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("intensity must lie in [0, 1]")
    clean = returns.astype(float).dropna(how="any")
    if len(clean) < 2:
        raise ValueError("at least two complete observations are required")
    sample = clean.cov()
    target = pd.DataFrame(np.diag(np.diag(sample)), index=sample.index, columns=sample.columns)
    return intensity * target + (1.0 - intensity) * sample


def nearest_positive_semidefinite(matrix: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by eigenvalue clipping."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or floor < 0:
        raise ValueError("matrix must be square and floor non-negative")
    symmetric = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def global_minimum_variance_weights(
    covariance: pd.DataFrame | np.ndarray,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Fully invested unconstrained global-minimum-variance weights."""
    sigma = np.asarray(covariance, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1] or ridge < 0:
        raise ValueError("covariance must be square and ridge non-negative")
    regularized = nearest_positive_semidefinite(sigma, floor=ridge)
    ones = np.ones(regularized.shape[0])
    solution = np.linalg.solve(regularized, ones)
    denominator = float(ones @ solution)
    if abs(denominator) < np.finfo(float).eps:
        raise ValueError("degenerate full-investment constraint")
    return solution / denominator


def volatility_risk_contributions(
    weights: Sequence[float],
    covariance: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Euler contributions to portfolio volatility."""
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    variance = float(w @ sigma @ w)
    if variance <= 0:
        raise ValueError("portfolio variance must be positive")
    return w * (sigma @ w) / math.sqrt(variance)


def nelson_siegel_loadings(maturities: Sequence[float], decay: float = 0.0609) -> np.ndarray:
    """Return level, slope, and curvature loadings for maturities in months."""
    tau = np.asarray(maturities, dtype=float)
    if np.any(tau <= 0) or decay <= 0:
        raise ValueError("maturities and decay must be positive")
    scaled = decay * tau
    slope = -np.expm1(-scaled) / scaled
    curvature = slope - np.exp(-scaled)
    return np.column_stack([np.ones_like(tau), slope, curvature])


def fit_nelson_siegel(
    yields: Sequence[float],
    maturities: Sequence[float],
    decay: float = 0.0609,
    weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted least-squares Nelson--Siegel factors, curve, and RMSE."""
    y = np.asarray(yields, dtype=float)
    loadings = nelson_siegel_loadings(maturities, decay)
    if y.shape != (loadings.shape[0],) or np.any(~np.isfinite(y)):
        raise ValueError("one finite yield is required for every maturity")
    if weights is None:
        root_weight = np.ones_like(y)
    else:
        root_weight = np.sqrt(np.asarray(weights, dtype=float))
        if root_weight.shape != y.shape or np.any(root_weight <= 0):
            raise ValueError("weights must be positive and align with yields")
    beta, *_ = np.linalg.lstsq(loadings * root_weight[:, None], y * root_weight, rcond=None)
    fitted = loadings @ beta
    return beta, fitted, float(np.sqrt(np.mean((y - fitted) ** 2)))


def black_scholes_call(spot: float, strike: float, maturity: float, rate: float, volatility: float) -> float:
    """European call value under Black-Scholes-Merton assumptions."""
    if min(spot, strike, maturity, volatility) <= 0:
        raise ValueError("spot, strike, maturity, and volatility must be positive")
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * math.sqrt(maturity))
    d2 = d1 - volatility * math.sqrt(maturity)
    return float(spot * norm.cdf(d1) - strike * math.exp(-rate * maturity) * norm.cdf(d2))


def black_scholes_put(spot: float, strike: float, maturity: float, rate: float, volatility: float) -> float:
    """European put value under Black--Scholes--Merton assumptions."""
    call = black_scholes_call(spot, strike, maturity, rate, volatility)
    return float(call - spot + strike * math.exp(-rate * maturity))


def black_scholes_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> dict[str, float]:
    """Call delta, gamma, vega, theta, and rho under the no-dividend model."""
    if min(spot, strike, maturity, volatility) <= 0:
        raise ValueError("spot, strike, maturity, and volatility must be positive")
    root_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    density = norm.pdf(d1)
    return {
        "delta": float(norm.cdf(d1)),
        "gamma": float(density / (spot * volatility * root_t)),
        "vega_per_unit": float(spot * density * root_t),
        "theta_per_year": float(
            -spot * density * volatility / (2.0 * root_t)
            - rate * strike * math.exp(-rate * maturity) * norm.cdf(d2)
        ),
        "rho_per_unit": float(strike * maturity * math.exp(-rate * maturity) * norm.cdf(d2)),
    }


def implied_volatility_call(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    lower: float = 1e-6,
    upper: float = 5.0,
) -> float:
    """Invert a European call after enforcing no-arbitrage bounds."""
    if min(spot, strike, maturity) <= 0 or lower <= 0 or upper <= lower:
        raise ValueError("invalid option inputs or volatility bracket")
    lower_price = max(0.0, spot - strike * math.exp(-rate * maturity))
    upper_price = spot
    tolerance = 1e-12
    if market_price < lower_price - tolerance or market_price > upper_price + tolerance:
        raise ValueError("market price violates no-arbitrage call bounds")
    objective = lambda sigma: black_scholes_call(spot, strike, maturity, rate, sigma) - market_price
    return float(brentq(objective, lower, upper))


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
