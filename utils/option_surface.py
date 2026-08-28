"""Arbitrage-aware option-surface and stochastic-volatility primitives.

Prices use forward coordinates whenever possible.  This keeps the discount factor
and forward explicit and avoids silently mixing spot, rates, and dividend inputs.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import least_squares, minimize
from scipy.stats import norm


@dataclass(frozen=True)
class ParityEstimate:
    """Robust put--call parity estimate and cross-sectional diagnostics."""

    forward: float
    discount: float
    rmse: float
    median_absolute_error: float
    observations: int


def black76_call(
    forward: float,
    strike: float,
    maturity: float,
    discount: float,
    volatility: float,
) -> float:
    """Discounted European call in Black's forward-price convention."""
    if min(forward, strike, maturity, discount, volatility) <= 0:
        raise ValueError("forward, strike, maturity, discount, and volatility must be positive")
    root_t = math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * volatility**2 * maturity) / (
        volatility * root_t
    )
    d2 = d1 - volatility * root_t
    return float(discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2)))


def black76_put(
    forward: float,
    strike: float,
    maturity: float,
    discount: float,
    volatility: float,
) -> float:
    """Discounted European put obtained from forward put--call parity."""
    call = black76_call(forward, strike, maturity, discount, volatility)
    return float(call - discount * (forward - strike))


def black76_vega(
    forward: float,
    strike: float,
    maturity: float,
    discount: float,
    volatility: float,
) -> float:
    """Derivative of a Black-76 option price with respect to unit volatility."""
    if min(forward, strike, maturity, discount, volatility) <= 0:
        raise ValueError("forward, strike, maturity, discount, and volatility must be positive")
    root_t = math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * volatility**2 * maturity) / (
        volatility * root_t
    )
    return float(discount * forward * norm.pdf(d1) * root_t)


def implied_volatility_black76(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    discount: float,
    lower: float = 1e-6,
    upper: float = 5.0,
) -> float:
    """Invert a call price after enforcing discounted forward-price bounds."""
    from scipy.optimize import brentq

    if min(forward, strike, maturity, discount) <= 0 or lower <= 0 or upper <= lower:
        raise ValueError("invalid option inputs or volatility bracket")
    intrinsic = discount * max(forward - strike, 0.0)
    maximum = discount * forward
    tolerance = 1e-10
    if price < intrinsic - tolerance or price > maximum + tolerance:
        raise ValueError("price violates Black-76 call bounds")
    if abs(price - intrinsic) <= tolerance:
        return lower
    objective = lambda sigma: black76_call(
        forward, strike, maturity, discount, sigma
    ) - price
    return float(brentq(objective, lower, upper))


def estimate_forward_discount(
    strikes: Iterable[float],
    call_mids: Iterable[float],
    put_mids: Iterable[float],
    weights: Iterable[float] | None = None,
    discount_bounds: tuple[float, float] = (0.50, 1.50),
) -> ParityEstimate:
    """Estimate ``C-P = discount * (forward-strike)`` using robust regression."""
    strike = np.asarray(list(strikes), dtype=float)
    calls = np.asarray(list(call_mids), dtype=float)
    puts = np.asarray(list(put_mids), dtype=float)
    if not (strike.shape == calls.shape == puts.shape):
        raise ValueError("strikes, calls, and puts must align")
    if weights is None:
        weight = np.ones_like(strike)
    else:
        weight = np.asarray(list(weights), dtype=float)
        if weight.shape != strike.shape:
            raise ValueError("weights must align with strikes")
    valid = (
        np.isfinite(strike)
        & np.isfinite(calls)
        & np.isfinite(puts)
        & np.isfinite(weight)
        & (strike > 0)
        & (calls >= 0)
        & (puts >= 0)
        & (weight > 0)
    )
    strike, calls, puts, weight = strike[valid], calls[valid], puts[valid], weight[valid]
    if len(strike) < 3:
        raise ValueError("at least three aligned call--put pairs are required")
    if not 0 < discount_bounds[0] < discount_bounds[1]:
        raise ValueError("discount bounds must be positive and ordered")
    weight = weight / np.mean(weight)
    difference = calls - puts
    initial_discount = 1.0
    initial_forward = float(np.median(difference + strike))

    def residual(parameters: np.ndarray) -> np.ndarray:
        forward, discount = parameters
        return np.sqrt(weight) * (difference - discount * (forward - strike))

    result = least_squares(
        residual,
        x0=np.array([max(initial_forward, 1e-6), initial_discount]),
        bounds=(
            np.array([1e-6, discount_bounds[0]]),
            np.array([np.inf, discount_bounds[1]]),
        ),
        loss="soft_l1",
        f_scale=max(float(np.median(np.abs(difference - np.median(difference)))), 0.05),
        max_nfev=2_000,
    )
    if not result.success:
        raise RuntimeError(f"put--call parity regression failed: {result.message}")
    forward, discount = map(float, result.x)
    raw_error = difference - discount * (forward - strike)
    return ParityEstimate(
        forward=forward,
        discount=discount,
        rmse=float(np.sqrt(np.mean(raw_error**2))),
        median_absolute_error=float(np.median(np.abs(raw_error))),
        observations=len(strike),
    )


def svi_total_variance(log_moneyness: Sequence[float] | np.ndarray, parameters: Sequence[float]) -> np.ndarray:
    """Raw-SVI total variance ``a+b(rho(k-m)+sqrt((k-m)^2+sigma^2))``."""
    k = np.asarray(log_moneyness, dtype=float)
    a, b, rho, m, sigma = np.asarray(parameters, dtype=float)
    if b < 0 or abs(rho) >= 1 or sigma <= 0:
        return np.full_like(k, np.nan)
    centered = k - m
    return a + b * (rho * centered + np.sqrt(centered**2 + sigma**2))


def svi_butterfly_g(log_moneyness: Sequence[float] | np.ndarray, parameters: Sequence[float]) -> np.ndarray:
    """Gatheral--Jacquier density condition; non-negative values exclude butterflies."""
    k = np.asarray(log_moneyness, dtype=float)
    a, b, rho, m, sigma = np.asarray(parameters, dtype=float)
    centered = k - m
    radius = np.sqrt(centered**2 + sigma**2)
    variance = svi_total_variance(k, parameters)
    first = b * (rho + centered / radius)
    second = b * sigma**2 / radius**3
    safe = np.maximum(variance, 1e-12)
    return (1.0 - k * first / (2.0 * safe)) ** 2 - 0.25 * first**2 * (
        1.0 / safe + 0.25
    ) + 0.5 * second


def fit_svi_slice(
    log_moneyness: Sequence[float],
    total_variance: Sequence[float],
    weights: Sequence[float] | None = None,
    grid: Sequence[float] | None = None,
) -> np.ndarray:
    """Fit a raw-SVI smile with positivity, wing, and dense-grid butterfly constraints."""
    k = np.asarray(log_moneyness, dtype=float)
    observed = np.asarray(total_variance, dtype=float)
    if k.shape != observed.shape:
        raise ValueError("log-moneyness and total variance must align")
    valid = np.isfinite(k) & np.isfinite(observed) & (observed > 0)
    k, observed = k[valid], observed[valid]
    if len(k) < 5:
        raise ValueError("at least five valid smile observations are required")
    if weights is None:
        weight = np.ones_like(k)
    else:
        all_weights = np.asarray(weights, dtype=float)
        if all_weights.shape != valid.shape:
            raise ValueError("weights must align with observations")
        weight = all_weights[valid]
        if np.any(~np.isfinite(weight)) or np.any(weight <= 0):
            raise ValueError("weights must be finite and positive")
    weight = weight / np.mean(weight)
    dense = np.asarray(
        grid if grid is not None else np.linspace(min(-0.60, k.min()), max(0.60, k.max()), 181),
        dtype=float,
    )
    minimum_index = int(np.argmin(observed))
    slope_scale = max((observed.max() - observed.min()) / max(np.ptp(k), 0.05), 0.02)
    initial = np.array([
        max(observed.min() * 0.70, 1e-5),
        min(slope_scale, 1.0),
        -0.30,
        float(k[minimum_index]),
        0.12,
    ])

    def objective(parameters: np.ndarray) -> float:
        fitted = svi_total_variance(k, parameters)
        return float(np.mean(weight * (fitted - observed) ** 2))

    constraints = [
        {"type": "ineq", "fun": lambda p: svi_total_variance(dense, p) - 1e-9},
        {"type": "ineq", "fun": lambda p: svi_butterfly_g(dense, p) + 1e-8},
        {"type": "ineq", "fun": lambda p: 1.999 - p[1] * (1.0 + p[2])},
        {"type": "ineq", "fun": lambda p: 1.999 - p[1] * (1.0 - p[2])},
    ]
    starts = [
        initial,
        np.array([max(float(np.median(observed)) - 2e-5, 1e-6), 1e-4, 0.0, 0.0, 0.20]),
        np.array([max(observed.min() * 0.50, 1e-6), 0.08, -0.50, 0.0, 0.20]),
        np.array([max(observed.min() * 0.75, 1e-6), 0.04, -0.20, float(k.mean()), 0.35]),
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds during a minimize step",
            category=RuntimeWarning,
        )
        results = [
            minimize(
                objective,
                start,
                method="SLSQP",
                bounds=[(-1.0, 5.0), (1e-7, 5.0), (-0.999, 0.999), (-2.0, 2.0), (1e-4, 3.0)],
                constraints=constraints,
                options={"ftol": 1e-14, "maxiter": 4_000},
            )
            for start in starts
        ]
    successful = [result for result in results if result.success]
    if not successful:
        messages = "; ".join(result.message for result in results)
        raise RuntimeError(f"constrained SVI calibration failed from all starts: {messages}")
    best = min(successful, key=lambda result: objective(result.x))
    return np.asarray(best.x, dtype=float)


def heston_characteristic_function(
    argument: complex | np.ndarray,
    forward: float,
    maturity: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    initial_variance: float,
) -> np.ndarray:
    """Characteristic function of log terminal forward under the Heston model."""
    if min(forward, maturity, kappa, theta, vol_of_vol, initial_variance) <= 0 or abs(rho) >= 1:
        raise ValueError("invalid Heston parameters")
    u = np.asarray(argument, dtype=complex)
    iu = 1j * u
    xi = vol_of_vol
    beta = kappa - rho * xi * iu
    discriminant = np.sqrt(beta**2 + xi**2 * (u**2 + iu))
    discriminant = np.where(np.real(discriminant) < 0, -discriminant, discriminant)
    g = (beta - discriminant) / (beta + discriminant)
    exponential = np.exp(-discriminant * maturity)
    log_term = np.log((1.0 - g * exponential) / (1.0 - g))
    c_term = (kappa * theta / xi**2) * (
        (beta - discriminant) * maturity - 2.0 * log_term
    )
    d_term = ((beta - discriminant) / xi**2) * (
        (1.0 - exponential) / (1.0 - g * exponential)
    )
    return np.exp(iu * math.log(forward) + c_term + d_term * initial_variance)


def heston_forward_calls(
    forward: float,
    strikes: Sequence[float],
    maturity: float,
    discount: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    initial_variance: float,
    integration_limit: float = 140.0,
    integration_points: int = 900,
) -> np.ndarray:
    """Vectorized Heston call prices using characteristic-function inversion."""
    strike = np.asarray(strikes, dtype=float)
    if strike.ndim != 1 or np.any(strike <= 0) or discount <= 0 or integration_points < 200:
        raise ValueError("invalid strikes, discount factor, or integration grid")
    u = np.linspace(1e-5, integration_limit, integration_points)
    common = dict(
        forward=forward,
        maturity=maturity,
        kappa=kappa,
        theta=theta,
        vol_of_vol=vol_of_vol,
        rho=rho,
        initial_variance=initial_variance,
    )
    phi_u = heston_characteristic_function(u, **common)
    phi_shifted = heston_characteristic_function(u - 1j, **common)
    phi_minus_i = complex(forward)
    phase = np.exp(-1j * np.outer(np.log(strike), u))
    integrand_two = np.real(phase * phi_u[None, :] / (1j * u))
    integrand_one = np.real(phase * phi_shifted[None, :] / (1j * u * phi_minus_i))
    probability_one = 0.5 + simpson(integrand_one, x=u, axis=1) / math.pi
    probability_two = 0.5 + simpson(integrand_two, x=u, axis=1) / math.pi
    prices = discount * (forward * probability_one - strike * probability_two)
    lower = discount * np.maximum(forward - strike, 0.0)
    upper = np.full_like(strike, discount * forward)
    return np.clip(np.real(prices), lower, upper)
