"""Numerical and mathematical invariants for the shared model library."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from utils.quant_models import (
    black_scholes_call,
    black_scholes_greeks,
    black_scholes_put,
    diagonal_covariance_shrinkage,
    fit_nelson_siegel,
    global_minimum_variance_weights,
    implied_volatility_call,
    kupiec_unconditional_coverage,
    monte_carlo_call,
    nearest_positive_semidefinite,
    nelson_siegel_loadings,
    normal_var_es,
    simple_returns,
    volatility_risk_contributions,
    volatility_target_returns,
)
from utils.option_surface import (
    black76_call,
    black76_put,
    estimate_forward_discount,
    fit_svi_slice,
    heston_forward_calls,
    implied_volatility_black76,
    svi_butterfly_g,
    svi_total_variance,
)


class ReturnAndRiskTests(unittest.TestCase):
    def test_simple_returns(self) -> None:
        prices = pd.Series([100.0, 110.0, 99.0])
        actual = simple_returns(prices)
        np.testing.assert_allclose(actual.to_numpy(), [0.10, -0.10], atol=1e-12)

    def test_normal_expected_shortfall_exceeds_var(self) -> None:
        var, expected_shortfall = normal_var_es(0.0, 1.0, 0.99)
        self.assertGreater(expected_shortfall, var)
        self.assertAlmostEqual(var, 2.326347874, places=8)

    def test_kupiec_returns_exception_count(self) -> None:
        losses = [0.0, 0.1, 0.2, 0.3]
        forecasts = [0.15] * 4
        statistic, p_value, exceptions = kupiec_unconditional_coverage(
            losses, forecasts, confidence=0.5
        )
        self.assertEqual(exceptions, 2)
        self.assertAlmostEqual(statistic, 0.0, places=12)
        self.assertAlmostEqual(p_value, 1.0, places=12)

    def test_volatility_target_uses_lagged_forecast(self) -> None:
        returns = pd.Series(
            [0.01, -0.01, 0.01, -0.01, 0.50, 0.00],
            index=pd.date_range("2020-01-01", periods=6),
        )
        result = volatility_target_returns(
            returns,
            target_volatility=0.10,
            lookback=3,
            max_leverage=2.0,
            cost_bps=0.0,
        )
        self.assertAlmostEqual(result["exposure"].iloc[4], result["exposure"].iloc[3])
        self.assertLess(result["exposure"].iloc[5], result["exposure"].iloc[4])


class PortfolioTests(unittest.TestCase):
    def test_psd_projection(self) -> None:
        matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
        projected = nearest_positive_semidefinite(matrix)
        self.assertGreaterEqual(np.linalg.eigvalsh(projected).min(), -1e-12)
        np.testing.assert_allclose(projected, projected.T, atol=1e-12)

    def test_covariance_shrinkage_and_gmv(self) -> None:
        rng = np.random.default_rng(5)
        returns = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("ABC"))
        returns["C"] *= 2.0
        covariance = diagonal_covariance_shrinkage(returns, intensity=0.4)
        weights = global_minimum_variance_weights(covariance)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        equal = np.full(3, 1.0 / 3.0)
        self.assertLessEqual(
            float(weights @ covariance.to_numpy() @ weights),
            float(equal @ covariance.to_numpy() @ equal) + 1e-12,
        )

    def test_risk_contributions_add_to_volatility(self) -> None:
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = np.array([0.6, 0.4])
        contributions = volatility_risk_contributions(weights, covariance)
        volatility = math.sqrt(float(weights @ covariance @ weights))
        self.assertAlmostEqual(float(contributions.sum()), volatility, places=12)


class CurveTests(unittest.TestCase):
    def test_nelson_siegel_recovers_exact_factors(self) -> None:
        maturities = [3, 6, 12, 24, 60, 120, 360]
        beta = np.array([4.0, -1.5, 0.8])
        yields = nelson_siegel_loadings(maturities) @ beta
        estimated, fitted, rmse = fit_nelson_siegel(yields, maturities)
        np.testing.assert_allclose(estimated, beta, atol=1e-10)
        np.testing.assert_allclose(fitted, yields, atol=1e-10)
        self.assertLess(rmse, 1e-10)


class OptionTests(unittest.TestCase):
    def test_put_call_parity(self) -> None:
        call = black_scholes_call(100.0, 105.0, 1.5, 0.03, 0.24)
        put = black_scholes_put(100.0, 105.0, 1.5, 0.03, 0.24)
        parity = 100.0 - 105.0 * math.exp(-0.03 * 1.5)
        self.assertAlmostEqual(call - put, parity, places=10)

    def test_implied_volatility_round_trip(self) -> None:
        price = black_scholes_call(100.0, 90.0, 0.75, 0.02, 0.31)
        implied = implied_volatility_call(price, 100.0, 90.0, 0.75, 0.02)
        self.assertAlmostEqual(implied, 0.31, places=9)

    def test_greeks_have_expected_signs(self) -> None:
        greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.02, 0.20)
        self.assertGreater(greeks["delta"], 0.0)
        self.assertLess(greeks["delta"], 1.0)
        self.assertGreater(greeks["gamma"], 0.0)
        self.assertGreater(greeks["vega_per_unit"], 0.0)

    def test_monte_carlo_agrees_with_closed_form(self) -> None:
        closed = black_scholes_call(100.0, 100.0, 1.0, 0.03, 0.20)
        estimate, standard_error = monte_carlo_call(
            100.0, 100.0, 1.0, 0.03, 0.20, simulations=200_000, seed=19
        )
        self.assertLess(abs(estimate - closed), 4.0 * standard_error)


class OptionSurfaceTests(unittest.TestCase):
    def test_black76_parity_and_implied_volatility(self) -> None:
        call = black76_call(102.0, 100.0, 0.75, 0.97, 0.28)
        put = black76_put(102.0, 100.0, 0.75, 0.97, 0.28)
        self.assertAlmostEqual(call - put, 0.97 * 2.0, places=11)
        implied = implied_volatility_black76(call, 102.0, 100.0, 0.75, 0.97)
        self.assertAlmostEqual(implied, 0.28, places=9)

    def test_put_call_parity_recovers_forward_and_discount(self) -> None:
        strikes = np.linspace(80.0, 120.0, 17)
        forward, discount, maturity, volatility = 103.0, 0.975, 0.8, 0.24
        calls = [black76_call(forward, k, maturity, discount, volatility) for k in strikes]
        puts = [black76_put(forward, k, maturity, discount, volatility) for k in strikes]
        estimate = estimate_forward_discount(strikes, calls, puts)
        self.assertAlmostEqual(estimate.forward, forward, places=8)
        self.assertAlmostEqual(estimate.discount, discount, places=8)
        self.assertLess(estimate.rmse, 1e-8)

    def test_constrained_svi_recovers_arbitrage_free_smile(self) -> None:
        log_moneyness = np.linspace(-0.30, 0.30, 31)
        parameters = np.array([0.025, 0.12, -0.45, 0.01, 0.18])
        variance = svi_total_variance(log_moneyness, parameters)
        fitted = fit_svi_slice(log_moneyness, variance)
        dense = np.linspace(-0.5, 0.5, 201)
        self.assertLess(np.sqrt(np.mean((svi_total_variance(log_moneyness, fitted) - variance) ** 2)), 2e-4)
        self.assertGreaterEqual(float(svi_total_variance(dense, fitted).min()), 0.0)
        self.assertGreaterEqual(float(svi_butterfly_g(dense, fitted).min()), -2e-7)

    def test_heston_prices_respect_static_call_bounds(self) -> None:
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices = heston_forward_calls(
            102.0,
            strikes,
            maturity=1.0,
            discount=0.97,
            kappa=1.8,
            theta=0.04,
            vol_of_vol=0.45,
            rho=-0.65,
            initial_variance=0.05,
            integration_points=700,
        )
        intrinsic = 0.97 * np.maximum(102.0 - strikes, 0.0)
        self.assertTrue(np.all(prices >= intrinsic - 1e-8))
        self.assertTrue(np.all(prices <= 0.97 * 102.0 + 1e-8))
        self.assertTrue(np.all(np.diff(prices) <= 1e-8))
        self.assertTrue(np.all(np.diff(prices, n=2) >= -1e-5))


if __name__ == "__main__":
    unittest.main()
