"""Out-of-sample volatility-model benchmarks under fixed strike splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, minimize_scalar

from utils.option_surface import (
    black76_call,
    fit_svi_slice,
    heston_forward_calls,
    svi_total_variance,
)

from .config import ExperimentConfig
from .splits import assign_split, strike_spanning_panel
from .surface import SVI_GRID


PARAMETER_NAMES = ("kappa", "theta", "vol_of_vol", "rho", "initial_variance")


def _black_prices(frame: pd.DataFrame, volatility: np.ndarray | float) -> np.ndarray:
    values = np.broadcast_to(np.asarray(volatility, dtype=float), len(frame))
    return np.array(
        [
            black76_call(row.forward, row.strike, row.maturity_years, row.discount, sigma)
            for row, sigma in zip(frame.itertuples(), values)
        ]
    )


def _flat_prediction(panel: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, object]]]:
    train = panel.loc[panel.split == "train"]
    scale = np.maximum(train.price_noise_scale.to_numpy(dtype=float), 0.05)
    fit = minimize_scalar(
        lambda sigma: float(
            np.mean(((_black_prices(train, sigma) - train.call_equivalent_price) / scale) ** 2)
        ),
        bounds=(0.03, 1.50),
        method="bounded",
    )
    sigma = float(fit.x)
    return _black_prices(panel, sigma), [{"parameter": "volatility", "estimate": sigma}]


def _svi_prediction(panel: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, object]]]:
    volatility = pd.Series(index=panel.index, dtype=float)
    parameters: list[dict[str, object]] = []
    for expiration, group in panel.groupby("expiration", sort=True):
        train = group.loc[group.split == "train"]
        weights = 1.0 / (train.iv_uncertainty_approx.to_numpy(dtype=float) ** 2 + 0.005**2)
        fitted = fit_svi_slice(
            train.log_moneyness,
            train.total_variance,
            weights=weights,
            grid=SVI_GRID,
        )
        variance = svi_total_variance(group.log_moneyness, fitted)
        volatility.loc[group.index] = np.sqrt(
            np.maximum(variance, 1e-10) / group.maturity_years.to_numpy(dtype=float)
        )
        parameters.extend(
            {
                "expiration": expiration,
                "parameter": name,
                "estimate": float(value),
            }
            for name, value in zip(("a", "b", "rho", "m", "sigma"), fitted)
        )
    return _black_prices(panel, volatility.to_numpy()), parameters


def _pchip_prediction(panel: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, object]]]:
    volatility = pd.Series(index=panel.index, dtype=float)
    parameters: list[dict[str, object]] = []
    for expiration, group in panel.groupby("expiration", sort=True):
        train = group.loc[group.split == "train"].sort_values("log_moneyness")
        if train.log_moneyness.nunique() < 2:
            raise ValueError(f"PCHIP requires two distinct training strikes for {expiration}")
        interpolator = PchipInterpolator(
            train.log_moneyness.to_numpy(),
            train.implied_volatility.to_numpy(),
            extrapolate=True,
        )
        predicted = interpolator(group.log_moneyness.to_numpy())
        predicted = np.clip(predicted, 0.03, 1.50)
        volatility.loc[group.index] = predicted
        parameters.append(
            {
                "expiration": expiration,
                "parameter": "training_knots",
                "estimate": float(len(train)),
            }
        )
    return _black_prices(panel, volatility.to_numpy()), parameters


def sabr_lognormal_volatility(
    forward: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    alpha: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Hagan lognormal SABR approximation with beta fixed at one."""
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    log_fk = np.log(forward / strike)
    z = (nu / alpha) * log_fk
    radical = np.sqrt(np.maximum(1.0 - 2.0 * rho * z + z * z, 1e-14))
    denominator = np.log(np.maximum((radical + z - rho) / (1.0 - rho), 1e-14))
    ratio = np.ones_like(z)
    non_atm = np.abs(z) > 1e-7
    ratio[non_atm] = z[non_atm] / denominator[non_atm]
    correction = 1.0 + (
        rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    ) * maturity
    return alpha * ratio * correction


def _sabr_prediction(panel: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, object]]]:
    volatility = pd.Series(index=panel.index, dtype=float)
    parameters: list[dict[str, object]] = []
    for expiration, group in panel.groupby("expiration", sort=True):
        train = group.loc[group.split == "train"]
        forward = train.forward.to_numpy(dtype=float)
        strike = train.strike.to_numpy(dtype=float)
        maturity = train.maturity_years.to_numpy(dtype=float)
        observed = train.implied_volatility.to_numpy(dtype=float)
        weights = 1.0 / np.sqrt(
            train.iv_uncertainty_approx.to_numpy(dtype=float) ** 2 + 0.005**2
        )

        def residual(values: np.ndarray) -> np.ndarray:
            predicted = sabr_lognormal_volatility(
                forward, strike, maturity, values[0], values[1], values[2]
            )
            if not np.all(np.isfinite(predicted)):
                return np.full(len(train), 1e4)
            return (predicted - observed) * weights

        atm = float(train.loc[train.log_moneyness.abs().idxmin(), "implied_volatility"])
        fit = least_squares(
            residual,
            x0=np.array([atm, -0.3, 0.5]),
            bounds=(np.array([0.01, -0.995, 0.001]), np.array([1.50, 0.995, 5.0])),
            loss="soft_l1",
            max_nfev=400,
        )
        predicted = sabr_lognormal_volatility(
            group.forward.to_numpy(dtype=float),
            group.strike.to_numpy(dtype=float),
            group.maturity_years.to_numpy(dtype=float),
            *fit.x,
        )
        volatility.loc[group.index] = np.clip(predicted, 0.03, 1.50)
        parameters.extend(
            {
                "expiration": expiration,
                "parameter": name,
                "estimate": float(value),
            }
            for name, value in zip(("alpha", "rho", "nu"), fit.x)
        )
    return _black_prices(panel, volatility.to_numpy()), parameters


def _heston_prices(frame: pd.DataFrame, values: np.ndarray, points: int) -> np.ndarray:
    prices = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("expiration", sort=False):
        prices.loc[group.index] = heston_forward_calls(
            forward=float(group.forward.iloc[0]),
            strikes=group.strike.to_numpy(dtype=float),
            maturity=float(group.maturity_years.iloc[0]),
            discount=float(group.discount.iloc[0]),
            kappa=float(values[0]),
            theta=float(values[1]),
            vol_of_vol=float(values[2]),
            rho=float(values[3]),
            initial_variance=float(values[4]),
            integration_points=points,
        )
    return prices.to_numpy()


def _heston_prediction(
    panel: pd.DataFrame, config: ExperimentConfig
) -> tuple[np.ndarray, list[dict[str, object]]]:
    train = panel.loc[panel.split == "train"].copy()
    lower = np.asarray(config.heston.lower_bounds, dtype=float)
    upper = np.asarray(config.heston.upper_bounds, dtype=float)
    scale = np.maximum(train.price_noise_scale.to_numpy(dtype=float), 0.05)

    def residual(values: np.ndarray) -> np.ndarray:
        try:
            prediction = _heston_prices(
                train, values, config.heston.calibration_integration_points
            )
            result = (prediction - train.call_equivalent_price.to_numpy(dtype=float)) / scale
            return result if np.all(np.isfinite(result)) else np.full(len(train), 1e4)
        except (FloatingPointError, ValueError, OverflowError):
            return np.full(len(train), 1e4)

    fits = []
    for index, start in enumerate(config.heston.starts, start=1):
        fit = least_squares(
            residual,
            x0=np.asarray(start, dtype=float),
            bounds=(lower, upper),
            loss="soft_l1",
            max_nfev=config.heston.maximum_function_evaluations,
            xtol=2e-7,
            ftol=2e-7,
            gtol=2e-7,
        )
        fits.append((index, fit, float(np.sum(residual(fit.x) ** 2))))
    selected_start, best, best_sse = min(fits, key=lambda item: item[2])
    records: list[dict[str, object]] = []
    for index, fit, sse in fits:
        for name, value in zip(PARAMETER_NAMES, fit.x):
            records.append(
                {
                    "start": index,
                    "selected": index == selected_start,
                    "success": bool(fit.success),
                    "function_evaluations": int(fit.nfev),
                    "scaled_residual_sse": sse,
                    "parameter": name,
                    "estimate": float(value),
                }
            )
        records.append(
            {
                "start": index,
                "selected": index == selected_start,
                "success": bool(fit.success),
                "function_evaluations": int(fit.nfev),
                "scaled_residual_sse": sse,
                "parameter": "feller_slack",
                "estimate": float(2.0 * fit.x[0] * fit.x[1] - fit.x[2] ** 2),
            }
        )
    if not np.isfinite(best_sse):
        raise RuntimeError("all Heston starts produced non-finite loss")
    return (
        _heston_prices(panel, best.x, config.heston.evaluation_integration_points),
        records,
    )


def _model_prediction(
    model: str, panel: pd.DataFrame, config: ExperimentConfig
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if model == "flat":
        return _flat_prediction(panel)
    if model == "svi":
        return _svi_prediction(panel)
    if model == "pchip":
        return _pchip_prediction(panel)
    if model == "sabr":
        return _sabr_prediction(panel)
    if model == "heston":
        return _heston_prediction(panel, config)
    raise ValueError(f"unknown model: {model}")


def validate_models(
    clean: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calibrate all configured models under every pre-declared split rule."""
    predictions: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    parameters: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for observation_date, date_frame in clean.groupby("observation_date", sort=True):
        base_panel = strike_spanning_panel(
            date_frame, config.study.maximum_quotes_per_maturity
        )
        for rule in config.splits.rules:
            try:
                panel = assign_split(
                    base_panel,
                    rule,
                    config.splits.seed,
                    config.splits.stratified_fraction,
                )
            except Exception as error:
                failures.append(
                    {
                        "observation_date": observation_date,
                        "stage": "split_assignment",
                        "expiration": "",
                        "split_rule": rule,
                        "reason": type(error).__name__,
                        "detail": str(error),
                    }
                )
                continue
            for model in config.models.names:
                try:
                    model_price, model_parameters = _model_prediction(model, panel, config)
                    result = panel.copy()
                    result["model"] = model
                    result["model_price"] = model_price
                    result["price_error"] = result.model_price - result.call_equivalent_price
                    result["iv_equivalent_error"] = (
                        result.price_error / result.black_vega.clip(lower=1e-8)
                    )
                    result["inside_noise_band"] = (
                        result.price_error.abs() <= result.price_noise_scale
                    )
                    predictions.append(result)
                    for record in model_parameters:
                        parameters.append(
                            {
                                "observation_date": observation_date,
                                "split_rule": rule,
                                "model": model,
                            }
                            | record
                        )
                    for sample, group in result.groupby("split", sort=True):
                        summaries.append(
                            {
                                "observation_date": observation_date,
                                "split_rule": rule,
                                "model": model,
                                "sample": sample,
                                "quotes": len(group),
                                "price_rmse": float(np.sqrt(np.mean(group.price_error**2))),
                                "iv_equivalent_rmse": float(
                                    np.sqrt(np.mean(group.iv_equivalent_error**2))
                                ),
                                "iv_equivalent_mae": float(
                                    np.mean(group.iv_equivalent_error.abs())
                                ),
                                "inside_noise_fraction": float(group.inside_noise_band.mean()),
                            }
                        )
                except Exception as error:
                    failures.append(
                        {
                            "observation_date": observation_date,
                            "stage": "model_validation",
                            "expiration": "",
                            "split_rule": rule,
                            "model": model,
                            "reason": type(error).__name__,
                            "detail": str(error),
                        }
                    )
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(summaries),
        pd.DataFrame(parameters),
        pd.DataFrame(failures),
    )
