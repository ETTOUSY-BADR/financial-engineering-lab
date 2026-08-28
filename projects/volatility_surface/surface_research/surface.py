"""Parity identification, IV inversion, constrained SVI, and failure recording."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from utils.option_surface import (
    black76_vega,
    estimate_forward_discount,
    fit_svi_slice,
    implied_volatility_black76,
    svi_butterfly_g,
    svi_total_variance,
)

from .config import ExperimentConfig
from .data import Snapshot


YEAR_SECONDS = 365.25 * 86_400.0
SVI_GRID = np.linspace(-0.45, 0.45, 241)


def maturity_years(expiration: str, valuation: pd.Timestamp) -> float:
    expiry = pd.Timestamp(expiration, tz="UTC") + pd.Timedelta(hours=20)
    return (expiry - valuation).total_seconds() / YEAR_SECONDS


def _valid_quotes(
    frame: pd.DataFrame, valuation: pd.Timestamp, config: ExperimentConfig
) -> pd.DataFrame:
    output = frame.copy()
    numeric = ("strike", "last_price", "bid", "ask", "volume", "open_interest")
    for column in numeric:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["last_trade_utc"] = pd.to_datetime(output["last_trade_utc"], utc=True, errors="coerce")
    trade = output["price_type"].eq("reference-session last trade")
    time_ok = output.last_trade_utc.between(
        valuation - pd.Timedelta(hours=config.study.session_hours),
        valuation + pd.Timedelta(minutes=config.study.post_close_minutes),
    )
    trade_ok = trade & time_ok & (output.last_price > 0) & (output.volume.fillna(0) > 0)
    nbbo_ok = (
        ~trade
        & time_ok
        & (output.bid > 0)
        & (output.ask >= output.bid)
        & np.isfinite(output.last_price)
    )
    return output.loc[trade_ok | nbbo_ok].copy()


def _pair_quality(
    group: pd.DataFrame, spot: float, valuation: pd.Timestamp, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    calls = group.loc[group.option_type == "call"].copy()
    puts = group.loc[group.option_type == "put"].copy()
    calls["trade_price"] = calls.last_price
    puts["trade_price"] = puts.last_price
    paired = calls.merge(puts, on="strike", suffixes=("_call", "_put"))
    paired = paired.loc[paired.strike.between(0.88 * spot, 1.12 * spot)].copy()
    gaps = (
        paired.last_trade_utc_call - paired.last_trade_utc_put
    ).abs().dt.total_seconds() / 60.0
    liquidity = np.sqrt(
        np.log1p(paired.volume_call.fillna(0).clip(lower=0))
        + np.log1p(paired.volume_put.fillna(0).clip(lower=0))
    )
    spread_call = (paired.ask_call - paired.bid_call).clip(lower=0)
    spread_put = (paired.ask_put - paired.bid_put).clip(lower=0)
    spread_penalty = 1.0 + (spread_call + spread_put).fillna(0) / max(spot * 0.001, 1e-8)
    weights = np.maximum(liquidity, 0.25) / (1.0 + gaps.fillna(390) / 30.0) / spread_penalty
    return paired, gaps, weights


def select_maturities(
    raw: pd.DataFrame,
    valuation: pd.Timestamp,
    spot: float,
    config: ExperimentConfig,
) -> tuple[list[str], pd.DataFrame]:
    """Apply deterministic quality-aware target-tenor selection."""
    valid = _valid_quotes(raw, valuation, config)
    candidates: list[dict[str, object]] = []
    for expiration, group in valid.groupby("expiration", sort=True):
        years = maturity_years(str(expiration), valuation)
        paired, gaps, _ = _pair_quality(group, spot, valuation, config)
        candidates.append(
            {
                "expiration": str(expiration),
                "days_to_expiration": years * 365.25,
                "parity_pairs_available": len(paired),
                "median_pair_gap_minutes": float(gaps.median()) if len(gaps) else np.nan,
                "eligible": (
                    years * 365.25 >= config.study.minimum_expiration_days
                    and len(paired) >= config.study.minimum_parity_pairs
                ),
                "selected_target_days": np.nan,
            }
        )
    audit = pd.DataFrame(candidates)
    selected: list[str] = []
    for target in config.study.target_days:
        eligible = audit.loc[audit.eligible & ~audit.expiration.isin(selected)].copy()
        if eligible.empty:
            break
        eligible["distance"] = (eligible.days_to_expiration - target).abs()
        chosen = eligible.sort_values(
            ["distance", "parity_pairs_available", "expiration"],
            ascending=[True, False, True],
        ).iloc[0]
        expiration = str(chosen.expiration)
        selected.append(expiration)
        audit.loc[audit.expiration == expiration, "selected_target_days"] = target
    return selected, audit


def prepare_snapshot(
    snapshot: Snapshot, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build identified OTM smiles while continuing past individual maturity failures."""
    raw = snapshot.quotes.copy()
    valuation = pd.Timestamp(str(snapshot.metadata["valuation_utc"]))
    valuation = valuation.tz_localize("UTC") if valuation.tzinfo is None else valuation.tz_convert("UTC")
    spot = float(snapshot.metadata["spot_reference"])
    selected, selection = select_maturities(raw, valuation, spot, config)
    parity_rows: list[dict[str, object]] = []
    option_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    valid = _valid_quotes(raw, valuation, config)
    for expiration in selected:
        try:
            maturity = maturity_years(expiration, valuation)
            expiry = valid.loc[valid.expiration.astype(str) == expiration].copy()
            paired, gaps, weights = _pair_quality(expiry, spot, valuation, config)
            if len(paired) < config.study.minimum_parity_pairs:
                raise ValueError(f"only {len(paired)} usable parity pairs")
            bound = math.exp(-config.study.maximum_rate * maturity)
            estimate = estimate_forward_discount(
                paired.strike,
                paired.trade_price_call,
                paired.trade_price_put,
                weights,
                discount_bounds=(bound, 1.0),
            )
            forward, discount = estimate.forward, estimate.discount
            parity_rows.append(
                {
                    "observation_date": snapshot.observation_date,
                    "expiration": expiration,
                    "maturity_years": maturity,
                    "forward": forward,
                    "discount": discount,
                    "implied_continuous_rate": -math.log(discount) / maturity,
                    "discount_at_lower_bound": abs(discount - bound) < 1e-6,
                    "discount_at_upper_bound": abs(discount - 1.0) < 1e-6,
                    "parity_rmse": estimate.rmse,
                    "parity_median_absolute_error": estimate.median_absolute_error,
                    "parity_pairs": estimate.observations,
                    "median_call_put_time_gap_minutes": float(gaps.median()),
                    "maximum_call_put_time_gap_minutes": float(gaps.max()),
                }
            )
            calls = expiry.loc[expiry.option_type == "call"]
            puts = expiry.loc[expiry.option_type == "put"]
            for strike in sorted(set(calls.strike).union(puts.strike)):
                option_type = "call" if strike >= forward else "put"
                source = calls if option_type == "call" else puts
                selected_quote = source.loc[source.strike == strike]
                if selected_quote.empty:
                    continue
                quote = selected_quote.sort_values("last_trade_utc").iloc[-1]
                price = float(quote.last_price)
                if price < config.study.minimum_option_price:
                    continue
                open_interest = float(quote.open_interest) if np.isfinite(quote.open_interest) else 0.0
                volume = float(quote.volume) if np.isfinite(quote.volume) else 0.0
                if open_interest < 1 and volume < 1 and "NBBO" not in str(quote.price_type):
                    continue
                call_price = price if option_type == "call" else price + discount * (forward - strike)
                intrinsic = discount * max(forward - strike, 0.0)
                if not intrinsic + 1e-8 < call_price < discount * forward:
                    continue
                k = math.log(strike / forward)
                if abs(k) > config.study.log_moneyness_limit:
                    continue
                try:
                    implied = implied_volatility_black76(
                        call_price, forward, strike, maturity, discount
                    )
                except ValueError:
                    continue
                if not (
                    config.study.minimum_implied_volatility
                    <= implied
                    <= config.study.maximum_implied_volatility
                ):
                    continue
                vega = black76_vega(forward, strike, maturity, discount, implied)
                noise = max(estimate.median_absolute_error, 0.05)
                iv_uncertainty = noise / max(vega, 1e-8)
                if iv_uncertainty > config.study.maximum_iv_uncertainty:
                    continue
                option_rows.append(
                    {
                        "observation_date": snapshot.observation_date,
                        "expiration": expiration,
                        "maturity_years": maturity,
                        "contract_symbol": str(quote.contract_symbol),
                        "last_trade_utc": pd.Timestamp(quote.last_trade_utc).isoformat(),
                        "minutes_from_reference_close": (
                            pd.Timestamp(quote.last_trade_utc) - valuation
                        ).total_seconds() / 60.0,
                        "source_option_type": option_type,
                        "source_price_type": str(quote.price_type),
                        "strike": float(strike),
                        "forward": forward,
                        "discount": discount,
                        "displayed_bid": float(quote.bid) if np.isfinite(quote.bid) else np.nan,
                        "displayed_ask": float(quote.ask) if np.isfinite(quote.ask) else np.nan,
                        "observed_price": price,
                        "open_interest": open_interest,
                        "volume": volume,
                        "call_equivalent_price": call_price,
                        "log_moneyness": k,
                        "implied_volatility": implied,
                        "total_variance": implied**2 * maturity,
                        "black_vega": vega,
                        "price_noise_scale": noise,
                        "iv_uncertainty_approx": iv_uncertainty,
                    }
                )
        except Exception as error:
            failures.append(
                {
                    "observation_date": snapshot.observation_date,
                    "stage": "surface_preparation",
                    "expiration": expiration,
                    "split_rule": "",
                    "reason": type(error).__name__,
                    "detail": str(error),
                }
            )
    parity = pd.DataFrame(parity_rows)
    clean = pd.DataFrame(option_rows)
    if not clean.empty:
        counts = clean.groupby("expiration").size()
        weak = counts[counts < config.study.minimum_smile_quotes]
        for expiration, count in weak.items():
            failures.append(
                {
                    "observation_date": snapshot.observation_date,
                    "stage": "liquidity_filter",
                    "expiration": expiration,
                    "split_rule": "",
                    "reason": "insufficient_smile_quotes",
                    "detail": f"{count} < {config.study.minimum_smile_quotes}",
                }
            )
        clean = clean.loc[~clean.expiration.isin(weak.index)].copy()
        parity = parity.loc[parity.expiration.isin(clean.expiration.unique())].copy()
    if clean.empty or clean.expiration.nunique() < config.study.minimum_valid_maturities:
        failures.append(
            {
                "observation_date": snapshot.observation_date,
                "stage": "date_eligibility",
                "expiration": "",
                "split_rule": "",
                "reason": "insufficient_valid_maturities",
                "detail": f"{clean.expiration.nunique() if not clean.empty else 0} valid maturities",
            }
        )
    return (
        parity.sort_values("maturity_years").reset_index(drop=True),
        clean.sort_values(["maturity_years", "strike"]).reset_index(drop=True),
        selection,
        pd.DataFrame(failures),
    )


def fit_svi_surface(
    clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit constrained slices, record repairs, and continue after failed maturities."""
    parameter_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    fitted_parts: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    previous: np.ndarray | None = None
    for expiration, group in clean.groupby("expiration", sort=True):
        try:
            group = group.sort_values("log_moneyness").copy()
            liquidity = np.log1p(group.volume) + np.log1p(group.open_interest)
            weights = np.maximum(liquidity, 0.25) / (
                group.iv_uncertainty_approx**2 + 0.005**2
            )
            weights = np.minimum(weights, np.quantile(weights, 0.90))
            parameters = fit_svi_slice(
                group.log_moneyness, group.total_variance, weights=weights, grid=SVI_GRID
            )
            pre_repair_gap = np.nan
            shift = 0.0
            if previous is not None:
                pre_repair_gap = float(
                    np.min(
                        svi_total_variance(SVI_GRID, parameters)
                        - svi_total_variance(SVI_GRID, previous)
                    )
                )
                if pre_repair_gap < 0:
                    shift = -pre_repair_gap + 1e-7
                    parameters[0] += shift
            grid_variance = svi_total_variance(SVI_GRID, parameters)
            butterfly = svi_butterfly_g(SVI_GRID, parameters)
            post_gap = np.nan if previous is None else float(
                np.min(grid_variance - svi_total_variance(SVI_GRID, previous))
            )
            if butterfly.min() < -1e-6:
                raise ValueError("calendar repair introduced butterfly arbitrage")
            maturity = float(group.maturity_years.iloc[0])
            fitted_variance = svi_total_variance(group.log_moneyness, parameters)
            fitted_vol = np.sqrt(fitted_variance / maturity)
            group["svi_total_variance"] = fitted_variance
            group["svi_implied_volatility"] = fitted_vol
            group["svi_iv_residual"] = fitted_vol - group.implied_volatility
            fitted_parts.append(group)
            base = {
                "observation_date": str(group.observation_date.iloc[0]),
                "expiration": expiration,
                "maturity_years": maturity,
            }
            parameter_rows.append(
                base
                | dict(zip(("a", "b", "rho", "m", "sigma"), parameters))
                | {
                    "calendar_vertical_shift": shift,
                    "calendar_repair_applied": shift > 0,
                }
            )
            diagnostic_rows.append(
                base
                | {
                    "quotes": len(group),
                    "iv_rmse": float(np.sqrt(np.mean(group.svi_iv_residual**2))),
                    "iv_mae": float(np.mean(group.svi_iv_residual.abs())),
                    "minimum_total_variance": float(grid_variance.min()),
                    "minimum_butterfly_g": float(butterfly.min()),
                    "minimum_calendar_gap_before_repair": pre_repair_gap,
                    "minimum_calendar_gap_after_repair": post_gap,
                    "left_wing_slope": float(parameters[1] * (1 - parameters[2])),
                    "right_wing_slope": float(parameters[1] * (1 + parameters[2])),
                    "calendar_repair_applied": shift > 0,
                }
            )
            previous = parameters.copy()
        except Exception as error:
            failures.append(
                {
                    "observation_date": str(group.observation_date.iloc[0]),
                    "stage": "svi_fit",
                    "expiration": expiration,
                    "split_rule": "",
                    "reason": type(error).__name__,
                    "detail": str(error),
                }
            )
    return (
        pd.DataFrame(parameter_rows),
        pd.concat(fitted_parts, ignore_index=True) if fitted_parts else pd.DataFrame(),
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(failures),
    )


def calendar_monotonicity(
    parameters: pd.DataFrame, grid: np.ndarray = SVI_GRID
) -> pd.DataFrame:
    """Recompute adjacent-slice calendar gaps from saved SVI parameters."""
    rows: list[dict[str, object]] = []
    for date, date_frame in parameters.groupby("observation_date", sort=True):
        previous = None
        previous_expiration = ""
        for row in date_frame.sort_values("maturity_years").itertuples():
            current = np.array([row.a, row.b, row.rho, row.m, row.sigma])
            if previous is not None:
                rows.append(
                    {
                        "observation_date": date,
                        "earlier_expiration": previous_expiration,
                        "later_expiration": row.expiration,
                        "minimum_calendar_variance_gap": float(
                            np.min(
                                svi_total_variance(grid, current)
                                - svi_total_variance(grid, previous)
                            )
                        ),
                    }
                )
            previous = current
            previous_expiration = row.expiration
    return pd.DataFrame(rows)
