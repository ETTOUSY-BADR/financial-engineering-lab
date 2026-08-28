"""Build an arbitrage-aware SPY volatility surface and validate Heston out of sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.option_surface import (
    black76_call,
    black76_put,
    black76_vega,
    estimate_forward_discount,
    fit_svi_slice,
    heston_forward_calls,
    implied_volatility_black76,
    svi_butterfly_g,
    svi_total_variance,
)

PROJECT = ROOT / "projects" / "volatility_surface"
OUTPUT = PROJECT / "output"
TABLES = OUTPUT / "tables"
TICKER = "^SPX"
TARGET_DAYS = (30, 90, 180, 365)
SVI_GRID = np.linspace(-0.45, 0.45, 241)
YEAR_SECONDS = 365.25 * 24.0 * 60.0 * 60.0
COLORS = ["#071B33", "#1676B8", "#C99700", "#C84C4C", "#26866A", "#7656A5"]


def select_expirations(expirations: tuple[str, ...], as_of: pd.Timestamp) -> list[str]:
    """Create a candidate pool around transparent target tenors."""
    available = []
    for value in expirations:
        expiry = pd.Timestamp(value, tz="UTC") + pd.Timedelta(hours=20)
        days = (expiry - as_of).total_seconds() / 86_400.0
        if days >= 14:
            available.append((value, days))
    if len(available) < len(TARGET_DAYS):
        raise RuntimeError("too few listed expirations beyond fourteen days")
    selected: list[str] = []
    standard_monthlies = [
        item
        for item in available
        if pd.Timestamp(item[0]).weekday() == 4 and 15 <= pd.Timestamp(item[0]).day <= 21
    ]
    for target in TARGET_DAYS:
        candidates = sorted(available, key=lambda item: abs(item[1] - target))[:4]
        candidates += sorted(standard_monthlies, key=lambda item: abs(item[1] - target))[:2]
        for candidate, _ in candidates:
            if candidate not in selected:
                selected.append(candidate)
    return selected


def download_snapshot() -> tuple[pd.DataFrame, dict[str, object]]:
    """Download a delayed public option snapshot through yfinance."""
    import yfinance as yf

    acquired = pd.Timestamp.now(tz="UTC")
    instrument = yf.Ticker(TICKER)
    candidate_expirations = select_expirations(instrument.options, acquired)
    history = instrument.history(period="5d", auto_adjust=False)
    close = history["Close"].dropna()
    if close.empty:
        raise RuntimeError("underlying reference close is unavailable")
    spot_reference = float(close.iloc[-1])
    rows: list[pd.DataFrame] = []
    for expiration in candidate_expirations:
        chain = instrument.option_chain(expiration)
        for option_type, frame in (("call", chain.calls), ("put", chain.puts)):
            selected = frame.copy()
            selected["option_type"] = option_type
            selected["expiration"] = expiration
            selected["acquired_utc"] = acquired.isoformat()
            selected["spot_reference"] = spot_reference
            rows.append(selected)
    raw = pd.concat(rows, ignore_index=True)
    if raw.empty:
        raise RuntimeError("downloaded option chain is empty")
    trade_time = pd.to_datetime(raw["lastTradeDate"], utc=True, errors="coerce")
    trade_volume = pd.to_numeric(raw["volume"], errors="coerce").fillna(0.0)
    coverage = (
        raw.loc[trade_time.notna() & (trade_volume > 0)]
        .assign(session=trade_time.loc[trade_time.notna() & (trade_volume > 0)].dt.date)
        .groupby(["session", "expiration", "option_type"])
        .size()
        .unstack("option_type", fill_value=0)
    )
    for required in ("call", "put"):
        if required not in coverage:
            coverage[required] = 0
    usable_by_expiry = coverage[["call", "put"]].min(axis=1) >= 5
    usable_count = usable_by_expiry.groupby(level="session").sum()
    eligible_sessions = usable_count.index[usable_count >= len(TARGET_DAYS)]
    completed_sessions = [
        session
        for session in eligible_sessions
        if pd.Timestamp(f"{session.isoformat()} 16:00", tz="America/New_York").tz_convert("UTC")
        <= acquired
    ]
    if len(completed_sessions) == 0:
        raise RuntimeError("no common option-trading session has adequate maturity coverage")
    session_date = max(completed_sessions).isoformat()
    aligned_close = close.loc[[timestamp.date().isoformat() == session_date for timestamp in close.index]]
    if aligned_close.empty:
        aligned_close = close.loc[[timestamp.date().isoformat() <= session_date for timestamp in close.index]]
    if aligned_close.empty:
        raise RuntimeError("underlying close cannot be aligned to the option-trading session")
    spot_reference = float(aligned_close.iloc[-1])
    valuation = pd.Timestamp(f"{session_date} 16:00", tz="America/New_York").tz_convert("UTC")
    raw["spot_reference"] = spot_reference
    session_time = pd.to_datetime(raw["lastTradeDate"], utc=True, errors="coerce")
    session_volume = pd.to_numeric(raw["volume"], errors="coerce").fillna(0.0)
    session_mask = (
        session_time.between(valuation - pd.Timedelta(hours=6.5), valuation + pd.Timedelta(minutes=20))
        & (session_volume > 0)
    )
    session_raw = raw.loc[session_mask]
    quality: dict[str, int] = {}
    for expiration, group in session_raw.groupby("expiration"):
        calls = set(group.loc[group.option_type == "call", "strike"])
        puts = set(group.loc[group.option_type == "put", "strike"])
        overlap = [strike for strike in calls & puts if 0.88 * spot_reference <= strike <= 1.12 * spot_reference]
        quality[str(expiration)] = len(overlap)
    available = []
    for expiration in candidate_expirations:
        expiry = pd.Timestamp(expiration, tz="UTC") + pd.Timedelta(hours=20)
        days = (expiry - valuation).total_seconds() / 86_400.0
        if quality.get(expiration, 0) >= 5:
            available.append((expiration, days, quality[expiration]))
    selected_expirations: list[str] = []
    for target in TARGET_DAYS:
        candidates = [item for item in available if item[0] not in selected_expirations]
        if not candidates:
            raise RuntimeError(f"no liquid expiration available near the {target}-day target")
        chosen = min(candidates, key=lambda item: (abs(item[1] - target), -item[2]))
        selected_expirations.append(chosen[0])
    raw = raw.loc[raw.expiration.isin(selected_expirations)].reset_index(drop=True)
    metadata: dict[str, object] = {
        "ticker": TICKER,
        "acquired_utc": acquired.isoformat(),
        "spot_reference": spot_reference,
        "reference_session_date": session_date,
        "valuation_utc": valuation.isoformat(),
        "expirations": selected_expirations,
        "candidate_expirations": candidate_expirations,
        "same_session_pair_counts": {key: quality[key] for key in selected_expirations},
        "source": "Yahoo Finance delayed option chain via yfinance",
        "evidence_class": "public delayed snapshot; not an exchange record",
    }
    return raw, metadata


def synthetic_snapshot() -> tuple[pd.DataFrame, dict[str, object]]:
    """Create a deterministic, visibly labeled quote panel for software validation."""
    acquired = pd.Timestamp("2026-01-02T16:00:00Z")
    spot = 600.0
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(2026)
    for days in TARGET_DAYS:
        maturity = days / 365.25
        expiration = (acquired + pd.Timedelta(days=days)).date().isoformat()
        discount = math.exp(-0.035 * maturity)
        forward = spot * math.exp(0.012 * maturity)
        for strike in np.arange(0.72 * spot, 1.29 * spot, 5.0):
            k = math.log(strike / forward)
            variance = 0.028 * maturity + 0.055 * maturity * (
                -0.45 * (k + 0.02) + math.sqrt((k + 0.02) ** 2 + 0.11**2)
            )
            volatility = math.sqrt(max(variance / maturity, 1e-8))
            call = black76_call(forward, strike, maturity, discount, volatility)
            put = black76_put(forward, strike, maturity, discount, volatility)
            for option_type, price in (("call", call), ("put", put)):
                spread = max(0.04, 0.015 * price) * (1.0 + 0.1 * rng.random())
                rows.append({
                    "contractSymbol": f"SYNTH{expiration}{option_type[0].upper()}{int(strike*1000):08d}",
                    "lastTradeDate": acquired.isoformat(),
                    "strike": strike,
                    "lastPrice": price,
                    "bid": max(0.01, price - spread / 2.0),
                    "ask": price + spread / 2.0,
                    "volume": 100,
                    "openInterest": 1_000,
                    "impliedVolatility": volatility,
                    "inTheMoney": (strike < forward) if option_type == "call" else (strike > forward),
                    "contractSize": "REGULAR",
                    "currency": "USD",
                    "option_type": option_type,
                    "expiration": expiration,
                    "acquired_utc": acquired.isoformat(),
                    "spot_reference": spot,
                })
    metadata: dict[str, object] = {
        "ticker": "SYNTHETIC_SPY_LIKE",
        "acquired_utc": acquired.isoformat(),
        "spot_reference": spot,
        "reference_session_date": acquired.date().isoformat(),
        "valuation_utc": acquired.isoformat(),
        "expirations": sorted({str(row["expiration"]) for row in rows}),
        "source": "Deterministic synthetic SVI fallback",
        "evidence_class": "software validation only; not empirical evidence",
    }
    return pd.DataFrame(rows), metadata


def snapshot(refresh: bool) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the committed snapshot unless a refresh is requested or no snapshot exists."""
    raw_path = OUTPUT / "raw_option_quotes.csv"
    metadata_path = OUTPUT / "snapshot_metadata.json"
    if raw_path.exists() and metadata_path.exists() and not refresh:
        return pd.read_csv(raw_path), json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        return download_snapshot()
    except Exception as error:
        print(f"Public option data unavailable ({error}); using labeled synthetic fallback.")
        return synthetic_snapshot()


def valid_near_close_trade(frame: pd.DataFrame, valuation: pd.Timestamp) -> pd.Series:
    """Identify positive trades printed during the complete reference session."""
    last = pd.to_numeric(frame["lastPrice"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    timestamp = pd.to_datetime(frame["lastTradeDate"], utc=True, errors="coerce")
    return (
        last.notna()
        & (last > 0)
        & volume.notna()
        & (volume > 0)
        & timestamp.between(valuation - pd.Timedelta(hours=6.5), valuation + pd.Timedelta(minutes=20))
    )


def expiry_maturity(expiration: str, acquired: pd.Timestamp) -> float:
    """Year fraction to the regular-session close on expiration day."""
    expiry = pd.Timestamp(expiration, tz="UTC") + pd.Timedelta(hours=20)
    return (expiry - acquired).total_seconds() / YEAR_SECONDS


def clean_surface(
    raw: pd.DataFrame,
    metadata: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infer parity inputs and create a single liquid OTM quote at each strike."""
    valuation = pd.Timestamp(str(metadata["valuation_utc"]))
    if valuation.tzinfo is None:
        valuation = valuation.tz_localize("UTC")
    else:
        valuation = valuation.tz_convert("UTC")
    spot = float(metadata["spot_reference"])
    parity_rows: list[dict[str, object]] = []
    option_rows: list[dict[str, object]] = []
    for expiration in sorted(raw["expiration"].astype(str).unique()):
        maturity = expiry_maturity(expiration, valuation)
        if maturity <= 0:
            continue
        expiry = raw.loc[raw["expiration"].astype(str) == expiration].copy()
        expiry = expiry.loc[valid_near_close_trade(expiry, valuation)]
        calls = expiry.loc[expiry.option_type == "call"].copy()
        puts = expiry.loc[expiry.option_type == "put"].copy()
        calls["trade_price"] = pd.to_numeric(calls.lastPrice, errors="coerce")
        puts["trade_price"] = pd.to_numeric(puts.lastPrice, errors="coerce")
        calls["trade_time"] = pd.to_datetime(calls.lastTradeDate, utc=True)
        puts["trade_time"] = pd.to_datetime(puts.lastTradeDate, utc=True)
        paired = calls.merge(puts, on="strike", suffixes=("_call", "_put"))
        paired = paired.loc[paired.strike.between(0.88 * spot, 1.12 * spot)].copy()
        if len(paired) < 5:
            raise RuntimeError(f"insufficient parity pairs for {expiration}: {len(paired)}")
        time_gap = (paired.trade_time_call - paired.trade_time_put).abs().dt.total_seconds() / 60.0
        pair_liquidity = np.sqrt(
            np.log1p(pd.to_numeric(paired.volume_call, errors="coerce").fillna(0.0))
            + np.log1p(pd.to_numeric(paired.volume_put, errors="coerce").fillna(0.0))
        )
        parity_weight = np.maximum(pair_liquidity, 0.25) / (1.0 + time_gap / 30.0)
        estimate = estimate_forward_discount(
            paired.strike,
            paired.trade_price_call,
            paired.trade_price_put,
            parity_weight,
            discount_bounds=(math.exp(-0.15 * maturity), 1.0),
        )
        forward, discount = estimate.forward, estimate.discount
        parity_rows.append({
            "expiration": expiration,
            "maturity_years": maturity,
            "forward": forward,
            "discount": discount,
            "implied_continuous_rate": -math.log(discount) / maturity,
            "discount_at_lower_bound": abs(discount - math.exp(-0.15 * maturity)) < 1e-6,
            "discount_at_upper_bound": abs(discount - 1.0) < 1e-6,
            "parity_rmse": estimate.rmse,
            "parity_median_absolute_error": estimate.median_absolute_error,
            "parity_pairs": estimate.observations,
            "median_call_put_time_gap_minutes": float(time_gap.median()),
        })

        for strike in sorted(set(calls.strike).union(puts.strike)):
            option_type = "call" if strike >= forward else "put"
            source = calls if option_type == "call" else puts
            selected = source.loc[source.strike == strike]
            if selected.empty:
                continue
            quote = selected.iloc[0]
            trade_price = float(quote.trade_price)
            if trade_price < 0.05:
                continue
            open_interest = float(pd.to_numeric(quote.get("openInterest", 0), errors="coerce"))
            volume = float(pd.to_numeric(quote.get("volume", 0), errors="coerce"))
            if not np.isfinite(open_interest):
                open_interest = 0.0
            if not np.isfinite(volume):
                volume = 0.0
            if open_interest < 1 and volume < 1:
                continue
            call_equivalent = (
                trade_price if option_type == "call"
                else trade_price + discount * (forward - strike)
            )
            intrinsic = discount * max(forward - strike, 0.0)
            if call_equivalent <= intrinsic + 1e-8 or call_equivalent >= discount * forward:
                continue
            log_moneyness = math.log(strike / forward)
            if abs(log_moneyness) > 0.35:
                continue
            try:
                implied = implied_volatility_black76(
                    call_equivalent, forward, strike, maturity, discount
                )
            except ValueError:
                continue
            if not 0.03 <= implied <= 1.50:
                continue
            vega = black76_vega(forward, strike, maturity, discount, implied)
            price_noise = max(estimate.median_absolute_error, 0.05)
            iv_uncertainty = price_noise / max(vega, 1e-8)
            if iv_uncertainty > 0.30:
                continue
            trade_time = pd.Timestamp(quote.trade_time)
            option_rows.append({
                "expiration": expiration,
                "maturity_years": maturity,
                "contract_symbol": quote.get("contractSymbol", ""),
                "last_trade_utc": trade_time.isoformat(),
                "minutes_from_reference_close": (trade_time - valuation).total_seconds() / 60.0,
                "source_option_type": option_type,
                "source_price_type": "reference-session last trade",
                "strike": float(strike),
                "forward": forward,
                "discount": discount,
                "displayed_bid": float(pd.to_numeric(quote.get("bid", np.nan), errors="coerce")),
                "displayed_ask": float(pd.to_numeric(quote.get("ask", np.nan), errors="coerce")),
                "last_trade_price": trade_price,
                "open_interest": open_interest,
                "volume": volume,
                "call_equivalent_mid": call_equivalent,
                "log_moneyness": log_moneyness,
                "implied_volatility": implied,
                "total_variance": implied**2 * maturity,
                "black_vega": vega,
                "price_noise_scale": price_noise,
                "iv_uncertainty_approx": iv_uncertainty,
            })
    parity = pd.DataFrame(parity_rows).sort_values("maturity_years").reset_index(drop=True)
    clean = pd.DataFrame(option_rows).sort_values(["maturity_years", "strike"]).reset_index(drop=True)
    counts = clean.groupby("expiration").size()
    if len(counts) < 3 or (counts < 9).any():
        raise RuntimeError(f"insufficient clean smile observations: {counts.to_dict()}")
    return parity, clean


def fit_svi_surface(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit constrained smiles and repair calendar total-variance crossings by vertical shifts."""
    parameter_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    fitted_parts: list[pd.DataFrame] = []
    previous: np.ndarray | None = None
    for expiration, group in clean.groupby("expiration", sort=False):
        group = group.sort_values("log_moneyness").copy()
        liquidity = np.log1p(group.volume.to_numpy()) + np.log1p(group.open_interest.to_numpy())
        weights = np.maximum(liquidity, 0.25) / (
            group.iv_uncertainty_approx.to_numpy() ** 2 + 0.005**2
        )
        weights = np.minimum(weights, np.quantile(weights, 0.90))
        parameters = fit_svi_slice(
            group.log_moneyness,
            group.total_variance,
            weights=weights,
            grid=SVI_GRID,
        )
        calendar_shift = 0.0
        if previous is not None:
            gap = svi_total_variance(SVI_GRID, parameters) - svi_total_variance(SVI_GRID, previous)
            if float(gap.min()) < 0:
                calendar_shift = -float(gap.min()) + 1e-7
                parameters[0] += calendar_shift
        grid_variance = svi_total_variance(SVI_GRID, parameters)
        butterfly = svi_butterfly_g(SVI_GRID, parameters)
        if float(butterfly.min()) < -1e-6:
            raise RuntimeError(f"calendar repair introduced butterfly arbitrage at {expiration}")
        maturity = float(group.maturity_years.iloc[0])
        fitted_variance = svi_total_variance(group.log_moneyness, parameters)
        fitted_vol = np.sqrt(fitted_variance / maturity)
        group["svi_total_variance"] = fitted_variance
        group["svi_implied_volatility"] = fitted_vol
        group["svi_iv_residual"] = fitted_vol - group.implied_volatility
        group["svi_inside_empirical_noise"] = (
            group.svi_iv_residual.abs() <= group.iv_uncertainty_approx
        )
        fitted_parts.append(group)
        calendar_gap = np.nan if previous is None else float(
            np.min(grid_variance - svi_total_variance(SVI_GRID, previous))
        )
        parameter_rows.append({
            "expiration": expiration,
            "maturity_years": maturity,
            "a": parameters[0],
            "b": parameters[1],
            "rho": parameters[2],
            "m": parameters[3],
            "sigma": parameters[4],
            "calendar_vertical_shift": calendar_shift,
        })
        diagnostics.append({
            "expiration": expiration,
            "quotes": len(group),
            "iv_rmse": float(np.sqrt(np.mean(group.svi_iv_residual**2))),
            "iv_mae": float(np.mean(np.abs(group.svi_iv_residual))),
            "inside_empirical_noise_fraction": float(group.svi_inside_empirical_noise.mean()),
            "minimum_total_variance": float(grid_variance.min()),
            "minimum_butterfly_g": float(butterfly.min()),
            "minimum_calendar_variance_gap": calendar_gap,
            "left_wing_slope": float(parameters[1] * (1.0 - parameters[2])),
            "right_wing_slope": float(parameters[1] * (1.0 + parameters[2])),
        })
        previous = parameters.copy()
    return (
        pd.DataFrame(parameter_rows),
        pd.concat(fitted_parts, ignore_index=True),
        pd.DataFrame(diagnostics),
    )


def calibration_subset(clean: pd.DataFrame, maximum_per_expiry: int = 25) -> pd.DataFrame:
    """Build a deterministic strike-spanning subset and hold out every fourth quote."""
    parts = []
    for _, group in clean.groupby("expiration", sort=False):
        ordered = group.sort_values("log_moneyness").reset_index(drop=True)
        if len(ordered) > maximum_per_expiry:
            locations = np.unique(np.linspace(0, len(ordered) - 1, maximum_per_expiry).round().astype(int))
            ordered = ordered.iloc[locations].reset_index(drop=True)
        ordered["split"] = np.where(ordered.index % 4 == 1, "holdout", "train")
        parts.append(ordered)
    return pd.concat(parts, ignore_index=True)


def heston_price_panel(frame: pd.DataFrame, parameters: np.ndarray, points: int = 520) -> np.ndarray:
    """Price a panel by grouping contracts that share maturity and parity inputs."""
    output = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("expiration", sort=False):
        output.loc[group.index] = heston_forward_calls(
            forward=float(group.forward.iloc[0]),
            strikes=group.strike.to_numpy(),
            maturity=float(group.maturity_years.iloc[0]),
            discount=float(group.discount.iloc[0]),
            kappa=float(parameters[0]),
            theta=float(parameters[1]),
            vol_of_vol=float(parameters[2]),
            rho=float(parameters[3]),
            initial_variance=float(parameters[4]),
            integration_points=points,
        )
    return output.to_numpy()


def calibrate_heston(
    clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Multi-start Heston calibration with deterministic strike holdout and flat-vol null."""
    panel = calibration_subset(clean)
    train = panel.loc[panel.split == "train"].copy()
    lower = np.array([0.05, 0.005, 0.05, -0.99, 0.005])
    upper = np.array([12.0, 0.60, 3.00, 0.99, 0.60])
    starts = [
        np.array([1.5, 0.04, 0.40, -0.70, 0.04]),
        np.array([0.5, 0.06, 0.80, -0.40, 0.05]),
        np.array([3.0, 0.03, 0.25, -0.85, 0.03]),
        np.array([5.0, 0.08, 1.20, -0.20, 0.08]),
        np.array([1.0, 0.12, 1.80, 0.20, 0.10]),
    ]

    def residual(parameters: np.ndarray) -> np.ndarray:
        try:
            model = heston_price_panel(train, parameters)
        except (FloatingPointError, ValueError):
            return np.full(len(train), 1e4)
        scale = np.maximum(train.black_vega.to_numpy() * 0.01, 0.20)
        values = (model - train.call_equivalent_mid.to_numpy()) / scale
        if np.any(~np.isfinite(values)):
            return np.full(len(train), 1e4)
        return values

    results = []
    for index, start in enumerate(starts, start=1):
        fit = least_squares(
            residual,
            x0=start,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=240,
            xtol=2e-7,
            ftol=2e-7,
            gtol=2e-7,
        )
        results.append((index, fit))
    best_index, best = min(results, key=lambda item: float(np.sum(residual(item[1].x) ** 2)))
    names = ["kappa", "theta", "vol_of_vol", "rho", "initial_variance"]
    multistart_rows = []
    for start_index, fit in results:
        row: dict[str, object] = {
            "start": start_index,
            "success": bool(fit.success),
            "function_evaluations": int(fit.nfev),
            "scaled_residual_sse": float(np.sum(residual(fit.x) ** 2)),
        }
        row.update(dict(zip(names, fit.x)))
        row["feller_slack"] = 2.0 * fit.x[0] * fit.x[1] - fit.x[2] ** 2
        multistart_rows.append(row)
    multistart = pd.DataFrame(multistart_rows)

    degrees = max(len(train) - len(best.x), 1)
    residual_variance = float(np.sum(best.fun**2) / degrees)
    covariance = residual_variance * np.linalg.pinv(best.jac.T @ best.jac)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    parameters = pd.DataFrame({
        "parameter": names,
        "estimate": best.x,
        "local_standard_error": standard_error,
        "lower_bound": lower,
        "upper_bound": upper,
    })
    parameters["near_boundary"] = (
        (parameters.estimate - parameters.lower_bound < 0.01 * (upper - lower))
        | (parameters.upper_bound - parameters.estimate < 0.01 * (upper - lower))
    )
    parameters["selected_start"] = best_index

    panel["heston_price"] = heston_price_panel(panel, best.x, points=760)
    flat_fit = minimize_scalar(
        lambda volatility: float(np.mean(
            ((np.array([
                black76_call(row.forward, row.strike, row.maturity_years, row.discount, volatility)
                for row in train.itertuples()
            ]) - train.call_equivalent_mid.to_numpy()) / np.maximum(train.black_vega, 0.20)) ** 2
        )),
        bounds=(0.03, 1.50),
        method="bounded",
    )
    panel["flat_volatility"] = float(flat_fit.x)
    panel["flat_price"] = [
        black76_call(row.forward, row.strike, row.maturity_years, row.discount, flat_fit.x)
        for row in panel.itertuples()
    ]
    panel["heston_price_error"] = panel.heston_price - panel.call_equivalent_mid
    panel["flat_price_error"] = panel.flat_price - panel.call_equivalent_mid
    panel["heston_iv_equivalent_error"] = panel.heston_price_error / panel.black_vega
    panel["flat_iv_equivalent_error"] = panel.flat_price_error / panel.black_vega
    panel["heston_inside_noise_band"] = panel.heston_price_error.abs() <= panel.price_noise_scale
    panel["flat_inside_noise_band"] = panel.flat_price_error.abs() <= panel.price_noise_scale

    summary_rows = []
    for split, group in panel.groupby("split"):
        for model in ("heston", "flat"):
            summary_rows.append({
                "model": model,
                "sample": split,
                "quotes": len(group),
                "price_rmse": float(np.sqrt(np.mean(group[f"{model}_price_error"] ** 2))),
                "iv_equivalent_rmse": float(np.sqrt(np.mean(group[f"{model}_iv_equivalent_error"] ** 2))),
                "iv_equivalent_mae": float(np.mean(np.abs(group[f"{model}_iv_equivalent_error"]))),
                "inside_empirical_noise_fraction": float(group[f"{model}_inside_noise_band"].mean()),
            })
    summary = pd.DataFrame(summary_rows).sort_values(["sample", "model"])
    return parameters, multistart, panel, summary


def save_figures(
    clean: pd.DataFrame,
    svi_parameters: pd.DataFrame,
    svi_fitted: pd.DataFrame,
    heston_validation: pd.DataFrame,
) -> None:
    """Save publication-style calibration and validation figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for (expiration, group), color in zip(clean.groupby("expiration"), COLORS):
        ax.scatter(group.log_moneyness, 100 * group.implied_volatility, s=18, alpha=0.75,
                   label=expiration, color=color)
    ax.set(title="Clean OTM SPX implied-volatility smiles", xlabel="Log-moneyness log(K/F)",
           ylabel="Black implied volatility (%)")
    ax.legend(title="Expiration", frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(OUTPUT / "clean_implied_smiles.png", dpi=190); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for (expiration, group), color in zip(svi_fitted.groupby("expiration"), COLORS):
        row = svi_parameters.loc[svi_parameters.expiration == expiration].iloc[0]
        parameters = row[["a", "b", "rho", "m", "sigma"]].to_numpy(dtype=float)
        grid = np.linspace(group.log_moneyness.min(), group.log_moneyness.max(), 180)
        vol = np.sqrt(svi_total_variance(grid, parameters) / float(group.maturity_years.iloc[0]))
        ax.scatter(group.log_moneyness, 100 * group.implied_volatility, s=13, alpha=.55, color=color)
        ax.plot(grid, 100 * vol, linewidth=2, color=color, label=expiration)
    ax.set(title="Constrained SVI fits", xlabel="Log-moneyness log(K/F)",
           ylabel="Black implied volatility (%)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(OUTPUT / "svi_fitted_smiles.png", dpi=190); plt.close(fig)

    holdout = heston_validation.loc[heston_validation.split == "holdout"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    axes[0].scatter(holdout.log_moneyness, 100 * holdout.heston_iv_equivalent_error,
                    c=holdout.maturity_years, cmap="viridis", s=32)
    axes[1].scatter(holdout.log_moneyness, 100 * holdout.flat_iv_equivalent_error,
                    c=holdout.maturity_years, cmap="viridis", s=32)
    for axis, title in zip(axes, ("Heston holdout residuals", "Flat-volatility holdout residuals")):
        axis.axhline(0, color="#071B33", linewidth=1)
        axis.set(title=title, xlabel="Log-moneyness", ylabel="IV-equivalent error (vol points)")
    fig.tight_layout(); fig.savefig(OUTPUT / "heston_holdout_residuals.png", dpi=190); plt.close(fig)

    representative = svi_parameters.iloc[len(svi_parameters) // 2]
    parameters = representative[["a", "b", "rho", "m", "sigma"]].to_numpy(dtype=float)
    forward = float(svi_fitted.loc[svi_fitted.expiration == representative.expiration, "forward"].iloc[0])
    discount = float(svi_fitted.loc[svi_fitted.expiration == representative.expiration, "discount"].iloc[0])
    maturity = float(representative.maturity_years)
    k_grid = np.linspace(-0.38, 0.38, 500)
    strikes = forward * np.exp(k_grid)
    vols = np.sqrt(svi_total_variance(k_grid, parameters) / maturity)
    calls = np.array([black76_call(forward, strike, maturity, discount, vol)
                      for strike, vol in zip(strikes, vols)])
    density = np.gradient(np.gradient(calls, strikes), strikes) / discount
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(strikes, density, color="#1676B8", linewidth=2)
    ax.axhline(0, color="#071B33", linewidth=1)
    ax.set(title=f"SVI-implied risk-neutral density: {representative.expiration}",
           xlabel="Terminal SPX level", ylabel="Density")
    fig.tight_layout(); fig.savefig(OUTPUT / "svi_state_price_density.png", dpi=190); plt.close(fig)


def write_tables(
    parity: pd.DataFrame,
    diagnostics: pd.DataFrame,
    parameters: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Create compact report-ready LaTeX tables from the exact CSV results."""
    TABLES.mkdir(parents=True, exist_ok=True)
    parity_view = parity[[
        "expiration", "maturity_years", "forward", "discount", "parity_rmse",
        "parity_pairs", "discount_at_upper_bound"
    ]]
    latex_options = {"index": False, "escape": True}
    parity_view.to_latex(
        TABLES / "parity.tex", float_format="%.4f", **latex_options
    )
    diagnostics.to_latex(
        TABLES / "svi_diagnostics.tex", float_format="%.5f", **latex_options
    )
    parameters.to_latex(
        TABLES / "heston_parameters.tex", float_format="%.5f", **latex_options
    )
    summary.to_latex(
        TABLES / "heston_validation.tex", float_format="%.5f", **latex_options
    )


def write_provenance(metadata: dict[str, object], raw_path: Path, counts: dict[str, int]) -> None:
    """Record data lineage, evidence limitations, and primary model references."""
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    lines = [
        f"Ticker: {metadata['ticker']}",
        f"Source: {metadata['source']}",
        f"Evidence class: {metadata['evidence_class']}",
        f"Acquired UTC: {metadata['acquired_utc']}",
        f"Reference session date: {metadata['reference_session_date']}",
        f"Valuation timestamp UTC: {metadata['valuation_utc']}",
        f"Underlying reference close: {metadata['spot_reference']}",
        f"Selected expirations: {', '.join(map(str, metadata['expirations']))}",
        f"Raw quote rows: {counts['raw']}",
        f"Clean OTM quote rows: {counts['clean']}",
        f"Raw CSV SHA-256: {raw_hash}",
        "Price convention: positive last trades during the complete reference session through twenty minutes after close.",
        "The transaction cross section is asynchronous and is not an executable simultaneous surface.",
        "Forward/discount convention: liquidity- and time-gap-weighted robust put-call parity regression.",
        "Discount identification is bounded to continuous rates between 0% and 15%; boundary hits are reported.",
        "Parity median absolute residual is propagated as an empirical price-noise scale.",
        "Expiration clock: 20:00 UTC regular-session approximation; ACT/365.25 year fraction.",
        "Yahoo Finance is a public distributor, not an exchange record or OPRA archive.",
        "Displayed bid/ask fields are retained but rejected as a calibration source when markets are absent.",
        "OCC/OIC quote reference: https://www.optionseducation.org/news/understanding-the-bid-and-ask-prices-for-options",
        "SVI reference: Gatheral and Jacquier (2014), https://arxiv.org/abs/1204.0646",
        "Heston reference: Heston (1993), https://doi.org/10.1093/rfs/6.2.327",
        "Synthetic fallback, when used, validates software only and cannot support market claims.",
    ]
    (OUTPUT / "provenance.txt").write_bytes(
        ("\n".join(lines) + "\n").encode("utf-8")
    )


def write_manifest() -> None:
    """Hash every generated CSV and metadata file for integrity checks."""
    targets = sorted([*OUTPUT.glob("*.csv"), *OUTPUT.glob("*.json")])
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in targets]
    (OUTPUT / "manifest.sha256").write_bytes(
        ("\n".join(lines) + "\n").encode("utf-8")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Download a new public snapshot")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    raw, metadata = snapshot(args.refresh)
    raw_path = OUTPUT / "raw_option_quotes.csv"
    raw.to_csv(raw_path, index=False, lineterminator="\n")
    (OUTPUT / "snapshot_metadata.json").write_bytes(
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    parity, clean = clean_surface(raw, metadata)
    svi_parameters, svi_fitted, svi_diagnostics = fit_svi_surface(clean)
    heston_parameters, heston_multistart, heston_validation, calibration_summary = calibrate_heston(clean)

    csv_options = {"index": False, "lineterminator": "\n"}
    parity.to_csv(OUTPUT / "parity_estimates.csv", **csv_options)
    clean.to_csv(OUTPUT / "clean_otm_quotes.csv", **csv_options)
    svi_parameters.to_csv(OUTPUT / "svi_parameters.csv", **csv_options)
    svi_fitted.to_csv(OUTPUT / "svi_fitted_quotes.csv", **csv_options)
    svi_diagnostics.to_csv(OUTPUT / "svi_diagnostics.csv", **csv_options)
    heston_parameters.to_csv(OUTPUT / "heston_parameters.csv", **csv_options)
    heston_multistart.to_csv(OUTPUT / "heston_multistart.csv", **csv_options)
    heston_validation.to_csv(OUTPUT / "heston_validation.csv", **csv_options)
    calibration_summary.to_csv(OUTPUT / "calibration_summary.csv", **csv_options)
    save_figures(clean, svi_parameters, svi_fitted, heston_validation)
    write_tables(parity, svi_diagnostics, heston_parameters, calibration_summary)
    write_provenance(metadata, raw_path, {"raw": len(raw), "clean": len(clean)})
    write_manifest()

    print("Put--call parity estimates")
    print(parity.round(5).to_string(index=False))
    print("\nConstrained SVI diagnostics")
    print(svi_diagnostics.round(5).to_string(index=False))
    print("\nHeston parameters")
    print(heston_parameters.round(5).to_string(index=False))
    print("\nHeston holdout validation")
    print(calibration_summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
