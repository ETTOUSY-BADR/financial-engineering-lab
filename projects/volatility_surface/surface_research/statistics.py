"""Date-level loss aggregation and explicitly gated inference."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import ExperimentConfig


def date_level_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quote errors before inference so quotes are not pseudo-replicates."""
    if predictions.empty:
        return pd.DataFrame()
    result = (
        predictions.assign(
            squared_iv_error=predictions.iv_equivalent_error**2,
            absolute_iv_error=predictions.iv_equivalent_error.abs(),
        )
        .groupby(
            ["observation_date", "split_rule", "model", "split"], as_index=False
        )
        .agg(
            quotes=("iv_equivalent_error", "size"),
            mean_squared_iv_error=("squared_iv_error", "mean"),
            mean_absolute_iv_error=("absolute_iv_error", "mean"),
        )
        .rename(columns={"split": "sample"})
    )
    return result


def _seed(base: int, *labels: str) -> int:
    digest = hashlib.sha256("|".join((str(base), *labels)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _hac_mean_test(values: np.ndarray) -> tuple[float, float, int]:
    count = len(values)
    centered = values - values.mean()
    lag = min(count - 1, max(1, int(math.floor(4 * (count / 100) ** (2 / 9)))))
    long_run = float(centered @ centered / count)
    for offset in range(1, lag + 1):
        covariance = float(centered[offset:] @ centered[:-offset] / count)
        long_run += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run, 0.0) / count)
    statistic = values.mean() / standard_error if standard_error > 0 else np.nan
    probability = 2.0 * norm.sf(abs(statistic)) if np.isfinite(statistic) else np.nan
    return float(statistic), float(probability), lag


def paired_model_inference(
    losses: pd.DataFrame, config: ExperimentConfig
) -> pd.DataFrame:
    """Compare each model with flat volatility using paired date losses."""
    columns = (
        "split_rule",
        "sample",
        "model",
        "dates",
        "mean_loss_difference_vs_flat",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_two_sided_p",
        "hac_statistic",
        "hac_two_sided_p",
        "hac_lags",
        "inference_status",
    )
    if losses.empty or "flat" not in set(losses.model):
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    benchmark = losses.loc[losses.model == "flat"]
    for (rule, sample, model), group in losses.loc[losses.model != "flat"].groupby(
        ["split_rule", "sample", "model"], sort=True
    ):
        flat = benchmark.loc[
            (benchmark["split_rule"] == rule) & (benchmark["sample"] == sample),
            ["observation_date", "mean_squared_iv_error"],
        ].rename(columns={"mean_squared_iv_error": "flat_loss"})
        paired = group.merge(flat, on="observation_date", how="inner").sort_values(
            "observation_date"
        )
        difference = (
            paired.mean_squared_iv_error - paired.flat_loss
        ).to_numpy(dtype=float)
        count = len(difference)
        row: dict[str, object] = {
            "split_rule": rule,
            "sample": sample,
            "model": model,
            "dates": count,
            "mean_loss_difference_vs_flat": float(difference.mean()) if count else np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "bootstrap_two_sided_p": np.nan,
            "hac_statistic": np.nan,
            "hac_two_sided_p": np.nan,
            "hac_lags": np.nan,
            "inference_status": "insufficient_dates_for_bootstrap",
        }
        if count >= config.statistics.minimum_dates_for_bootstrap:
            rng = np.random.default_rng(
                _seed(config.statistics.seed, str(rule), str(sample), str(model))
            )
            draws = rng.choice(
                difference,
                size=(config.statistics.bootstrap_replications, count),
                replace=True,
            ).mean(axis=1)
            row["bootstrap_ci_low"], row["bootstrap_ci_high"] = np.quantile(
                draws, [0.025, 0.975]
            )
            row["bootstrap_two_sided_p"] = min(
                1.0,
                2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0))),
            )
            row["inference_status"] = "bootstrap_only"
        if count >= config.statistics.minimum_dates_for_hac:
            statistic, probability, lag = _hac_mean_test(difference)
            row["hac_statistic"] = statistic
            row["hac_two_sided_p"] = probability
            row["hac_lags"] = lag
            row["inference_status"] = "bootstrap_and_hac"
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
