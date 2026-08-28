"""Deterministic train/holdout protocols fixed before model calibration."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def strike_spanning_panel(frame: pd.DataFrame, maximum_per_maturity: int) -> pd.DataFrame:
    """Select an evenly spaced strike panel before any holdout rule is applied."""
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("expiration", sort=True):
        ordered = group.sort_values("log_moneyness").reset_index(drop=True)
        if len(ordered) > maximum_per_maturity:
            locations = np.unique(
                np.linspace(0, len(ordered) - 1, maximum_per_maturity).round().astype(int)
            )
            ordered = ordered.iloc[locations].reset_index(drop=True)
        ordered["strike_rank"] = np.arange(len(ordered), dtype=int)
        parts.append(ordered)
    if not parts:
        raise ValueError("cannot construct a calibration panel from an empty surface")
    return pd.concat(parts, ignore_index=True)


def _stable_seed(base_seed: int, date: str, rule: str, expiration: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{date}|{rule}|{expiration}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def assign_split(
    panel: pd.DataFrame,
    rule: str,
    seed: int,
    stratified_fraction: float = 0.25,
) -> pd.DataFrame:
    """Return a copy with a deterministic split column."""
    output = panel.copy()
    output["split"] = "train"
    if rule.startswith("every_"):
        step = int(rule.rsplit("_", 1)[1])
        for _, indices in output.groupby("expiration", sort=True).groups.items():
            ordered = output.loc[indices].sort_values("strike_rank")
            holdout = ordered.index[ordered.strike_rank % step == 1]
            output.loc[holdout, "split"] = "holdout"
    elif rule == "seeded_stratified":
        date = str(output["observation_date"].iloc[0])
        for expiration, indices in output.groupby("expiration", sort=True).groups.items():
            ordered = output.loc[indices].sort_values("strike_rank")
            buckets = pd.qcut(
                ordered["log_moneyness"].rank(method="first"),
                q=min(4, len(ordered)),
                labels=False,
                duplicates="drop",
            )
            for bucket in sorted(buckets.dropna().unique()):
                candidates = ordered.index[buckets == bucket].to_numpy()
                rng = np.random.default_rng(_stable_seed(seed, date, rule, str(expiration)))
                count = max(1, int(round(len(candidates) * stratified_fraction)))
                chosen = np.sort(rng.choice(candidates, size=min(count, len(candidates)), replace=False))
                output.loc[chosen, "split"] = "holdout"
    elif rule == "maturity_longest":
        longest = output.groupby("expiration").maturity_years.first().idxmax()
        output.loc[output.expiration == longest, "split"] = "holdout"
    else:
        raise ValueError(f"unknown split rule: {rule}")
    counts = output.groupby(["expiration", "split"]).size().unstack(fill_value=0)
    if rule != "maturity_longest" and (
        "holdout" not in counts or (counts.get("holdout", 0) < 1).any() or (counts.get("train", 0) < 4).any()
    ):
        raise ValueError(f"split {rule} leaves an unusable maturity: {counts.to_dict()}")
    output["split_rule"] = rule
    return output
