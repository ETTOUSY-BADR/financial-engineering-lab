"""Regression tests for the modular multi-date surface framework."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from projects.volatility_surface.surface_research.config import load_config
from projects.volatility_surface.surface_research.data import (
    CboeEODAdapter,
    LegacyYahooAdapter,
    StudyPaths,
    archive_snapshot,
)
from projects.volatility_surface.surface_research.models import sabr_lognormal_volatility
from projects.volatility_surface.surface_research.splits import assign_split
from projects.volatility_surface.surface_research.statistics import (
    date_level_losses,
    paired_model_inference,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects" / "volatility_surface" / "config" / "multidate.toml"


class ConfigurationTests(unittest.TestCase):
    def test_configuration_hash_is_stable(self) -> None:
        first = load_config(CONFIG)
        second = load_config(CONFIG)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(len(first.sha256), 64)


class SplitTests(unittest.TestCase):
    def test_seeded_split_is_deterministic_and_preserves_training_quotes(self) -> None:
        panel = pd.DataFrame(
            {
                "observation_date": ["2026-08-27"] * 24,
                "expiration": ["2026-12-18"] * 12 + ["2027-03-19"] * 12,
                "strike_rank": list(range(12)) * 2,
                "log_moneyness": np.tile(np.linspace(-0.25, 0.25, 12), 2),
            }
        )
        first = assign_split(panel, "seeded_stratified", 1234, 0.25)
        second = assign_split(panel, "seeded_stratified", 1234, 0.25)
        self.assertListEqual(first.split.tolist(), second.split.tolist())
        counts = first.groupby(["expiration", "split"]).size().unstack(fill_value=0)
        self.assertTrue((counts.train >= 4).all())
        self.assertTrue((counts.holdout >= 1).all())


class ArchiveTests(unittest.TestCase):
    def test_raw_snapshot_is_immutable(self) -> None:
        frame = pd.DataFrame(
            {
                "contractSymbol": ["TEST"],
                "lastTradeDate": ["2026-08-27T19:59:00+00:00"],
                "strike": [100.0],
                "lastPrice": [2.0],
                "bid": [1.9],
                "ask": [2.1],
                "volume": [10],
                "openInterest": [20],
                "option_type": ["call"],
                "expiration": ["2026-09-18"],
                "spot_reference": [100.0],
                "acquired_utc": ["2026-08-28T10:00:00+00:00"],
            }
        )
        metadata = {
            "reference_session_date": "2026-08-27",
            "spot_reference": 100.0,
            "acquired_utc": "2026-08-28T10:00:00+00:00",
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = StudyPaths(
                root=root,
                raw=root / "raw",
                processed=root / "processed",
                calibration=root / "calibration",
                diagnostics=root / "diagnostics",
                figures=root / "figures",
                tables=root / "tables",
                manifests=root / "manifests",
                logs=root / "logs",
            )
            paths.create()
            archive_snapshot(paths, LegacyYahooAdapter(), frame, metadata)
            changed = frame.copy()
            changed.loc[0, "lastPrice"] = 2.5
            with self.assertRaises(RuntimeError):
                archive_snapshot(paths, LegacyYahooAdapter(), changed, metadata)

    def test_cboe_clock_respects_new_york_daylight_saving(self) -> None:
        frame = pd.DataFrame(
            {
                "quote_date": ["2026-01-15", "2026-07-15"],
                "bid_1545": [2.0, 2.0],
                "ask_1545": [2.2, 2.2],
                "put_call": ["C", "P"],
                "strike": [100.0, 100.0],
                "expiration": ["2026-02-20", "2026-08-21"],
            }
        )
        normalized = CboeEODAdapter().normalize(
            frame,
            {"spot_reference": 100.0, "acquired_utc": "2026-08-01T00:00:00Z"},
        )
        observed = pd.to_datetime(normalized.last_trade_utc, utc=True)
        self.assertEqual(observed.iloc[0].hour, 20)
        self.assertEqual(observed.iloc[1].hour, 19)


class ModelAndInferenceTests(unittest.TestCase):
    def test_sabr_atm_limit_is_finite(self) -> None:
        volatility = sabr_lognormal_volatility(
            np.array([100.0, 100.0]),
            np.array([100.0, 110.0]),
            np.array([0.5, 0.5]),
            alpha=0.20,
            rho=-0.4,
            nu=0.6,
        )
        self.assertTrue(np.isfinite(volatility).all())
        self.assertTrue((volatility > 0).all())

    def test_one_date_is_counted_but_inference_is_gated(self) -> None:
        config = load_config(CONFIG)
        predictions = pd.DataFrame(
            {
                "observation_date": ["2026-08-27"] * 4,
                "split_rule": ["every_4"] * 4,
                "model": ["flat", "flat", "svi", "svi"],
                "split": ["holdout"] * 4,
                "iv_equivalent_error": [0.04, -0.03, 0.01, -0.01],
            }
        )
        losses = date_level_losses(predictions)
        inference = paired_model_inference(losses, config)
        self.assertEqual(int(inference.loc[0, "dates"]), 1)
        self.assertEqual(
            inference.loc[0, "inference_status"],
            "insufficient_dates_for_bootstrap",
        )
        self.assertLess(float(inference.loc[0, "mean_loss_difference_vs_flat"]), 0.0)


if __name__ == "__main__":
    unittest.main()
