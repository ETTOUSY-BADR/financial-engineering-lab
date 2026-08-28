"""Typed TOML configuration and deterministic configuration hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


@dataclass(frozen=True)
class StudyConfig:
    name: str
    symbol: str
    random_seed: int
    target_days: tuple[int, ...]
    minimum_expiration_days: int
    minimum_parity_pairs: int
    minimum_smile_quotes: int
    minimum_valid_maturities: int
    maximum_quotes_per_maturity: int
    log_moneyness_limit: float
    minimum_option_price: float
    minimum_implied_volatility: float
    maximum_implied_volatility: float
    maximum_iv_uncertainty: float
    maximum_rate: float
    session_hours: float
    post_close_minutes: int


@dataclass(frozen=True)
class PathConfig:
    study_root: str
    legacy_quotes: str
    legacy_metadata: str


@dataclass(frozen=True)
class SplitConfig:
    rules: tuple[str, ...]
    seed: int
    stratified_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    names: tuple[str, ...]


@dataclass(frozen=True)
class HestonConfig:
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    starts: tuple[tuple[float, ...], ...]
    maximum_function_evaluations: int
    calibration_integration_points: int
    evaluation_integration_points: int


@dataclass(frozen=True)
class StatisticsConfig:
    bootstrap_replications: int
    minimum_dates_for_bootstrap: int
    minimum_dates_for_hac: int
    seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    study: StudyConfig
    paths: PathConfig
    splits: SplitConfig
    models: ModelConfig
    heston: HestonConfig
    statistics: StatisticsConfig
    source_path: Path

    @property
    def canonical_json(self) -> str:
        payload = asdict(self)
        payload.pop("source_path")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _tuple_values(mapping: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    output = dict(mapping)
    for name in names:
        output[name] = tuple(output[name])
    return output


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration."""
    source = Path(path).resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    study = StudyConfig(**_tuple_values(raw["study"], ("target_days",)))
    splits = SplitConfig(**_tuple_values(raw["splits"], ("rules",)))
    models = ModelConfig(**_tuple_values(raw["models"], ("names",)))
    heston_raw = _tuple_values(raw["heston"], ("lower_bounds", "upper_bounds"))
    heston_raw["starts"] = tuple(tuple(values) for values in heston_raw["starts"])
    heston = HestonConfig(**heston_raw)
    config = ExperimentConfig(
        study=study,
        paths=PathConfig(**raw["paths"]),
        splits=splits,
        models=models,
        heston=heston,
        statistics=StatisticsConfig(**raw["statistics"]),
        source_path=source,
    )
    if len(config.study.target_days) < 3:
        raise ValueError("at least three target maturities are required")
    if not 0.0 < config.splits.stratified_fraction < 0.5:
        raise ValueError("stratified holdout fraction must lie between zero and one half")
    if len(config.heston.lower_bounds) != 5 or len(config.heston.upper_bounds) != 5:
        raise ValueError("Heston bounds must contain five parameters")
    if any(len(start) != 5 for start in config.heston.starts):
        raise ValueError("every Heston start must contain five parameters")
    if any(lower >= upper for lower, upper in zip(config.heston.lower_bounds, config.heston.upper_bounds)):
        raise ValueError("every Heston lower bound must be below its upper bound")
    for start in config.heston.starts:
        if any(
            not lower <= value <= upper
            for value, lower, upper in zip(
                start, config.heston.lower_bounds, config.heston.upper_bounds
            )
        ):
            raise ValueError("every Heston start must lie within the configured bounds")
    allowed_splits = {"every_3", "every_4", "every_5", "seeded_stratified", "maturity_longest"}
    unknown = set(config.splits.rules) - allowed_splits
    if unknown:
        raise ValueError(f"unsupported split rules: {sorted(unknown)}")
    allowed_models = {"flat", "svi", "pchip", "sabr", "heston"}
    unknown_models = set(config.models.names) - allowed_models
    if unknown_models:
        raise ValueError(f"unsupported models: {sorted(unknown_models)}")
    if len(set(config.study.target_days)) != len(config.study.target_days):
        raise ValueError("target maturities must be unique")
    if config.statistics.bootstrap_replications < 100:
        raise ValueError("at least 100 bootstrap replications are required")
    return config
