"""Immutable snapshot archive and replaceable vendor-schema adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .config import ExperimentConfig


@dataclass(frozen=True)
class StudyPaths:
    root: Path
    raw: Path
    processed: Path
    calibration: Path
    diagnostics: Path
    figures: Path
    tables: Path
    manifests: Path
    logs: Path

    @classmethod
    def from_config(cls, project: Path, config: ExperimentConfig) -> "StudyPaths":
        root = project / config.paths.study_root
        return cls(
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

    def create(self) -> None:
        for path in (
            self.raw,
            self.processed,
            self.calibration,
            self.diagnostics,
            self.figures,
            self.tables,
            self.manifests,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class Snapshot:
    observation_date: str
    quotes: pd.DataFrame
    metadata: dict[str, object]
    raw_directory: Path


class VendorAdapter(Protocol):
    """Contract required for a new licensed or public data source."""

    name: str

    def normalize(
        self, frame: pd.DataFrame, metadata: dict[str, object]
    ) -> pd.DataFrame:
        """Map a vendor frame into the canonical quote schema."""


CANONICAL_COLUMNS = (
    "contract_symbol",
    "last_trade_utc",
    "strike",
    "last_price",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "option_type",
    "expiration",
    "spot_reference",
    "acquired_utc",
    "price_type",
)


class LegacyYahooAdapter:
    """Normalize the original yfinance snapshot without changing its evidence class."""

    name = "yahoo_delayed"

    def normalize(
        self, frame: pd.DataFrame, metadata: dict[str, object]
    ) -> pd.DataFrame:
        rename = {
            "contractSymbol": "contract_symbol",
            "lastTradeDate": "last_trade_utc",
            "lastPrice": "last_price",
            "openInterest": "open_interest",
        }
        output = frame.rename(columns=rename).copy()
        output["price_type"] = "reference-session last trade"
        for column in CANONICAL_COLUMNS:
            if column not in output:
                output[column] = np.nan
        output["option_type"] = output["option_type"].astype(str).str.lower()
        return output.loc[:, CANONICAL_COLUMNS]


class CboeEODAdapter:
    """Adapter for licensed Cboe Option EOD Summary files at the 15:45 snapshot."""

    name = "cboe_eod_1545"

    def normalize(
        self, frame: pd.DataFrame, metadata: dict[str, object]
    ) -> pd.DataFrame:
        lookup = {column.lower().replace(" ", "_"): column for column in frame.columns}

        def source(*names: str) -> pd.Series:
            for name in names:
                if name in lookup:
                    return frame[lookup[name]]
            return pd.Series(np.nan, index=frame.index)

        bid = pd.to_numeric(source("bid_1545", "bid_1545_et"), errors="coerce")
        ask = pd.to_numeric(source("ask_1545", "ask_1545_et"), errors="coerce")
        quote_date = pd.to_datetime(source("quote_date", "quotedate"), errors="coerce")
        local_clock = pd.to_datetime(
            quote_date.dt.strftime("%Y-%m-%d") + " 15:45:00", errors="coerce"
        )
        observed = (
            local_clock.dt.tz_localize(
                "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
            )
            .dt.tz_convert("UTC")
            .astype(str)
        )
        option_type = source("option_type", "put_call").astype(str).str.lower().str[0]
        output = pd.DataFrame(
            {
                "contract_symbol": source("option_symbol", "root").astype(str),
                "last_trade_utc": observed,
                "strike": pd.to_numeric(source("strike", "strike_price"), errors="coerce"),
                "last_price": (bid + ask) / 2.0,
                "bid": bid,
                "ask": ask,
                "volume": pd.to_numeric(source("trade_volume", "volume"), errors="coerce"),
                "open_interest": pd.to_numeric(source("open_interest"), errors="coerce"),
                "option_type": option_type.map({"c": "call", "p": "put"}),
                "expiration": pd.to_datetime(source("expiration", "expiration_date")).dt.date.astype(str),
                "spot_reference": float(metadata["spot_reference"]),
                "acquired_utc": str(metadata["acquired_utc"]),
                "price_type": "15:45 ET NBBO midpoint",
            }
        )
        return output.loc[:, CANONICAL_COLUMNS]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text_lf(path: Path, value: str) -> None:
    path.write_bytes(value.replace("\r\n", "\n").encode("utf-8"))


def _write_csv_lf(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def archive_snapshot(
    paths: StudyPaths,
    adapter: VendorAdapter,
    frame: pd.DataFrame,
    metadata: dict[str, object],
) -> Path:
    """Write a normalized snapshot once and refuse later byte changes."""
    normalized = adapter.normalize(frame, metadata)
    observation_date = str(metadata["reference_session_date"])
    target = paths.raw / adapter.name / observation_date
    target.mkdir(parents=True, exist_ok=True)
    quotes_path = target / "quotes.csv"
    metadata_path = target / "metadata.json"
    candidate_csv = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    enriched = dict(metadata)
    enriched["adapter"] = adapter.name
    enriched["canonical_schema_version"] = 1
    candidate_metadata = (
        json.dumps(enriched, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    for path, candidate in (
        (quotes_path, candidate_csv),
        (metadata_path, candidate_metadata),
    ):
        if path.exists() and path.read_bytes() != candidate:
            raise RuntimeError(f"immutable raw snapshot would change: {path}")
        if not path.exists():
            path.write_bytes(candidate)
    hashes = {
        "metadata.json": sha256(metadata_path),
        "quotes.csv": sha256(quotes_path),
    }
    _write_text_lf(target / "raw.sha256", "\n".join(
        f"{digest}  {name}" for name, digest in sorted(hashes.items())
    ) + "\n")
    return target


def bootstrap_legacy_snapshot(
    project: Path, paths: StudyPaths, config: ExperimentConfig
) -> Path:
    """Archive the committed one-date reference study in the new immutable layout."""
    quote_path = project / config.paths.legacy_quotes
    metadata_path = project / config.paths.legacy_metadata
    if not quote_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("legacy empirical snapshot is unavailable")
    frame = pd.read_csv(quote_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return archive_snapshot(paths, LegacyYahooAdapter(), frame, metadata)


class SnapshotRepository:
    """Discover and verify all normalized date partitions in the raw archive."""

    def __init__(self, paths: StudyPaths):
        self.paths = paths

    def directories(self) -> list[Path]:
        return sorted(self.paths.raw.glob("*/*"))

    def load_all(self) -> list[Snapshot]:
        snapshots: list[Snapshot] = []
        for directory in self.directories():
            quotes_path = directory / "quotes.csv"
            metadata_path = directory / "metadata.json"
            hash_path = directory / "raw.sha256"
            if not (quotes_path.exists() and metadata_path.exists() and hash_path.exists()):
                continue
            expected = {}
            for line in hash_path.read_text(encoding="utf-8").splitlines():
                digest, name = line.split(maxsplit=1)
                expected[name] = digest
            actual = {
                "quotes.csv": sha256(quotes_path),
                "metadata.json": sha256(metadata_path),
            }
            if actual != expected:
                raise RuntimeError(f"raw snapshot integrity failure in {directory}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            date = str(metadata["reference_session_date"])
            quotes = pd.read_csv(quotes_path)
            missing = set(CANONICAL_COLUMNS) - set(quotes.columns)
            if missing:
                raise ValueError(f"canonical columns missing in {directory}: {sorted(missing)}")
            snapshots.append(Snapshot(date, quotes, metadata, directory))
        return snapshots
