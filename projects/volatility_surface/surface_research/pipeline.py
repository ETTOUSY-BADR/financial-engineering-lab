"""Deterministic orchestration for the multi-date surface data layer.

This module deliberately separates data preparation and static-arbitrage checks
from model validation. A failed date or maturity is recorded and does not erase
the usable evidence from the rest of the archive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import ExperimentConfig, load_config
from .data import SnapshotRepository, StudyPaths, bootstrap_legacy_snapshot, sha256
from .models import validate_models
from .statistics import date_level_losses, paired_model_inference
from .surface import calendar_monotonicity, fit_svi_surface, prepare_snapshot


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "config" / "multidate.toml"
FAILURE_COLUMNS = (
    "observation_date",
    "stage",
    "expiration",
    "split_rule",
    "model",
    "reason",
    "detail",
)


@dataclass(frozen=True)
class ExperimentResult:
    """Compact, programmatic summary of a completed preparation run."""

    paths: StudyPaths
    config_sha256: str
    snapshot_count: int
    eligible_date_count: int
    failure_count: int


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write stable LF-delimited CSV, including an empty file for empty results."""
    frame.to_csv(path, index=False, lineterminator="\n")


def _failure(
    observation_date: str,
    stage: str,
    error: Exception,
    expiration: str = "",
) -> dict[str, object]:
    return {
        "observation_date": observation_date,
        "stage": stage,
        "expiration": expiration,
        "split_rule": "",
        "reason": type(error).__name__,
        "detail": str(error),
    }


def _derived_manifest(paths: StudyPaths, config: ExperimentConfig) -> dict[str, str]:
    files: list[Path] = [config.source_path]
    for directory in (
        paths.processed,
        paths.calibration,
        paths.diagnostics,
        paths.figures,
        paths.tables,
        paths.logs,
    ):
        files.extend(path for path in directory.rglob("*") if path.is_file())
    return {
        path.relative_to(paths.root.parent).as_posix(): sha256(path)
        for path in sorted(set(files))
    }


def _write_manifest(paths: StudyPaths, config: ExperimentConfig) -> None:
    manifest = _derived_manifest(paths, config)
    content = "".join(f"{digest}  {name}\n" for name, digest in manifest.items())
    (paths.manifests / "derived.sha256").write_bytes(content.encode("utf-8"))


def run_experiment(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project: str | Path = PROJECT,
    include_legacy_snapshot: bool = True,
    run_model_validation: bool = True,
) -> ExperimentResult:
    """Prepare every archived date and save an auditable surface data set.

    The committed Yahoo snapshot is bootstrapped into the immutable archive by
    default. Additional vendor snapshots can be archived through ``data.py`` and
    are discovered automatically on the next run.
    """
    project_path = Path(project).resolve()
    config_source = Path(config_path)
    if not config_source.is_absolute():
        candidate = project_path / config_source
        config_source = candidate if candidate.exists() else config_source
    config = load_config(config_source)
    paths = StudyPaths.from_config(project_path, config)
    paths.create()

    if include_legacy_snapshot:
        bootstrap_legacy_snapshot(project_path, paths, config)

    snapshots = SnapshotRepository(paths).load_all()
    parity_frames: list[pd.DataFrame] = []
    clean_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    svi_parameter_frames: list[pd.DataFrame] = []
    svi_fitted_frames: list[pd.DataFrame] = []
    svi_diagnostic_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    for snapshot in snapshots:
        try:
            parity, clean, selection, failures = prepare_snapshot(snapshot, config)
        except Exception as error:
            failure_frames.append(pd.DataFrame([_failure(snapshot.observation_date, "snapshot", error)]))
            status_rows.append(
                {
                    "observation_date": snapshot.observation_date,
                    "adapter": snapshot.metadata.get("adapter", "unknown"),
                    "eligible": False,
                    "valid_maturities": 0,
                    "clean_quotes": 0,
                    "svi_maturities": 0,
                }
            )
            continue

        parity_frames.append(parity)
        clean_frames.append(clean)
        if not selection.empty:
            selection = selection.copy()
            selection.insert(0, "observation_date", snapshot.observation_date)
        selection_frames.append(selection)
        failure_frames.append(failures)

        valid_maturities = int(clean.expiration.nunique()) if not clean.empty else 0
        eligible = valid_maturities >= config.study.minimum_valid_maturities
        svi_count = 0
        if eligible:
            parameters, fitted, diagnostics, svi_failures = fit_svi_surface(clean)
            svi_parameter_frames.append(parameters)
            svi_fitted_frames.append(fitted)
            svi_diagnostic_frames.append(diagnostics)
            failure_frames.append(svi_failures)
            svi_count = int(parameters.expiration.nunique()) if not parameters.empty else 0

        status_rows.append(
            {
                "observation_date": snapshot.observation_date,
                "adapter": snapshot.metadata.get("adapter", "unknown"),
                "eligible": eligible,
                "valid_maturities": valid_maturities,
                "clean_quotes": len(clean),
                "svi_maturities": svi_count,
            }
        )

    parity = _concat(parity_frames)
    clean = _concat(clean_frames)
    selections = _concat(selection_frames)
    svi_parameters = _concat(svi_parameter_frames)
    svi_fitted = _concat(svi_fitted_frames)
    svi_diagnostics = _concat(svi_diagnostic_frames)
    failures = _concat(failure_frames)
    failures = failures.reindex(columns=FAILURE_COLUMNS)
    date_status = pd.DataFrame(status_rows)
    calendar = (
        calendar_monotonicity(svi_parameters)
        if not svi_parameters.empty
        else pd.DataFrame()
    )
    predictions = pd.DataFrame()
    model_summary = pd.DataFrame()
    model_parameters = pd.DataFrame()
    date_losses = pd.DataFrame()
    inference = pd.DataFrame()
    if run_model_validation and not clean.empty:
        eligible_dates = set(date_status.loc[date_status.eligible, "observation_date"])
        validation_input = clean.loc[clean.observation_date.isin(eligible_dates)].copy()
        predictions, model_summary, model_parameters, model_failures = validate_models(
            validation_input, config
        )
        failures = _concat((failures, model_failures))
        failures = failures.reindex(columns=FAILURE_COLUMNS)
        date_losses = date_level_losses(predictions)
        inference = paired_model_inference(date_losses, config)

    outputs = {
        paths.processed / "maturity_selection.csv": selections,
        paths.processed / "parity_estimates.csv": parity,
        paths.processed / "clean_otm_quotes.csv": clean,
        paths.calibration / "svi_parameters.csv": svi_parameters,
        paths.calibration / "svi_fitted_quotes.csv": svi_fitted,
        paths.calibration / "model_parameters.csv": model_parameters,
        paths.calibration / "model_predictions.csv": predictions,
        paths.diagnostics / "svi_diagnostics.csv": svi_diagnostics,
        paths.diagnostics / "calendar_monotonicity.csv": calendar,
        paths.diagnostics / "failures.csv": failures,
        paths.diagnostics / "date_status.csv": date_status,
        paths.diagnostics / "model_summary_by_date.csv": model_summary,
        paths.diagnostics / "date_level_losses.csv": date_losses,
        paths.diagnostics / "paired_model_inference.csv": inference,
    }
    for path, frame in outputs.items():
        _write_csv(path, frame)

    run_record = {
        "study": config.study.name,
        "config_sha256": config.sha256,
        "snapshot_count": len(snapshots),
        "observation_dates": sorted(snapshot.observation_date for snapshot in snapshots),
        "eligible_date_count": int(date_status.eligible.sum()) if not date_status.empty else 0,
        "failure_count": len(failures),
        "model_validation_enabled": run_model_validation,
        "configured_models": list(config.models.names),
        "configured_split_rules": list(config.splits.rules),
        "raw_snapshot_hashes": {
            directory.relative_to(paths.root).as_posix(): sha256(directory / "raw.sha256")
            for directory in SnapshotRepository(paths).directories()
            if (directory / "raw.sha256").exists()
        },
    }
    run_json = json.dumps(run_record, sort_keys=True, indent=2) + "\n"
    (paths.manifests / "run.json").write_bytes(run_json.encode("utf-8"))
    _write_manifest(paths, config)

    return ExperimentResult(
        paths=paths,
        config_sha256=config.sha256,
        snapshot_count=len(snapshots),
        eligible_date_count=run_record["eligible_date_count"],
        failure_count=len(failures),
    )
