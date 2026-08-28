"""Command-line entry point for the modular multi-date surface pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from projects.volatility_surface.surface_research import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/multidate.toml",
        help="configuration path, absolute or relative to the volatility-surface project",
    )
    parser.add_argument(
        "--no-legacy-snapshot",
        action="store_true",
        help="do not bootstrap the committed public snapshot into the raw archive",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="prepare surfaces and static-arbitrage diagnostics without model calibration",
    )
    arguments = parser.parse_args()
    project = Path(__file__).resolve().parent
    result = run_experiment(
        arguments.config,
        project=project,
        include_legacy_snapshot=not arguments.no_legacy_snapshot,
        run_model_validation=not arguments.prepare_only,
    )
    print(
        f"Prepared {result.snapshot_count} snapshot(s); "
        f"{result.eligible_date_count} eligible date(s); "
        f"{result.failure_count} recorded failure(s)."
    )
    print(f"Study artifacts: {result.paths.root}")


if __name__ == "__main__":
    main()
