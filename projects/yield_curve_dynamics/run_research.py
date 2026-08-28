"""Fit and forecast the U.S. Treasury curve with PCA and Nelson--Siegel factors."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.quant_models import fit_nelson_siegel, nelson_siegel_loadings

OUTPUT = ROOT / "projects" / "yield_curve_dynamics" / "output"
SERIES = {
    "DGS3MO": 3,
    "DGS6MO": 6,
    "DGS1": 12,
    "DGS2": 24,
    "DGS3": 36,
    "DGS5": 60,
    "DGS7": 84,
    "DGS10": 120,
    "DGS20": 240,
    "DGS30": 360,
}
DECAY = 0.0609
START = "1990-01-01"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(SERIES)


def synthetic_fallback() -> pd.DataFrame:
    """Generate a deterministic dynamic Nelson--Siegel panel for code validation."""
    rng = np.random.default_rng(73)
    dates = pd.date_range("1990-01-31", periods=440, freq="ME")
    transition = np.diag([0.985, 0.94, 0.90])
    center = np.array([4.0, -1.2, 0.4])
    innovation = np.array([0.18, 0.22, 0.16])
    factors = np.empty((len(dates), 3))
    factors[0] = center
    for t in range(1, len(dates)):
        factors[t] = center + transition @ (factors[t - 1] - center)
        factors[t] += rng.normal(0.0, innovation)
    loadings = nelson_siegel_loadings(list(SERIES.values()), DECAY)
    yields = factors @ loadings.T + rng.normal(0.0, 0.035, (len(dates), len(SERIES)))
    return pd.DataFrame(yields, index=dates, columns=SERIES)


def load_yields() -> tuple[pd.DataFrame, str]:
    """Load FRED constant-maturity Treasury yields and aggregate to month-end."""
    try:
        raw = pd.read_csv(FRED_CSV)
        date_column = "observation_date" if "observation_date" in raw else raw.columns[0]
        raw[date_column] = pd.to_datetime(raw[date_column])
        data = raw.set_index(date_column).reindex(columns=SERIES)
        data = data.apply(pd.to_numeric, errors="coerce").loc[START:]
        monthly = data.resample("ME").last().dropna(how="any")
        if len(monthly) < 120:
            raise RuntimeError("FRED returned too little complete curve history")
        return monthly, "FRED constant-maturity Treasury series via fredgraph CSV"
    except Exception as error:
        print(f"FRED data unavailable ({error}); using labeled synthetic fallback.")
        return synthetic_fallback(), "Synthetic dynamic Nelson-Siegel fallback (not market evidence)"


def fit_curve_history(yields: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    maturities = list(SERIES.values())
    factor_rows: list[np.ndarray] = []
    fitted_rows: list[np.ndarray] = []
    rmses: list[float] = []
    for row in yields.to_numpy():
        beta, fitted, rmse = fit_nelson_siegel(row, maturities, DECAY)
        factor_rows.append(beta)
        fitted_rows.append(fitted)
        rmses.append(rmse)
    factors = pd.DataFrame(
        factor_rows,
        index=yields.index,
        columns=["level", "slope", "curvature"],
    )
    factors["cross_sectional_rmse_pct_points"] = rmses
    fitted = pd.DataFrame(fitted_rows, index=yields.index, columns=yields.columns)
    return factors, fitted


def principal_components(yields: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    centered = yields - yields.mean()
    _, singular, right = np.linalg.svd(centered.to_numpy(), full_matrices=False)
    loadings = pd.DataFrame(
        right[:3].T,
        index=[SERIES[name] for name in yields.columns],
        columns=["PC1", "PC2", "PC3"],
    )
    for column in loadings:
        anchor = loadings[column].abs().idxmax()
        if loadings.loc[anchor, column] < 0:
            loadings[column] *= -1
    variance = singular**2 / np.sum(singular**2)
    explained = pd.Series(variance[:3], index=loadings.columns, name="explained_variance_ratio")
    return loadings, explained


def expanding_factor_forecast(
    yields: pd.DataFrame,
    factors: pd.DataFrame,
    min_train: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-month recursive VAR(1) forecast versus a random-walk curve."""
    core = factors[["level", "slope", "curvature"]]
    loadings = nelson_siegel_loadings(list(SERIES.values()), DECAY)
    ns_predictions: list[np.ndarray] = []
    rw_predictions: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    for i in range(min_train, len(core)):
        train = core.iloc[:i].to_numpy()
        design = np.column_stack([np.ones(len(train) - 1), train[:-1]])
        coefficients, *_ = np.linalg.lstsq(design, train[1:], rcond=None)
        state = np.r_[1.0, train[-1]]
        factor_forecast = state @ coefficients
        ns_predictions.append(loadings @ factor_forecast)
        rw_predictions.append(yields.iloc[i - 1].to_numpy())
        dates.append(yields.index[i])
    ns = pd.DataFrame(ns_predictions, index=dates, columns=yields.columns)
    rw = pd.DataFrame(rw_predictions, index=dates, columns=yields.columns)
    return ns, rw


def forecast_report(
    actual: pd.DataFrame,
    ns: pd.DataFrame,
    rw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = actual.reindex(ns.index)
    errors = pd.concat(
        {
            "Nelson_Siegel_VAR": ns - aligned,
            "random_walk": rw - aligned,
        },
        axis=1,
    )
    rows = []
    for model, forecast in {"Nelson_Siegel_VAR": ns, "random_walk": rw}.items():
        error = forecast - aligned
        for series in actual.columns:
            rows.append({
                "model": model,
                "series": series,
                "maturity_months": SERIES[series],
                "rmse_pct_points": np.sqrt(np.mean(error[series] ** 2)),
                "mae_pct_points": np.mean(np.abs(error[series])),
                "mean_error_pct_points": np.mean(error[series]),
            })
    return pd.DataFrame(rows).set_index(["model", "series"]), errors


def save_figures(
    yields: pd.DataFrame,
    factors: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    report: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    yields[["DGS3MO", "DGS2", "DGS10", "DGS30"]].plot(ax=ax, linewidth=1.0)
    ax.set(title="U.S. constant-maturity Treasury yields", ylabel="Percent", xlabel="")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "yield_history.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, name in zip(axes, ["level", "slope", "curvature"]):
        factors[name].plot(ax=axis, linewidth=1.0, color="#135E96")
        axis.set_ylabel(name.title())
        axis.grid(alpha=0.22)
    axes[0].set_title("Nelson--Siegel factors")
    fig.tight_layout()
    fig.savefig(OUTPUT / "nelson_siegel_factors.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    pca_loadings.plot(ax=ax, marker="o")
    ax.set(title="Principal-component loadings", xlabel="Maturity (months)", ylabel="Loading")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "pca_loadings.png", dpi=180)
    plt.close(fig)

    rmse = report["rmse_pct_points"].unstack(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    rmse.plot(ax=ax, marker="o")
    ax.set(title="Recursive one-month forecast RMSE", xlabel="Maturity", ylabel="Percentage points")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "forecast_rmse.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    yields, source = load_yields()
    factors, fitted = fit_curve_history(yields)
    loadings, explained = principal_components(yields)
    ns_forecast, rw_forecast = expanding_factor_forecast(yields, factors)
    report, forecast_errors = forecast_report(yields, ns_forecast, rw_forecast)

    yields.to_csv(OUTPUT / "monthly_yields.csv")
    factors.to_csv(OUTPUT / "nelson_siegel_factors.csv")
    fitted.to_csv(OUTPUT / "nelson_siegel_fitted_yields.csv")
    loadings.to_csv(OUTPUT / "pca_loadings.csv")
    explained.to_csv(OUTPUT / "pca_explained_variance.csv")
    ns_forecast.to_csv(OUTPUT / "recursive_ns_var_forecasts.csv")
    rw_forecast.to_csv(OUTPUT / "random_walk_forecasts.csv")
    forecast_errors.to_csv(OUTPUT / "forecast_errors.csv")
    report.to_csv(OUTPUT / "forecast_summary.csv")
    save_figures(yields, factors, loadings, report)

    provenance = [
        f"Source: {source}",
        f"URL: {FRED_CSV}",
        f"Series: {', '.join(SERIES)}",
        f"Requested start: {START}",
        f"Complete monthly rows: {len(yields)}",
        f"First month: {yields.index.min()}",
        f"Last month: {yields.index.max()}",
        f"Generated UTC: {pd.Timestamp.utcnow().isoformat()}",
        "Values are constant-maturity par yields in percent, not zero-coupon rates.",
        "Monthly observations use the last available daily value.",
        "Treasury changed its official par-curve methodology on 2021-12-06.",
        "Synthetic fallback validates software only and cannot support market claims.",
    ]
    (OUTPUT / "provenance.txt").write_text("\n".join(provenance) + "\n", encoding="utf-8")
    print("PCA explained variance")
    print(explained.round(5).to_string())
    print("\nRecursive forecast report")
    print(report.round(5).to_string())


if __name__ == "__main__":
    main()
