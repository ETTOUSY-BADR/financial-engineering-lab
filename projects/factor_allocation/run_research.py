"""Walk-forward robust allocation across public Fama--French factor portfolios."""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.quant_models import diagonal_covariance_shrinkage, performance_summary

OUTPUT = ROOT / "projects" / "factor_allocation" / "output"
FIVE_FACTOR_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
MOMENTUM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def parse_monthly_zip(payload: bytes) -> pd.DataFrame:
    """Extract the first 6-digit-date monthly table from a French-library ZIP."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = archive.namelist()[0]
        text = archive.read(member).decode("utf-8", errors="replace")
    rows: list[list[str]] = []
    header: list[str] | None = None
    for raw_line in text.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if fields and fields[0] == "" and len(fields) > 1:
            header = ["date"] + fields[1:]
            continue
        if header and fields and len(fields[0]) == 6 and fields[0].isdigit():
            rows.append(fields[: len(header)])
        elif rows:
            break
    if header is None or not rows:
        raise ValueError("monthly factor table not found in archive")
    frame = pd.DataFrame(rows, columns=header)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    frame = frame.set_index("date").apply(pd.to_numeric, errors="coerce") / 100.0
    return frame


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "financial-engineering-lab/1.0 educational research"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def synthetic_fallback() -> pd.DataFrame:
    """Deterministic heavy-tailed factor panel for software validation."""
    rng = np.random.default_rng(2026)
    dates = pd.date_range("1963-07-31", periods=750, freq="ME")
    means = np.array([0.0050, 0.0017, 0.0022, 0.0025, 0.0018, 0.0042])
    vol = np.array([0.045, 0.030, 0.032, 0.026, 0.025, 0.042])
    common = rng.normal(size=(len(dates), 2))
    loadings = np.array([
        [0.8, -0.1],
        [0.3, 0.5],
        [0.2, 0.7],
        [0.2, 0.4],
        [0.1, 0.5],
        [0.4, -0.4],
    ])
    innovations = common @ loadings.T + rng.normal(size=(len(dates), len(FACTORS)))
    innovations /= innovations.std(axis=0, ddof=1)
    tail_scale = np.sqrt(8.0 / rng.chisquare(8.0, size=len(dates)))[:, None]
    returns = means + innovations * vol * tail_scale
    frame = pd.DataFrame(returns, index=dates, columns=FACTORS)
    frame["RF"] = 0.002
    return frame


def load_factors() -> tuple[pd.DataFrame, str]:
    try:
        five = parse_monthly_zip(download(FIVE_FACTOR_URL))
        momentum = parse_monthly_zip(download(MOMENTUM_URL))
        momentum_column = next(column for column in momentum if column.strip().lower().startswith("mom"))
        combined = five.join(momentum[[momentum_column]].rename(columns={momentum_column: "Mom"}))
        combined = combined.reindex(columns=FACTORS + ["RF"]).dropna()
        if len(combined) < 300:
            raise RuntimeError("download returned too little complete factor history")
        return combined, "Kenneth R. French Data Library monthly research factors"
    except Exception as error:
        print(f"Factor data unavailable ({error}); using labeled synthetic fallback.")
        return synthetic_fallback(), "Synthetic heavy-tailed factor fallback (not market evidence)"


def long_only_minimum_variance(covariance: np.ndarray, cap: float = 0.40) -> np.ndarray:
    """Solve long-only minimum variance with a maximum sleeve weight."""
    n_assets = covariance.shape[0]
    initial = np.full(n_assets, 1.0 / n_assets)
    result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        initial,
        method="SLSQP",
        bounds=[(0.0, cap)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"portfolio optimization failed: {result.message}")
    return np.asarray(result.x)


def walk_forward_allocation(
    factors: pd.DataFrame,
    lookback: int = 120,
    shrinkage: float = 0.50,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Monthly allocation with trailing covariance, no future information, and costs."""
    factor_returns = factors[FACTORS]
    records: list[dict[str, float | pd.Timestamp]] = []
    previous = np.zeros(len(FACTORS))
    for i in range(lookback, len(factor_returns)):
        train = factor_returns.iloc[i - lookback : i]
        covariance = diagonal_covariance_shrinkage(train, shrinkage).to_numpy()
        weights = long_only_minimum_variance(covariance)
        realized = factor_returns.iloc[i].to_numpy()
        turnover = 0.5 * np.sum(np.abs(weights - previous))
        gross = float(weights @ realized)
        cost = turnover * cost_bps * 1e-4
        row: dict[str, float | pd.Timestamp] = {
            "date": factor_returns.index[i],
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross - cost,
        }
        row.update({f"weight_{name}": value for name, value in zip(FACTORS, weights)})
        records.append(row)
        previous = weights
    return pd.DataFrame(records).set_index("date")


def factor_diagnostics(factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame({
        name: performance_summary(factors[name], periods_per_year=12)
        for name in FACTORS
    }).T
    correlation = factors[FACTORS].corr()
    return summary, correlation


def save_figures(factors: pd.DataFrame, strategy: pd.DataFrame) -> None:
    factor_wealth = (1.0 + factors[FACTORS]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    factor_wealth.plot(ax=ax, logy=True, linewidth=1.0)
    ax.set(title="Growth of factor research portfolios", ylabel="Growth of one", xlabel="")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "factor_wealth.png", dpi=180)
    plt.close(fig)

    equal_weight = factors.loc[strategy.index, FACTORS].mean(axis=1)
    wealth = pd.DataFrame({
        "Robust minimum variance, net": (1.0 + strategy["net_return"]).cumprod(),
        "Equal-weight factors": (1.0 + equal_weight).cumprod(),
    })
    fig, ax = plt.subplots(figsize=(10, 5.8))
    wealth.plot(ax=ax, logy=True, linewidth=1.3)
    ax.set(title="Walk-forward factor allocation", ylabel="Growth of one", xlabel="")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "allocation_wealth.png", dpi=180)
    plt.close(fig)

    weight_columns = [f"weight_{name}" for name in FACTORS]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    strategy[weight_columns].plot.area(ax=ax, linewidth=0.0)
    ax.set(title="Long-only factor-sleeve weights", ylabel="Weight", xlabel="", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(OUTPUT / "allocation_weights.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    factors, source = load_factors()
    summary, correlation = factor_diagnostics(factors)
    strategy = walk_forward_allocation(factors)
    equal_weight = factors.loc[strategy.index, FACTORS].mean(axis=1)
    strategy_summary = pd.DataFrame({
        "robust_minimum_variance_net": performance_summary(strategy["net_return"], periods_per_year=12),
        "equal_weight_factors": performance_summary(equal_weight, periods_per_year=12),
    }).T
    strategy_summary["average_monthly_turnover"] = [
        strategy["turnover"].mean(),
        0.0,
    ]

    factors.to_csv(OUTPUT / "monthly_factors.csv")
    summary.to_csv(OUTPUT / "factor_summary.csv")
    correlation.to_csv(OUTPUT / "factor_correlation.csv")
    strategy.to_csv(OUTPUT / "walk_forward_allocation.csv")
    strategy_summary.to_csv(OUTPUT / "strategy_summary.csv")
    save_figures(factors, strategy)

    provenance = [
        f"Source: {source}",
        f"Five-factor URL: {FIVE_FACTOR_URL}",
        f"Momentum URL: {MOMENTUM_URL}",
        f"Rows: {len(factors)}",
        f"First month: {factors.index.min()}",
        f"Last month: {factors.index.max()}",
        f"Generated UTC: {pd.Timestamp.utcnow().isoformat()}",
        "Input percentage returns are converted to decimal returns.",
        "French-library methodology and CRSP input format can change over time.",
        "Factor portfolios are research returns, not costless directly traded assets.",
        "Allocation charges 10 bps per unit of one-way turnover.",
        "Synthetic fallback validates software only and cannot support market claims.",
    ]
    (OUTPUT / "provenance.txt").write_text("\n".join(provenance) + "\n", encoding="utf-8")
    print("Factor summary")
    print(summary.round(4).to_string())
    print("\nStrategy summary")
    print(strategy_summary.round(4).to_string())


if __name__ == "__main__":
    main()
