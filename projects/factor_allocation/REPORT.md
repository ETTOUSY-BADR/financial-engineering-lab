# Empirical report: robust public-factor allocation

## Snapshot

The live run generated on 28 August 2026 contains 756 monthly observations from July
1963 through June 2026 for the Fama--French market, size, value, profitability, and
investment factors plus momentum. Data come from the Kenneth R. French Data Library
and are converted from percentage to decimal returns.

The walk-forward evaluation uses the previous 120 months at each rebalance. Sample
covariance is shrunk 50% toward its diagonal, and a long-only minimum-variance
program caps each factor sleeve at 40%. Ten basis points are charged per unit of
one-way turnover.

## Main evidence

| Allocation | Geometric return | Volatility | Sharpe | Maximum drawdown | Monthly turnover |
|---|---:|---:|---:|---:|---:|
| Robust minimum variance, net | 3.20% | 3.60% | 0.889 | -11.72% | 0.65% |
| Equal-weight factors | 4.47% | 4.14% | 1.079 | -12.37% | 0.00% |

The optimized allocation reduced volatility and slightly reduced drawdown, but its
lower compound return left it with a worse sample Sharpe than equal weighting.
This is another useful negative result: covariance shrinkage and constraints can
stabilize risk without overcoming the opportunity cost of defensive weights.

Among individual factors, momentum and the market had the highest sample compound
returns, while investment and profitability had lower standalone volatility.
Momentum had pronounced negative skew (-1.30) and excess kurtosis (9.79);
profitability also had excess kurtosis above 10. Ordinary covariance does not
summarize those tails.

## Limitations and next test

- Factor portfolios are academic research returns, not costless traded funds.
- The cost model applies only to allocation turnover, not underlying factor
  construction, borrow, spread, or capacity.
- A single 50% shrinkage intensity and 40% cap are predeclared sensitivities, not
  universally optimal values.
- The current experiment evaluates the full historical panel and should be repeated
  across publication eras and the 2025 CRSP-format change.
- Sharpe differences need block-bootstrap or other dependent-sample uncertainty.

Next tests should reconstruct investable factor proxies, add volatility targeting,
compare Ledoit--Wolf and factor covariance, and lock an independent international
sample.
