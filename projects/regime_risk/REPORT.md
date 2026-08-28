# Empirical report: regime-aware cross-asset risk

## Snapshot

This report summarizes the reproducible run generated on 28 August 2026 from 4,189
Yahoo Finance adjusted-close observations spanning 4 January 2010 through 27 August
2026. The analytical return sample contains 4,188 days. Results are educational and
conditional on the convenience feed, proxy choice, cost rule, and sample.

## Main evidence

| Strategy | Geometric return | Volatility | Sharpe | Maximum drawdown | Average daily turnover |
|---|---:|---:|---:|---:|---:|
| SPY buy-and-hold | 14.22% | 17.08% | 0.833 | -33.72% | 0.00% |
| SPY 10% volatility target, net | 9.86% | 11.07% | 0.891 | -14.38% | 2.99% |
| Cross-asset inverse volatility, net | 9.12% | 8.84% | 1.032 | -21.99% | 0.52% |

The lagged volatility target materially reduced realized volatility and maximum
drawdown, but it also reduced compound return. Its higher Sharpe is therefore a
risk-scaling result, not a claim of free alpha. The inverse-volatility allocation had
the lowest ordinary volatility and highest sample Sharpe, but still experienced a
21.99% drawdown and retained excess kurtosis of 4.65.

## Observable regimes

The stress state is known before each return: yesterday's VIX must exceed its
trailing 75th percentile. SPY annualized volatility was 27.58% in 900 stress-state
days versus 12.78% in 3,288 calm-state days. TLT's SPY correlation became more
negative in the stress state (-0.382 versus -0.219), while gold's sample correlation
was near zero in stress.

SPY's annualized mean was higher after high-VIX observations. This should not be
read as a causal stress premium: the state definition is persistent and captures
recovery rallies after volatility spikes. The one-percent daily SPY return was much
worse in stress (-4.50%) than calm (-2.32%), which better reveals the conditional
tail.

## Limitations and next test

- Adjusted ETF closes do not model executable bid, ask, intraday path, taxes,
  capacity, or institutional data corrections.
- The two-basis-point linear charge is a sensitivity point, not an impact model.
- The VIX state is observable but ad hoc; threshold search was not optimized.
- TLT's long-duration exposure and behavior changed with the rate regime.
- A stronger replication should use licensed total-return data, Treasury futures,
  multiple volatility estimators, a pre-2020/post-2020 split, and block-bootstrap
  uncertainty.

The CSV files contain every weight, exposure, forecast, cost, regime, and return
needed to audit these statements.
