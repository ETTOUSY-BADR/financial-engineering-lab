# Empirical report: Treasury yield-curve dynamics

## Snapshot

The live run generated on 28 August 2026 uses 395 complete monthly observations from
October 1993 through the August 2026 month-end label. Inputs are ten FRED
constant-maturity Treasury series from three months through thirty years. They are
par yields in percentage points, not zero-coupon yields or transaction prices.

## Cross-sectional structure

| Principal component | Explained variance |
|---|---:|
| PC1 | 92.972% |
| PC2 | 6.422% |
| PC3 | 0.505% |
| First three | 99.900% |

The familiar low-dimensional curve structure is strongly visible: three orthogonal
directions describe almost all sample variation. The economic labels level, slope,
and curvature remain interpretations; PCA signs and rotations are statistical.

Nelson--Siegel provides a separate smooth representation with explicit maturity
loadings. Its monthly factor history and cross-sectional fitting errors are saved,
allowing the user to inspect periods where three factors do not fit the par curve
well.

## Genuine recursive forecast

After an initial 120-month window, a VAR(1) is re-estimated at every month using only
earlier Nelson--Siegel factors. Its next-month curve forecast is compared with the
previous month's curve.

Average maturity-level RMSE was 0.249 percentage points for the Nelson--Siegel VAR
and 0.229 percentage points for the random walk. The random walk had lower RMSE at
every included maturity. The structured model therefore fails to beat the simple
benchmark in this design. This negative result is informative: excellent
cross-sectional dimension reduction does not imply superior time-series forecasts.

## Limitations and next test

- Complete-case filtering starts the joint panel in 1993 and can select dates.
- Month-end uses the last available daily value, not a synchronized transaction.
- FRED series are par-curve constructs and include a Treasury methodology change on
  6 December 2021.
- The fixed Nelson--Siegel decay was not tuned; tuning would require nested
  validation.
- Forecast comparison reports point errors without formal serial-dependence tests.

Next steps are an arbitrage-consistent par-to-zero bootstrap, state-space dynamic
Nelson--Siegel estimation, forecast-combination shrinkage, pre/post-methodology
analysis, and evaluation through bond price or duration-scaled loss.
