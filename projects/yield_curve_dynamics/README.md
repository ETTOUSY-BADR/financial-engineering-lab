# Treasury yield-curve dynamics

This project fits the U.S. Treasury curve with two complementary mathematical
representations:

- principal components, which estimate statistical level, slope, and curvature
  directions;
- Nelson--Siegel factors, which impose smooth maturity loadings and are forecast by
  a recursive VAR(1).

The VAR forecast is evaluated one month ahead against a random-walk curve with an
expanding window. Every prediction is generated only from earlier months.

## Data

The default input is the public FRED CSV endpoint for 3-month through 30-year
constant-maturity Treasury yields. These are par-curve yields in percent, not
zero-coupon rates and not transaction prices. Monthly data use the last available
daily observation. The provenance file records the exact endpoint, series, sample,
generation time, units, and the December 2021 Treasury methodology break.

If FRED is unavailable, a deterministic dynamic Nelson--Siegel panel is generated
and unmistakably labeled synthetic. It tests the pipeline and cannot support an
empirical claim.

## Run

    python projects/yield_curve_dynamics/run_research.py

Outputs include fitted factors and curves, PCA loadings and explained variance,
immutable recursive forecasts, maturity-level forecast errors, and four figures.
