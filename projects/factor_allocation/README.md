# Robust public-factor allocation

This laboratory downloads the Fama--French five research factors and momentum from
the Kenneth R. French Data Library, parses the monthly tables, and performs a
strictly walk-forward allocation across six factor sleeves.

At each month the model uses the previous 120 months, shrinks sample covariance 50%
toward its diagonal, and solves a long-only minimum-variance program with a 40%
maximum sleeve weight. The next month's return is then observed. The experiment
charges ten basis points per unit of turnover and compares against equal weighting.

## Research questions

- How stable are factor correlations and individual tail risks?
- Does covariance shrinkage plus concentration control improve realized risk?
- How much of any improvement survives a transparent turnover charge?

## Data caution

The factor library is a high-quality public research benchmark, but its portfolios
are not costless directly traded securities. The library's construction methodology
and upstream CRSP format have changed. The exact URLs, sample, transformations, and
generation time are recorded in provenance.txt.

If download fails, a deterministic heavy-tailed factor panel is labeled synthetic.
It validates the pipeline only.

## Run

    python projects/factor_allocation/run_research.py

Outputs include clean factors, diagnostics, correlations, monthly out-of-sample
weights and returns, strategy comparisons, and three figures.
