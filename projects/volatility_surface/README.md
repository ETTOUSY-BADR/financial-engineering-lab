# Arbitrage-aware volatility surface and Heston validation

This project turns a public S&P 500 index (SPX) option-chain snapshot into an auditable volatility
surface. It is designed as a research artifact, not a screenshot of implied
volatility.

## Research questions

1. Can discount factors and forwards be inferred internally from put--call parity?
2. Can raw SVI fit each smile while satisfying dense-grid butterfly restrictions?
3. Does the repaired surface remain calendar-monotone across observed maturities?
4. Does one global Heston parameter vector generalize to held-out strikes better
   than a flat-volatility benchmark?
5. Which parameters are unstable across calibration starts even when pricing loss
   is similar?

## Defensibility controls

- raw quotes, acquisition time, source, expirations, and SHA-256 checksum are saved;
- displayed bid/ask fields, volume, open interest, and last-trade time remain visible;
- only trades from one completed reference session enter the study;
- forwards and discount factors come from recency- and liquidity-weighted robust parity regression;
- only OTM observations with positive same-session trades enter the calibration;
- SVI positivity, Lee wing slopes, butterfly density, and calendar variance are audited;
- Heston uses deterministic multi-start calibration and a strike holdout set;
- a flat-volatility model is retained as the null benchmark;
- local Jacobian uncertainty, Feller slack, boundary hits, and multi-start dispersion
  are reported rather than hidden;
- a deterministic synthetic fallback validates software only and is labeled as such.

SPX options are European-style, which makes equality-based put--call parity coherent
with the contract exercise convention. Yahoo Finance is a convenient public distributor, not an exchange record or an
OPRA-licensed historical database. Its displayed bid/ask fields were empirically
too sparse for this snapshot, so the study uses timestamped same-session transactions
and propagates parity residuals as a noise scale. The cross section is asynchronous,
is retained for educational model validation, and does not claim executable prices.

## Run

Use the committed snapshot and reproduce every derived output:

```powershell
python projects/volatility_surface/run_research.py
```

Explicitly request a new public snapshot:

```powershell
python projects/volatility_surface/run_research.py --refresh
```

Build the formal report after the pipeline completes:

```powershell
powershell -ExecutionPolicy Bypass -File projects/volatility_surface/build_report.ps1
```

The compiled [research paper](report.pdf) presents the assumptions, identification
proof, SVI arbitrage conditions, Heston specification, held-out evidence, and model
risk register. The
[executed audit notebook](../../notebooks/07_spx_surface_heston_validation.ipynb)
rechecks the manifest and the decisive numerical claims from a fresh kernel.

## Outputs

The output folder contains raw and cleaned quotes, parity estimates, constrained SVI
parameters, static-arbitrage diagnostics, Heston multi-start results, held-out
validation errors, report-ready LaTeX tables, figures, a provenance record, and a
checksum manifest.
