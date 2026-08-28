# Authoritative data-source registry

Every project should prefer the primary producer, preserve the exact query and
vintage, and review redistribution terms. Convenience feeds are useful for prototypes
but should not silently become production evidence.

| Domain | Preferred public source | What to preserve |
|---|---|---|
| U.S. macro and vintages | FRED and ALFRED: https://fred.stlouisfed.org/docs/api/fred/ | series ID, underlying agency, units, seasonal adjustment, realtime vintage, release timestamp |
| U.S. Treasury curve | U.S. Treasury: https://home.treasury.gov/resource-center/data-chart-center/interest-rates | curve type, par-versus-zero meaning, date, methodology, December 2021 method break |
| Monetary policy and reference rates | Federal Reserve and New York Fed: https://www.newyorkfed.org/markets | business date, publication time, operation or reference-rate definition |
| U.S. company filings | SEC EDGAR APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces | CIK mapping, filing accession, fiscal period, acceptance timestamp, amendment, fair-access compliance |
| Academic factor returns | Kenneth R. French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html | downloaded archive, checksum, construction notes, frequency, percent-to-decimal conversion, methodology version |
| Options and exchange statistics | Cboe: https://www.cboe.com/data/market_statistics/ | exchange, symbol, contract specification, timestamp, bid/ask, license |
| European macro and finance | ECB Data Portal: https://data.ecb.europa.eu/ | dataflow key, area composition, frequency, units, adjustment, vintage |
| Global banking and credit | BIS Data Portal: https://data.bis.org/ | country coverage, break flags, currency and consolidation basis |

## Repository policy

- Raw downloads are immutable or reproducibly re-downloadable.
- Derived datasets live beside the project that creates them.
- Every run writes a provenance file with source, endpoint, requested sample,
  returned sample, transformations, generation time, and limitations.
- Synthetic fallbacks are labeled in filenames or manifests and may test software
  only. They are never market evidence.
- Secrets and API keys are read from the environment and never committed.
- A point-in-time study stores event time, publication time, ingestion time, and
  decision time separately.

Yahoo Finance through yfinance is used in the open cross-asset prototype because it
is convenient and reproducible for many readers. It is not treated as the primary
exchange record or as licensed institutional point-in-time data.

The SPX volatility-surface project also preserves a Yahoo Finance delayed-chain
snapshot because a redistributable exchange-certified historical NBBO is not
available in this open laboratory. Contract transaction timestamps, displayed
bid/ask fields, volume, open interest, acquisition time, source classification, and
checksums are retained. The study uses reference-session last transactions because
the displayed bid/ask cross-section was too sparse; it therefore labels prices as
asynchronous research observations rather than executable quotes. The Options
Industry Council's bid/ask discussion is recorded as market-structure context:
https://www.optionseducation.org/news/understanding-the-bid-and-ask-prices-for-options
