# Data

The input data are licensed (CRSP via WRDS, FactSet, Refinitiv) and are not included.
Place the files below in this folder; the code reads them from `data/crsp` and `data/market`.

| File | Source | Used for |
| --- | --- | --- |
| `crsp/daily_2015_to_2026_09.csv.gz` | CRSP daily stock file (WRDS) | unadjusted closes, total returns (`DlyRet`), trading status |
| `crsp/daily_2014_12_31.csv` | CRSP daily stock file | opening prices for the first signal |
| `crsp/distributions_2015_to_2026_09.csv` | CRSP distributions | dividends, splits, spin-offs, mergers |
| `crsp/delistings_2015_to_2026_09.csv` | CRSP delistings | delisting dates and proceeds |
| `crsp/spin_off_receiver_daily_2017_2026_09.csv.gz`, `crsp/spin_off_*.csv`, `crsp/split_terms.csv` | CRSP | prices and terms of securities received in corporate actions |
| `crsp/confirmed_close_halts.csv`, `crsp/spy_temporary_holdings_2017.csv`, `crsp/spy_*` | CRSP / issuer filings | trading halts and SPY-specific events |
| `market/spy_month_end_holdings.csv.gz` | CRSP mutual-fund holdings, FactSet (Aug. 2016), Refinitiv (Jan. 2021) | reference weights; built by `analysis/prepare_spy_holdings.py` |
| `market/sp500_total_return_usd_daily.csv` | Yahoo Finance | S&P 500 TR benchmark |
| `market/spy_market_total_return_2015_2025.csv`, `market/spy_yahoo_adjusted_close_2015_2025.csv` | CRSP / Yahoo Finance | SPY market-price benchmark |
| `market/sp500_observed_trading_dates.csv` | exchange calendar | trading calendar check |

The full folder used for the report (about 570 MB) is in the project archive.
