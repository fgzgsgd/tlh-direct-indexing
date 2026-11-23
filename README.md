# Direct Indexing with Tax-Loss Harvesting

A two-step tax- and wash-sale-aware optimization strategy on the S&P 500, backtested from cash over 2016–2025 with a lot-level tax ledger and the 30-day wash-sale rule.
The report is [`report/report.pdf`](report/report.pdf) (9 pages).

## Results (Jan. 2016 – Dec. 2025, $1 million initial cash)

| | Strategy | No-TLH control | S&P 500 TR |
| --- | ---: | ---: | ---: |
| Pre-tax CAGR | 15.76% | 14.77% | 15.01% |
| After-tax CAGR | 15.95% | 14.21% | – |
| Tax alpha, per year | 0.75% | – | – |
| Realized tracking error | 1.24% | 0.50% | – |
| After-tax wealth, end of 2025 | $4.39M | $3.77M | – |

Tax alpha is the reduction in tax drag (after-tax minus pre-tax CAGR) against a No-TLH account that replicates SPY monthly under the same costs and tax rules.

## Strategy

Each month-end the optimizer chooses target weights that minimize estimated tax plus trading cost, subject to long-only, fully invested and 1% ex-ante tracking-error constraints. It makes no return forecasts.

1. **Step 1, direction search.** SLSQP from three starting portfolios on a wash-sale-adjusted objective. Every stock holding a loss lot is labeled BUY, SELL or HOLD. Up to three distinct direction patterns are kept.
2. **Step 2, lot-level refinement.** With directions fixed the problem is a convex second-order cone program, solved with CVXPY/Clarabel. The pattern with the lowest cost is traded.
3. **Execution.** Targets are floored to whole shares, sale lots are chosen least-tax-first-out (LTFO), and orders fill at the next trading day's close.

Risk uses a 252-day Ledoit–Wolf covariance. Taxes: short-term 40.8%, long-term and dividends 23.8%, losses offset outside gains in full each year. Costs: 5 bp per side.

## Repository layout

| Path | Contents |
| --- | --- |
| `tlh/` | Tax-lot ledger, wash-sale matching, corporate actions and backtest engine (standard library only) |
| `tlh/optimization/` | Risk model, LTFO tax curves, the two-step optimizer (`wash_a_strategy.py`) |
| `analysis/` | Experiment entry point (`wash_a.py`, `tax_aware.py`), reporting and data preparation |
| `scripts/run_no_tlh.py` | No-TLH control account |
| `tests/`, `examples/` | Unit and integration tests, and a small weights file they use |
| `results/strategy/` | Frozen outputs of the final run used in the report (daily and annual reports, monthly optimizer decisions) |
| `results/no_tlh/` | Frozen outputs of the No-TLH control |
| `results/wash/` | Wash-sale trace of the final run: matches, deferrals and releases |
| `report/` | LaTeX source, figures and `make_results.py`, which builds every figure and number in Section 3 from `results/` |
| `data/` | Input data (not included, see `data/README.md`) |

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# copy the licensed input data into data/ (see data/README.md)

python -m unittest discover -s tests -q                                    # tests
python -m analysis.wash_a --output runs/strategy --solver-cache .cache     # final strategy, about 1 hour on 3 cores
python scripts/run_no_tlh.py --output runs/no_tlh                         # No-TLH control, about 1 minute

python report/figures/make_results.py                                     # figures and numbers from results/
cd report && pdflatex -output-directory=build main.tex                     # report (run twice)
```

The figures read the frozen outputs in `results/`. To rebuild them from a new run, copy the new reports into `results/strategy` and `results/no_tlh`. `results/strategy/comparison.json` holds the combined strategy and No-TLH metrics, including annual tax alpha.

## Use of AI tools

AI assistants (ChatGPT and Claude) were used to collect papers and industry reports, review and debug code, write test cases, reorganize the code, and draft parts of the LaTeX source. All design choices and reported numbers were checked against the code and outputs in this repository.
