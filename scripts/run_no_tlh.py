"""No-TLH control: monthly SPY replication with the same costs, timing, LTFO relief and tax rules.

Usage (from the repository root):  python scripts/run_no_tlh.py --output runs/no_tlh
"""
import argparse
import json
import sys
from datetime import date
from decimal import Decimal
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import spy_benchmark as benchmark  # noqa: E402
from tlh.lot_relief import hifo  # noqa: E402
from tlh.optimization.two_stage_ltfo import ltfo_with_fees  # noqa: E402


def main(output: Path):
    benchmark.write_strategy_report = partial(benchmark.write_strategy_report, workbook=False)
    benchmark.LOT_METHODS['ltfo'] = partial(ltfo_with_fees, fee_rate=Decimal('0.0005'))
    original_run = benchmark.run

    def matched_run(**kwargs):
        kwargs['lot_relief'] = hifo  # forced exits use HIFO, as in the active strategy
        return original_run(**kwargs)

    benchmark.run = matched_run
    output.mkdir(parents=True, exist_ok=False)
    summary = benchmark.run_benchmark(ROOT / 'data/market/spy_month_end_holdings.csv.gz',
                                      ROOT / 'data/crsp/daily_2014_12_31.csv', output,
                                      date(2016, 1, 4), date(2025, 12, 31), lot_method='ltfo')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
