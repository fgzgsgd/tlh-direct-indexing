"""合并按 PERMNO、日期排序的 WRDS 日行情，保留成分期外记录和全部原始字段。"""

import csv
import gzip
from contextlib import ExitStack
from heapq import merge
from pathlib import Path
from tempfile import TemporaryDirectory

from tlh.market import crsp_table


DATA = Path(__file__).resolve().parents[1] / "data" / "crsp"
DAILY_PRICES = DATA / "daily_2015_to_2026_09.csv.gz"


def security_day(row):
    return int(row["permno"]), row["dlycaldt"]


def merge_daily(sources, target, *, omit_columns=()):
    kept = duplicates = 0
    previous_key = previous_row = None
    with TemporaryDirectory(dir=target.parent) as tmp, ExitStack() as files:
        tables = [files.enter_context(crsp_table(path)) for path in sources]
        columns = list(dict.fromkeys(name for table in tables for name in table.fieldnames if name not in omit_columns))
        output = Path(tmp) / target.name
        with gzip.open(output, "wt", encoding="utf-8", newline="", compresslevel=5) as out:
            writer = csv.DictWriter(out, columns, lineterminator="\n")
            writer.writeheader()
            for row in merge(*tables, key=security_day):
                for name in omit_columns:
                    row.pop(name, None)
                key = security_day(row)
                if key == previous_key:
                    if row != previous_row:
                        raise ValueError(f"Conflicting CRSP daily rows for {key}.")
                    duplicates += 1
                    continue
                if previous_key is not None and key < previous_key:
                    raise ValueError(f"CRSP export is out of order at {key}.")
                writer.writerow(row)
                previous_key, previous_row = key, row
                kept += 1
        output.replace(target)
    return kept, duplicates


if __name__ == "__main__":
    sources = [DATA / "raw" / name for name in
               ("daily_base.zip", "daily_supplement.zip", "daily_spin_off_receivers.zip",
                "daily_spin_off_extended.zip", "daily_receiver_history.zip")]
    # 新导出多选的两个上市区间字段保留在原始 ZIP，行情按同一组原始报价字段对账。
    kept, duplicates = merge_daily(sources, DAILY_PRICES, omit_columns=("securitybegdt", "securityenddt"))
    print(f"{DAILY_PRICES.name}: {kept:,} unique security-days, {duplicates:,} identical rows removed.")
