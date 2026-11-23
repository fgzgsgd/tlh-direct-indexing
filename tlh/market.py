"""读取 WRDS 导出的 CRSP CIZ 个股日数据。"""

import argparse
import csv
import gzip
from bisect import bisect_right
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from functools import cache
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile


@contextmanager
def crsp_table(path: Path) -> Iterator[csv.DictReader]:
    """读取 CSV、压缩 CSV 或单表 ZIP。列名统一为小写，数值原样保留。"""
    with ExitStack() as files:
        if path.suffix.lower() == ".zip":
            archive = files.enter_context(ZipFile(path))
            name, = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            source = files.enter_context(TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline=""))
        elif path.suffix.lower() == ".gz":
            source = files.enter_context(gzip.open(path, "rt", encoding="utf-8-sig", newline=""))
        else:
            source = files.enter_context(path.open(encoding="utf-8-sig", newline=""))
        rows = csv.DictReader(source)
        rows.fieldnames = [name.strip().lower() for name in rows.fieldnames or ()]
        yield rows


@cache
def nyse_trading_days() -> tuple[date, ...]:
    """项目已核实的 NYSE 交易日历，独立于任何一次回测的起止日。"""
    path = Path(__file__).resolve().parents[1] / "data/market/sp500_observed_trading_dates.csv"
    with crsp_table(path) as rows:
        return tuple(date.fromisoformat(row["trading_date"]) for row in rows)


def last_trading_day_of_year(year: int) -> date:
    year_end = date(year, 12, 31)
    calendar = nyse_trading_days()
    if calendar[0] <= year_end <= calendar[-1]:
        return calendar[bisect_right(calendar, year_end) - 1]
    # 日历尚未覆盖年底时，仅跳过确定休市的周末，避免把文件末行误作全年结束。
    return year_end - timedelta(days=max(0, year_end.weekday() - 4))


@dataclass(frozen=True, slots=True)
class StockDay:
    permno: int
    day: date
    total_return: Decimal | None
    raw_open: Decimal | None
    raw_close: Decimal | None
    ordinary_dividend: Decimal | None = None
    nonordinary_distribution: Decimal | None = None
    price_factor: Decimal | None = None
    trading_status: str | None = None
    distribution_flag: str | None = None


def read_crsp_daily(path: Path, *, require_prices: bool) -> Iterator[StockDay]:
    """DlyRet 含分配，仅供收益分析。成交和持仓估值取独立的交易价格列。"""
    with crsp_table(path) as rows:
        required = {"permno", "dlycaldt", "dlyclose" if require_prices else "dlyret"}
        missing = required.difference(rows.fieldnames)
        if missing:
            raise ValueError(f"{path.name}: missing CRSP columns: {', '.join(sorted(missing))}")
        for row in rows:
            yield StockDay(
                int(row["permno"]),
                date.fromisoformat(row["dlycaldt"]),
                _number(row.get("dlyret")),
                _number(row.get("dlyopen")),
                _number(row.get("dlyclose")),
                _number(row.get("dlyorddivamt")),
                _number(row.get("dlynonorddivamt")),
                _number(row.get("dlyfacprc")),
                row.get("tradingstatusflg"),
                row.get("dlydistretflg"),
            )


def _number(value: str | None) -> Decimal | None:
    # WRDS 空值表示没有观测；零收益则是一个有效观测。
    return Decimal(value) if value else None


def read_vwaps(path: Path) -> dict[date, dict[int, Decimal]]:
    """读独立行情表 date,permno,vwap；每行是同一指定成交时段的未复权 USD/share。"""
    prices: dict[date, dict[int, Decimal]] = {}
    with crsp_table(path) as rows:
        for row in rows:
            day, permno = date.fromisoformat(row["date"]), int(row["permno"])
            by_security = prices.setdefault(day, {})
            if permno in by_security:
                raise ValueError(f"{day} {permno}: duplicate VWAP in {path.name}.")
            price = Decimal(row["vwap"])
            if not price.is_finite() or price <= 0:
                raise ValueError(f"{day} {permno}: VWAP must be a positive trade price.")
            by_security[permno] = price
    return prices


def read_benchmark_weights(path: Path, *, date_column: str = "date", id_column: str = "permno",
                           weight_column: str = "weight") -> dict[date, dict[int, Decimal]]:
    """只读取供应商给出的逐日成分权重（小数）。缺少的日期或成分没有替代值。"""
    weights: dict[date, dict[int, Decimal]] = {}
    with crsp_table(path) as rows:
        columns = date_column.lower(), id_column.lower(), weight_column.lower()
        if not set(columns).issubset(rows.fieldnames):
            raise ValueError(f"{path.name}: expected benchmark weight columns {columns}.")
        for row in rows:
            d, p, raw = (row[k] for k in columns)
            day, permno = date.fromisoformat(d), int(p)
            if not raw.strip():
                raise ValueError(f"{day} {permno}: missing benchmark weight.")
            weight = Decimal(raw)
            daily = weights.setdefault(day, {})
            if permno in daily or not weight.is_finite() or not 0 <= weight <= 1:
                raise ValueError(f"{day} {permno}: duplicate or invalid benchmark weight.")
            daily[permno] = weight
    return weights


@dataclass(frozen=True, slots=True)
class Membership:
    permno: int
    start: date
    end: date


def read_membership(path: Path) -> list[Membership]:
    """保留每段历史成员区间；MbrEndDt 不是证券退市日期。"""
    with crsp_table(path) as rows:
        permno, start, end = "permno", "mbrstartdt", "mbrenddt"
        if "first_in_sample_trading_date" in rows.fieldnames:
            permno, start, end = ("permno_if_available", "first_in_sample_trading_date",
                                  "last_in_sample_trading_date_inclusive")
        members = []
        for r in rows:
            if not r[permno]:
                raise ValueError(f"{path.name}: missing PERMNO for interval starting {r[start]}.")
            members.append(Membership(int(r[permno]), date.fromisoformat(r[start]), date.fromisoformat(r[end])))
        return members


@dataclass(frozen=True, slots=True)
class Delisting:
    permno: int
    day: date
    reason: str


def read_delistings(path: Path) -> list[Delisting]:
    """DelistingDt 是 CRSP 跟踪交易所的最后价格日。DelRet 不产生账户现金流。"""
    with crsp_table(path) as rows:
        return [Delisting(int(r["permno"]), date.fromisoformat(r["delistingdt"]),
                          f"{r['delactiontype']}/{r['delreasontype']}") for r in rows]


def read_close_halts() -> dict[date, frozenset[int]]:
    """在本地 CRSP 仍标 Active 的日期，加入已核实妨碍收盘成交的停牌。"""
    path = Path(__file__).resolve().parents[1] / "data/crsp/confirmed_close_halts.csv"
    by_day: dict[date, set[int]] = {}
    with crsp_table(path) as rows:
        for row in rows:
            by_day.setdefault(date.fromisoformat(row["date"]), set()).add(int(row["permno"]))
    return {d: frozenset(names) for d, names in by_day.items()}


def describe(path: Path) -> str:
    first = last = None
    securities = set()
    count = returns = opens = closes = 0
    for row in read_crsp_daily(path, require_prices=False):
        first = row.day if first is None else min(first, row.day)
        last = row.day if last is None else max(last, row.day)
        securities.add(row.permno)
        count += 1
        returns += row.total_return is not None
        opens += row.raw_open is not None
        closes += row.raw_close is not None
    return (f"{path.name}: rows={count}, PERMNOs={len(securities)}, dates={first}..{last}, "
            f"DlyRet={returns}, DlyOpen={opens}, DlyClose={closes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect CRSP daily observations and price coverage.")
    parser.add_argument("paths", type=Path, nargs="+", metavar="CSV")
    for file in parser.parse_args().paths:
        print(describe(file))
