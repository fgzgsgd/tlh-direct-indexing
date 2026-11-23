"""Read CRSP CIZ daily stock data exported from WRDS."""

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
    """Read a CSV, compressed CSV or single-table ZIP. Column names are lowercased and values kept as is."""
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
    """The project's verified NYSE trading calendar, independent of any backtest's start and end dates."""
    path = Path(__file__).resolve().parents[1] / "data/market/sp500_observed_trading_dates.csv"
    with crsp_table(path) as rows:
        return tuple(date.fromisoformat(row["trading_date"]) for row in rows)


def last_trading_day_of_year(year: int) -> date:
    year_end = date(year, 12, 31)
    calendar = nyse_trading_days()
    if calendar[0] <= year_end <= calendar[-1]:
        return calendar[bisect_right(calendar, year_end) - 1]
    # If the calendar does not yet cover year-end, skip only weekends, which are certain closures, so the file's last row is not mistaken for the end of the year.
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
    """DlyRet includes distributions and is for return analysis only. Fills and valuations use a separate trade-price column."""
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
    # A WRDS null means no observation; a zero return is a valid observation.
    return Decimal(value) if value else None


def read_vwaps(path: Path) -> dict[date, dict[int, Decimal]]:
    """Read a separate price table date,permno,vwap; each row is an unadjusted USD/share for the same specified trading session."""
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
    """Read only the vendor's daily constituent weights (decimals). Missing dates or constituents have no substitute."""
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
    """Keep each historical membership interval; MbrEndDt is not the security's delisting date."""
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
    """DelistingDt is the last price date on the CRSP-tracked exchange. DelRet creates no account cash flow."""
    with crsp_table(path) as rows:
        return [Delisting(int(r["permno"]), date.fromisoformat(r["delistingdt"]),
                          f"{r['delactiontype']}/{r['delreasontype']}") for r in rows]


def read_close_halts() -> dict[date, frozenset[int]]:
    """On dates the local CRSP still marks Active, add verified halts that prevent trading at the close."""
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
