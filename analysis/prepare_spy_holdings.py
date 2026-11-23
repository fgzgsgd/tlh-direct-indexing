"""Rebuild the 132 SPY month-end equity snapshots from the original vendor exports."""

import argparse
import csv
import gzip
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from tlh.market import crsp_table, nyse_trading_days, read_delistings
from tlh.spy_benchmark import monthly_signals
from analysis.spy_benchmark import CRSP, MARKET, PRIOR, START, END


def prepare(destination: Path):
    calendar = (PRIOR, *nyse_trading_days())
    dates = {d.isoformat()[:7]: d for d in monthly_signals(calendar, START, END).values()}
    ended = {d.permno: d.day for d in read_delistings(CRSP / "delistings_2015_to_2026_09.csv")}
    with crsp_table(MARKET / "spy_security_links.csv") as rows:
        links = {(r["source"], r["source_security_id"], r["security_name"], r["reported_permno"]): int(r["permno"])
                 for r in rows}
    amounts = defaultdict(Decimal)
    by_report = defaultdict(list)
    with crsp_table(MARKET / "spy_crsp_reported_holdings_2014_12_2025_11.zip") as rows:
        for row in rows:
            if row["crsp_portno"] != "1021980":
                raise ValueError("Unexpected CRSP fund.")
            by_report[dates[row["report_dt"][:7]]].append(row)
    for day, positions in sorted(by_report.items()):
        for row in positions:
            name, cusip, raw = row["security_name"], row["cusip"], row["permno"]
            kind, p = "equity", links.get(("CRSP", cusip, name, raw), int(raw) if raw else None)
            if name in {"USD CASH", "GBP CASH", "JPY CASH"}:
                kind, p = "cash", None
            elif (name == "OTHER ASSETS LESS LIABILITIES" or cusip in {"11012215", "67459916", "25470M11"}
                  or name == "MILLROSE PROPERTIES INC WI" or (p in ended and ended[p] <= day)
                  or (not p and name in {"AETNA ORD", "TWITTER INC ORD", "FIRST NIAGARA FINANCIAL GROUP INC ORD"})):
                # Non-common-stock rights, merger receivables, and final-day balances cannot be bought next session.
                kind, p = "other_net_assets", None
            elif day == date(2015, 9, 30) and name == "ALPHABET INC ORD":
                same = [v for v in positions if v["permno"] == "14542" and v["market_val"] == row["market_val"]
                        and v["nbr_shares"] == row["nbr_shares"]]
                if len(same) != 1:
                    raise ValueError("Review the duplicate Google class C position.")
                continue
            elif day == date(2015, 10, 30) and name == "HEWLETT PACKARD ENTERPRISE CO":
                # Both when-issued legs appear on the statement. The regular-way parent receives HPE at the next open.
                p = 27828
            if kind == "equity" and not p:
                raise ValueError(f"{day}: unresolved {name} ({cusip}).")
            amounts[day, kind, str(p or ""), "CRSP reported market value"] += Decimal(row["market_val"])

    factset_day = date(2016, 8, 31)
    with crsp_table(MARKET / "spy_factset_holdings_2016-08-31.csv") as rows:
        for row in rows:
            if (row["factset_fund_id"], row["report_date"]) != ("04BH50-E", str(factset_day)):
                raise ValueError("Unexpected FactSet fund or report date.")
            key = ("FactSet", row["fsym_id"], row["security_name"], "")
            p = links[key]
            amounts[factset_day, "equity", str(p), "FactSet reported market value"] += Decimal(row["reported_mv"])

    from openpyxl import load_workbook
    workspace_day = date(2021, 1, 29)
    workbook = load_workbook(MARKET / "spy_workspace_holdings_2021-01-31.xlsx", data_only=True, read_only=True)
    try:
        sheet = workbook.active
        sheet.reset_dimensions()
        contents = list(sheet.values)
    finally:
        workbook.close()
    if contents[1][1] != "31-Jan-2021":
        raise ValueError("Unexpected Workspace report date.")
    for ric, name, _, weight, *_ in contents[4:]:
        kind, code = ("cash", "") if name == "USD CASH" else (
            "equity", str(links["Workspace", ric or "", name, ""]))
        amounts[workspace_day, kind, code, "Workspace Weight"] += Decimal(str(weight))

    by_date = defaultdict(set)
    for (day, kind, code, _), amount in amounts.items():
        if kind == "equity" and amount > 0:
            by_date[day].add(code)
    if set(by_date) != set(dates.values()) or any(len(s) < 490 for s in by_date.values()):
        raise ValueError("Unexpected monthly equity coverage.")
    if len(by_date[factset_day]) != 505 or len(by_date[workspace_day]) != 505:
        raise ValueError("A supplemental snapshot did not map all 505 equity positions.")
    with gzip.open(destination, "wt", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(("as_of", "kind", "permno", "amount", "source"))
        writer.writerows((day, kind, p, v, source) for (day, kind, p, source), v in sorted(amounts.items()))
    return len(by_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MARKET / "spy_month_end_holdings.csv.gz")
    print(f"{prepare(parser.parse_args().output)} verified month-end snapshots")
