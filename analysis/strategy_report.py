"""Build period and daily diagnostics from a completed, unliquidated stock account."""

import csv
from collections import defaultdict
from decimal import Decimal
from itertools import groupby
from math import prod, sqrt
from pathlib import Path
from statistics import stdev

from analysis.report_workbook import export_workbook
from tlh.backtest import Run


ZERO = Decimal(0)
ONE = Decimal(1)
FLOW_FIELDS = (
    "st_gross_loss", "lt_gross_loss", "st_recognized_loss", "lt_recognized_loss",
    "st_realized_gain", "lt_realized_gain", "wash_deferred", "buys", "sales", "fees",
    "initial_turnover", "turnover", "initial_fees", "fee_nav_ratio",
)


def ledger_flows(run):
    flows = defaultdict(lambda: dict.fromkeys(FLOW_FIELDS, ZERO))
    for sale in run.ledger.sales:
        if sale.gain is None:
            raise ValueError(f"{sale.day}: missing tax basis for sale of {sale.permno}.")
        day = flows[sale.day]
        term = "lt" if sale.is_long_term else "st"
        day[term + "_gross_loss"] += max(-sale.gain, ZERO)
        day[term + "_recognized_loss"] += max(-sale.taxable_gain, ZERO)
        day[term + "_realized_gain"] += max(sale.taxable_gain, ZERO)
        day["wash_deferred"] += sale.disallowed_loss
    active_fees = defaultdict(Decimal)
    for entry in run.ledger.entries:
        day = flows[entry.day]
        day["fees"] += entry.fee
        if entry.kind == "BUY":
            day["buys"] += -entry.cash_change - entry.fee
        elif entry.kind in {"TRADE", "PRE_DELISTING_EXIT"}:
            day["sales"] += entry.cash_change + entry.fee
        else:
            continue
        active_fees[entry.day] += entry.fee
    return flows, active_fees


def compound(returns):
    return prod((ONE + r for r in returns), start=ONE) - ONE


def cagr(total_return, start, end):
    if total_return is None or start == end:
        return None
    return (ONE + total_return) ** (Decimal("365.2425") / (end - start).days) - ONE


def difference(left, right):
    return None if left is None or right is None else left - right


def wealth_return(end, start):
    return None if end is None or start is None or start <= 0 else end / start - ONE


def check_opening(run, initial_cash):
    if not run.history or initial_cash <= 0:
        raise ValueError("A cash-funded account needs a positive initial balance and at least one valuation.")
    v = run.history[0]
    if v.daily_return is None or v.nav is None:
        raise ValueError("The first session needs a complete valuation.")
    opening_check = (initial_cash + v.tax_cash_flow) * (ONE + v.daily_return)
    if abs(v.nav - opening_check) > Decimal("0.000001"):
        raise ValueError("The supplied initial balance does not reconcile to the first NAV and daily return.")


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def write_strategy_report(run: Run, benchmarks, output: Path, *, initial_cash: Decimal,
                          no_tlh: Run | None = None, ex_ante_te=None, pretrade_nav=None,
                          enter_at_close=True, account_name="Strategy Account", ex_ante_benchmark=None,
                          workbook=True):
    """Benchmark inputs are dated total returns; ex_ante_te holds dated annualized forecasts.

    no_tlh must use matching capital, market, dates, costs, tax assumptions and terminal policy.
    A forecast is reported only on its own as-of date. VWAP trading requires an explicit pre-fill NAV.
    """
    check_opening(run, initial_cash)
    if not benchmarks:
        raise ValueError("Supply at least one dated total-return benchmark.")
    dates = [v.day for v in run.history]
    if dates != sorted(set(dates)):
        raise ValueError("Valuations must be unique and chronological.")
    if no_tlh is not None and no_tlh is not run:
        check_opening(no_tlh, initial_cash)
        if (dates != [v.day for v in no_tlh.history] or no_tlh.tax_scenario != run.tax_scenario
                or any(s.execution_source != a.execution_source for s, a in zip(run.history, no_tlh.history))):
            raise ValueError("No-TLH comparison requires matching sessions, execution convention and tax scenario.")
    forecasts = ex_ante_te or {}
    forecast_ref = ex_ante_benchmark or next(iter(benchmarks))
    if forecasts and forecast_ref not in benchmarks:
        raise ValueError("Name the comparison benchmark for the risk model forecast.")
    if forecasts.keys() - set(dates) or any(not Decimal(str(v)).is_finite() or v < 0 for v in forecasts.values()):
        raise ValueError("Supply nonnegative annualized tracking-error forecasts on observed dates.")
    reports = {v.as_of: v for v in run.annual_wealth()}
    baseline_reports = (reports if no_tlh is run else
                        {v.as_of: v for v in no_tlh.annual_wealth()} if no_tlh is not None else {})
    flows, active_fees = ledger_flows(run)
    daily = []
    cum_st = cum_lt = ZERO
    for v in run.history:
        if v.daily_return is None or v.nav is None or v.nav <= 0:
            raise ValueError(f"{v.day}: a complete daily return and positive account NAV are required.")
        flow = flows[v.day]
        notional = flow["buys"] + flow["sales"]
        if notional or flow["fees"]:
            before = (pretrade_nav or {}).get(v.day)
            if before is None and v.execution_source == "CLOSE":
                # At a common closing mark, fills exchange stock for cash. Only order fees change NAV.
                before = v.nav + active_fees[v.day]
            if before is None or before <= 0:
                raise ValueError(f"{v.day}: supply the positive pre-fill NAV at the execution price.")
            flow["initial_turnover" if v.day == dates[0] else "turnover"] = notional / (2 * before)
            flow["fee_nav_ratio"] = flow["fees"] / before
            if v.day == dates[0]:
                flow["initial_fees"] = flow["fees"]
        cum_st += flow["st_recognized_loss"]
        cum_lt += flow["lt_recognized_loss"]
        unreal = difference(v.stock_market_value, v.total_basis)
        row = dict(date=v.day, pretax_return=v.daily_return, nav=v.nav, cash=v.cash,
                   stock_market_value=v.stock_market_value, basis=v.total_basis, net_unrealized=unreal,
                   basis_nav=None if v.total_basis is None else v.total_basis / v.nav,
                   unrealized_nav=None if unreal is None else unreal / v.nav,
                   holdings=sum(q != 0 for q in v.holdings.values()), **flow,
                   cumulative_st_loss=cum_st, cumulative_lt_loss=cum_lt, ex_ante_te=forecasts.get(v.day))
        for name, returns in benchmarks.items():
            observation = ZERO if enter_at_close and v.day == dates[0] else returns.get(v.day)
            r = None if observation is None else Decimal(str(observation))
            if r is None or not r.is_finite():
                raise ValueError(f"{v.day}: missing total return for {name}.")
            row[name + "_return"] = r
        daily.append(row)

    periods, comparisons = [], []
    previous_after = previous_base = initial_cash
    groups = [(str(year), list(rows)) for year, rows in groupby(daily, key=lambda row: row["date"].year)]
    for label, rows in [*groups, ("full", daily)]:
        full = label == "full"
        start, end = rows[0]["date"], rows[-1]["date"]
        final = rows[-1]
        tax = reports[end]
        pre = compound(r["pretax_return"] for r in rows)
        after = wealth_return(tax.after_tax, initial_cash if full else previous_after)
        base = baseline_reports.get(end)
        base_pre = pre if no_tlh is run else no_tlh.pre_tax_twr(start, end) if no_tlh is not None else None
        base_after = wealth_return(base.after_tax, initial_cash if full else previous_base) if base else None
        after_gap = difference(after, base_after)
        tax_alpha = difference(after_gap, difference(pre, base_pre))
        pre_cagr = cagr(pre, start, end) if full else None
        after_cagr = cagr(after, start, end) if full else None
        base_pre_cagr = cagr(base_pre, start, end) if full else None
        base_after_cagr = cagr(base_after, start, end) if full else None
        forecasts_in_period = [r for r in rows if r["ex_ante_te"] is not None]
        last_forecast = forecasts_in_period[-1] if forecasts_in_period else {}
        periods.append(dict(
            period=label, start=start, end=end, sessions=len(rows), pretax_return=pre, aftertax_return=after,
            no_tlh_pretax_return=base_pre, no_tlh_aftertax_return=base_after, aftertax_gap=after_gap,
            tax_alpha=tax_alpha, pretax_cagr=pre_cagr, aftertax_cagr=after_cagr,
            aftertax_gap_cagr=difference(after_cagr, base_after_cagr),
            tax_alpha_cagr=difference(difference(after_cagr, base_after_cagr), difference(pre_cagr, base_pre_cagr)),
            ex_ante_as_of=last_forecast.get("date"), ex_ante_te=last_forecast.get("ex_ante_te"),
            ex_ante_benchmark=forecast_ref if last_forecast else None,
            **{k: sum((r[k] for r in rows), ZERO) for k in FLOW_FIELDS},
            **{k: final[k] for k in ("cumulative_st_loss", "cumulative_lt_loss", "nav", "cash",
                                      "stock_market_value", "basis", "net_unrealized", "basis_nav",
                                      "unrealized_nav", "holdings")},
            min_holdings=min(r["holdings"] for r in rows), max_holdings=max(r["holdings"] for r in rows),
            aftertax_wealth=tax.after_tax, tax=tax.tax.cumulative_tax if full else tax.tax.net_tax,
            st_carryforward=tax.tax.st_carryforward, lt_carryforward=tax.tax.lt_carryforward,
            tax_final=tax.tax.final, no_tlh_tax_final=base.tax.final if base else None,
            tax_review_through=tax.tax.review_end, unverified_events="|".join(tax.unverified_split_ratios),
            account=account_name, tax_scenario=run.tax_scenario.value, initial_cash=initial_cash,
            no_tlh_reference="self" if no_tlh is run else "paired" if no_tlh is not None else "none",
            entry_at_close=enter_at_close,
        ))
        for name in benchmarks:
            rs = [r[name + "_return"] for r in rows]
            benchmark_return = compound(rs)
            active = [float(r["pretax_return"] - b) for r, b in zip(rows, rs)]
            bench_cagr = cagr(benchmark_return, start, end) if full else None
            comparisons.append(dict(period=label, benchmark=name, benchmark_return=benchmark_return,
                                    pretax_alpha=pre - benchmark_return, benchmark_cagr=bench_cagr,
                                    pretax_alpha_cagr=difference(pre_cagr, bench_cagr),
                                    realized_te=stdev(active) * sqrt(252) if len(active) > 1 else None))
        previous_after, previous_base = tax.after_tax, base.after_tax if base else None

    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "report_daily.csv", daily)
    write_rows(output / "report_periods.csv", periods)
    write_rows(output / "report_benchmarks.csv", comparisons)
    if workbook:
        export_workbook(output)
    return daily, periods, comparisons
