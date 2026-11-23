"""Run the SPY monthly stock-replication baseline and compare with the traded ETF."""

import argparse
import csv
import gzip
import json
import math
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from itertools import chain
from pathlib import Path
from statistics import correlation, stdev

from tlh.backtest import INITIAL_CASH, Run, run, sessions_from_crsp
from analysis.spy_replication_events import benchmark_actions
from analysis.strategy_report import write_strategy_report
from tlh.actions import CorporateAction, PropertyDividend, SpinOff
from tlh.lot_relief import hifo, ltfo
from tlh.market import crsp_table, nyse_trading_days, read_crsp_daily, read_delistings
from tlh.spy_benchmark import SpyReplication, monthly_signals, read_monthly_holdings


ROOT = Path(__file__).resolve().parents[1]
CRSP = ROOT / "data/crsp"
MARKET = ROOT / "data/market"
START = date(2015, 1, 2)
END = date(2025, 12, 31)
PRIOR = date(2014, 12, 31)
SPY_RETURNS = MARKET / "spy_market_total_return_2015_2025.csv"
LOT_METHODS = {"hifo": hifo, "ltfo": ltfo}


def write_table(path, fields, records):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def read_etf_returns(column="market_total_return") -> dict[date, Decimal]:
    with crsp_table(SPY_RETURNS) as rows:
        return {date.fromisoformat(r["date"]): Decimal(r[column]) for r in rows if r[column]}


def adjusted_close_returns() -> dict[date, Decimal]:
    with crsp_table(MARKET / "spy_yahoo_adjusted_close_2015_2025.csv") as rows:
        closes = [(date.fromisoformat(r["date"]), Decimal(r["adjusted_close_usd"])) for r in rows]
    return {day: price / prev - 1 for (_, prev), (day, price) in zip(closes, closes[1:])}


def write_signal_weights(output: Path, signals, weights):
    """Export the normalized SPY weights before security-specific execution restrictions."""
    with gzip.open(output / "monthly_spy_weights.csv.gz", "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("signal_close", "trade_date", "permno", "stock_weight"))
        for trade, signal in sorted(signals.items()):
            writer.writerows((signal, trade, p, w) for p, w in sorted(weights[signal].items()))


def compare(result: Run, spy: Mapping[date, Decimal], output: Path, held_orders=(), nav_returns=None,
            *, lot_method=None):
    daily, annual, active = [], [], []
    twr = etf = start_twr = start_etf = Decimal(1)
    start_day = result.history[0].day
    reports = {v.as_of: v for v in result.annual_wealth()}
    for v in result.history:
        if v.nav is None or v.daily_return is None:
            raise ValueError(f"{v.day}: incomplete account valuation.")
        # Both accounts start as cash and enter at the first close. The stock account pays its entry commission.
        etf_return = Decimal(0) if v.day == start_day else spy[v.day]
        twr *= 1 + v.daily_return
        etf *= 1 + etf_return
        active.append(float(v.daily_return - etf_return))
        daily.append(dict(date=v.day, replica_pretax_return=v.daily_return, spy_market_return=etf_return,
                          replica_twr_growth=twr, spy_growth=etf, account_nav=v.nav, cash=v.cash,
                          tax_cash_flow=v.tax_cash_flow, stocks=len(v.holdings)))
        if v.day in reports:
            year = reports[v.day]
            if not year.tax_amount_known or year.after_tax is None:
                raise ValueError(f"{v.day}: account tax or tax basis is incomplete.")
            annual.append(dict(year=v.day.year, as_of=v.day, replica_pretax_return=twr / start_twr - 1,
                               spy_market_return=etf / start_etf - 1, net_tax_usd=year.tax.net_tax,
                               cumulative_tax_usd=year.tax.cumulative_tax, nav_before_current_tax_usd=v.nav,
                               wealth_after_tax_usd=year.after_tax, tax_final=year.tax.final,
                               tax_review_through=year.tax.review_end,
                               unverified_ratios="|".join(year.unverified_split_ratios)))
            start_twr, start_etf = twr, etf
    end = result.history[-1]
    years = (end.day - start_day).days / 365.2425
    final = reports[end.day]
    terminal_tax_due = end.nav - final.after_tax
    if terminal_tax_due > end.cash:
        raise ValueError(f"{end.day}: insufficient cash to settle the closing tax without selling shares.")
    ret_account = float(twr) ** (1 / years) - 1
    ret_spy = float(etf) ** (1 / years) - 1
    summary = dict(first_trade=str(start_day), end=str(end.day), sessions=len(daily),
                   rebalances=sum(target.rebalance for target in result.targets.values()), initial_cash_usd=float(INITIAL_CASH),
                   pretax_replica_return=float(twr - 1), spy_market_total_return=float(etf - 1),
                   cumulative_gap_percentage_points=float((twr - etf) * 100),
                   replica_cagr=ret_account, spy_cagr=ret_spy, cagr_gap_bp=(ret_account - ret_spy) * 10000,
                   annualized_daily_tracking_error_bp=stdev(active) * math.sqrt(252) * 10000,
                   end_stock_count=len(end.holdings), end_cash_before_current_tax_usd=float(end.cash),
                   end_nav_before_current_tax_usd=float(end.nav), terminal_year_tax_usd=float(final.tax.net_tax),
                   end_cash_after_tax_settlement_usd=float(end.cash - terminal_tax_due),
                   cumulative_tax_usd=float(final.tax.cumulative_tax), end_after_tax_usd=float(final.after_tax),
                   after_tax_return=float(final.after_tax / INITIAL_CASH - 1),
                   terminal_tax_final=final.tax.final, tax_review_through=str(final.tax.review_end),
                   tax_scenario=result.tax_scenario.value, execution_exceptions=len(result.exceptions),
                   pre_delisting_sales=sum(e.shares > 0 for e in result.exits),
                   average_end_of_day_cash_weight=float(sum(v.cash / v.nav for v in result.history) / len(daily)),
                   total_commission_usd=float(sum(e.fee for e in result.ledger.entries)),
                   ending_unrealized_gain_usd=float(end.nav - end.cash - end.total_basis),
                   held_corporate_action_orders=len(held_orders),
                   income_tax_cash_paid_before_final_usd=float(-sum(v.tax_cash_flow for v in result.history)),
                   provisional_corporate_actions=list(final.unverified_split_ratios))
    if lot_method is not None:
        summary.update(lot_selection=lot_method.upper(), reference_lot_selection="HIFO", active_tlh=False)
    if nav_returns is not None:
        growth = last_year_nav = Decimal(1)
        nav_active, fund_returns, stock_returns = [], [], []
        annual_by_day = {r["as_of"]: r for r in annual}
        for row in daily:
            d = row["date"]
            r = Decimal(0) if d == start_day else nav_returns[d]
            growth *= 1 + r
            row.update(spy_nav_return=r, spy_nav_growth=growth)
            nav_active.append(float(row["replica_pretax_return"] - r))
            fund_returns.append(float(row["spy_market_return"]))
            stock_returns.append(float(row["replica_pretax_return"]))
            if d in annual_by_day:
                annual_by_day[d]["spy_nav_return"] = growth / last_year_nav - 1
                last_year_nav = growth
        summary.update(spy_nav_total_return=float(growth - 1), spy_nav_cagr=float(growth) ** (1/years) - 1,
                       annualized_nav_tracking_error_bp=stdev(nav_active) * math.sqrt(252) * 10000,
                       daily_return_correlation=correlation(stock_returns, fund_returns))
    output.mkdir(exist_ok=True, parents=True)
    write_table(output / "daily.csv", daily[0], daily)
    write_table(output / "annual.csv", annual[0], annual)
    closing = dict(date=end.day, stocks_held=len(end.holdings), cash_before_usd=end.cash,
                   terminal_tax_payment_usd=terminal_tax_due, cash_after_usd=end.cash - terminal_tax_due,
                   stock_value_usd=end.nav - end.cash, wealth_after_tax_usd=final.after_tax,
                   tax_final=final.tax.final)
    write_table(output / "terminal_settlement.csv", closing, [closing])
    order_fields = ("trade_date", "permno", "previous_shares", "shares_without_restriction", "raw_equity_target")
    write_table(output / "corporate_action_orders.csv", order_fields,
                (dict(zip(order_fields, r)) for r in held_orders))
    position_fields = ("permno", "shares", "closing_value_usd", "unrealized_gain_usd")
    by_name = result.ledger.trading_lots()
    basis = {}
    for lots in by_name.values():
        for lot in lots:
            basis[lot.permno] = basis.get(lot.permno, Decimal(0)) + lot.basis
    write_table(output / "ending_positions.csv", position_fields,
                (dict(zip(position_fields, (p, q, end.market_values[p], end.market_values[p] - basis[p])))
                 for p, q in sorted(end.holdings.items())))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_benchmark(holdings: Path, opening_prices: Path, output: Path, start=START, end=END, *, lot_method="hifo"):
    selector = LOT_METHODS[lot_method]
    calendar = (PRIOR, *nyse_trading_days())
    signals = monthly_signals(calendar, start, end)
    weights = read_monthly_holdings(holdings, signals)
    if end not in calendar:
        raise ValueError(f"{end}: choose a trading session for the final valuation.")
    following = next(d for d in calendar if d > end and (d.year, d.month) != (end.year, end.month))
    first_signal = signals[start]
    tradable = {d: set() for d in calendar if first_signal <= d <= end}
    actions = benchmark_actions(CRSP)
    market_file = CRSP / "daily_2015_to_2026_09.csv.gz"
    paths = [market_file, CRSP / "spy_temporary_holdings_2017.csv"]
    if first_signal == PRIOR:
        paths.insert(0, opening_prices)

    def quotes():
        for row in chain.from_iterable(read_crsp_daily(p, require_prices=True) for p in paths):
            if row.day in tradable:
                tradable[row.day].add(row.permno)
                yield row

    days = sessions_from_crsp(quotes(), tradable=tradable, actions_by_day={d: actions.get(d, []) for d in tradable})
    missing = weights[first_signal].keys() - days[first_signal].raw_closes.keys()
    if missing:
        raise ValueError(f"{first_signal}: missing opening signal closes for PERMNOs {sorted(missing)}.")
    if any(not m.raw_closes for m in days.values()):
        raise ValueError("One or more trading sessions have no CRSP coverage.")
    freeze = {}
    for trade in signals:
        restrictions = {}
        for event in actions.get(trade, ()):
            if isinstance(event, SpinOff) or (isinstance(event, PropertyDividend) and not event.cash_at_fmv):
                restrictions[event.spin_off_child] = event.permno
            elif isinstance(event, CorporateAction) and event.kind == "NONTAXABLE_STOCK_DIVIDEND":
                restrictions[event.permno] = None
        freeze[trade] = restrictions
    sessions = [d for d in calendar if first_signal <= d <= following]
    delistings = read_delistings(CRSP / "delistings_2015_to_2026_09.csv")

    def simulate(policy):
        strategy = SpyReplication(signals, weights, freeze, lot_relief=policy)
        account = run(sessions=sessions, first_trade=start, end=end, days=days, delistings=delistings,
                      delistings_known_through=date(2026, 9, 30), rebalance_days=signals,
                      next_rebalance=following, strategy=strategy, lot_relief=policy)
        return account, strategy.held_orders

    result, held_orders = simulate(selector)
    baseline = result if lot_method == "hifo" else simulate(hifo)[0]
    spy = read_etf_returns()
    nav_returns = read_etf_returns("nav_total_return")
    summary = compare(result, spy, output, held_orders, nav_returns, lot_method=lot_method)
    write_signal_weights(output, signals, weights)
    with crsp_table(MARKET / "sp500_total_return_usd_daily.csv") as rows:
        sp500 = {date.fromisoformat(r["date"]): Decimal(r["daily_total_return"]) for r in rows}
    name = "SPY Monthly Replication" if lot_method == "hifo" else "SPY Monthly Replication (LTFO vs. HIFO)"
    write_strategy_report(result, {"sp500_tr": sp500, "spy_adj_close": adjusted_close_returns(),
                                   "spy_market": spy, "spy_nav": nav_returns}, output,
                          initial_cash=INITIAL_CASH, no_tlh=baseline, account_name=name)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdings", type=Path, default=MARKET / "spy_month_end_holdings.csv.gz")
    parser.add_argument("--opening-prices", type=Path, default=CRSP / "daily_2014_12_31.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lot-relief", type=str.lower, choices=LOT_METHODS, default="hifo",
                        help="Lot selection for partial sales. LTFO is paired with an independent HIFO account.")
    parser.add_argument("--start", type=date.fromisoformat, default=START)
    parser.add_argument("--end", type=date.fromisoformat, default=END)
    args = parser.parse_args()
    suffix = "" if args.lot_relief == "hifo" else "_" + args.lot_relief
    output = args.output or ROOT / "outputs" / f"spy_replication{suffix}_{args.start.year}_{args.end.year}"
    print(json.dumps(run_benchmark(args.holdings, args.opening_prices, output, args.start, args.end,
                                   lot_method=args.lot_relief), indent=2))
