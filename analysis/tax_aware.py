"""Run the isolated two-stage LTFO experiment with the existing account engine.

Example: python -m analysis.tax_aware --end 2016-12-30 --output outputs/tax_aware_trial
Cash shortages and corporate-action conflicts are errors, not automatic strategy overrides.
"""

import argparse
import csv
import hashlib
import json
from datetime import date
from decimal import Decimal
from functools import partial
from itertools import chain
from importlib.metadata import version
from pathlib import Path

from analysis.spy_replication_events import benchmark_actions
from analysis.strategy_report import write_strategy_report
from tlh.backtest import INITIAL_CASH, run, sessions_from_crsp
from tlh.execution import proportional_commission
from tlh.lot_relief import hifo
from tlh.market import crsp_table, nyse_trading_days, read_crsp_daily, read_delistings
from tlh.optimization.risk import RollingLedoitWolf
from tlh.optimization.two_stage_ltfo import TwoStageLtfo
from tlh.optimization.replay import read_replay_decisions
from tlh.optimization.tax_timing import ConfirmedTaxTimingEngine, read_tax_timing
from tlh.positions import write_weights
from tlh.spy_benchmark import monthly_signals, read_monthly_holdings


ROOT = Path(__file__).resolve().parents[1]
CRSP, MARKET = ROOT / "data/crsp", ROOT / "data/market"
START, END = date(2016, 1, 4), date(2025, 12, 31)


def run_experiment(output: Path, *, start=START, end=END, gamma=.5, te_limit=.01,
                   fee_rate=Decimal("0.0005"), max_iterations=400, solver_workers=1, solver_cache=None,
                   retry_max_iterations=None, final_retry_max_iterations=None,
                   fallback_te_limit=None, risk_fallback_start=None, replay_decisions=None,
                   accept_feasible_fallback=False, tax_timing_overrides=None):
    tax_timing = read_tax_timing(tax_timing_overrides)
    replay = read_replay_decisions(replay_decisions)
    # Never overwrite another experiment, including a previous failed run's diagnostics.
    output.mkdir(parents=True, exist_ok=False)
    config = dict(start=str(start), end=str(end), initial_cash_usd=str(INITIAL_CASH), gamma=gamma,
                  annual_te_limit=te_limit, fee_rate=str(fee_rate), risk_window=252,
                  return_field="DlyRet", lot_relief="LTFO net of sale fees", wash_sale_in_objective=False,
                  cash_reserve=False, integer_constraint_repair=False, solver="stage1 multistart SLSQP; stage2 CVXPY+Clarabel",
                  max_iterations=max_iterations, solver_workers=solver_workers, global_optimality_guaranteed=False,
                  retry_max_iterations=retry_max_iterations,
                  final_retry_max_iterations=final_retry_max_iterations,
                  fallback_te_limit=fallback_te_limit,
                  accept_feasible_fallback=accept_feasible_fallback,
                  fallback_acceptance=("prefer_converged_else_feasible" if accept_feasible_fallback else "converged_only"),
                  risk_fallback_start=str(risk_fallback_start) if risk_fallback_start is not None else None,
                  risk_fallback_policy=("Each new month starts at base TE; on no converged feasible candidate retry once at fallback TE with the same iteration cap; then stop on failure."
                                        if fallback_te_limit is not None else "disabled"),
                  replay_decisions_file=str(replay_decisions.resolve()) if replay_decisions is not None else None,
                  replay_decisions_sha256=hashlib.sha256(replay_decisions.read_bytes()).hexdigest() if replay_decisions is not None else None,
                  replay_decisions_count=len(replay),
                  replay_iteration_limits=sorted({r["solver_max_iterations_used"] for r in replay.values()}),
                  retry_policy=("up to two retries only when each prior stage has no converged feasible candidate; same starts and acceptance checks"
                                if final_retry_max_iterations is not None else
                                "one retry only when no converged feasible candidate; same starts and acceptance checks"
                                if retry_max_iterations is not None else "disabled"),
                  universe="PERMNOs in the signal-date SPY weight table",
                  tax_scenario="external_gains", pre_delisting=True,
                  short_history_pre_delisting="benchmark weight in continuous solve; zero execution target",
                  risk_constraint_stage="before execution overrides and integer rounding",
                  solver_cache=str(solver_cache.resolve()) if solver_cache is not None else None,
                  split_day_order_policy="Cancel all discretionary orders in a splitting security; process corporate actions and CIL normally.",
                  confirmed_tax_timing_overrides=[r.as_dict() for r in tax_timing],
                  packages={name: version(name) for name in ("numpy", "scipy", "scikit-learn", "cvxpy", "clarabel")})
    config.update(stage_two_acceptance="optimal or optimal_inaccurate; independent constraints, objectives, dual residual <= 1e-8 and absolute/relative gap <= 1e-10", stage_two_feasibility_tolerance=1e-8, stage_two_variable_scale=100, historical_prefix="Jan 2016-Dec 2020 exact validated replay; tolerances recorded per historical decision", stage_one_lot_relief="LTFO", stage_two_lot_relief="LTFO",
                  stage_one_convergence_required=False,
                  stage_one_candidates="initial, feasible major iterates, endpoints; original objective ranking",
                  stage_two_convergence_required=True, stage_two_solver="CVXPY + Clarabel",
                  stage_two_te_limit=te_limit, beta_constraint=False,
                  loss_definition="existing lot: shares * signal close < adjusted basis",
                  direction_rule="loss stocks only; sign(stage1 weight - current weight); BUY/SELL/HOLD",
                  execute_stage="stage2 only", stop_on_failure=True,
                  stage_two_global_optimality="conditional on stage1 directions only")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    try:
        return _simulate(output, start, end, gamma, te_limit, fee_rate, max_iterations, solver_workers,
                         solver_cache, retry_max_iterations, final_retry_max_iterations,
                         fallback_te_limit, risk_fallback_start, replay, accept_feasible_fallback, tax_timing)
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps(
            dict(status="failed", error_type=type(exc).__name__, message=str(exc)), indent=2) + "\n")
        raise


def _simulate(output, start, end, gamma, te_limit, fee_rate, max_iterations, solver_workers=1,
              solver_cache=None, retry_max_iterations=None, final_retry_max_iterations=None,
              fallback_te_limit=None, risk_fallback_start=None, replay_decisions=None, accept_feasible_fallback=False,
              tax_timing=()):
    calendar = nyse_trading_days()
    signals = monthly_signals(calendar, start, end)
    if end not in calendar:
        raise ValueError(f"{end}: final valuation must be a trading session.")
    benchmarks = read_monthly_holdings(MARKET / "spy_month_end_holdings.csv.gz", signals)
    following = next(d for d in calendar if d > end and (d.year, d.month) != (end.year, end.month))
    first_signal = signals[start]
    names = set().union(*(weights.keys() for weights in benchmarks.values()))
    risk = RollingLedoitWolf.from_crsp(CRSP / "daily_2015_to_2026_09.csv.gz", calendar, names)
    # A first-date estimate here catches insufficient calendar coverage before running any account.
    risk.estimate(first_signal, benchmarks[first_signal])
    actions = benchmark_actions(CRSP)
    tradable = {d: set() for d in calendar if first_signal <= d <= end}

    def quotes():
        paths = (CRSP / "daily_2015_to_2026_09.csv.gz", CRSP / "spy_temporary_holdings_2017.csv")
        for row in chain.from_iterable(read_crsp_daily(p, require_prices=True) for p in paths):
            if row.day in tradable:
                tradable[row.day].add(row.permno)
                yield row

    days = sessions_from_crsp(quotes(), tradable=tradable,
                              actions_by_day={d: actions.get(d, ()) for d in tradable})
    with (output / "benchmark_weights.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("signal_date", "permno", "benchmark_weight"))
        for day, weights in sorted(benchmarks.items()):
            writer.writerows((day, p, w) for p, w in sorted(weights.items()))

    with ((output / "decisions.jsonl").open("w") as diagnostics,
          (output / "solver_events.jsonl").open("w") as solver_events,
          (output / "continuous_targets.csv").open("w", newline="") as targets,
          (output / "execution_targets.csv").open("w", newline="") as execution_targets):
        writer = csv.writer(targets)
        writer.writerow(("signal_date", "trade_date", "permno", "continuous_weight"))
        execution_writer = csv.writer(execution_targets)
        execution_writer.writerow(("signal_date", "trade_date", "permno", "execution_weight"))

        def solver_event(event):
            solver_events.write(json.dumps(event) + "\n")
            solver_events.flush()
            if event["event"] == "risk_fallback_started":
                print(f"{event['trade_date']}: no converged feasible candidate at TE limit "
                      f"{event['initial_te_limit']:.2%}; retrying at {event['fallback_te_limit']:.2%} "
                      f"with the same {event['max_iterations']} iterations", flush=True)
            else:
                print(f"{event['trade_date']}: no converged feasible candidate at "
                      f"{event['initial_max_iterations']} iterations; retrying all starts with "
                      f"{event['retry_max_iterations']} iterations", flush=True)

        def record(decision, weights):
            diagnostics.write(json.dumps(decision) + "\n")
            diagnostics.flush()
            writer.writerows((decision["signal_date"], decision["trade_date"], p, w)
                             for p, w in sorted(weights.items()))
            targets.flush()
            overrides = decision["pre_delisting_execution_overrides"]
            execution_writer.writerows((decision["signal_date"], decision["trade_date"], p,
                                        overrides.get(str(p), {}).get("execution_weight", w))
                                       for p, w in sorted(weights.items()))
            execution_targets.flush()
            print(f"{decision['trade_date']}: continuous TE={decision['continuous_te']:.6%}, "
                  f"limit={decision['annual_te_limit']:.2%}, source={decision['solver_result_source']}, "
                  f"selected_converged={decision['selected_converged']}, "
                  f"estimated tax=${decision['estimated_tax_usd']:,.2f}, "
                  f"short-history fixed={len(decision['incomplete_history'])}", flush=True)

        strategy = TwoStageLtfo(signals, benchmarks, risk, gamma=gamma, te_limit=te_limit,
                                fee_rate=fee_rate, max_iterations=max_iterations,
                                solver_workers=solver_workers, solver_cache=solver_cache,
                                retry_max_iterations=retry_max_iterations, on_solver_event=solver_event,
                                final_retry_max_iterations=final_retry_max_iterations,
                                fallback_te_limit=fallback_te_limit, risk_fallback_start=risk_fallback_start,
                                replay_decisions=replay_decisions,
                                accept_feasible_fallback=accept_feasible_fallback,
                                on_decision=record)
        account = run(sessions=[d for d in calendar if first_signal <= d <= following],
                      first_trade=start, end=end, days=days,
                      delistings=read_delistings(CRSP / "delistings_2015_to_2026_09.csv"),
                      delistings_known_through=date(2026, 9, 30), rebalance_days=signals,
                      next_rebalance=following, strategy=strategy, lot_relief=hifo,
                      engine_factory=partial(ConfirmedTaxTimingEngine, tax_after_sales=tax_timing),
                      fee=proportional_commission(fee_rate))
    if tax_timing:
        timing_audit = []
        for rule in tax_timing:
            if not start <= rule.trade_date <= end:
                continue
            entries = [e for e in account.ledger.entries if e.day == rule.trade_date]
            positions = [i for i, e in enumerate(entries) if e.kind == "TAX_SETTLEMENT"]
            if len(positions) != 1:
                raise ValueError(f"{rule.trade_date}: expected exactly one confirmed tax payment.")
            tax_index, = positions
            payment = entries[tax_index]
            if (payment.cash_change != -rule.tax_usd or payment.reference != str(rule.tax_year)
                    or any(i > tax_index for i, e in enumerate(entries) if e.kind == "TRADE")
                    or any(i < tax_index for i, e in enumerate(entries) if e.kind == "BUY")):
                raise ValueError(f"{rule.trade_date}: confirmed tax ordering or amount did not reconcile.")
            mark = next(v for v in account.history if v.day == rule.trade_date)
            timing_audit.append(dict(**rule.as_dict(), original_sales_before_tax=True, original_buys_after_tax=True,
                                     closing_cash_usd=str(mark.cash), tax_cash_flow_usd=str(mark.tax_cash_flow)))
        (output / "tax_timing_audit.json").write_text(json.dumps(timing_audit, indent=2) + "\n")
    write_weights(output / "weights.csv", account.targets)
    (output / "split_order_cancellations.json").write_text(json.dumps([
        dict(date=str(day), permno=p, reason=reason) for day, p, reason in account.exceptions
        if reason.startswith("SKIPPED_SPLIT_")], indent=2) + "\n")
    with crsp_table(MARKET / "sp500_total_return_usd_daily.csv") as rows:
        sp500 = {date.fromisoformat(r["date"]): Decimal(r["daily_total_return"]) for r in rows}
    with crsp_table(MARKET / "spy_market_total_return_2015_2025.csv") as rows:
        spy = {date.fromisoformat(r["date"]): Decimal(r["market_total_return"])
               for r in rows if r["market_total_return"]}
    forecasts = {date.fromisoformat(r["trade_date"]): r["continuous_te"] for r in strategy.decisions}
    write_strategy_report(account, {"sp500_tr": sp500, "spy_market": spy}, output,
                          initial_cash=INITIAL_CASH, ex_ante_te=forecasts, ex_ante_benchmark="spy_market",
                          account_name="Two-stage LTFO (continuous-target risk)", workbook=False)
    summary = dict(status="completed", start=str(start), end=str(end),
                   rebalances=len(strategy.decisions), end_nav_usd=str(account.history[-1].nav),
                   end_cash_usd=str(account.history[-1].cash),
                   annual_te_limits_used=sorted({r["annual_te_limit"] for r in strategy.decisions}),
                   risk_fallback_months=[r["trade_date"] for r in strategy.decisions if r.get("solver_risk_fallback")],
                   replayed_decisions=sum(r.get("replayed_decision", False) for r in strategy.decisions),
                   nonconverged_accepted_months=[r["trade_date"] for r in strategy.decisions
                                                if r.get("accepted_without_convergence")],
                   risk_measure="continuous-target ex-ante TE before execution overrides and integer rounding; not realized TE")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=START)
    parser.add_argument("--end", type=date.fromisoformat, default=END)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten.")
    parser.add_argument("--gamma", type=float, default=.5)
    parser.add_argument("--te-limit", type=float, default=.01)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--retry-max-iterations", type=int,
                        help="Optional larger cap for one retry after no converged feasible candidate.")
    parser.add_argument("--final-retry-max-iterations", type=int,
                        help="Optional final larger cap, used only if the first retry also has no converged feasible candidate.")
    parser.add_argument("--fallback-te-limit", type=float,
                        help="Retry once with this higher TE limit instead of increasing the iteration cap.")
    parser.add_argument("--risk-fallback-start", type=date.fromisoformat,
                        help="First trade date using the monthly TE fallback rule.")
    parser.add_argument("--replay-decisions", type=Path,
                        help="Previously accepted decisions; jump directly to their exact-input successful cache entries.")
    parser.add_argument("--accept-feasible-fallback", action="store_true",
                        help="Only in the higher-TE stage, prefer converged feasible candidates; if none, accept the best feasible candidate with its nonconvergence flag intact.")
    parser.add_argument("--solver-workers", type=int, choices=(1, 3), default=1)
    parser.add_argument("--solver-cache", type=Path, help="Reuse successful solves with identical inputs and code.")
    parser.add_argument("--tax-timing-overrides", type=Path,
                        help="Explicit exact-date tax payments after original sales and before original buys.")
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.output, start=args.start, end=args.end, gamma=args.gamma,
                                   te_limit=args.te_limit, max_iterations=args.max_iterations,
                                   solver_workers=args.solver_workers, solver_cache=args.solver_cache,
                                   retry_max_iterations=args.retry_max_iterations,
                                   final_retry_max_iterations=args.final_retry_max_iterations,
                                   fallback_te_limit=args.fallback_te_limit, risk_fallback_start=args.risk_fallback_start,
                                   replay_decisions=args.replay_decisions,
                                   accept_feasible_fallback=args.accept_feasible_fallback,
                                   tax_timing_overrides=args.tax_timing_overrides), indent=2))
