"""LTFO direction heuristic followed by a direction-constrained convex lot solve.

Both stages use the same net-of-sale-fee estimated tax. No hypothetical stage-one
trade touches the ledger. Future wash-sale matches remain outside the objective.
"""

from dataclasses import replace
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from functools import partial
from pathlib import Path
from time import perf_counter
import hashlib

import cvxpy as cp
import cloudpickle
import numpy as np
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

from tlh.execution import floor_shares, lock_orders
from tlh.lot_relief import take_shares
from tlh.optimization.best_feasible import solve_best_feasible
from tlh.optimization.problem import PortfolioProblem
from tlh.optimization.tax_cost import HifoTaxCurve, TradeCosts
from tlh.shares import sum_shares
from tlh.strategy import Target
from tlh.tax import LT_RATE, ST_RATE, is_long_term


def marginal_tax(state, lot, fee_rate):
    if state.trade_day is None or lot.basis is None:
        raise ValueError("LTFO requires known basis and a scheduled sale date.")
    price = state.closes[lot.permno]
    tax = sum(((price * (1 - fee_rate) * part.shares - part.basis)
               * (LT_RATE if is_long_term(part.holding_start, state.trade_day) else ST_RATE)
               for part in lot.segments or (lot,)), Decimal(0))
    return tax / lot.shares


def ordered_lots(state, permno, fee_rate):
    lots = state.eligible_lots(permno)
    if sum_shares(l.shares for l in lots) != sum_shares(
            l.shares for l in state.lots_by_permno.get(permno, ())):
        raise ValueError(f"{state.day} {permno}: unresolved tax lots.")
    return sorted(lots, key=lambda l: (marginal_tax(state, l, fee_rate),
                  min(p.acquired for p in l.segments) if l.segments else l.acquired, l.id))


def ltfo_with_fees(state, permno, shares_to_sell, *, fee_rate):
    return take_shares(ordered_lots(state, permno, fee_rate), shares_to_sell)


class LtfoCosts(TradeCosts):
    def __init__(self, state, permnos, *, fee_rate=Decimal("0.0005")):
        self.permnos = tuple(permnos)
        self.nav, self.fee_rate = float(state.investable_nav), float(fee_rate)
        self.prices = np.array([float(state.closes[p]) for p in permnos])
        self.shares = np.array([float(sum_shares(l.shares for l in state.lots_by_permno.get(p, ())))
                                for p in permnos])
        self.current_weights = self.shares * self.prices / self.nav
        self.curves, self.lots = [], []
        for p in permnos:
            lots = ordered_lots(state, p, fee_rate)
            self.lots.append(lots)
            quantities = np.array([float(l.shares) for l in lots])
            slopes = np.array([float(marginal_tax(state, l, fee_rate)) for l in lots])
            self.curves.append(HifoTaxCurve(tuple(l.id for l in lots), np.cumsum(quantities),
                                           slopes, np.cumsum(slopes * quantities)))


def loss_directions(problem, first_weights):
    """A loss is assessed on each whole trading lot's gross MV vs adjusted basis."""
    directions = {}
    for i, p in enumerate(problem.permnos):
        if any(l.shares * problem.state.closes[p] < l.basis for l in problem.costs.lots[i]):
            change = first_weights[i] - problem.costs.current_weights[i]
            # Do not arbitrarily turn a small feasible stage-one trade into a hold.
            directions[i] = "BUY" if change > 0 else "SELL" if change < 0 else "HOLD"
    return directions


def validate_optimality(info, original_objective, model_objective, nav):
    """Require numerical optimality evidence separately from the status label.

Original portfolio/trade feasibility is checked separately. The internal primal
residual is retained for diagnosis, not used to override those independent checks.
    """
    names = ('res_primal', 'res_dual', 'gap_abs', 'gap_rel', 'cost_primal', 'cost_dual')
    values = [float(getattr(info, key)) for key in names]
    if not np.isfinite(values).all() or any(v < 0 for v in values[:4]):
        raise ValueError('Stage two has invalid optimality diagnostics.')
    if info.res_dual > 1e-8 or (info.gap_abs > 1e-10 and info.gap_rel > 1e-10):
        raise ValueError('Stage two failed dual-residual or optimality-gap validation.')
    tolerance = max(.005, nav * 1e-8)
    primal_usd, dual_usd = info.cost_primal / 10000 * nav, info.cost_dual / 10000 * nav
    primal_error = abs(model_objective - primal_usd)
    dual_error = abs(original_objective - dual_usd)
    if max(primal_error, dual_error) > tolerance:
        raise ValueError('Stage two primal/dual objective does not reconcile in original dollars.')
    return dict(internal_primal_residual=float(info.res_primal), dual_residual=float(info.res_dual),
                absolute_gap=float(info.gap_abs), relative_gap=float(info.gap_rel),
                dual_objective_usd=float(dual_usd), model_vs_primal_error_usd=float(primal_error),
                original_vs_dual_error_usd=float(dual_error), dollar_tolerance=tolerance,
                dual_residual_limit=1e-8, absolute_gap_limit=1e-10, relative_gap_limit=1e-10)


def solve_lots(problem, first_weights, directions):
    """Use percentage-point weights and trades for numerical conditioning; validate in NAV units."""
    scale = 100.0
    n, costs = len(problem.permnos), problem.costs
    lot_rows, lot_ids, caps, coefficients = [], [], [], []
    for i, (lots, curve) in enumerate(zip(costs.lots, costs.curves)):
        for j, lot in enumerate(lots):
            lot_rows.append(i)
            lot_ids.append(lot.id)
            caps.append(float(lot.shares) * costs.prices[i] / costs.nav)
            coefficients.append(curve.marginal_tax[j] / costs.prices[i])
    caps, coefficients = np.asarray(caps) * scale, np.asarray(coefficients)
    w, buys = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    constraints = [cp.sum(w) == scale]
    sales = cp.Variable(len(caps), nonneg=True) if len(caps) else None
    if sales is not None:
        incidence = csc_matrix((np.ones(len(caps)), (lot_rows, np.arange(len(caps)))), shape=(n, len(caps)))
        sold = incidence @ sales
        constraints.append(sales <= caps)
        tax = coefficients @ sales
        turnover = cp.sum(buys) + cp.sum(sales)
    else:
        sold, tax, turnover = np.zeros(n), 0., cp.sum(buys)
    constraints.append(w == costs.current_weights * scale + buys - sold)
    for i, value in problem.fixed.items():
        constraints.append(w[i] == float(value) * scale)
    for i, direction in directions.items():
        if direction in ("SELL", "HOLD"):
            constraints.append(buys[i] == 0)
        if direction in ("BUY", "HOLD") and sales is not None:
            constraints.append(sold[i] == 0)
    if len(problem.risk_indices):
        # covariance = C C.T, hence variance = ||C.T active||^2.
        with threadpool_limits(limits=1):
            root = np.linalg.cholesky(problem.risk.covariance).T
        constraints.append(cp.norm(root @ (w - problem.benchmark_vector * scale)[problem.risk_indices]) <= problem.te_limit * scale)
    model = cp.Problem(cp.Minimize(10000 / scale * (problem.gamma * tax + costs.fee_rate * turnover)), constraints)
    if not model.is_dcp():
        raise ValueError("Stage two is not a convex DCP problem.")
    started = perf_counter()
    with threadpool_limits(limits=1):
        model.solve(solver="CLARABEL", max_iter=300, tol_gap_abs=1e-10,
                    tol_gap_rel=1e-10, tol_feas=1e-8, max_threads=1)
    if model.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise ValueError(f"Stage two rejected solver status: {model.status}")
    vector = np.asarray(w.value).ravel() / scale
    b = np.asarray(buys.value).ravel() / scale
    s = np.asarray(sales.value).ravel() / scale if sales is not None else np.empty(0)
    caps = caps / scale
    stock_sales = np.bincount(lot_rows, weights=s, minlength=n)
    problem.check(vector, tolerance=1e-7)
    if not all(np.isfinite(x).all() for x in (vector, b, s)):
        raise ValueError("Stage two returned nonfinite values.")
    nonnegative_error = max(0., -float(b.min()), -float(s.min()) if len(s) else 0.)
    balance_error = float(np.max(np.abs(vector - costs.current_weights - b + stock_sales)))
    cap_error = float(max(0., np.max(s - caps))) if len(caps) else 0.
    overlap = float(np.max(np.minimum(b, stock_sales)))
    direction_error = max([0.] + [max(b[i] if d in ("SELL", "HOLD") else 0.,
                                     stock_sales[i] if d in ("BUY", "HOLD") else 0.)
                                    for i, d in directions.items()])
    if max(balance_error, cap_error, overlap, direction_error, nonnegative_error) > 1e-9:
        raise ValueError(f"Stage two failed trade validation: {balance_error, cap_error, overlap, direction_error, nonnegative_error}")
    original_objective = problem.objective(vector)[0] * costs.nav
    lot_objective = float(model.value / 10000 * costs.nav)
    objective_tolerance = max(.005, costs.nav * 1e-8)
    if abs(lot_objective - original_objective) > objective_tolerance:
        raise ValueError(f"Stage two lot vs LTFO objective mismatch: {lot_objective} vs {original_objective}")
    first_objective = problem.objective(first_weights)[0] * costs.nav
    if original_objective > first_objective + objective_tolerance:
        raise ValueError("Stage two is worse than its feasible stage-one anchor.")
    info = model._solver_cache['CLARABEL'].get_info()
    optimality_checks = validate_optimality(info, original_objective, lot_objective, costs.nav)
    return vector, dict(status=model.status, solver="CLARABEL", iterations=model.solver_stats.num_iters,
                        accepted_optimal_inaccurate=model.status == cp.OPTIMAL_INACCURATE,
                        optimality_checks=optimality_checks,
                        wall_seconds=perf_counter() - started, objective_usd=original_objective,
                        model_objective_usd=lot_objective, stage_one_objective_usd=first_objective,
                        objective_improvement_usd=first_objective - original_objective,
                        max_balance_error_weight=balance_error, max_lot_cap_error_weight=cap_error,
                        max_simultaneous_buy_sell_weight=overlap, max_direction_error_weight=direction_error,
                        max_negative_trade_weight=nonnegative_error,
                        lot_sales=[dict(permno=problem.permnos[i], lot_id=lid, sale_usd=float(q * costs.nav),
                                        limit_usd=float(cap * costs.nav), tax_per_dollar=float(c))
                                   for i, lid, q, cap, c in zip(lot_rows, lot_ids, s, caps, coefficients)],
                        globally_optimal_conditional_on_directions=True)


def target_decimals(problem, vector, directions):
    """Normalize only floating point noise, preserving directional boundaries."""
    unit = Decimal("1e-18")
    weights = {p: max(Decimal(0), Decimal(str(float(v))).quantize(unit, rounding=ROUND_DOWN))
               for p, v in zip(problem.permnos, vector)}
    lower, upper = {}, {}
    for i, value in problem.fixed.items():
        p = problem.permnos[i]
        weights[p] = lower[p] = upper[p] = value
    for i, d in directions.items():
        p = problem.permnos[i]
        held = sum_shares(l.shares for l in problem.costs.lots[i])
        current = held * problem.state.closes[p] / problem.state.investable_nav
        if d in ("BUY", "HOLD"):
            lower[p] = current.quantize(unit, rounding=ROUND_UP)
            weights[p] = max(weights[p], lower[p])
        if d == "SELL":
            upper[p] = current
            weights[p] = min(weights[p], upper[p])
        if d == "HOLD":
            weights[p] = lower[p]
            upper[p] = lower[p]
    residual = 1 - sum(weights.values())
    for p in sorted(weights, key=weights.get, reverse=True):
        if lower.get(p, Decimal(0)) <= weights[p] + residual <= upper.get(p, Decimal(1)):
            weights[p] += residual
            break
    if sum(weights.values()) != 1:
        raise ValueError("Cannot normalize decimal weights within directional constraints.")
    result = np.array([float(weights[p]) for p in problem.permnos])
    problem.check(result, tolerance=1e-7)
    if np.max(np.abs(result - vector)) > 1e-8:
        raise ValueError("Decimal normalization would materially change the solution.")
    return weights


class TwoStageLtfo:
    def __init__(self, signals, benchmarks, risk_model, *, gamma=.5, te_limit=.01,
                 fee_rate=Decimal("0.0005"), max_iterations=400, solver_workers=3,
                 solver_cache=None, on_decision=None, **legacy_options):
        if any(v for v in legacy_options.values() if not callable(v)):
            raise ValueError("Two-stage experiment does not support legacy fallbacks/replays.")
        self.signals, self.benchmarks, self.risk_model = signals, benchmarks, risk_model
        self.gamma, self.te_limit, self.fee_rate = gamma, te_limit, fee_rate
        self.max_iterations, self.solver_workers = max_iterations, solver_workers
        self.on_decision, self.decisions = on_decision, []
        # Includes LTFO and direction-model source; never reuse HIFO cache entries.
        namespace = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:20]
        self.solver_cache = Path(solver_cache) / namespace if solver_cache else None

    def __call__(self, state):
        if state.trade_day is None or self.signals.get(state.trade_day) != state.day:
            raise ValueError("Unexpected signal/trade date.")
        problem = PortfolioProblem.build(state, self.benchmarks[state.day], self.risk_model,
                                         gamma=self.gamma, te_limit=self.te_limit, fee_rate=self.fee_rate)
        problem = replace(problem, costs=LtfoCosts(state, problem.permnos, fee_rate=self.fee_rate))
        print(f"{state.trade_day}: stage 1 LTFO starting (3 starts, cap {self.max_iterations})", flush=True)
        started = perf_counter()
        first = solve_best_feasible(problem, max_iterations=self.max_iterations, workers=self.solver_workers,
                                    cache_dir=self.solver_cache, retain_feasible_iterates=True)
        stage_one_seconds = perf_counter() - started
        directions = loss_directions(problem, first.weights)
        if self.solver_cache:
            audit = self.solver_cache.parent / "stage_two_inputs"
            audit.mkdir(parents=True, exist_ok=True)
            with (audit / f"{state.trade_day}.pkl").open("wb") as stream:
                cloudpickle.dump((problem, first.weights, directions), stream)
        print(f"{state.trade_day}: stage 1 accepted {first.selected_start}; stage 2 Clarabel starting", flush=True)
        vector, second = solve_lots(problem, first.weights, directions)
        weights = target_decimals(problem, vector, directions)
        vector = np.array([float(weights[p]) for p in problem.permnos])
        tax, fees, _, _ = problem.costs.evaluate(vector)
        execution = dict(weights)
        overrides = {}
        for p in sorted(state.blocked):
            if execution.get(p, 0):
                overrides[str(p)] = dict(continuous_weight=str(weights[p]), execution_weight="0")
                execution[p] = Decimal(0)
        relief = partial(ltfo_with_fees, fee_rate=self.fee_rate)
        orders = lock_orders(state, execution, floor_shares, relief, set())
        # An actual whole-share order may never reverse its chosen direction.
        for order in orders:
            d = directions.get(problem.permnos.index(order.permno))
            if order.permno not in state.blocked and (
                    d == "HOLD" or (d == "BUY" and order.quantity < 0) or (d == "SELL" and order.quantity > 0)):
                raise ValueError(f"{state.day} {order.permno}: integer order violates {d} direction.")
        converged = first.selected_start == "all_fixed" or any(
            a["start"] == first.selected_start and a["success"] for a in first.attempts)
        record = dict(signal_date=str(state.day), trade_date=str(state.trade_day),
                      window_start=str(problem.risk.start), window_end=str(problem.risk.end),
                      risk_stocks=len(problem.risk.permnos), shrinkage=problem.risk.shrinkage,
                      incomplete_history={str(p): problem.risk.observations[p] for p in problem.risk.incomplete},
                      pre_delisting_blocked=sorted(state.blocked), forced_exits=list(problem.forced_exits),
                      available_cash_usd=float(state.available_cash), investable_nav_usd=problem.costs.nav,
                      estimated_tax_usd=tax, estimated_fees_usd=fees, objective_usd=self.gamma * tax + fees,
                      continuous_te=float(np.sqrt(max(0., problem.variance(vector)))),
                      annual_te_limit=self.te_limit, base_annual_te_limit=self.te_limit,
                      pre_delisting_execution_overrides=overrides, execution_target_weight_sum=str(sum(execution.values())),
                      selected_start=first.selected_start, selected_converged=converged,
                      accepted_without_convergence=not converged, attempts=list(first.attempts),
                      solver_result_source="stage1_cache+stage2_computed" if first.from_cache else "computed",
                      solver_cache_key=first.cache_key, solver_workers=self.solver_workers,
                      solve_wall_seconds=perf_counter() - started, stage_one_wall_seconds=stage_one_seconds,
                      solver_max_iterations_used=self.max_iterations, solver_acceptance_policy="best_feasible_initial_iterate_endpoint",
                      stage_one_weights={str(p): float(w) for p, w in zip(problem.permnos, first.weights)},
                      current_weights={str(p): float(w) for p, w in zip(problem.permnos, problem.costs.current_weights)},
                      loss_stock_directions={str(problem.permnos[i]): d for i, d in directions.items()},
                      stage_two=second, global_optimality_guaranteed=False,
                      integer_orders=[dict(permno=o.permno, quantity=str(o.quantity),
                                           lots={str(l): str(q) for l, q in o.lots.items()}) for o in orders])
        self.decisions.append(record)
        if self.on_decision:
            self.on_decision(record, weights)
        return Target(execution, lot_relief=relief)
