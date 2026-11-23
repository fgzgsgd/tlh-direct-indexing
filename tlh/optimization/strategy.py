"""Monthly tax-aware HIFO targets through the existing Target / account interfaces."""

from dataclasses import replace
from decimal import Decimal, ROUND_DOWN
from time import perf_counter

import numpy as np

from tlh.lot_relief import hifo
from tlh.optimization.problem import PortfolioProblem
from tlh.optimization.solver import solve
from tlh.optimization.replay import load_cached_decision
from tlh.optimization.feasible_fallback import FEASIBLE_POLICY, solve_feasible_fallback
from tlh.strategy import CloseSnapshot, Target


class TaxAwareHifo:
    def __init__(self, signals, benchmarks, risk_model, *, gamma=.5, te_limit=.01,
                 fee_rate=Decimal("0.0005"), max_iterations=400, solver_workers=1,
                 solver_cache=None, on_decision=None,
                 retry_max_iterations=None, on_solver_event=None, final_retry_max_iterations=None,
                 fallback_te_limit=None, risk_fallback_start=None, replay_decisions=None,
                 accept_feasible_fallback=False):
        if retry_max_iterations is not None and retry_max_iterations <= max_iterations:
            raise ValueError("Retry iteration limit must exceed the initial limit.")
        if final_retry_max_iterations is not None and (
                retry_max_iterations is None or final_retry_max_iterations <= retry_max_iterations):
            raise ValueError("Final retry requires a first retry and a strictly larger iteration limit.")
        if fallback_te_limit is not None:
            if not np.isfinite(fallback_te_limit) or fallback_te_limit <= te_limit:
                raise ValueError("Fallback TE must be finite and larger than the base TE.")
            if retry_max_iterations is not None or final_retry_max_iterations is not None:
                raise ValueError("Risk fallback cannot be combined with iteration escalation.")
        elif risk_fallback_start is not None:
            raise ValueError("A risk fallback start requires a fallback TE limit.")
        if accept_feasible_fallback and fallback_te_limit is None:
            raise ValueError("Nonconverged feasible acceptance requires a TE fallback stage.")
        self.accept_feasible_fallback = accept_feasible_fallback
        self.fallback_te_limit = fallback_te_limit
        self.risk_fallback_start = risk_fallback_start
        self.replay_decisions = dict(replay_decisions or {})
        self.signals = signals
        self.benchmarks = benchmarks
        self.risk_model = risk_model
        self.gamma, self.te_limit, self.fee_rate = gamma, te_limit, fee_rate
        self.max_iterations = max_iterations
        self.retry_max_iterations = retry_max_iterations
        self.final_retry_max_iterations = final_retry_max_iterations
        self.solver_workers = solver_workers
        self.solver_cache = solver_cache
        self.on_decision = on_decision
        self.on_solver_event = on_solver_event
        self.decisions = []

    def _solve_iterations(self, problem, state, started):
        used_iterations, retry, final_retry = self.max_iterations, None, None
        try:
            solution = solve(problem, max_iterations=used_iterations, workers=self.solver_workers,
                             cache_dir=self.solver_cache)
        except ValueError as exc:
            # Only the existing solver's exhausted candidate set authorizes a retry.
            # Input, cache, accounting and other feasibility errors still stop immediately.
            if (self.retry_max_iterations is None
                    or not str(exc).startswith("No converged, feasible continuous solution: ")):
                raise
            retry = dict(initial_max_iterations=self.max_iterations,
                         initial_error=str(exc), initial_wall_seconds=perf_counter() - started,
                         retry_max_iterations=self.retry_max_iterations)
            if self.on_solver_event:
                self.on_solver_event(dict(event="retry_started", signal_date=str(state.day),
                                          trade_date=str(state.trade_day), **retry))
            used_iterations = self.retry_max_iterations
            # Each optional stage restarts the same deterministic starts. A successful
            # earlier stage is final; larger caps never reconsider its selected result.
            retry_started = perf_counter()
            try:
                solution = solve(problem, max_iterations=used_iterations, workers=self.solver_workers,
                                 cache_dir=self.solver_cache)
            except ValueError as retry_exc:
                if (self.final_retry_max_iterations is None
                        or not str(retry_exc).startswith("No converged, feasible continuous solution: ")):
                    raise
                final_retry = dict(initial_max_iterations=used_iterations,
                                   initial_error=str(retry_exc),
                                   initial_wall_seconds=perf_counter() - retry_started,
                                   retry_max_iterations=self.final_retry_max_iterations)
                if self.on_solver_event:
                    self.on_solver_event(dict(event="final_retry_started", signal_date=str(state.day),
                                              trade_date=str(state.trade_day), **final_retry))
                used_iterations = self.final_retry_max_iterations
                # Exhaustion or unrelated errors here propagate; no further fallback.
                solution = solve(problem, max_iterations=used_iterations, workers=self.solver_workers,
                                 cache_dir=self.solver_cache)
        return solution, used_iterations, retry, final_retry

    def __call__(self, state: CloseSnapshot) -> Target:
        if state.trade_day is None or self.signals.get(state.trade_day) != state.day:
            raise ValueError(f"{state.trade_day}: expected a scheduled previous-month close.")
        replay = self.replay_decisions.get(state.trade_day)
        if self.replay_decisions and state.trade_day <= max(self.replay_decisions) and replay is None:
            raise ValueError(f"Missing historical replay decision: {state.trade_day}")
        effective_te = replay.get("annual_te_limit", self.te_limit) if replay is not None else self.te_limit
        problem = PortfolioProblem.build(state, self.benchmarks[state.day], self.risk_model,
                                         gamma=self.gamma, te_limit=effective_te,
                                         fee_rate=self.fee_rate)
        started = perf_counter()
        retry, final_retry, risk_fallback = None, None, None
        acceptance_policy = "converged_only"
        if replay is not None:
            solution, used_iterations = load_cached_decision(problem, replay, self.solver_cache)
            retry, final_retry = replay.get("solver_retry"), replay.get("solver_final_retry")
            risk_fallback = replay.get("solver_risk_fallback")
            acceptance_policy = replay.get("solver_acceptance_policy", "converged_only")
        elif self.fallback_te_limit is not None and (
                self.risk_fallback_start is None or state.trade_day >= self.risk_fallback_start):
            used_iterations = self.max_iterations
            try:
                solution = solve(problem, max_iterations=used_iterations, workers=self.solver_workers,
                                 cache_dir=self.solver_cache)
            except ValueError as exc:
                if not str(exc).startswith("No converged, feasible continuous solution: "):
                    raise
                risk_fallback = dict(initial_te_limit=problem.te_limit,
                                     fallback_te_limit=self.fallback_te_limit,
                                     max_iterations=used_iterations, initial_error=str(exc),
                                     initial_wall_seconds=perf_counter() - started)
                if self.on_solver_event:
                    self.on_solver_event(dict(event="risk_fallback_started", signal_date=str(state.day),
                                              trade_date=str(state.trade_day), **risk_fallback))
                problem = replace(problem, te_limit=self.fallback_te_limit)
                # The next month starts again at self.te_limit; never escalate iterations.
                fallback_solver = solve_feasible_fallback if self.accept_feasible_fallback else solve
                acceptance_policy = FEASIBLE_POLICY if self.accept_feasible_fallback else "converged_only"
                solution = fallback_solver(problem, max_iterations=used_iterations, workers=self.solver_workers,
                                           cache_dir=self.solver_cache)
        else:
            solution, used_iterations, retry, final_retry = self._solve_iterations(problem, state, started)
        solve_seconds = perf_counter() - started
        selected_converged = (solution.selected_start == "all_fixed" or any(
            a["start"] == solution.selected_start and a["success"] for a in solution.attempts))
        if not selected_converged and not (
                acceptance_policy == FEASIBLE_POLICY and risk_fallback is not None
                and problem.te_limit == self.fallback_te_limit):
            raise ValueError("Nonconverged candidate is not authorized outside the TE fallback stage.")
        weights = decimal_weights(problem, solution.weights)
        vector = np.array([float(weights[p]) for p in problem.permnos])
        problem.check(vector, tolerance=1e-7)
        tax, fees, _, _ = problem.costs.evaluate(vector)
        execution_weights = dict(weights)
        overrides = {}
        for p in sorted(state.blocked):
            if execution_weights.get(p, 0):
                overrides[str(p)] = dict(continuous_weight=str(weights[p]), execution_weight="0")
                execution_weights[p] = Decimal(0)
        # The user gives exits priority after the continuous solve. Do not reinvest
        # the removed weight or claim its post-override risk satisfies the constraint.
        execution_vector = np.array([float(execution_weights[p]) for p in problem.permnos])
        execution_tax, execution_fees, _, _ = problem.costs.evaluate(execution_vector)
        forced = [problem.permnos.index(p) for p in problem.forced_exits]
        record = dict(signal_date=str(state.day), trade_date=str(state.trade_day),
                      window_start=str(problem.risk.start), window_end=str(problem.risk.end),
                      risk_stocks=len(problem.risk.permnos), shrinkage=problem.risk.shrinkage,
                      incomplete_history={str(p): problem.risk.observations[p] for p in problem.risk.incomplete},
                      pre_delisting_blocked=sorted(state.blocked),
                      forced_exits=list(problem.forced_exits),
                      forced_sale_gross_usd=float(sum(problem.costs.shares[i] * problem.costs.prices[i] for i in forced)),
                      available_cash_usd=float(state.available_cash), investable_nav_usd=problem.costs.nav,
                      estimated_tax_usd=tax, estimated_fees_usd=fees,
                      objective_usd=self.gamma * tax + fees,
                      pre_delisting_execution_overrides=overrides,
                      execution_target_weight_sum=str(sum(execution_weights.values())),
                      execution_estimated_tax_usd=execution_tax,
                      execution_estimated_fees_usd=execution_fees,
                      continuous_te=float(np.sqrt(max(0., problem.variance(vector)))),
                      selected_start=solution.selected_start, attempts=list(solution.attempts),
                      solver_workers=self.solver_workers, solve_wall_seconds=solve_seconds,
                      solver_max_iterations_used=used_iterations, solver_retry=retry,
                      solver_final_retry=final_retry,
                      base_annual_te_limit=self.te_limit, annual_te_limit=problem.te_limit,
                      solver_risk_fallback=risk_fallback, replayed_decision=replay is not None,
                      solver_history_source="historical_decision" if replay is not None else "current_run",
                      solver_acceptance_policy=acceptance_policy, selected_converged=selected_converged,
                      accepted_without_convergence=not selected_converged,
                      solver_result_source="cache" if solution.from_cache else "computed",
                      solver_cache_key=solution.cache_key,
                      global_optimality_guaranteed=False)
        self.decisions.append(record)
        if self.on_decision:
            self.on_decision(record, weights)
        # Keep zero targets explicit. The account locks integer shares and actual HIFO lots.
        return Target(execution_weights, lot_relief=hifo)


def decimal_weights(problem: PortfolioProblem, vector) -> dict[int, Decimal]:
    """Restore exact fixed weights and unit sum after floating-point numerical solving."""
    # Match the existing benchmark reader's 18-decimal weights so 500+ terms sum exactly.
    weights = {p: max(Decimal(0), Decimal(str(float(value))).quantize(Decimal("1e-18"), rounding=ROUND_DOWN))
               for p, value in zip(problem.permnos, vector)}
    for i, value in problem.fixed.items():
        weights[problem.permnos[i]] = value
    free = problem.free_indices
    if len(free):
        largest = max((problem.permnos[i] for i in free), key=weights.__getitem__)
        weights[largest] += 1 - sum(weights.values())
    if sum(weights.values()) != 1 or any(w < 0 for w in weights.values()):
        raise ValueError("Cannot convert continuous solution to valid decimal weights.")
    return weights
