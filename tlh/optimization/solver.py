"""Deterministic multistart SLSQP on the exact, piecewise HIFO objective.

This is a local numerical method, not a certificate of global optimality.
"""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from tlh.optimization.problem import PortfolioProblem
from tlh.optimization.cache import fingerprint, read_entry, write_entry


@dataclass(frozen=True)
class Solution:
    weights: np.ndarray
    attempts: tuple[dict, ...]
    selected_start: str
    from_cache: bool = False
    cache_key: str | None = None


def _attempt(problem, name, weights, free, template, budget, max_iterations):
    """One independent start; keep all numerical settings identical in either mode."""
    def expand(x):
        proposed = template.copy()
        proposed[free] = x / 100
        return proposed

    def objective(x):
        value, gradient = problem.objective(expand(x))
        return value * 10000, gradient[free] * 100

    constraints = [
        {"type": "eq", "fun": lambda x: x.sum() - budget * 100,
         "jac": lambda x: np.ones(len(x))},
        {"type": "ineq", "fun": lambda x: 1 - problem.variance(expand(x)) / problem.te_limit ** 2,
         "jac": lambda x: -problem.variance_gradient(expand(x))[free] / (100 * problem.te_limit ** 2)},
    ]
    with threadpool_limits(limits=1):
        started = perf_counter()
        result = minimize(objective, weights * 100, jac=True, method="SLSQP",
                          bounds=[(0., budget * 100)] * len(free), constraints=constraints,
                          options={"ftol": 1e-9, "maxiter": max_iterations})
        proposed = expand(result.x)
        error = None
        try:
            problem.check(proposed, tolerance=1e-7)
        except ValueError as exc:
            error = str(exc)
        value = problem.objective(proposed)[0] if np.isfinite(proposed).all() else float("nan")
        attempt = dict(start=name, success=bool(result.success), status=int(result.status),
                       iterations=int(result.nit), function_evaluations=int(result.get("nfev", 0)),
                       elapsed_seconds=perf_counter() - started, objective_nav=value,
                       continuous_te=float(np.sqrt(max(0., problem.variance(proposed)))),
                       feasible=error is None, message=str(result.message), constraint_error=error)
    return attempt, proposed


def _solve(problem: PortfolioProblem, *, max_iterations=400, workers=1) -> Solution:
    if workers not in (1, 3):
        raise ValueError("Use one sequential worker or three independent start workers.")
    free = problem.free_indices
    template = np.zeros(len(problem.permnos))
    for i, weight in problem.fixed.items():
        template[i] = float(weight)
    budget = 1 - template.sum()
    if budget < -1e-12 or (not len(free) and abs(budget) > 1e-12):
        raise ValueError("Fixed positions leave no feasible fully invested portfolio.")
    if not len(free) or budget <= 0:
        problem.check(template)
        return Solution(template, (), "all_fixed")

    benchmark = problem.benchmark_vector[free]
    benchmark = budget * benchmark / benchmark.sum() if benchmark.sum() else np.full(len(free), budget / len(free))
    current = problem.costs.current_weights[free]
    # Retain current proportions where possible and invest existing cash in benchmark proportions.
    current = (current + (budget - current.sum()) * benchmark / budget if current.sum() <= budget
               else current * budget / current.sum())
    equal = np.full(len(free), budget / len(free))
    starts = [("benchmark", benchmark), ("current_plus_cash", current),
              ("diversified", .9 * benchmark + .1 * equal)]
    if workers == 1:
        results = [_attempt(problem, name, weights, free, template, budget, max_iterations)
                   for name, weights in starts]
    else:
        # Process isolation avoids the Python GIL and shared solver state. BLAS stays
        # single-threaded inside each process, as in the original serial solver.
        with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=1):
            results = Parallel()(delayed(_attempt)(problem, name, weights, free, template, budget, max_iterations)
                                 for name, weights in starts)
    candidates, attempts = [], []
    # Parallel returns in submission order, preserving deterministic tie breaking.
    for attempt, proposed in results:
        attempts.append(attempt)
        if attempt["success"] and attempt["feasible"]:
            candidates.append((attempt["objective_nav"], attempt["start"], proposed))
    if not candidates:
        raise ValueError(f"No converged, feasible continuous solution: {attempts}")
    best_value = min(candidate[0] for candidate in candidates)
    # Ignore sub-microdollar floating noise for a $1m account; from cash all weights cost the same.
    tied = [candidate for candidate in candidates if candidate[0] <= best_value + 1e-12]
    _, name, weights = min(tied, key=lambda candidate: problem.variance(candidate[2]))
    return Solution(weights, tuple(attempts), name)


def solve(problem: PortfolioProblem, *, max_iterations=400, workers=1, cache_dir=None) -> Solution:
    if workers not in (1, 3):
        raise ValueError("Use one sequential worker or three independent start workers.")
    key = fingerprint(problem, max_iterations) if cache_dir is not None else None
    entry = read_entry(cache_dir, key) if cache_dir is not None else None
    if entry is not None:
        weights = np.asarray(entry["weights"], dtype=float)
        problem.check(weights, tolerance=1e-7)
        if entry["selected_start"] != "all_fixed" and not any(
                a["start"] == entry["selected_start"] and a["success"] and a["feasible"]
                for a in entry["attempts"]):
            raise ValueError("Solver cache does not contain a converged, feasible selected candidate.")
        return Solution(weights, tuple(entry["attempts"]), entry["selected_start"], True, key)
    solution = _solve(problem, max_iterations=max_iterations, workers=workers)
    if cache_dir is not None:
        solution = replace(solution, cache_key=key)
        write_entry(cache_dir, key, solution)
    return solution
