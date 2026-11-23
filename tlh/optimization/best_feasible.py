"""Fixed-risk experiment: select by the original objective, without requiring convergence.

The three starts, SLSQP equations and numerical settings match the original solver.
Optional observation of initial/iteration points never changes the optimizer's path.
This module has its own cache namespace and does not reuse convergence-only results.
"""

import hashlib
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from tlh.optimization.cache import fingerprint, read_entry, write_entry
from tlh.optimization.solver import Solution, _attempt

BEST_FEASIBLE_POLICY = "best_feasible_original_objective"


def best_feasible_fingerprint(problem, max_iterations, retain_feasible_iterates):
    digest = hashlib.sha256(BEST_FEASIBLE_POLICY.encode())
    digest.update(str(bool(retain_feasible_iterates)).encode())
    digest.update(fingerprint(problem, max_iterations).encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _best(candidates, problem):
    best_value = min(a["objective_nav"] for a, _ in candidates)
    tied = [(a, w) for a, w in candidates if a["objective_nav"] <= best_value + 1e-12]
    return min(tied, key=lambda pair: problem.variance(pair[1]))


def _observed_attempt(problem, name, weights, free, template, budget, max_iterations):
    """Retain feasible points without modifying x, constraints, or stopping criteria."""
    candidates = []
    iteration = 0

    def expand(x):
        proposed = template.copy()
        proposed[free] = x / 100
        return proposed

    def objective(x):
        value, gradient = problem.objective(expand(x))
        return value * 10000, gradient[free] * 100

    def observe(x, source, step):
        proposed = expand(x)
        try:
            problem.check(proposed, tolerance=1e-7)
        except ValueError:
            return
        value = problem.objective(proposed)[0]
        if not np.isfinite(value):
            raise ValueError("A feasible candidate has a non-finite objective.")
        candidate = dict(start=f"{name}:{source}:{step}", parent_start=name,
                         candidate_source=source, optimizer_endpoint=False,
                         success=False, status=None, iterations=step,
                         function_evaluations=0, elapsed_seconds=0., objective_nav=float(value),
                         continuous_te=float(np.sqrt(max(0., problem.variance(proposed)))),
                         feasible=True, message="Retained feasible point; convergence not certified.",
                         constraint_error=None)
        candidates.append((candidate, proposed))

    def callback(x):
        nonlocal iteration
        iteration += 1
        observe(x, "iterate", iteration)

    constraints = [
        {"type": "eq", "fun": lambda x: x.sum() - budget * 100,
         "jac": lambda x: np.ones(len(x))},
        {"type": "ineq", "fun": lambda x: 1 - problem.variance(expand(x)) / problem.te_limit ** 2,
         "jac": lambda x: -problem.variance_gradient(expand(x))[free] / (100 * problem.te_limit ** 2)},
    ]
    with threadpool_limits(limits=1):
        started = perf_counter()
        observe(weights * 100, "initial", 0)
        result = minimize(objective, weights * 100, jac=True, method="SLSQP",
                          bounds=[(0., budget * 100)] * len(free), constraints=constraints,
                          callback=callback, options={"ftol": 1e-9, "maxiter": max_iterations})
        proposed = expand(result.x)
        error = None
        try:
            problem.check(proposed, tolerance=1e-7)
        except ValueError as exc:
            error = str(exc)
        value = problem.objective(proposed)[0] if np.isfinite(proposed).all() else float("nan")
        endpoint = dict(start=name, parent_start=name, candidate_source="final", optimizer_endpoint=True,
                        success=bool(result.success), status=int(result.status),
                        iterations=int(result.nit), function_evaluations=int(result.get("nfev", 0)),
                        elapsed_seconds=perf_counter() - started, objective_nav=value,
                        continuous_te=float(np.sqrt(max(0., problem.variance(proposed)))),
                        feasible=error is None, message=str(result.message), constraint_error=error,
                        feasible_observations=len(candidates))
        # Keep the actual final status, even when a different feasible point is selected.
        results = [(endpoint, proposed)]
        if candidates:
            retained = _best(candidates, problem)
            # Identical final weights inherit the actual final status instead of being
            # duplicated as an uncertified callback point.
            if not np.array_equal(retained[1], proposed):
                results.append(retained)
        return results


def solve_best_feasible(problem, *, max_iterations=400, workers=1, cache_dir=None,
                        retain_feasible_iterates=False):
    if workers not in (1, 3):
        raise ValueError("Use one sequential worker or three independent start workers.")
    key = (best_feasible_fingerprint(problem, max_iterations, retain_feasible_iterates)
           if cache_dir is not None else None)
    entry = read_entry(cache_dir, key) if cache_dir is not None else None
    if entry is not None:
        weights = np.asarray(entry["weights"], dtype=float)
        problem.check(weights, tolerance=1e-7)
        if entry["selected_start"] != "all_fixed":
            selected = next((a for a in entry["attempts"] if a["start"] == entry["selected_start"]), None)
            if selected is None or not selected["feasible"]:
                raise ValueError("Best-feasible cache has no feasible selected candidate.")
            objective = problem.objective(weights)[0]
            if not np.isfinite(objective) or abs(objective - selected["objective_nav"]) > 1e-12:
                raise ValueError("Best-feasible cache objective does not match selected weights.")
            if any(a["feasible"] and a["objective_nav"] < objective - 1e-12 for a in entry["attempts"]):
                raise ValueError("Best-feasible cache ignored a better feasible candidate.")
        return Solution(weights, tuple(entry["attempts"]), entry["selected_start"], True, key)

    free = problem.free_indices
    template = np.zeros(len(problem.permnos))
    for i, weight in problem.fixed.items():
        template[i] = float(weight)
    budget = 1 - template.sum()
    if budget < -1e-12 or (not len(free) and abs(budget) > 1e-12):
        raise ValueError("Fixed positions leave no feasible fully invested portfolio.")
    if not len(free) or budget <= 0:
        problem.check(template)
        solution = Solution(template, (), "all_fixed")
    else:
        benchmark = problem.benchmark_vector[free]
        benchmark = budget * benchmark / benchmark.sum() if benchmark.sum() else np.full(len(free), budget / len(free))
        current = problem.costs.current_weights[free]
        current = (current + (budget - current.sum()) * benchmark / budget if current.sum() <= budget
                   else current * budget / current.sum())
        equal = np.full(len(free), budget / len(free))
        starts = [("benchmark", benchmark), ("current_plus_cash", current),
                  ("diversified", .9 * benchmark + .1 * equal)]
        attempt = _observed_attempt if retain_feasible_iterates else _attempt
        if workers == 1:
            outcomes = [attempt(problem, name, weights, free, template, budget, max_iterations)
                        for name, weights in starts]
        else:
            with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=1):
                outcomes = Parallel()(delayed(attempt)(problem, name, weights, free, template, budget, max_iterations)
                                      for name, weights in starts)
        results = ([pair for outcome in outcomes for pair in outcome]
                   if retain_feasible_iterates else outcomes)
        feasible = [(a, w) for a, w in results if a["feasible"]]
        if not feasible:
            raise ValueError(f"No feasible continuous solution at fixed TE: {[a for a, _ in results]}")
        if any(not np.isfinite(a["objective_nav"]) for a, _ in feasible):
            raise ValueError("A feasible candidate has a non-finite objective.")
        selected, weights = _best(feasible, problem)
        problem.check(weights, tolerance=1e-7)
        solution = Solution(weights, tuple(a for a, _ in results), selected["start"])
    if cache_dir is not None:
        solution = replace(solution, cache_key=key)
        write_entry(cache_dir, key, solution)
    return solution
