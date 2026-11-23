"""Approved fallback acceptance; preserve each candidate's true convergence status.

The numerical attempts and three start formulas match solver._solve. Only candidate
acceptance differs when none of the three candidates is converged and feasible.
Existing strict-solver source files and cache fingerprints remain unchanged.
"""

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config

from tlh.optimization.cache import fingerprint, read_entry, write_entry
from tlh.optimization.solver import Solution, _attempt, solve

FEASIBLE_POLICY = "prefer_converged_else_feasible"


def fallback_fingerprint(problem, max_iterations):
    digest = hashlib.sha256(FEASIBLE_POLICY.encode())
    digest.update(fingerprint(problem, max_iterations).encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def validate_selected_candidate(entry, *, allow_nonconverged=False):
    if entry["selected_start"] == "all_fixed":
        return
    selected = next((a for a in entry["attempts"] if a["start"] == entry["selected_start"]), None)
    if selected is None or not selected["feasible"]:
        raise ValueError("Cached selected candidate is not feasible.")
    if not selected["success"]:
        if not allow_nonconverged:
            raise ValueError("Cached selected candidate did not converge.")
        if any(a["success"] and a["feasible"] for a in entry["attempts"]):
            raise ValueError("Cached fallback ignored a converged feasible candidate.")


def solve_feasible_fallback(problem, *, max_iterations=400, workers=1, cache_dir=None):
    if workers not in (1, 3):
        raise ValueError("Use one sequential worker or three independent start workers.")
    # A prior strict success is already accepted and can be reused immediately.
    strict_key = fingerprint(problem, max_iterations) if cache_dir is not None else None
    if cache_dir is not None and read_entry(cache_dir, strict_key) is not None:
        return solve(problem, max_iterations=max_iterations, workers=workers, cache_dir=cache_dir)
    key = fallback_fingerprint(problem, max_iterations) if cache_dir is not None else None
    entry = read_entry(cache_dir, key) if cache_dir is not None else None
    if entry is not None:
        weights = np.asarray(entry["weights"], dtype=float)
        problem.check(weights, tolerance=1e-7)
        validate_selected_candidate(entry, allow_nonconverged=True)
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
        if workers == 1:
            results = [_attempt(problem, name, weights, free, template, budget, max_iterations)
                       for name, weights in starts]
        else:
            with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=1):
                results = Parallel()(delayed(_attempt)(problem, name, weights, free, template, budget, max_iterations)
                                     for name, weights in starts)
        feasible = [(a, w) for a, w in results if a["feasible"]]
        converged = [(a, w) for a, w in feasible if a["success"]]
        candidates = converged or feasible
        if not candidates:
            raise ValueError(f"No feasible continuous solution: {[a for a, _ in results]}")
        best = min(a["objective_nav"] for a, _ in candidates)
        tied = [(a, w) for a, w in candidates if a["objective_nav"] <= best + 1e-12]
        selected, weights = min(tied, key=lambda pair: problem.variance(pair[1]))
        problem.check(weights, tolerance=1e-7)
        solution = Solution(weights, tuple(a for a, _ in results), selected["start"])
    if cache_dir is not None:
        solution = replace(solution, cache_key=key)
        write_entry(cache_dir, key, solution)
    return solution
