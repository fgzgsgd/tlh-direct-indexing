"""Fixed 1% risk, truthful nonconvergence, observed iterates, and cache isolation."""

import importlib.util
import unittest
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tlh.strategy import CloseSnapshot, LotView

READY = all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "sklearn"))
if READY:
    import numpy as np
    from scipy.optimize import OptimizeResult
    from tlh.optimization.best_feasible import (
        BEST_FEASIBLE_POLICY, _observed_attempt, best_feasible_fingerprint, solve_best_feasible,
    )
    from tlh.optimization.cache import fingerprint
    from tlh.optimization.problem import PortfolioProblem
    from tlh.optimization.risk import RollingLedoitWolf
    from tlh.optimization.solver import _attempt
    from tlh.optimization.two_stage_ltfo import TwoStageLtfo


@unittest.skipUnless(READY, "Install the optimizer dependencies.")
class BestFeasibleTests(unittest.TestCase):
    signal, trade = date(2015, 12, 31), date(2016, 1, 4)

    def inputs(self):
        days = tuple(self.signal - timedelta(days=i) for i in reversed(range(252)))
        amplitude = np.sqrt(.04 * 2 / 252)
        returns = np.tile(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) * amplitude, (63, 1))
        risk = RollingLedoitWolf(days, (1, 2), returns)
        old = date(2015, 10, 1)
        state = CloseSnapshot(self.signal, D(1000), D(2000), {1: D(100), 2: D(100)},
                              (LotView(1, 1, old, old, D(10), D(1200)),), frozenset(), self.trade)
        return state, {1: D('.5'), 2: D('.5')}, risk

    def problem(self):
        return PortfolioProblem.build(*self.inputs(), te_limit=.01)

    def fake_attempt(self, problem, name, *args):
        weights = np.array({'benchmark': [.49, .51], 'current_plus_cash': [.47, .53],
                            'diversified': [.5, .5]}[name])
        problem.check(weights, tolerance=1e-7)
        success = name == 'benchmark'
        return dict(start=name, success=success, feasible=True, status=0 if success else 9,
                    iterations=400, objective_nav=problem.objective(weights)[0],
                    continuous_te=float(np.sqrt(problem.variance(weights)))), weights

    def test_lower_objective_nonconverged_beats_converged_and_is_cached_separately(self):
        problem = self.problem()
        with TemporaryDirectory() as tmp:
            with patch('tlh.optimization.best_feasible._attempt', side_effect=self.fake_attempt):
                result = solve_best_feasible(problem, cache_dir=tmp)
            self.assertEqual(result.selected_start, 'current_plus_cash')
            self.assertFalse(next(a for a in result.attempts if a['start'] == result.selected_start)['success'])
            self.assertNotEqual(result.cache_key, fingerprint(problem, 400))
            self.assertNotEqual(result.cache_key, best_feasible_fingerprint(problem, 400, True))
            self.assertNotEqual(result.cache_key, best_feasible_fingerprint(replace(problem, te_limit=.05), 400, False))
            with patch('tlh.optimization.best_feasible._attempt', side_effect=AssertionError('cache expected')):
                cached = solve_best_feasible(problem, cache_dir=tmp)
            self.assertTrue(cached.from_cache)
            np.testing.assert_array_equal(cached.weights, result.weights)

    def test_feasible_iteration_survives_infeasible_endpoint_without_changing_risk_limit(self):
        def failed_attempt(fun, x0, *, callback, **kwargs):
            callback(np.array([47., 53.]))
            callback(np.array([44., 56.]))
            return OptimizeResult(x=np.array([44., 56.]), success=False, status=9, nit=2,
                                  nfev=2, message='Iteration limit reached')

        problem = self.problem()
        with patch('tlh.optimization.best_feasible.minimize', side_effect=failed_attempt):
            result = solve_best_feasible(problem, retain_feasible_iterates=True)
        np.testing.assert_array_equal(result.weights, [.47, .53])
        selected = next(a for a in result.attempts if a['start'] == result.selected_start)
        self.assertEqual(selected['candidate_source'], 'iterate')
        self.assertFalse(selected['success'])
        finals = [a for a in result.attempts if a['optimizer_endpoint']]
        self.assertEqual(len(finals), 3)
        self.assertTrue(all(not a['feasible'] and not a['success'] for a in finals))
        self.assertLessEqual(np.sqrt(problem.variance(result.weights)), .01)

    def test_initial_benchmark_is_available_if_no_feasible_iterates(self):
        def failed_attempt(fun, x0, *, callback, **kwargs):
            return OptimizeResult(x=np.array([44., 56.]), success=False, status=9, nit=1,
                                  nfev=1, message='Iteration limit reached')
        with patch('tlh.optimization.best_feasible.minimize', side_effect=failed_attempt):
            result = solve_best_feasible(self.problem(), retain_feasible_iterates=True)
        np.testing.assert_array_equal(result.weights, [.5, .5])
        selected = next(a for a in result.attempts if a['start'] == result.selected_start)
        self.assertEqual(selected['candidate_source'], 'initial')
        self.assertFalse(selected['success'])

    def test_callback_does_not_change_numerical_endpoint(self):
        problem = self.problem()
        free, template = problem.free_indices, np.zeros(2)
        strict, strict_weights = _attempt(problem, 'benchmark', np.array([.5, .5]), free, template, 1., 20)
        observed = _observed_attempt(problem, 'benchmark', np.array([.5, .5]), free, template, 1., 20)
        endpoint, weights = observed[0]
        np.testing.assert_array_equal(weights, strict_weights)
        for key in strict.keys() - {'elapsed_seconds'}:
            self.assertEqual(endpoint[key], strict[key], key)

    def test_no_feasible_candidates_stops_without_cache(self):
        def invalid(problem, name, *args):
            return dict(start=name, success=False, feasible=False, objective_nav=-1.), np.array([.44, .56])
        with TemporaryDirectory() as tmp, patch('tlh.optimization.best_feasible._attempt', side_effect=invalid):
            with self.assertRaisesRegex(ValueError, 'No feasible continuous solution at fixed TE'):
                solve_best_feasible(self.problem(), cache_dir=tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_strategy_accepts_at_fixed_one_percent_and_forbids_history_with_other_risk(self):
        state, benchmark, risk = self.inputs()
        args = ({self.trade: self.signal}, {self.signal: benchmark}, risk)
        strategy = TwoStageLtfo(*args, te_limit=.01, solver_workers=1)
        with patch('tlh.optimization.best_feasible._observed_attempt',
                   side_effect=lambda *args: [self.fake_attempt(*args)]):
            target = strategy(state)
        self.assertEqual(sum(target.weights.values()), 1)
        record = strategy.decisions[0]
        self.assertEqual(record['annual_te_limit'], .01)
        self.assertEqual(record['stage_two']['status'], 'optimal')
        self.assertEqual(record['solver_acceptance_policy'], 'best_feasible_initial_iterate_endpoint')
        self.assertFalse(record['selected_converged'])
        self.assertTrue(record['accepted_without_convergence'])
        for extra in ({'fallback_te_limit': .10}, {'retry_max_iterations': 2000},
                      {'replay_decisions': {self.trade: {'annual_te_limit': .05}}}):
            with self.assertRaisesRegex(ValueError, 'does not support'):
                TwoStageLtfo(*args, te_limit=.01, **extra)


if __name__ == '__main__':
    unittest.main()
