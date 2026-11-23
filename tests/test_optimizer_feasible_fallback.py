"""Verify the explicitly approved feasible-only exception at the higher TE stage."""

import importlib.util
import unittest
from datetime import date, timedelta
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

READY = all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "sklearn"))
if READY:
    import numpy as np
    from tlh.optimization.cache import fingerprint
    from tlh.optimization.feasible_fallback import FEASIBLE_POLICY, solve_feasible_fallback
    from tlh.optimization.problem import PortfolioProblem
    from tlh.optimization.replay import load_cached_decision
    from tlh.optimization.risk import RollingLedoitWolf
    from tlh.optimization.solver import solve
    from tlh.optimization.strategy import TaxAwareHifo
from tlh.strategy import CloseSnapshot, LotView


@unittest.skipUnless(READY, "Install the optimizer dependencies.")
class FeasibleFallbackTests(unittest.TestCase):
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
        state, benchmark, risk = self.inputs()
        return PortfolioProblem.build(state, benchmark, risk, te_limit=.10)

    def fake_attempt(self, *, converged=(), infeasible=False):
        vectors = {'benchmark': [.4, .6], 'current_plus_cash': [.2, .8], 'diversified': [.3, .7]}

        def attempt(problem, name, *args):
            weights = np.array([.99, .01] if infeasible else vectors[name])
            error = None
            try:
                problem.check(weights, tolerance=1e-7)
            except ValueError as exc:
                error = str(exc)
            success = name in converged
            record = dict(start=name, success=success, status=0 if success else 9, iterations=400,
                          function_evaluations=400, elapsed_seconds=0., objective_nav=problem.objective(weights)[0],
                          continuous_te=float(np.sqrt(problem.variance(weights))), feasible=error is None,
                          message='success' if success else 'Iteration limit reached', constraint_error=error)
            return record, weights
        return attempt

    def test_real_numerical_attempts_match_strict_solver_exactly(self):
        problem = self.problem()
        strict = solve(problem)
        candidate = solve_feasible_fallback(problem, workers=3)
        np.testing.assert_array_equal(strict.weights, candidate.weights)
        self.assertEqual(strict.selected_start, candidate.selected_start)
        clean = lambda result: [{k: v for k, v in a.items() if k != 'elapsed_seconds'} for a in result.attempts]
        self.assertEqual(clean(strict), clean(candidate))

    def test_converged_feasible_candidate_has_priority_over_cheaper_nonconverged_one(self):
        with patch('tlh.optimization.feasible_fallback._attempt', side_effect=self.fake_attempt(converged=('benchmark',))):
            result = solve_feasible_fallback(self.problem())
        self.assertEqual(result.selected_start, 'benchmark')
        self.assertFalse(next(a for a in result.attempts if a['start'] == 'current_plus_cash')['success'])

    def test_best_feasible_candidate_keeps_failure_flag_and_has_separate_replayable_cache(self):
        problem = self.problem()
        with TemporaryDirectory() as tmp:
            with patch('tlh.optimization.feasible_fallback._attempt', side_effect=self.fake_attempt()):
                result = solve_feasible_fallback(problem, cache_dir=tmp)
            self.assertEqual(result.selected_start, 'current_plus_cash')
            self.assertTrue(all(not a['success'] and a['feasible'] for a in result.attempts))
            self.assertNotEqual(result.cache_key, fingerprint(problem, 400))
            self.assertFalse((Path(tmp) / (fingerprint(problem, 400) + '.json')).exists())
            with patch('tlh.optimization.feasible_fallback._attempt', side_effect=AssertionError('must use cache')):
                cached = solve_feasible_fallback(problem, cache_dir=tmp)
            self.assertTrue(cached.from_cache)
            np.testing.assert_array_equal(cached.weights, result.weights)
            record = dict(signal_date=str(self.signal), trade_date=str(self.trade), solver_max_iterations_used=400,
                          solver_cache_key=result.cache_key, selected_start=result.selected_start,
                          attempts=list(result.attempts), solver_acceptance_policy=FEASIBLE_POLICY)
            # Even if a later strict success exists, exact historical replay must retain its accepted result.
            solve(problem, cache_dir=tmp)
            replayed, limit = load_cached_decision(problem, record, tmp)
            self.assertEqual(limit, 400)
            np.testing.assert_array_equal(replayed.weights, result.weights)
            self.assertTrue(all(not a['success'] for a in replayed.attempts))

    def test_infeasible_candidates_are_rejected_and_not_cached(self):
        with TemporaryDirectory() as tmp, patch('tlh.optimization.feasible_fallback._attempt',
                                               side_effect=self.fake_attempt(infeasible=True)):
            with self.assertRaisesRegex(ValueError, 'No feasible continuous solution'):
                solve_feasible_fallback(self.problem(), cache_dir=tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_strategy_applies_exception_only_after_base_risk_failure_and_labels_it(self):
        state, benchmark, risk = self.inputs()
        strategy = TaxAwareHifo({self.trade: self.signal}, {self.signal: benchmark}, risk,
                                te_limit=.05, fallback_te_limit=.10, accept_feasible_fallback=True)
        with patch('tlh.optimization.strategy.solve', side_effect=ValueError('No converged, feasible continuous solution: []')) as strict:
            with patch('tlh.optimization.feasible_fallback._attempt', side_effect=self.fake_attempt()):
                target = strategy(state)
        self.assertEqual(strict.call_count, 1)
        self.assertEqual(strict.call_args.args[0].te_limit, .05)
        self.assertEqual(sum(target.weights.values()), 1)
        decision = strategy.decisions[0]
        self.assertEqual(decision['annual_te_limit'], .10)
        self.assertTrue(decision['accepted_without_convergence'])
        self.assertFalse(decision['selected_converged'])
        self.assertEqual(decision['solver_acceptance_policy'], FEASIBLE_POLICY)
        self.assertEqual(decision['solver_max_iterations_used'], 400)
        with self.assertRaisesRegex(ValueError, 'requires a TE fallback'):
            TaxAwareHifo({self.trade: self.signal}, {self.signal: benchmark}, risk, accept_feasible_fallback=True)


if __name__ == '__main__':
    unittest.main()
