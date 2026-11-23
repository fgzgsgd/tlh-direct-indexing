import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cvxpy as cp
import numpy as np
from test_two_stage_ltfo import fixture
from tlh.optimization.two_stage_ltfo import solve_lots, validate_optimality


def diagnostics(**changes):
    values = dict(res_primal=1e-6, res_dual=1e-14, gap_abs=1e-12, gap_rel=1e-12,
                  cost_primal=1., cost_dual=1.)
    return SimpleNamespace(**(values | changes))


class ValidatedStatusTests(unittest.TestCase):
    def test_internal_primal_residual_is_diagnostic_when_other_checks_pass(self):
        r = validate_optimality(diagnostics(), 100., 100., 1e6)
        self.assertEqual(r['internal_primal_residual'], 1e-6)
        self.assertEqual(r['original_vs_dual_error_usd'], 0)

    def test_bad_dual_residual_is_rejected_even_with_zero_gap(self):
        with self.assertRaisesRegex(ValueError, 'dual-residual'):
            validate_optimality(diagnostics(res_dual=1e-5, gap_abs=0., gap_rel=0.), 100., 100., 1e6)

    def test_large_absolute_and_relative_gap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'optimality-gap'):
            validate_optimality(diagnostics(gap_abs=.001, gap_rel=.001), 100., 100., 1e6)

    def test_relative_gap_can_pass_when_absolute_gap_is_larger(self):
        validate_optimality(diagnostics(cost_primal=1e5, cost_dual=1e5-1e-6,
                                       gap_abs=1e-6, gap_rel=1e-11), 1e7, 1e7, 1e6)

    def test_nonfinite_diagnostics_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'invalid optimality'):
            validate_optimality(diagnostics(res_dual=float('nan')), 100., 100., 1e6)

    def test_original_objective_must_reconcile_with_dual(self):
        with self.assertRaisesRegex(ValueError, 'original dollars'):
            validate_optimality(diagnostics(), 110., 100., 1e6)

    def test_inaccurate_status_is_accepted_with_full_checks(self):
        original = cp.Problem.solve
        def solve(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            model._status = cp.OPTIMAL_INACCURATE
            return result
        with patch.object(cp.Problem, 'solve', solve):
            _, r = solve_lots(fixture(), np.array([.48, .52]), {0: 'SELL'})
        self.assertTrue(r['accepted_optimal_inaccurate'])
        self.assertLessEqual(r['optimality_checks']['dual_residual'], 1e-8)

    def test_inaccurate_status_does_not_bypass_risk_check(self):
        original = cp.Problem.solve
        def solve(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            min(model.variables(), key=lambda v: v.id).value = np.array([60., 40.])
            model._status = cp.OPTIMAL_INACCURATE
            return result
        with patch.object(cp.Problem, 'solve', solve):
            with self.assertRaisesRegex(ValueError, 'tracking-error'):
                solve_lots(fixture(), np.array([.48, .52]), {0: 'SELL'})


if __name__ == '__main__':
    unittest.main()
