"""Hand-solvable optimization and unchanged account-engine integration."""

import importlib.util
import csv
import unittest
from datetime import date, timedelta
from dataclasses import replace
from decimal import Decimal as D
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

READY = all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "sklearn"))
if READY:
    import numpy as np
    from tlh.optimization.problem import PortfolioProblem
    from tlh.optimization.risk import RollingLedoitWolf
    from tlh.optimization.solver import solve
    from tlh.optimization.strategy import TaxAwareHifo
from tlh.actions import CorporateAction
from tlh.backtest import AccountEngine, Session
from tlh.execution import DEFAULT_COMMISSION
from tlh.ledger import Ledger
from tlh.market import Delisting
from tlh.positions import write_weights
from tlh.strategy import CloseSnapshot, LotView, Target


@unittest.skipUnless(READY, "Install requirements-optimizer.txt in the optimizer environment.")
class OptimizerStrategyTests(unittest.TestCase):
    signal, trade = date(2015, 12, 31), date(2016, 1, 4)

    def risk(self, *, incomplete=False):
        days = tuple(self.signal - timedelta(days=i) for i in reversed(range(252)))
        amplitude = np.sqrt(.04 * 2 / 252)
        returns = np.tile(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) * amplitude, (63, 1))
        if incomplete:
            returns[:200, 1] = np.nan
        return RollingLedoitWolf(days, (1, 2), returns)

    def state(self, *, extra=False):
        old = date(2015, 10, 1)
        lots = [LotView(1, 1, old, old, D(10), D(1200))]
        if extra:
            lots.append(LotView(2, 3, old, old, D(5), D(600)))
        return CloseSnapshot(self.signal, D(1000), D(2500 if extra else 2000),
                             {1: D(100), 2: D(100), 3: D(100)}, tuple(lots), frozenset(), self.trade)

    def strategy(self, risk=None):
        return TaxAwareHifo({self.trade: self.signal}, {self.signal: {1: D('.5'), 2: D('.5')}},
                            risk or self.risk())

    def test_two_stock_solution_matches_analytic_risk_boundary(self):
        state = self.state()
        problem = PortfolioProblem.build(state, {1: D('.5'), 2: D('.5')}, self.risk())
        solution = solve(problem)
        expected = .5 - .01 / np.sqrt(.08)
        self.assertAlmostEqual(solution.weights[0], expected, places=6)
        self.assertAlmostEqual(solution.weights.sum(), 1)
        self.assertLessEqual(np.sqrt(problem.variance(solution.weights)), .01 + 1e-9)
        self.assertEqual(len(solution.attempts), 3)
        self.assertTrue(any(a['success'] and a['feasible'] for a in solution.attempts))

    def test_parallel_starts_match_serial_weights_and_all_candidate_results(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete):
                problem = PortfolioProblem.build(self.state(extra=True), {1: D('.5'), 2: D('.5')},
                                                 self.risk(incomplete=incomplete))
                serial, parallel = solve(problem), solve(problem, workers=3)
                np.testing.assert_array_equal(serial.weights, parallel.weights)
                self.assertEqual(serial.selected_start, parallel.selected_start)
                clean = lambda result: [{k: v for k, v in a.items() if k != 'elapsed_seconds'}
                                        for a in result.attempts]
                self.assertEqual(clean(serial), clean(parallel))

    def test_cache_reuses_exact_solve_and_invalidates_for_risk_tax_or_iteration_changes(self):
        state = self.state()
        benchmark = {1: D('.5'), 2: D('.5')}
        problem = PortfolioProblem.build(state, benchmark, self.risk())
        with TemporaryDirectory() as tmp:
            first = solve(problem, cache_dir=tmp)
            self.assertFalse(first.from_cache)
            with patch('tlh.optimization.solver.minimize', side_effect=RuntimeError('fresh solve')):
                cached = solve(problem, cache_dir=tmp, workers=3)
                self.assertTrue(cached.from_cache)
                np.testing.assert_array_equal(first.weights, cached.weights)
                self.assertEqual(first.attempts, cached.attempts)
                changed_risk = replace(problem, risk=replace(problem.risk,
                                                            covariance=problem.risk.covariance * 1.1))
                changed_lot = replace(state.lots[0], basis=state.lots[0].basis + 1)
                changed_tax = PortfolioProblem.build(replace(state, lots=(changed_lot,)), benchmark, self.risk())
                for changed in (changed_risk, changed_tax):
                    with self.assertRaisesRegex(RuntimeError, 'fresh solve'):
                        solve(changed, cache_dir=tmp)
                with self.assertRaisesRegex(RuntimeError, 'fresh solve'):
                    solve(problem, cache_dir=tmp, max_iterations=401)

    def test_missing_history_fixed_at_benchmark_and_outside_holding_forced_to_zero(self):
        state = self.state(extra=True)
        strategy = self.strategy(self.risk(incomplete=True))
        target = strategy(state)
        self.assertEqual(target.weights[2], D('.5'))
        self.assertEqual(target.weights[3], 0)
        self.assertEqual(sum(target.weights.values()), 1)
        self.assertEqual(strategy.decisions[0]['forced_exits'], [3])
        self.assertEqual(strategy.decisions[0]['forced_sale_gross_usd'], 500)
        self.assertEqual(state.cash, 1000)
        self.assertEqual(state.lots[-1].shares, 5)

    def test_unheld_nonbenchmark_delisting_name_does_not_enter_optimization(self):
        state = self.state(extra=True)
        base = self.strategy(self.risk(incomplete=True))(state)
        # 99 has no quote, benchmark weight, or lot. The held outside name 3 still exits.
        strategy = self.strategy(self.risk(incomplete=True))
        target = strategy(replace(state, blocked=frozenset({3, 99})))
        self.assertEqual(target.weights, base.weights)
        self.assertNotIn(99, target.weights)
        self.assertEqual(target.weights[3], 0)
        self.assertEqual(strategy.decisions[0]['forced_exits'], [3])

    def test_final_targets_are_floored_by_engine_and_excess_integer_te_is_allowed(self):
        # A $100 account cannot buy fractional shares of either $60 stock.
        # Continuous benchmark weights are feasible; the resulting all-cash position is accepted.
        engine = AccountEngine(D(100), start=self.signal)
        state = engine.close_snapshot(self.signal, {1: D(60), 2: D(60)}, trade_day=self.trade)
        strategy = self.strategy()
        target = strategy(state)
        orders = engine.prepare(self.signal, state.closes, self.trade, target, set(), snapshot=state)
        self.assertEqual(orders, ())
        engine.advance(self.trade, Session(dict(state.closes), dict(state.closes), ()), orders, {}, ())
        self.assertEqual(engine.result.history[-1].cash, 100)
        self.assertFalse(engine.result.history[-1].holdings)

    def test_short_history_stays_at_benchmark_in_solve_but_exit_overrides_execution(self):
        engine = AccountEngine(D(3000), start=self.signal)
        book = engine.result.ledger
        book.buy(date(2015, 10, 1), 1, D(10), D(100), D(0))
        book.buy(date(2015, 10, 1), 2, D(10), D(120), D(0))
        prices = {1: D(100), 2: D(100)}
        engine.last_nav = book.nav(prices)
        state = engine.close_snapshot(self.signal, prices, trade_day=self.trade, blocked=frozenset({2}))
        risk = self.risk(incomplete=True)
        strategy = self.strategy(risk)
        recorded = []
        strategy.on_decision = lambda decision, weights: recorded.append((decision, dict(weights)))
        target = strategy(state)
        decision, continuous = recorded[0]
        self.assertEqual(continuous, {1: D('.5'), 2: D('.5')})
        self.assertEqual(target.weights, {1: D('.5'), 2: D(0)})
        self.assertEqual(decision['continuous_te'], 0)
        self.assertEqual(decision['incomplete_history'], {'2': 52})
        self.assertEqual(decision['forced_exits'], [2])
        self.assertEqual(decision['forced_sale_gross_usd'], 1000)
        self.assertEqual(decision['pre_delisting_execution_overrides'],
                         {'2': {'continuous_weight': '0.5', 'execution_weight': '0'}})
        self.assertEqual(D(decision['execution_target_weight_sum']), D('.5'))
        self.assertEqual(decision['estimated_tax_usd'], 0)
        self.assertAlmostEqual(decision['execution_estimated_tax_usd'], -81.804)
        self.assertAlmostEqual(decision['execution_estimated_fees_usd'], .7)
        with self.assertRaisesRegex(ValueError, 'nonzero active weight'):
            risk.estimate(self.signal, (1, 2)).tracking_error(target.weights, continuous)
        orders = engine.prepare(self.signal, prices, self.trade, target, {2}, snapshot=state)
        # Blocked names exit through the engine's dedicated forced-sale path.
        self.assertEqual({o.permno: o.quantity for o in orders}, {1: 4})
        exits = {2: [Delisting(2, self.trade + timedelta(days=1), 'MER')]}
        engine.advance(self.trade, Session(prices, prices, ()), orders, exits, ())
        self.assertEqual(book.holdings(), {1: 14})
        self.assertEqual(book.cash, D('1399.3'))
        self.assertEqual(engine.result.history[-1].nav, D('2799.3'))
        self.assertEqual([(s.permno, s.shares, s.reason) for s in book.sales],
                         [(2, D(10), 'PRE_DELISTING_EXIT')])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.csv'
            write_weights(path, {self.signal: Target(continuous, False), **engine.result.targets})
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            exited = next(r for r in rows if r['date'] == str(self.trade) and r['permno'] == '2')
            self.assertEqual(D(exited['target_weight']), 0)

    def test_unheld_short_history_delisting_name_is_not_bought_or_reallocated(self):
        engine = AccountEngine(D(1000), start=self.signal)
        prices = {1: D(100), 2: D(100)}
        state = engine.close_snapshot(self.signal, prices, trade_day=self.trade, blocked=frozenset({2}))
        strategy = self.strategy(self.risk(incomplete=True))
        target = strategy(state)
        orders = engine.prepare(self.signal, prices, self.trade, target, {2}, snapshot=state)
        self.assertEqual({o.permno: o.quantity for o in orders}, {1: 5})
        self.assertEqual(strategy.decisions[0]['forced_exits'], [])
        engine.advance(self.trade, Session(prices, prices, ()), orders, {}, ())
        self.assertEqual(engine.result.ledger.holdings(), {1: 5})
        self.assertEqual(engine.result.ledger.cash, D('499.75'))

    def test_split_cancellation_is_in_account_layer_and_keeps_optimizer_targets(self):
        engine = AccountEngine(D(200), start=self.signal)
        prices = {1: D(100), 2: D(100)}
        state = engine.close_snapshot(self.signal, prices, trade_day=self.trade)
        strategy = self.strategy(self.risk(incomplete=True))
        saved = []
        strategy.on_decision = lambda decision, weights: saved.append((decision, dict(weights)))
        target = strategy(state)
        decision, continuous = saved[0]
        self.assertEqual(continuous, {1: D('.5'), 2: D('.5')})
        self.assertEqual(target.weights, {1: D('.5'), 2: D('.5')})
        self.assertAlmostEqual(decision['estimated_fees_usd'], .1)
        self.assertAlmostEqual(decision['execution_estimated_fees_usd'], .1)
        self.assertEqual(decision['continuous_te'], 0)
        orders = engine.prepare(self.signal, prices, self.trade, target, set(), snapshot=state)
        engine.advance(self.trade, Session({1: D(100), 2: D(50)}, {1: D(100), 2: D(50)},
                                           (CorporateAction('split', 2, 'SPLIT', D(2)),)), orders, {}, ())
        self.assertEqual(engine.result.ledger.holdings(), {1: D(1)})
        self.assertEqual(engine.result.ledger.cash, D('99.95'))
        self.assertEqual(engine.result.exceptions, [(self.trade, 2, 'SKIPPED_SPLIT_BUY 1/0 pre-split shares')])
        self.assertEqual(state.cash, 200)

    def test_forced_exit_proceeds_and_existing_cash_fund_buys_once(self):
        engine = AccountEngine(D(2800), start=self.signal)
        book = engine.result.ledger
        book.buy(date(2015, 10, 1), 1, D(10), D(120), D(0))
        book.buy(date(2015, 10, 1), 3, D(5), D(120), D(0))
        prices = {1: D(100), 2: D(100), 3: D(100)}
        engine.last_nav = book.nav(prices)
        state = engine.close_snapshot(self.signal, prices, trade_day=self.trade)
        target = self.strategy(self.risk(incomplete=True))(state)
        orders = engine.prepare(self.signal, prices, self.trade, target, set(), snapshot=state)
        self.assertEqual({o.permno: o.quantity for o in orders}, {1: 2, 2: 12, 3: -5})
        engine.advance(self.trade, Session(prices, prices, ()), orders, {}, ())
        self.assertEqual(book.holdings(), {1: 12, 2: 12})
        # $1000 opening cash + $500 forced proceeds - $1400 buys - $0.95 total commissions.
        self.assertEqual(book.cash, D('99.05'))
        self.assertEqual(engine.result.history[-1].nav, D('2499.05'))
        self.assertEqual(sum(s.shares for s in book.sales if s.permno == 3), 5)

    def test_cash_shortage_raises_without_adjusting_full_investment_or_changing_ledger(self):
        engine = AccountEngine(D(100), start=self.signal, fee=DEFAULT_COMMISSION)
        state = engine.close_snapshot(self.signal, {1: D(10), 2: D(10)}, trade_day=self.trade)
        # All starts have identical cost from cash. Fix both weights via the confirmed short-history rule.
        risk = self.risk()
        risk.returns[:] = np.nan
        target = self.strategy(risk)(state)
        orders = engine.prepare(self.signal, state.closes, self.trade, target, set(), snapshot=state)
        self.assertEqual(sum(o.quantity for o in orders), 10)
        with self.assertRaisesRegex(ValueError, "available cash"):
            engine.advance(self.trade, Session(dict(state.closes), dict(state.closes), ()), orders, {}, ())
        self.assertEqual(engine.result.ledger.cash, 100)
        self.assertFalse(engine.result.ledger.lots)
        self.assertFalse(engine.result.history)

    def test_wash_sale_is_permitted_and_account_deferment_is_not_predicted(self):
        book = Ledger(D(10000))
        old = book.buy(date(2015, 1, 2), 1, D(10), D(120), D(0))
        replacement = book.buy(date(2015, 12, 20), 1, D(10), D(110), D(0))
        state = CloseSnapshot.from_ledger(self.signal, book, {1: D(100), 2: D(100)},
                                          book.cash + 2000, set(), trade_day=self.trade)
        problem = PortfolioProblem.build(state, {1: D('.1'), 2: D('.9')}, self.risk())
        estimate = problem.costs.curves[0].value_and_slope(10)[0]
        sales = book.sell(self.trade, 1, {old.id: D(10)}, D(100), {old.id: D('.5')}, 'TRADE')
        self.assertLess(estimate, 0)
        self.assertEqual(sum(s.taxable_gain for s in sales), 0)
        self.assertEqual(sum(s.disallowed_loss for s in sales), D('200.5'))
        self.assertEqual(book.lots[replacement.id].basis, D('1300.5'))

    def test_no_converged_solution_is_an_error_not_a_silent_benchmark_fallback(self):
        from scipy.optimize import OptimizeResult
        problem = PortfolioProblem.build(self.state(), {1: D('.5'), 2: D('.5')}, self.risk())
        failure = OptimizeResult(x=np.array([50., 50.]), success=False, status=9, nit=1, message='limit')
        with TemporaryDirectory() as tmp, patch('tlh.optimization.solver.minimize', return_value=failure):
            with self.assertRaisesRegex(ValueError, 'No converged, feasible'):
                solve(problem, cache_dir=tmp)
            self.assertFalse(list(Path(tmp).iterdir()))

    def test_retry_restarts_same_starts_then_accepts_converged_feasible_solution(self):
        from scipy.optimize import OptimizeResult, minimize as scipy_minimize
        strategy = self.strategy()
        strategy.retry_max_iterations = 2000
        strategy.final_retry_max_iterations = 5000
        events, starts, limits = [], [], []
        strategy.on_solver_event = events.append

        def initial_failure(fun, x0, **kwargs):
            limit = kwargs['options']['maxiter']
            limits.append(limit)
            starts.append(x0.copy())
            if limit == 400:
                return OptimizeResult(x=x0, success=False, status=9, nit=400, message='Iteration limit reached')
            return scipy_minimize(fun, x0, **kwargs)

        with patch('tlh.optimization.solver.minimize', side_effect=initial_failure):
            target = strategy(self.state())
        self.assertEqual(limits, [400] * 3 + [2000] * 3)
        for first, retried in zip(starts[:3], starts[3:]):
            np.testing.assert_array_equal(first, retried)
        self.assertEqual(sum(target.weights.values()), 1)
        decision = strategy.decisions[0]
        self.assertLessEqual(decision['continuous_te'], .01 + 1e-9)
        self.assertEqual(decision['solver_max_iterations_used'], 2000)
        self.assertEqual(decision['solver_retry']['initial_max_iterations'], 400)
        self.assertIsNone(decision['solver_final_retry'])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event'], 'retry_started')
        self.assertTrue(any(a['success'] and a['feasible'] for a in decision['attempts']))

    def test_retry_rejects_nonconverged_feasible_candidates_and_stops_after_one_retry(self):
        from scipy.optimize import OptimizeResult
        strategy = self.strategy()
        strategy.retry_max_iterations = 2000
        failure = OptimizeResult(x=np.array([50., 50.]), success=False, status=9, nit=400, message='limit')
        with patch('tlh.optimization.solver.minimize', return_value=failure) as minimize:
            with self.assertRaisesRegex(ValueError, 'No converged, feasible'):
                strategy(self.state())
        self.assertEqual([call.kwargs['options']['maxiter'] for call in minimize.call_args_list],
                         [400] * 3 + [2000] * 3)
        self.assertEqual(strategy.decisions, [])

    def test_retry_preserves_initial_success_and_does_not_mask_unrelated_errors(self):
        strategy = self.strategy()
        strategy.retry_max_iterations = 2000
        strategy.final_retry_max_iterations = 5000
        with patch('tlh.optimization.strategy.solve', wraps=solve) as wrapped:
            strategy(self.state())
        self.assertEqual(wrapped.call_count, 1)
        self.assertIsNone(strategy.decisions[0]['solver_retry'])
        self.assertIsNone(strategy.decisions[0]['solver_final_retry'])
        self.assertEqual(strategy.decisions[0]['solver_max_iterations_used'], 400)
        for message in ('Invalid input', 'Solver cache fingerprint mismatch'):
            with self.subTest(message=message):
                strategy = self.strategy()
                strategy.retry_max_iterations = 2000
                strategy.final_retry_max_iterations = 5000
                with patch('tlh.optimization.strategy.solve', side_effect=ValueError(message)) as wrapped:
                    with self.assertRaisesRegex(ValueError, message):
                        strategy(self.state())
                self.assertEqual(wrapped.call_count, 1)
                self.assertEqual(strategy.decisions, [])

    def test_final_retry_uses_same_starts_and_preserves_both_failure_diagnostics(self):
        from scipy.optimize import OptimizeResult, minimize as scipy_minimize
        strategy = self.strategy()
        strategy.retry_max_iterations, strategy.final_retry_max_iterations = 2000, 5000
        starts, limits, events = [], [], []
        strategy.on_solver_event = events.append

        def fail_first_two_stages(fun, x0, **kwargs):
            limit = kwargs['options']['maxiter']
            limits.append(limit)
            starts.append(x0.copy())
            if limit < 5000:
                return OptimizeResult(x=x0, success=False, status=9, nit=limit, message='Iteration limit reached')
            return scipy_minimize(fun, x0, **kwargs)

        with patch('tlh.optimization.solver.minimize', side_effect=fail_first_two_stages):
            target = strategy(self.state())
        self.assertEqual(limits, [400] * 3 + [2000] * 3 + [5000] * 3)
        for i in range(3):
            np.testing.assert_array_equal(starts[i], starts[i + 3])
            np.testing.assert_array_equal(starts[i], starts[i + 6])
        decision = strategy.decisions[0]
        self.assertEqual(sum(target.weights.values()), 1)
        self.assertLessEqual(decision['continuous_te'], .01 + 1e-9)
        self.assertEqual(decision['solver_max_iterations_used'], 5000)
        self.assertEqual(decision['solver_retry']['initial_max_iterations'], 400)
        self.assertEqual(decision['solver_final_retry']['initial_max_iterations'], 2000)
        self.assertEqual(decision['solver_final_retry']['retry_max_iterations'], 5000)
        self.assertIn('No converged, feasible continuous solution:', decision['solver_final_retry']['initial_error'])
        self.assertEqual([e['event'] for e in events], ['retry_started', 'final_retry_started'])
        self.assertTrue(any(a['start'] == decision['selected_start'] and a['success'] and a['feasible']
                            for a in decision['attempts']))

    def test_final_retry_exhaustion_does_not_execute_nonconverged_feasible_weights(self):
        from scipy.optimize import OptimizeResult
        strategy = self.strategy()
        strategy.retry_max_iterations, strategy.final_retry_max_iterations = 2000, 5000
        state = self.state()

        def failed(fun, x0, **kwargs):
            return OptimizeResult(x=np.array([50., 50.]), success=False, status=9,
                                  nit=kwargs['options']['maxiter'], message='Iteration limit reached')

        with patch('tlh.optimization.solver.minimize', side_effect=failed) as minimize:
            with self.assertRaisesRegex(ValueError, 'No converged, feasible'):
                strategy(state)
        self.assertEqual([call.kwargs['options']['maxiter'] for call in minimize.call_args_list],
                         [400] * 3 + [2000] * 3 + [5000] * 3)
        self.assertEqual(strategy.decisions, [])
        self.assertEqual(state.cash, 1000)
        self.assertEqual(state.lots[0].shares, 10)

    def test_final_retry_does_not_mask_unrelated_second_stage_errors(self):
        for message in ('Invalid input', 'Solver cache fingerprint mismatch'):
            with self.subTest(message=message):
                strategy = self.strategy()
                strategy.retry_max_iterations, strategy.final_retry_max_iterations = 2000, 5000
                failures = [ValueError('No converged, feasible continuous solution: []'), ValueError(message)]
                with patch('tlh.optimization.strategy.solve', side_effect=failures) as wrapped:
                    with self.assertRaisesRegex(ValueError, message):
                        strategy(self.state())
                self.assertEqual([call.kwargs['max_iterations'] for call in wrapped.call_args_list], [400, 2000])
                self.assertEqual(strategy.decisions, [])

    def test_final_retry_limit_must_follow_an_enabled_smaller_retry_limit(self):
        for retry, final in ((None, 5000), (2000, 2000), (2000, 1000)):
            with self.subTest(retry=retry, final=final):
                with self.assertRaisesRegex(ValueError, 'Final retry requires'):
                    TaxAwareHifo({self.trade: self.signal}, {self.signal: {1: D('.5'), 2: D('.5')}},
                                 self.risk(), retry_max_iterations=retry, final_retry_max_iterations=final)

    def test_505_float_targets_convert_to_exact_decimal_budget(self):
        from tlh.optimization.strategy import decimal_weights
        vector = np.random.default_rng(45).dirichlet(np.ones(505) * .2)
        problem = SimpleNamespace(permnos=tuple(range(505)), fixed={0: D('.000001234567891234')},
                                  free_indices=np.arange(1, 505))
        weights = decimal_weights(problem, vector)
        self.assertEqual(sum(weights.values()), D(1))
        self.assertEqual(weights[0], problem.fixed[0])
        self.assertTrue(all(w >= 0 for w in weights.values()))

    def test_replay_jumps_directly_to_second_or_third_stage_successful_cache(self):
        from scipy.optimize import OptimizeResult, minimize as scipy_minimize
        for accepted_limit in (2000, 5000):
            with self.subTest(accepted_limit=accepted_limit), TemporaryDirectory() as tmp:
                old = self.strategy()
                old.solver_cache, old.retry_max_iterations = tmp, 2000
                old.final_retry_max_iterations = 5000 if accepted_limit == 5000 else None

                def fail_earlier_stages(fun, x0, **kwargs):
                    limit = kwargs['options']['maxiter']
                    if limit < accepted_limit:
                        return OptimizeResult(x=x0, success=False, status=9, nit=limit, message='limit')
                    return scipy_minimize(fun, x0, **kwargs)

                with patch('tlh.optimization.solver.minimize', side_effect=fail_earlier_stages):
                    expected = old(self.state())
                accepted = old.decisions[0]
                new = self.strategy()
                new.solver_cache = tmp
                new.replay_decisions = {self.trade: accepted}
                with patch('tlh.optimization.solver.minimize', side_effect=AssertionError('must use cache')):
                    actual = new(self.state())
                    self.assertEqual(actual.weights, expected.weights)
                    self.assertTrue(new.decisions[0]['replayed_decision'])
                    self.assertEqual(new.decisions[0]['solver_max_iterations_used'], accepted_limit)
                    self.assertEqual(new.decisions[0]['solver_history_source'], 'historical_decision')
                    changed_lot = replace(self.state().lots[0], basis=D(1201))
                    with self.assertRaisesRegex(ValueError, 'Replay numerical inputs changed'):
                        new(replace(self.state(), lots=(changed_lot,)))
                    (Path(tmp) / (accepted['solver_cache_key'] + '.json')).unlink()
                    with self.assertRaisesRegex(ValueError, 'Missing accepted replay cache'):
                        new(self.state())

    def test_risk_fallback_keeps_iteration_cap_and_resets_to_base_at_next_decision(self):
        strategy = self.strategy()
        strategy.te_limit, strategy.fallback_te_limit = .05, .10
        strategy.risk_fallback_start = self.trade
        calls, events, problems = [], [], []
        strategy.on_solver_event = events.append

        def first_base_failure(problem, **kwargs):
            calls.append((problem.te_limit, kwargs['max_iterations']))
            problems.append(problem)
            if len(calls) == 1:
                raise ValueError('No converged, feasible continuous solution: []')
            return solve(problem, **kwargs)

        with patch('tlh.optimization.strategy.solve', side_effect=first_base_failure):
            strategy(self.state())
            strategy(self.state())
        self.assertEqual(calls, [(.05, 400), (.10, 400), (.05, 400)])
        self.assertIs(problems[0].costs, problems[1].costs)
        self.assertIs(problems[0].risk, problems[1].risk)
        self.assertEqual(strategy.te_limit, .05)
        self.assertEqual([r['annual_te_limit'] for r in strategy.decisions], [.10, .05])
        self.assertTrue(all(r['solver_retry'] is None and r['solver_final_retry'] is None
                            for r in strategy.decisions))
        self.assertEqual(events[0]['event'], 'risk_fallback_started')
        self.assertEqual(len(events), 1)
        self.assertLessEqual(strategy.decisions[0]['continuous_te'], .10 + 1e-8)
        self.assertLessEqual(strategy.decisions[1]['continuous_te'], .05 + 1e-8)

    def test_risk_fallback_exhaustion_and_unrelated_errors_stop(self):
        failed = 'No converged, feasible continuous solution: []'
        for messages, expected_calls in (([failed, failed], 2), (['Invalid input'], 1),
                                         ([failed, 'Solver cache fingerprint mismatch'], 2)):
            with self.subTest(messages=messages):
                strategy = self.strategy()
                strategy.te_limit, strategy.fallback_te_limit = .05, .10
                with patch('tlh.optimization.strategy.solve', side_effect=[ValueError(m) for m in messages]) as wrapped:
                    with self.assertRaisesRegex(ValueError, messages[-1].split(':')[0]):
                        strategy(self.state())
                self.assertEqual(wrapped.call_count, expected_calls)
                self.assertTrue(all(c.kwargs['max_iterations'] == 400 for c in wrapped.call_args_list))
                self.assertEqual(strategy.decisions, [])

    def test_risk_fallback_boundary_and_no_iteration_escalation_combination(self):
        strategy = self.strategy()
        strategy.te_limit, strategy.fallback_te_limit = .05, .10
        strategy.risk_fallback_start = date(2016, 2, 1)
        with patch('tlh.optimization.strategy.solve', side_effect=ValueError('No converged, feasible continuous solution: []')) as wrapped:
            with self.assertRaisesRegex(ValueError, 'No converged, feasible'):
                strategy(self.state())
        self.assertEqual(wrapped.call_count, 1)
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            TaxAwareHifo({self.trade: self.signal}, {self.signal: {1: D('.5'), 2: D('.5')}}, self.risk(),
                         te_limit=.05, fallback_te_limit=.10, retry_max_iterations=2000)


if __name__ == '__main__':
    unittest.main()
