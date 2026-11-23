import unittest
from dataclasses import replace
from datetime import date
from decimal import Decimal as D
from unittest.mock import patch

import numpy as np

from tlh.strategy import CloseSnapshot, LotView, SegmentView
from tlh.optimization.problem import PortfolioProblem
from tlh.optimization.risk import RiskEstimate
from tlh.optimization.two_stage_ltfo import (LtfoCosts, TwoStageLtfo, loss_directions,
                                           solve_lots, target_decimals, ltfo_with_fees)
from tlh.execution import floor_shares, lock_orders
from functools import partial

OLD, RECENT = date(2020, 1, 2), date(2025, 1, 2)


def fixture(lots=None, cash=D(0), benchmark=None):
    lots = lots if lots is not None else (
        LotView(1, 1, OLD, OLD, D(50), D(6000)),
        LotView(2, 1, RECENT, RECENT, D(50), D(5750)),
        LotView(3, 2, OLD, OLD, D(100), D(8000)))
    nav = cash + sum(l.shares * 100 for l in lots)
    state = CloseSnapshot(date(2025, 1, 31), cash, nav, {1: D(100), 2: D(100)},
                          tuple(lots), frozenset(), date(2025, 2, 3))
    risk = RiskEstimate(date(2024, 1, 31), state.day, (1, 2), np.eye(2) * .04,
                        {1: 252, 2: 252}, .2)
    costs = LtfoCosts(state, (1, 2))
    return PortfolioProblem(state, (1, 2), benchmark or {1: D('.5'), 2: D('.5')},
                            risk, costs, {}, (), np.array([0, 1]), .5, .01)


class TwoStageTests(unittest.TestCase):
    def test_ltfo_sorts_tax_not_basis_and_execution_uses_same_order(self):
        p = fixture()
        self.assertEqual(p.costs.curves[0].lot_ids, (2, 1))
        self.assertEqual(ltfo_with_fees(p.state, 1, D(10), fee_rate=D('.0005')), {2: D(10)})
        self.assertAlmostEqual(p.costs.curves[0].value_and_slope(10)[0], (100 * .9995 - 115) * .408 * 10)

    def test_loss_detection_is_any_lot_not_net_position_and_directions_are_relative_to_current(self):
        p = fixture(lots=(LotView(1, 1, OLD, OLD, D(50), D(6000)),
                          LotView(2, 1, OLD, OLD, D(50), D(1000)),
                          LotView(3, 2, OLD, OLD, D(100), D(8000))))
        self.assertEqual(loss_directions(p, np.array([.49, .51])), {0: 'SELL'})
        self.assertEqual(loss_directions(p, np.array([.51, .49])), {0: 'BUY'})
        self.assertEqual(loss_directions(p, np.array([.5, .5])), {0: 'HOLD'})

    def test_sell_direction_solves_known_risk_boundary_and_lot_order(self):
        p = fixture()
        w, record = solve_lots(p, np.array([.48, .52]), {0: 'SELL'})
        self.assertAlmostEqual(w[0], .5 - .01 / np.sqrt(.08), places=7)
        self.assertLess(record['objective_usd'], record['stage_one_objective_usd'])
        sold = {row['lot_id']: row['sale_usd'] for row in record['lot_sales']}
        self.assertLess(sold[1], .001)
        self.assertGreater(sold[2], 700)
        self.assertLess(sold[3], .001)
        self.assertEqual(record['status'], 'optimal')

    def test_buy_and_hold_prevent_artificial_roundtrip_loss_harvesting(self):
        p = fixture()
        for direction in ('BUY', 'HOLD'):
            w, record = solve_lots(p, np.array([.5, .5]), {0: direction})
            np.testing.assert_allclose(w, [.5, .5], atol=1e-8)
            self.assertAlmostEqual(record['objective_usd'], 0, places=4)
            weights = target_decimals(p, w, {0: direction})
            orders = lock_orders(p.state, weights, floor_shares,
                                partial(ltfo_with_fees, fee_rate=D('.0005')), set())
            self.assertFalse(any(o.permno == 1 and o.quantity < 0 for o in orders))

    def test_empty_portfolio_has_no_sale_variables(self):
        p = fixture(lots=(), cash=D(20000))
        w, r = solve_lots(p, np.array([.5, .5]), {})
        p.check(w, tolerance=1e-7)
        self.assertEqual(r['lot_sales'], [])
        self.assertAlmostEqual(r['objective_usd'], 10)

    def test_forced_exit_is_preserved(self):
        p = fixture(benchmark={1: D(0), 2: D(1)})
        p = replace(p, fixed={0: D(0)}, forced_exits=(1,))
        w, _ = solve_lots(p, np.array([0., 1.]), {0: 'SELL'})
        self.assertLess(abs(w[0]), 1e-9)

    def test_infeasible_stage_two_stops(self):
        p = fixture(benchmark={1: D(0), 2: D(1)})
        p = replace(p, fixed={0: D(0)})
        with self.assertRaisesRegex(ValueError, 'rejected solver status'):
            solve_lots(p, np.array([0., 1.]), {0: 'BUY'})

    def test_stage_one_failure_never_calls_stage_two_or_emits_target(self):
        p = fixture()
        class Risk:
            def estimate(self, *_):
                return p.risk
        strategy = TwoStageLtfo({p.state.trade_day: p.state.day}, {p.state.day: p.benchmark}, Risk())
        with patch('tlh.optimization.two_stage_ltfo.solve_best_feasible', side_effect=ValueError('no feasible candidate')):
            with patch('tlh.optimization.two_stage_ltfo.solve_lots') as second:
                with self.assertRaisesRegex(ValueError, 'no feasible candidate'):
                    strategy(p.state)
                second.assert_not_called()
        self.assertEqual(strategy.decisions, [])

    def test_bound_lot_uses_each_segments_holding_period(self):
        parts = (SegmentView(11, 1, RECENT, OLD, D(40), D(4400)),
                 SegmentView(12, 2, RECENT, RECENT, D(60), D(6600)))
        p = fixture(lots=(LotView(1, 1, None, None, D(100), D(11000), parts),))
        expected = (40 * 99.95 - 4400) * .238 + (60 * 99.95 - 6600) * .408
        self.assertAlmostEqual(p.costs.curves[0].value_and_slope(100)[0], expected)


if __name__ == '__main__':
    unittest.main()
