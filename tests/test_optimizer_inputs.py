"""Risk-window and continuous HIFO tests. Install requirements-optimizer.txt to run."""

import importlib.util
import unittest
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal as D

READY = all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "sklearn"))
if READY:
    import numpy as np
    from tlh.optimization.risk import RollingLedoitWolf
    from tlh.optimization.tax_cost import HifoTaxCurve, TradeCosts
from tlh.strategy import CloseSnapshot, LotView, SegmentView
from tlh.shares import sum_shares


@unittest.skipUnless(READY, "Install requirements-optimizer.txt in the optimizer environment.")
class RiskWindowTests(unittest.TestCase):
    def test_window_is_252_sessions_and_never_reads_future_returns(self):
        days = tuple(date(2015, 1, 1) + timedelta(days=i) for i in range(260))
        returns = np.random.default_rng(8).normal(0, .01, (260, 3))
        model = RollingLedoitWolf(days, (1, 2, 3), returns)
        result = model.estimate(days[251], (1, 2, 3))
        changed = returns.copy()
        changed[252:] = 1000
        other = RollingLedoitWolf(days, (1, 2, 3), changed).estimate(days[251], (1, 2, 3))
        np.testing.assert_array_equal(result.covariance, other.covariance)
        self.assertEqual((result.start, result.end, result.observations),
                         (days[0], days[251], {1: 252, 2: 252, 3: 252}))
        self.assertEqual(model.estimate(days[259], (1, 2, 3)).start, days[8])
        with self.assertRaisesRegex(ValueError, "252 sessions"):
            model.estimate(days[250], (1, 2))

    def test_isotropic_returns_have_known_annual_covariance(self):
        days = tuple(date(2015, 1, 1) + timedelta(days=i) for i in range(252))
        returns = np.tile(np.array([[.01, 0], [-.01, 0], [0, .01], [0, -.01]]), (63, 1))
        risk = RollingLedoitWolf(days, (1, 2), returns).estimate(days[-1], (1, 2))
        np.testing.assert_allclose(risk.covariance, np.eye(2) * .00005 * 252, atol=1e-15)
        self.assertAlmostEqual(risk.tracking_error({1: .6, 2: .4}, {1: .5, 2: .5}),
                               np.sqrt(.02 * .00005 * 252))

    def test_missing_history_is_not_zero_filled_or_pairwise_deleted(self):
        days = tuple(date(2015, 1, 1) + timedelta(days=i) for i in range(253))
        returns = np.random.default_rng(3).normal(0, .01, (253, 3))
        returns[0, 1] = np.nan
        returns[:210, 2] = np.nan
        model = RollingLedoitWolf(days, (1, 2, 3), returns)
        risk = model.estimate(days[251], (1, 2, 3, 4))
        self.assertEqual(risk.permnos, (1,))
        self.assertEqual(risk.observations, {1: 252, 2: 251, 3: 42, 4: 0})
        self.assertEqual(risk.incomplete, (2, 3, 4))
        self.assertEqual(model.estimate(days[252], (1, 2, 3)).permnos, (1, 2))
        with self.assertRaisesRegex(ValueError, "nonzero active weight"):
            risk.tracking_error({2: 0}, {2: .01})


@unittest.skipUnless(READY, "Install requirements-optimizer.txt in the optimizer environment.")
class HifoCostTests(unittest.TestCase):
    def state(self, lots, *, cash=D(10000), price=D(90)):
        shares = sum_shares(lot.shares for lot in lots)
        return CloseSnapshot(date(2025, 1, 31), cash, cash + price * shares,
                             {1: price}, tuple(lots), frozenset(), date(2025, 2, 3))

    def test_hand_calculated_hifo_piecewise_cost(self):
        old, recent = date(2020, 1, 2), date(2025, 1, 2)
        state = self.state([LotView(1, 1, recent, recent, D(100), D(11000)),
                            LotView(2, 1, old, old, D(100), D(9500)),
                            LotView(3, 1, old, old, D(100), D(7000))])
        curve = HifoTaxCurve.from_snapshot(state, 1, D(0))
        self.assertEqual(curve.lot_ids, (1, 2, 3))
        for shares, expected in ((0, 0), (50, -408), (150, -875.5), (250, -697)):
            self.assertAlmostEqual(curve.value_and_slope(shares)[0], expected)
        self.assertAlmostEqual(curve.value_and_slope(50.5)[0], -412.08)

    def test_hifo_order_is_not_replaced_by_lower_tax_order(self):
        old, recent = date(2020, 1, 2), date(2025, 1, 2)
        state = self.state([LotView(1, 1, old, old, D(10), D(1100)),
                            LotView(2, 1, recent, recent, D(10), D(1050))])
        curve = HifoTaxCurve.from_snapshot(state, 1, D(0))
        self.assertEqual(curve.lot_ids, (1, 2))
        self.assertAlmostEqual(curve.value_and_slope(10)[0], -47.6)
        self.assertLess(curve.marginal_tax[1], curve.marginal_tax[0])  # Nonconvex kink.

    def test_fractional_lots_total_exact_whole_position_before_continuous_solve(self):
        # Actual 16692 quantities after the 2020-03-02 spin-off/CIL. Ordinary
        # 28-digit Decimal addition rounds their total upward by 1e-26 shares.
        quantities = tuple(map(D, (
            '12.3451507106044041855380290488', '5.2873652975167890051538341403',
            '12.3451507106044041855380290489', '4.4021552397313759175386537560',
            '4.4021552397313759175386537561', '11.4599406528189910979228486647',
            '0.8613150085897235670779322192', '7.91910042167733874746212712798',
            '8.80431047946275183507730751202', '14.1155708261752303607683898173',
            '7.0577854130876151803841949087')))
        old = date(2020, 1, 2)
        lots = tuple(LotView(i, 1, old, old, q, q * 100) for i, q in enumerate(quantities, 1))
        state = self.state(lots)
        costs = TradeCosts(state, (1,))
        self.assertEqual(costs.shares[0], 89)
        self.assertEqual(set(costs.curves[0].lot_ids), set(range(1, 12)))
        self.assertAlmostEqual(costs.curves[0].ends[-1], 89)
        tax, fees, _, _ = costs.evaluate([0])
        self.assertAlmostEqual(tax, 89 * (90 * .9995 - 100) * .238)
        self.assertAlmostEqual(fees, 89 * 90 * .0005)
        self.assertEqual(state.lots, lots)

    def test_exact_share_check_does_not_hide_tiny_unresolved_lot(self):
        old = date(2020, 1, 2)
        state = self.state([LotView(1, 1, old, old, D(89), D(8900)),
                            LotView(2, 1, old, old, D('1e-30'), D('1e-28'))])
        state = replace(state, pending_lots=frozenset({2}))
        with self.assertRaisesRegex(ValueError, 'unresolved lots'):
            HifoTaxCurve.from_snapshot(state, 1, D('.0005'))

    def test_bound_lot_segments_use_their_own_tax_holding_periods(self):
        old, recent = date(2020, 1, 2), date(2025, 1, 2)
        parts = (SegmentView(11, 1, recent, old, D('.4'), D(44)),
                 SegmentView(12, 2, recent, recent, D('.6'), D(66)))
        lot = LotView(1, 1, None, None, D(1), D(110), parts)
        curve = HifoTaxCurve.from_snapshot(self.state([lot]), 1, D('.0005'))
        expected = (.4 * 90 * .9995 - 44) * .238 + (.6 * 90 * .9995 - 66) * .408
        self.assertAlmostEqual(curve.value_and_slope(1)[0], expected)

    def test_cost_includes_forced_sale_and_cash_without_changing_snapshot(self):
        old = date(2020, 1, 2)
        state = self.state([LotView(1, 1, old, old, D(10), D(1000))])
        cost = TradeCosts(state, (1,))
        tax, fees, _, _ = cost.evaluate([0])
        self.assertAlmostEqual(tax, (900 * .9995 - 1000) * .238)
        self.assertAlmostEqual(fees, .45)
        self.assertEqual(state.cash, D(10000))
        self.assertEqual(state.lots[0].shares, D(10))
        tax, fees, _, _ = cost.evaluate([1])
        self.assertEqual(tax, 0)
        self.assertAlmostEqual(fees, 10000 * .0005)


if __name__ == "__main__":
    unittest.main()
