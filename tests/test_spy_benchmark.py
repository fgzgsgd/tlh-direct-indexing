import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.spy_benchmark import END, PRIOR, START, compare
from tlh.actions import CorporateAction
from tlh.backtest import INITIAL_CASH, Session, run
from tlh.lot_relief import ltfo
from tlh.market import nyse_trading_days
from tlh.spy_benchmark import SpyReplication, monthly_signals, read_monthly_holdings
from tlh.strategy import CloseSnapshot, LotView


class SpyBenchmarkTests(unittest.TestCase):
    def test_spy_partial_sale_honors_lot_policy_with_and_without_restricted_orders(self):
        signal, trade = date(2025, 1, 31), date(2025, 2, 3)
        old, new = date(2020, 1, 2), date(2025, 1, 2)
        d = Decimal
        lots = (LotView(1, 1, old, old, d(1), d(70)), LotView(2, 1, new, new, d(1), d(75)),
                LotView(3, 2, old, old, d(1), d(100)))
        state = CloseSnapshot(signal, d(700), d(1000), {1: d(100), 2: d(100)}, lots, frozenset(), trade_day=trade)
        weights = {signal: {1: d(".1"), 2: d(".2")}}
        for restrictions in ({}, {trade: {3: None}}, {trade: {2: None}}):
            with self.subTest(restrictions=restrictions):
                default = SpyReplication({trade: signal}, weights, restrictions)(state)
                selected = SpyReplication({trade: signal}, weights, restrictions, lot_relief=ltfo)(state)
                self.assertEqual(default.weights, selected.weights)
                self.assertEqual(default.lot_relief(state, 1, d(1)), {2: d(1)})
                self.assertEqual(selected.lot_relief(state, 1, d(1)), {1: d(1)})

    def test_conflicting_orders_keep_shares_and_preserve_the_other_targets(self):
        signal, trade = date(2025, 1, 31), date(2025, 2, 3)
        dates = {trade: signal}
        values = {signal: {1: Decimal(".1"), 2: Decimal(".1"), 3: Decimal(".8")}}
        strategy = SpyReplication(dates, values, {trade: {1: None, 2: None}})
        lots = tuple(LotView(i, i, signal, signal, Decimal(q), Decimal(q * 10))
                     for i, q in ((1, 20), (2, 10)))
        state = CloseSnapshot(signal, Decimal(700), Decimal(1000), {i: Decimal(10) for i in (1, 2, 3)},
                              lots, frozenset(), trade_day=trade)
        weights = strategy(state).active_weights(trade)
        self.assertEqual([int(weights[i] * 100) for i in (1, 2, 3)], [20, 10, 70])
        self.assertEqual(len(strategy.held_orders), 1)

    def test_calendar_and_cash_redistribution(self):
        schedule = monthly_signals((PRIOR, *nyse_trading_days()), START, END)
        self.assertEqual((len(schedule), schedule[START], schedule[date(2015, 2, 2)],
                          schedule[date(2025, 12, 1)]), (132, PRIOR, date(2015, 1, 30), date(2025, 11, 28)))
        two = {d: s for d, s in schedule.items() if d <= date(2015, 2, 2)}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "reports.csv"
            path.write_text("as_of,kind,permno,amount\n2014-12-31,equity,1,60\n"
                            "2014-12-31,equity,2,30\n2014-12-31,cash,,12\n2014-12-31,other_net_assets,,-2\n")
            with self.assertRaisesRegex(ValueError, "Missing 1.*2015-01-30"):
                read_monthly_holdings(path, two)
            with path.open("a") as f:
                f.write("2015-01-31,equity,2,49\n2015-01-31,equity,3,49\n2015-01-31,cash,,2\n")
            weights = read_monthly_holdings(path, two)
            self.assertEqual(sum(weights[PRIOR].values()), 1)
            self.assertAlmostEqual(weights[PRIOR][1], Decimal(2) / 3)
            self.assertEqual(weights[date(2015, 1, 30)], {2: Decimal(".5"), 3: Decimal(".5")})
            with path.open("a") as f:
                f.write("2015-01-29,equity,1,99\n")
            with self.assertRaisesRegex(ValueError, "expected one month-end"):
                read_monthly_holdings(path, two)

    def test_actual_engine_keeps_shares_and_settles_only_dividend_tax(self):
        signal, first, end = map(date.fromisoformat, ("2025-11-28", "2025-12-01", "2025-12-31"))
        d = Decimal
        prices = {1: d(100), 2: d(50)}
        last = {1: d(130), 2: d(55)}
        markets = {signal: Session(prices, prices, ()), first: Session(prices, prices, ()),
                   end: Session(last, last, (CorporateAction("div", 1, "CASH_DIVIDEND", d(1)),))}
        signals = monthly_signals(tuple(markets), first, end)
        result = run(sessions=tuple(markets), first_trade=first, end=end, days=markets,
                     delistings=(), delistings_known_through=end, pre_delisting=False,
                     strategy=SpyReplication(signals, {signal: {1: d(".5"), 2: d(".5")}}),
                     rebalance_days=signals, fee=lambda *_: d(0))
        self.assertEqual(result.history[-1].holdings, {1: 5000, 2: 10000})
        self.assertFalse(result.ledger.sales)
        with TemporaryDirectory() as tmp:
            summary = compare(result, {first: d(".1"), end: d(".205")}, Path(tmp))
        self.assertEqual(summary["end_after_tax_usd"], float(INITIAL_CASH + 205000 - 1190))
        self.assertEqual(summary["terminal_year_tax_usd"], 1190)
        self.assertAlmostEqual(summary["annualized_daily_tracking_error_bp"], 0)


if __name__ == "__main__":
    unittest.main()
