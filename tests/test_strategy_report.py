import unittest
from datetime import date
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.strategy_report import write_strategy_report
from tlh.backtest import AccountEngine, Session
from tlh.execution import Order


class StrategyReportTests(unittest.TestCase):
    def test_paired_tax_alpha_long_loss_and_later_wash(self):
        days = list(map(date.fromisoformat, ("2024-01-02", "2024-12-31", "2025-02-07", "2025-02-20")))
        before = date(2023, 12, 29)
        def account(harvest):
            engine = AccountEngine(D(1000), start=before, fee=lambda *_: D(0))
            for i, d in enumerate(days):
                px = {1: D(100 if i < 2 else 80)}
                orders = [Order(1, D(4), {})] if i == 0 else []
                if harvest and i == 2:
                    orders = [Order(1, D(-2), {1: D(2)})]
                if harvest and i == 3:
                    orders = [Order(1, D(1), {})]
                engine.advance(d, Session(px, px, ()), orders, {}, ())
            return engine.result

        strategy, baseline = account(True), account(False)
        benchmark = dict(zip(days, map(D, (".50", "0", "-.08", "0"))))
        with TemporaryDirectory() as tmp:
            daily, periods, refs = write_strategy_report(
                strategy, {"index": benchmark}, Path(tmp), initial_cash=D(1000), no_tlh=baseline,
                ex_ante_te={days[2]: D(".014")}, workbook=False)
        self.assertEqual(daily[0]["index_return"], 0)
        self.assertEqual((daily[2]["lt_gross_loss"], daily[2]["lt_recognized_loss"], daily[2]["wash_deferred"]),
                         (40, 20, 20))
        self.assertEqual((daily[3]["cumulative_lt_loss"], daily[3]["basis"], daily[3]["net_unrealized"]),
                         (20, 300, -60))
        total = periods[-1]
        self.assertEqual((periods[0]["tax_alpha"], total["tax_alpha"], total["aftertax_wealth"]),
                         (0, D(".00476"), D("924.76")))
        self.assertEqual((total["initial_turnover"], total["holdings"], refs[-1]["realized_te"]), (D(".2"), 1, 0))
        self.assertAlmostEqual(total["turnover"], D(240) / D(1840))
        self.assertEqual((periods[0]["ex_ante_te"], daily[3]["ex_ante_te"], total["ex_ante_as_of"],
                          total["ex_ante_te"], total["ex_ante_benchmark"]), (None, None, days[2], D(".014"), "index"))

    def test_fees_use_trade_day_nav_and_passive_redemption_excludes_turnover(self):
        dates = list(map(date.fromisoformat, ("2024-01-02", "2024-03-01", "2024-03-04")))
        engine = AccountEngine(D(100), start=date(2023, 12, 29), fee=lambda *_: D(1))
        from tlh.actions import CorporateAction
        for d, px, orders, events in zip(
            dates, [10, 8, 8], [[Order(1, D(9), {})], [Order(1, D(-3), {1: D(3)})], []],
            [(), (), (CorporateAction("redeem", 1, "CASH_ACQUISITION", D(8)),)]):
            market = {1: D(px)}
            engine.advance(d, Session(market, market, events), orders, {}, ())
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            benchmarks = {"index": dict.fromkeys(dates, D(0))}
            daily, periods, _ = write_strategy_report(engine.result, benchmarks, out, initial_cash=D(100), workbook=False)
            with self.assertRaisesRegex(ValueError, "missing total return"):
                write_strategy_report(engine.result, {"index": {}}, out, initial_cash=D(100), workbook=False)
        self.assertEqual(daily[0]["initial_turnover"], D(".45"))
        self.assertEqual(daily[1]["turnover"], D(24) / D(162))
        self.assertEqual((daily[2]["sales"], daily[2]["turnover"], daily[2]["holdings"]), (0, 0, 0))
        self.assertGreater(daily[2]["st_recognized_loss"], 0)
        total = periods[-1]
        self.assertEqual((total["buys"], total["sales"], total["fees"], total["initial_fees"]), (90, 24, 2, 1))
        self.assertEqual(total["fee_nav_ratio"], D(1) / 100 + D(1) / 81)
        self.assertIsNone(total["tax_alpha"])


if __name__ == "__main__":
    unittest.main()
