import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from tlh.actions import CaseReview, CorporateAction
from tlh.backtest import AccountEngine, Session, run
from tlh.execution import Order, proportional_commission
from tlh.positions import write_weights
from tlh.strategy import Target
from tlh.tax import Scenario


D = Decimal
day = date.fromisoformat
NO_FEE = proportional_commission(D(0))


def market(prices, *events):
    return Session(prices, prices, events)


class JanuarySettlementTests(unittest.TestCase):
    def test_tax_with_incomplete_basis_names_the_event_before_posting_any_cash(self):
        dec, close, jan = map(day, ("2023-12-28", "2023-12-29", "2024-01-02"))
        engine = AccountEngine(D(200), start=dec, fee=NO_FEE)
        book = engine.result.ledger
        lot = book.buy(dec, 1, D(1), D(100), D(0))
        pending = CaseReview("missing-terms", 1, "BASIS_PENDING", "Basis needs review", lambda book, day: (lot.id,))
        engine.advance(close, market({1: D(100)}, pending), (), {}, ())
        with self.assertRaisesRegex(ValueError, "TAX_SETTLEMENT for 2023.*missing-terms"):
            engine.advance(jan, market({1: D(100)}), (), {}, ())
        self.assertEqual((book.cash, engine.result.observed_through, engine.result.tax_settlements), (100, close, []))

    def test_january_rebalance_uses_refund_and_replays_each_scenario(self):
        signal, first, sale, dec, jan, up = dates = list(map(day, (
            "2023-09-29", "2023-10-02", "2023-11-15", "2023-12-29", "2024-01-02", "2024-01-03")))
        prices = {1: D(400), 2: D(600), 3: D(510)}
        sold = {1: D(500), 2: D(440), 3: D(510)}
        days = {d: market(prices if d <= first else sold) for d in dates}
        days[up] = market({**sold, 3: D(520)})
        base = dict(sessions=dates, first_trade=first, end=up, days=days, delistings=(),
                    delistings_known_through=up, pre_delisting=False, initial_cash=D(1060), fee=NO_FEE)
        for scenario, flow, shares in ((Scenario.EXTERNAL_GAINS, D("24.48"), 2), (Scenario.LIMITED_USE, D(0), 1)):
            def strategy(snapshot):
                if snapshot.trade_day == first:
                    return Target({1: D(".4"), 2: D(".6")})
                if snapshot.trade_day == sale:
                    return Target({})
                self.assertEqual((snapshot.cash, snapshot.nav, snapshot.scheduled_tax_flow), (1000, 1000, flow))
                self.assertEqual((snapshot.investable_nav, snapshot.available_cash), (1000 + flow, 1000 + flow))
                return Target({3: D(1)})

            r = run(**base, strategy=strategy, rebalance_days=[first, sale, jan], tax_scenario=scenario)
            last_dec, first_jan, last = r.history[-3:]
            self.assertEqual((last_dec.nav, first_jan.nav, first_jan.tax_cash_flow), (1000, 1000 + flow, flow))
            self.assertEqual((first_jan.holdings, first_jan.cash, first_jan.daily_return),
                             ({3: shares}, 1000 + flow - 510 * shares, 0))
            self.assertEqual(r.pre_tax_twr(jan, up), (last.nav / first_jan.nav) - 1)
            prior, current = r.annual_wealth()
            self.assertEqual((prior.after_tax, current.after_tax), (1000 + flow, last.nav))
            if scenario == Scenario.LIMITED_USE:
                self.assertEqual(prior.tax.st_carryforward, 60)
            other = Scenario.LIMITED_USE if flow else Scenario.EXTERNAL_GAINS
            with self.assertRaisesRegex(ValueError, "separate backtest"):
                r.annual_wealth(other)
            with TemporaryDirectory() as folder:
                path = Path(folder) / "weights.csv"
                write_weights(path, r.targets)
                replay = run(**base, weights_file=path, tax_scenario=scenario)
            self.assertEqual((replay.history, replay.tax_settlements), (r.history, r.tax_settlements))

    def test_dividend_tax_debit_precedes_trades_and_rolls_back_with_failed_day(self):
        dec, ex, jan = map(day, ("2023-12-28", "2023-12-29", "2024-01-02"))
        engine = AccountEngine(D(110), start=dec, fee=NO_FEE)
        book = engine.result.ledger
        book.buy(dec, 1, D(1), D(100), D(0))
        engine.advance(ex, market({1: D(100)}, CorporateAction("div", 1, "CASH_DIVIDEND", D(10))), (), {}, ())
        # Only 17.62 is left after the transfer. When an 18 buy order fails, the tax payment rolls back with the whole day.
        prices = market({1: D(100), 2: D(18)})
        with self.assertRaisesRegex(ValueError, "available cash"):
            engine.advance(jan, prices, [Order(2, 1, {})], {}, ())
        self.assertEqual((book.cash, engine.result.tax_settlements, engine.result.observed_through), (20, [], ex))
        engine.advance(jan, prices, (), {}, ())
        mark = engine.result.history[-1]
        self.assertEqual((mark.cash, mark.tax_cash_flow, mark.daily_return), (D("17.62"), D("-2.38"), 0))
        self.assertEqual(engine.result.liquidated_wealth().after_tax, D("117.62"))

        short = AccountEngine(D(100), start=dec, fee=NO_FEE)
        pos = short.result.ledger.buy(dec, 1, D(1), D(100), D(0))
        short.advance(ex, market({1: D(100), 2: D(10)}, CorporateAction("div", 1, "CASH_DIVIDEND", D(10))),
                      [Order(2, 1, {})], {}, ())
        with self.assertRaisesRegex(ValueError, "TAX_SETTLEMENT.*2023.*2.38.*opening cash 0"):
            short.advance(jan, prices, [Order(1, -1, {pos.id: 1})], {}, ())
        self.assertEqual((short.result.ledger.cash, len(short.result.ledger.sales)), (0, 0))

    def test_december_wash_window_posts_only_the_final_difference(self):
        dec, sold, jan, buy, last_window, true_up, later = map(day, (
            "2023-10-02", "2023-12-29", "2024-01-02", "2024-01-03", "2024-01-26", "2024-01-29", "2024-01-30"))
        for short_of_cash in (False, True):
            engine = AccountEngine(D(1000), start=dec, fee=NO_FEE)
            book = engine.result.ledger
            lot = book.buy(dec, 1, D(2), D(50), D(0))
            prices = market({1: D(20), 2: D(100)})
            engine.advance(sold, prices, [Order(1, -2, {lot.id: 2})], {}, ())
            engine.advance(jan, prices, (), {}, ())
            self.assertEqual((book.cash, engine.result.history[-1].daily_return), (D("964.48"), 0))
            orders = [Order(1, 1, {})]
            if short_of_cash:
                # 4.48 is left, which cannot repay 12.24. Stock planned for sale the next day cannot be used to fund it.
                orders.append(Order(2, 1, {}))
                prices = market({1: D(20), 2: D(940)})
            engine.advance(buy, prices, orders, {}, ())
            engine.advance(last_window, prices, (), {}, ())
            self.assertEqual(len(engine.result.tax_settlements), 1)
            if short_of_cash:
                with self.assertRaisesRegex(ValueError, "12.24.*opening cash 4.48"):
                    engine.advance(true_up, prices, (), {}, ())
                self.assertEqual(engine.result.observed_through, last_window)
                continue
            engine.advance(true_up, prices, (), {}, ())
            engine.advance(later, prices, (), {}, ())
            self.assertEqual([(s.day, s.cash_flow, s.final) for s in engine.result.tax_settlements],
                             [(jan, D("24.48"), False), (true_up, D("-12.24"), True)])
            self.assertEqual(engine.result.pre_tax_twr(jan, later), 0)
            prior, current = engine.result.annual_wealth()
            self.assertEqual((prior.tax.net_tax, prior.after_tax, current.after_tax), (D("-12.24"), D("952.24"), D("952.24")))
