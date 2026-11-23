"""All split-day discretionary buys and sales are cancelled, without a size threshold."""

import unittest
from datetime import date, timedelta
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory

from tlh.actions import CorporateAction
from tlh.backtest import AccountEngine, Session, run
from tlh.execution import Order, proportional_commission
from tlh.positions import write_weights
from tlh.strategy import Target


class SplitOrderTests(unittest.TestCase):
    def test_buy_is_skipped_for_split_and_reverse_split_and_replay(self):
        signal, first, ex = date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 2)
        for factor, price, held, cash in [(D(2), D(5), 40, 390), (D('.125'), D(80), 2, 430)]:
            with self.subTest(factor=factor):
                before, after = {10: D(10), 20: D(10)}, {10: price, 20: D(10)}
                markets = {signal: Session(before, before, ()), first: Session(before, before, ()),
                           ex: Session(after, after, (CorporateAction('split', 10, 'SPLIT', factor),))}
                weights = {first: {10: D('.20'), 20: D('.40')}, ex: {10: D('.22'), 20: D('.41')}}
                inputs = dict(sessions=[signal, first, ex], first_trade=first, end=ex, days=markets,
                              delistings=[], delistings_known_through=ex, initial_cash=D(1000),
                              fee=proportional_commission(D(0)), pre_delisting=False)
                result = run(strategy=lambda s: Target(weights[s.trade_day]), rebalance_days=weights, **inputs)
                self.assertEqual(result.exceptions, [(ex, 10, 'SKIPPED_SPLIT_BUY 2/20 pre-split shares')])
                self.assertEqual(result.plans[first][0].quantity, 2)  # Original intention remains auditable.
                self.assertEqual(result.history[-1].holdings, {10: held, 20: 41})
                self.assertEqual((result.history[-1].cash, result.history[-1].nav), (cash, 1000))
                self.assertFalse([e for e in result.ledger.entries if e.day == ex and e.permno == 10 and e.kind == 'BUY'])
                with TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'weights.csv'
                    write_weights(path, result.targets)
                    replay = run(weights_file=path, **inputs)
                self.assertEqual((replay.history, replay.exceptions), (result.history, result.exceptions))

    def test_large_buy_zero_holdings_and_same_security_acquisition_are_cancelled(self):
        ex = date(2024, 3, 4)
        split = CorporateAction('split', 10, 'SPLIT', D(2))
        acquisition = CorporateAction('buyout', 10, 'CASH_ACQUISITION', D(5))
        for held, events in [(19, [split]), (0, [split]), (20, [split, acquisition])]:
            with self.subTest(held=held, events=events):
                engine = AccountEngine(D(1000), start=ex-timedelta(days=1))
                if held:
                    engine.result.ledger.buy(date(2023, 1, 3), 10, D(held), D(10), D(0))
                cash = engine.result.ledger.cash
                engine.advance(ex, Session({10: D(5)}, {10: D(5)}, events), [Order(10, D(2), {})], {}, ())
                expected = {} if acquisition in events or not held else {10: held * 2}
                self.assertEqual(engine.result.ledger.holdings(), expected)
                self.assertEqual(engine.result.ledger.cash, cash + (D(held * 10) if acquisition in events else 0))
                self.assertEqual(engine.result.exceptions, [(ex, 10, f'SKIPPED_SPLIT_BUY 2/{held} pre-split shares')])
                self.assertFalse([e for e in engine.result.ledger.entries if e.day == ex and e.kind == 'BUY'])

    def test_both_sides_ignore_trade_size_for_split_and_reverse_split(self):
        ex = date(2024, 3, 4)
        for held, ratio in [(20, D(2)), (28, D(3)), (20, D('.125'))]:
            for quantity in (D(2), D(-2), D(100), D(-held)):
                with self.subTest(held=held, ratio=ratio, quantity=quantity):
                    engine = AccountEngine(D(1000), start=ex-timedelta(days=1))
                    engine.result.ledger.buy(date(2023, 1, 3), 10, D(held), D(12), D(0))
                    lot_id = next(iter(engine.result.ledger.lots))
                    lots = {lot_id: -quantity} if quantity < 0 else {}
                    cash = engine.result.ledger.cash
                    price = D(12) / ratio
                    event = CorporateAction('split', 10, 'SPLIT', ratio)
                    engine.advance(ex, Session({10: price}, {10: price}, [event]),
                                   [Order(10, quantity, lots)], {}, ())
                    side = 'BUY' if quantity > 0 else 'SALE'
                    self.assertEqual(engine.result.exceptions,
                                     [(ex, 10, f'SKIPPED_SPLIT_{side} {abs(quantity)}/{held} pre-split shares')])
                    self.assertEqual(engine.result.ledger.holdings(), {10: int(held * ratio)})
                    self.assertEqual(engine.result.ledger.cash, cash + (held * ratio % 1) * price)
                    self.assertFalse([e for e in engine.result.ledger.entries
                                      if e.day == ex and e.kind in ('BUY', 'TRADE')])

    def test_another_security_conflict_rolls_back_the_day_and_skip_record(self):
        ex = date(2024, 3, 4)
        engine = AccountEngine(D(1000), start=ex-timedelta(days=1))
        engine.result.ledger.buy(date(2023, 1, 3), 10, D(20), D(10), D(0))
        cash = engine.result.ledger.cash
        events = [CorporateAction('split', 10, 'SPLIT', D(2)),
                  CorporateAction('buyout', 20, 'CASH_ACQUISITION', D(5))]
        with self.assertRaisesRegex(ValueError, 'PERMNO 20 BUY 1 shares.*cash acquisition'):
            engine.advance(ex, Session({10: D(5), 20: D(5)}, {10: D(5), 20: D(5)}, events),
                           [Order(10, D(100), {}), Order(20, D(1), {})], {}, ())
        self.assertEqual(engine.result.ledger.holdings(), {10: 20})
        self.assertEqual(engine.result.ledger.cash, cash)
        self.assertFalse(engine.result.exceptions)

    def test_cancellation_does_not_raise_cash_by_resizing_unrelated_buys(self):
        ex = date(2024, 3, 4)
        engine = AccountEngine(D(100), start=ex-timedelta(days=1))
        lot = engine.result.ledger.buy(date(2023, 1, 3), 10, D(10), D(10), D(0))
        split = CorporateAction('split', 10, 'SPLIT', D(2))
        with self.assertRaisesRegex(ValueError, 'available cash'):
            engine.advance(ex, Session({10: D(5), 20: D(10)}, {10: D(5), 20: D(10)}, [split]),
                           [Order(10, D(-10), {lot.id: D(10)}), Order(20, D(1), {})], {}, ())
        self.assertEqual(engine.result.ledger.holdings(), {10: 10})
        self.assertEqual(engine.result.ledger.cash, 0)
        self.assertFalse(engine.result.exceptions)


if __name__ == '__main__':
    unittest.main()
