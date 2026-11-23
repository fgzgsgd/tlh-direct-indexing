"""Every tax day uses original sales, scheduled settlement, then original buys."""

import unittest
from datetime import date, timedelta
from decimal import Decimal as D

from tlh.actions import CorporateAction
from tlh.uniform_account import AccountEngine, Session
from tlh.execution import Order, proportional_commission
from tlh.market import Delisting


class TaxOrderTests(unittest.TestCase):
    start, close, trade = date(2023, 12, 28), date(2023, 12, 29), date(2024, 1, 2)

    def account(self, initial_cash=D(100)):
        engine = AccountEngine(initial_cash, start=self.start, fee=proportional_commission(D(0)))
        lot = engine.result.ledger.buy(self.start, 1, D(1), D(100), D(0))
        prices = {1: D(100), 2: D(10), 3: D(90)}
        engine.advance(self.close, Session(prices, prices, (CorporateAction('div', 1, 'CASH_DIVIDEND', D(10)),)),
                       [Order(2, D(1), {})], {}, ())
        self.assertEqual(engine.result.ledger.cash, initial_cash - D(100))
        return engine, lot, prices

    def test_sales_tax_buys_order_and_return_reconcile_without_changing_orders(self):
        engine, lot, prices = self.account()
        orders = [Order(1, D(-1), {lot.id: D(1)}), Order(3, D(1), {})]
        snapshot = engine.close_snapshot(self.close, prices, trade_day=self.trade)
        self.assertEqual((snapshot.investable_nav, snapshot.available_cash), (D('107.62'), D('-2.38')))
        engine.advance(self.trade, Session(prices, prices, ()), orders, {}, ())
        result = engine.result
        entries = [e for e in result.ledger.entries if e.day == self.trade]
        self.assertEqual([e.kind for e in entries], ['TRADE', 'TAX_SETTLEMENT', 'BUY'])
        self.assertEqual([(e.permno, e.shares) for e in entries if e.kind != 'TAX_SETTLEMENT'], [(1, -1), (3, 1)])
        self.assertEqual(result.ledger.cash, D('7.62'))
        mark = result.history[-1]
        self.assertEqual((mark.nav, mark.daily_return, mark.tax_cash_flow), (D('107.62'), 0, D('-2.38')))
        self.assertEqual(len(result.tax_settlements), 1)
        engine.advance(date(2024, 1, 3), Session(prices, prices, ()), (), {}, ())
        self.assertEqual((result.ledger.cash, len(result.tax_settlements)), (D('7.62'), 1))

    def test_cash_funded_tax_is_paid_before_buys_without_any_sale(self):
        engine, _, prices = self.account(D(110))
        prices = {**prices, 3: D(7)}
        engine.advance(self.trade, Session(prices, prices, ()), [Order(3, D(1), {})], {}, ())
        entries = [e for e in engine.result.ledger.entries if e.day == self.trade]
        self.assertEqual([e.kind for e in entries], ['TAX_SETTLEMENT', 'BUY'])
        self.assertEqual(engine.result.ledger.cash, D('.62'))

    def test_no_sale_and_insufficient_cash_stops_without_adding_an_order(self):
        engine, _, prices = self.account()
        before = list(engine.result.ledger.entries)
        with self.assertRaisesRegex(ValueError, 'TAX_SETTLEMENT.*after original sales.*available cash 0'):
            engine.advance(self.trade, Session(prices, prices, ()), (), {}, ())
        self.assertEqual(engine.result.ledger.entries, before)
        self.assertEqual(engine.result.observed_through, self.close)
        self.assertEqual(engine.result.ledger.holdings(), {1: 1, 2: 1})

    def test_insufficient_sale_proceeds_roll_back_and_retry_pays_once(self):
        engine, lot, prices = self.account()
        orders = [Order(1, D(-1), {lot.id: D(1)})]
        low = {**prices, 1: D(1)}
        before = list(engine.result.ledger.entries)
        with self.assertRaisesRegex(ValueError, 'TAX_SETTLEMENT.*after original sales'):
            engine.advance(self.trade, Session(low, low, ()), orders, {}, ())
        self.assertEqual(engine.result.ledger.entries, before)
        self.assertEqual((engine.result.ledger.cash, engine.result.observed_through), (0, self.close))
        engine.advance(self.trade, Session(prices, prices, ()), orders, {}, ())
        self.assertEqual((engine.result.ledger.cash, len(engine.result.tax_settlements)), (D('97.62'), 1))

    def test_failed_buy_rolls_back_sales_and_tax_before_retry(self):
        engine, lot, prices = self.account()
        orders = [Order(1, D(-1), {lot.id: D(1)}), Order(3, D(1), {})]
        high = {**prices, 3: D(100)}
        before = list(engine.result.ledger.entries)
        with self.assertRaisesRegex(ValueError, 'available cash 97.62'):
            engine.advance(self.trade, Session(high, high, ()), orders, {}, ())
        self.assertEqual(engine.result.ledger.entries, before)
        self.assertEqual((engine.result.ledger.cash, engine.result.observed_through, engine.result.tax_settlements),
                         (0, self.close, []))
        engine.advance(self.trade, Session(prices, prices, ()), orders, {}, ())
        self.assertEqual((engine.result.ledger.cash, len(engine.result.tax_settlements)), (D('7.62'), 1))

    def test_split_cancelled_sale_cannot_fund_tax_and_whole_day_rolls_back(self):
        engine, lot, prices = self.account()
        prices = {**prices, 1: D(50)}
        split = CorporateAction('split', 1, 'SPLIT', D(2))
        with self.assertRaisesRegex(ValueError, 'TAX_SETTLEMENT.*available cash 0'):
            engine.advance(self.trade, Session(prices, prices, (split,)),
                           [Order(1, D(-1), {lot.id: D(1)})], {}, ())
        self.assertEqual(engine.result.ledger.holdings(), {1: 1, 2: 1})
        self.assertEqual(engine.result.ledger.cash, 0)
        self.assertFalse(engine.result.exceptions)
        self.assertFalse(engine.result.tax_settlements)

    def test_original_forced_exit_proceeds_can_fund_tax(self):
        engine, _, prices = self.account()
        exits = {1: [Delisting(1, self.trade + timedelta(days=1), 'MER')]}
        engine.advance(self.trade, Session(prices, prices, ()), (), exits, ())
        self.assertEqual(engine.result.ledger.cash, D('97.62'))
        self.assertEqual(engine.result.ledger.holdings(), {2: 1})
        entries = [e for e in engine.result.ledger.entries if e.day == self.trade]
        self.assertEqual([e.kind for e in entries], ['PRE_DELISTING_EXIT', 'TAX_SETTLEMENT'])

    def test_same_day_cash_action_precedes_tax_without_new_sales(self):
        engine, _, prices = self.account()
        event = CorporateAction('new-year-div', 1, 'CASH_DIVIDEND', D(5))
        engine.advance(self.trade, Session(prices, prices, (event,)), (), {}, ())
        self.assertEqual(engine.result.ledger.cash, D('2.62'))
        self.assertFalse(engine.result.ledger.sales)
        entries = [e for e in engine.result.ledger.entries if e.day == self.trade]
        self.assertEqual([e.kind for e in entries], ['CASH_DIVIDEND', 'TAX_SETTLEMENT'])


if __name__ == '__main__':
    unittest.main()
