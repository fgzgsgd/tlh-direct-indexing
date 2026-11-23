import json
import unittest
from dataclasses import replace
from datetime import date
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory

from tlh.actions import CorporateAction
from tlh.backtest import AccountEngine, Session
from tlh.execution import Order, proportional_commission
from tlh.optimization.tax_timing import ConfirmedTaxTimingEngine, TaxAfterSales, read_tax_timing


class ConfirmedTaxTimingTests(unittest.TestCase):
    start, close, trade = date(2023, 12, 28), date(2023, 12, 29), date(2024, 1, 2)

    def rule(self):
        return TaxAfterSales(self.trade, 2023, D('2.38'), D(0), 'Single-day confirmed exception')

    def account(self, rules=None):
        cls = AccountEngine if rules is None else ConfirmedTaxTimingEngine
        kwargs = {} if rules is None else dict(tax_after_sales=rules)
        engine = cls(D(100), start=self.start, fee=proportional_commission(D(0)), **kwargs)
        lot = engine.result.ledger.buy(self.start, 1, D(1), D(100), D(0))
        prices = {1: D(100), 2: D(10), 3: D(90)}
        engine.advance(self.close, Session(prices, prices, (CorporateAction('div', 1, 'CASH_DIVIDEND', D(10)),)),
                       [Order(2, D(1), {})], {}, ())
        self.assertEqual(engine.result.ledger.cash, 0)
        return engine, lot, prices

    def test_sales_tax_buys_order_and_return_reconcile_without_changing_orders(self):
        engine, lot, prices = self.account((self.rule(),))
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

    def test_default_engine_and_unapproved_date_still_require_opening_cash(self):
        for rules in (None, (), (replace(self.rule(), trade_date=date(2024, 1, 3)),)):
            engine, lot, prices = self.account(rules)
            with self.assertRaisesRegex(ValueError, 'TAX_SETTLEMENT.*opening cash 0'):
                engine.advance(self.trade, Session(prices, prices, ()), [Order(1, D(-1), {lot.id: D(1)})], {}, ())
            self.assertEqual((engine.result.ledger.cash, engine.result.observed_through), (0, self.close))

    def test_changed_cash_amount_or_year_rejected_before_any_sale(self):
        for rule in (replace(self.rule(), opening_cash_usd=D(1)), replace(self.rule(), tax_usd=D(3)),
                     replace(self.rule(), tax_year=2022)):
            engine, lot, prices = self.account((rule,))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                engine.advance(self.trade, Session(prices, prices, ()), [Order(1, D(-1), {lot.id: D(1)})], {}, ())
            self.assertEqual(len(engine.result.ledger.sales), 0)
            self.assertEqual(engine.result.ledger.cash, 0)

    def test_insufficient_sale_proceeds_roll_back_and_retry_pays_once(self):
        engine, lot, prices = self.account((self.rule(),))
        orders = [Order(1, D(-1), {lot.id: D(1)})]
        low = {**prices, 1: D(1)}
        before = list(engine.result.ledger.entries)
        with self.assertRaisesRegex(ValueError, 'TAX_SETTLEMENT after original sales'):
            engine.advance(self.trade, Session(low, low, ()), orders, {}, ())
        self.assertEqual(engine.result.ledger.entries, before)
        self.assertEqual((engine.result.ledger.cash, engine.result.observed_through), (0, self.close))
        engine.advance(self.trade, Session(prices, prices, ()), orders, {}, ())
        self.assertEqual((engine.result.ledger.cash, len(engine.result.tax_settlements)), (D('97.62'), 1))

    def test_failed_buy_rolls_back_sales_and_tax_before_retry(self):
        engine, lot, prices = self.account((self.rule(),))
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

    def test_configuration_is_explicit_and_rejects_duplicates(self):
        self.assertEqual(read_tax_timing(None), ())
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'rules.json'
            path.write_text(json.dumps({'tax_after_sales': [self.rule().as_dict()]}))
            self.assertEqual(read_tax_timing(path), (self.rule(),))
            path.write_text(json.dumps({'tax_after_sales': [self.rule().as_dict()] * 2}))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_tax_timing(path)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'configuration'):
                read_tax_timing(path)


if __name__ == '__main__':
    unittest.main()
