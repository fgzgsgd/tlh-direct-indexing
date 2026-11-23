"""Opt-in, exact-date tax-payment exceptions for the isolated research account."""

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from tlh.backtest import AccountEngine
from tlh.ledger import Entry
from tlh.tax import ZERO


@dataclass(frozen=True)
class TaxAfterSales:
    trade_date: date
    tax_year: int
    tax_usd: Decimal
    opening_cash_usd: Decimal
    reason: str

    def __post_init__(self):
        if (not self.tax_usd.is_finite() or self.tax_usd <= 0
                or not self.opening_cash_usd.is_finite() or self.opening_cash_usd < 0
                or self.tax_year >= self.trade_date.year or not self.reason.strip()):
            raise ValueError("A tax-timing exception requires a prior tax year, positive tax, nonnegative cash and reason.")

    def check(self, day, due, cash):
        if (day != self.trade_date or cash != self.opening_cash_usd or len(due) != 1
                or due[0].tax_year != self.tax_year or due[0].net_tax != self.tax_usd
                or due[0].cash_flow != -self.tax_usd):
            raise ValueError(f"{day}: confirmed tax-timing exception does not match the tax year, amount or opening cash.")

    def as_dict(self):
        return dict(trade_date=str(self.trade_date), tax_year=self.tax_year, tax_usd=str(self.tax_usd),
                    opening_cash_usd=str(self.opening_cash_usd), reason=self.reason)


def read_tax_timing(path):
    if path is None:
        return ()
    rows = json.loads(path.read_text())
    if set(rows) != {"tax_after_sales"} or not rows["tax_after_sales"]:
        raise ValueError("Tax-timing configuration must contain nonempty tax_after_sales records.")
    result = tuple(TaxAfterSales(date.fromisoformat(r["trade_date"]), int(r["tax_year"]),
                                Decimal(r["tax_usd"]), Decimal(r["opening_cash_usd"]), r["reason"])
                   for r in rows["tax_after_sales"])
    if len({r.trade_date for r in result}) != len(result):
        raise ValueError("Duplicate confirmed tax-timing date.")
    return result


class ConfirmedTaxTimingEngine(AccountEngine):
    """Keep the ordinary engine unchanged; defer only explicitly matched debits.

The original advance transaction still encloses actions, sales, tax, buys and
valuation. Any failure rolls the entire day back. Snapshot budgets already
include scheduled tax, so no target, share count, or lot selection is changed.
"""

    def __init__(self, *args, tax_after_sales=(), **kwargs):
        super().__init__(*args, **kwargs)
        self._tax_timing_rules = {r.trade_date: r for r in tax_after_sales}
        if len(self._tax_timing_rules) != len(tax_after_sales):
            raise ValueError("Duplicate confirmed tax-timing date.")
        self._deferred_tax = None
        self._deferred_tax_posted = False

    def _opening_tax(self, day):
        if self._deferred_tax is not None and day == self._deferred_tax[0]:
            return [], self._deferred_tax[2]
        return super()._opening_tax(day)

    def advance(self, day, market, orders, forced_exits, expired):
        rule = self._tax_timing_rules.get(day)
        if rule is None:
            return super().advance(day, market, orders, forced_exits, expired)
        due, review = super()._opening_tax(day)
        rule.check(day, due, self.result.ledger.cash)
        if not any(o.quantity < 0 for o in orders):
            raise ValueError(f"{day}: confirmed tax timing requires the original sale orders.")
        self._deferred_tax = day, due, review
        self._deferred_tax_posted = False
        try:
            super().advance(day, market, orders, forced_exits, expired)
            self.result.tax_settlements.extend(due)
        finally:
            self._deferred_tax = None
            self._deferred_tax_posted = False

    def _execute(self, day, market, orders):
        if self._deferred_tax is None:
            return super()._execute(day, market, orders)
        sales = [o for o in orders if o.quantity < 0]
        buys = [o for o in orders if o.quantity > 0]
        super()._execute(day, market, sales)
        due = self._deferred_tax[1]
        flow = sum((s.cash_flow for s in due), ZERO)
        book = self.result.ledger
        if flow < -book.cash:
            raise ValueError(f"{day} TAX_SETTLEMENT after original sales: required {-flow}, available cash {book.cash}.")
        for settlement in due:
            book.post_cash(settlement.cash_flow)
            book.entries.append(Entry(day, "TAX_SETTLEMENT", None, ZERO, settlement.cash_flow,
                                      str(settlement.tax_year)))
        self._deferred_tax_posted = True
        super()._execute(day, market, buys)

    def _mark_close(self, day, market, marks, tax_flow=ZERO):
        if self._deferred_tax is not None:
            if not self._deferred_tax_posted:
                raise ValueError(f"{day}: deferred tax must be paid before valuation.")
            tax_flow += sum((s.cash_flow for s in self._deferred_tax[1]), ZERO)
        return super()._mark_close(day, market, marks, tax_flow)
