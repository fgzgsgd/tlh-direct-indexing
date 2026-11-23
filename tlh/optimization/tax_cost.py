"""Continuous HIFO sale costs; never post hypothetical trades to the account ledger."""

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from tlh.lot_relief import hifo
from tlh.shares import sum_shares
from tlh.strategy import CloseSnapshot
from tlh.tax import LT_RATE, ST_RATE, is_long_term


@dataclass(frozen=True)
class HifoTaxCurve:
    lot_ids: tuple[int, ...]
    ends: np.ndarray              # Cumulative shares sold in mandatory HIFO order.
    marginal_tax: np.ndarray      # Dollars of tax per additional share sold.
    cumulative_tax: np.ndarray

    @classmethod
    def from_snapshot(cls, state: CloseSnapshot, permno: int, fee_rate: Decimal):
        lots = state.lots_by_permno.get(permno, ())
        shares = sum_shares(lot.shares for lot in lots)
        if not shares:
            return cls((), np.empty(0), np.empty(0), np.empty(0))
        if state.trade_day is None:
            raise ValueError("HIFO tax estimation needs the scheduled sale date.")
        eligible = {lot.id: lot for lot in state.eligible_lots(permno)}
        if sum_shares(lot.shares for lot in eligible.values()) != shares:
            raise ValueError(f"{state.day} {permno}: cannot estimate tax with unresolved lots.")
        # Reuse the account's HIFO ordering, including its date/ID tie breakers.
        order = hifo(state, permno, shares)
        price = state.closes[permno]
        marginal, quantities = [], []
        for lot_id in order:
            lot = eligible[lot_id]
            tax = Decimal(0)
            for part in lot.segments or (lot,):
                rate = LT_RATE if is_long_term(part.holding_start, state.trade_day) else ST_RATE
                tax += (part.shares * price * (1 - fee_rate) - part.basis) * rate
            marginal.append(float(tax / lot.shares))
            quantities.append(float(lot.shares))
        slopes = np.asarray(marginal)
        return cls(tuple(order), np.cumsum(quantities), slopes, np.cumsum(slopes * quantities))

    def value_and_slope(self, shares_sold: float) -> tuple[float, float]:
        if not len(self.ends):
            if shares_sold > 1e-9:
                raise ValueError("Cannot sell an empty position.")
            return 0., 0.
        if shares_sold < -1e-9 or shares_sold > self.ends[-1] + 1e-8:
            raise ValueError("Continuous sale exceeds the current position.")
        sold = np.clip(shares_sold, 0., self.ends[-1])
        j = min(int(np.searchsorted(self.ends, sold, side="right")), len(self.ends) - 1)
        prior_shares = self.ends[j - 1] if j else 0.
        prior_tax = self.cumulative_tax[j - 1] if j else 0.
        return float(prior_tax + (sold - prior_shares) * self.marginal_tax[j]), float(self.marginal_tax[j])


class TradeCosts:
    """Evaluate dollar tax and fees for target weights, including fixed forced exits."""
    def __init__(self, state: CloseSnapshot, permnos, *, fee_rate=Decimal("0.0005")):
        self.permnos = tuple(permnos)
        self.nav = float(state.investable_nav)
        self.fee_rate = float(fee_rate)
        self.prices = np.array([float(state.closes[p]) for p in self.permnos])
        self.shares = np.array([float(sum_shares(lot.shares for lot in state.lots_by_permno.get(p, ())))
                                for p in self.permnos])
        self.current_weights = self.shares * self.prices / self.nav
        self.curves = [HifoTaxCurve.from_snapshot(state, p, fee_rate) for p in self.permnos]

    def evaluate(self, weights):
        quantity = np.asarray(weights) * self.nav / self.prices
        sold = np.maximum(self.shares - quantity, 0.)
        taxes, slopes = zip(*(curve.value_and_slope(s) for curve, s in zip(self.curves, sold)))
        tax = float(sum(taxes))
        fees = float(self.fee_rate * np.abs(quantity - self.shares) @ self.prices)
        # At a kink use the left derivative; the solver still evaluates the exact cost.
        tax_gradient = np.where(quantity <= self.shares, -np.asarray(slopes) * self.nav / self.prices, 0.)
        fee_gradient = self.fee_rate * self.nav * np.sign(quantity - self.shares)
        return tax, fees, tax_gradient, fee_gradient
