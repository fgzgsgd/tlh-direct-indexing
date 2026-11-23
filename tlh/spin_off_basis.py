"""Allocate a tax-free spin-off's original lot basis using the first common raw CLOSE after the ex-date."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date
from fractions import Fraction
from typing import TYPE_CHECKING

from .actions import CorporateAction, SpinOff, SpinOffBasis

if TYPE_CHECKING:
    from .backtest import Session


def _is_split(event) -> bool:
    return isinstance(event, CorporateAction) and event.kind in {"SPLIT", "NONTAXABLE_STOCK_DIVIDEND"}


def _valuation_sessions(ex_date: date, market: Session, history: Mapping[date, Session]):
    yield ex_date, market
    for day in sorted(d for d in history if d > ex_date):
        yield day, history[day]


def close_allocations(ex_date: date, market: Session, history: Mapping[date, Session]) -> dict[str, SpinOffBasis]:
    """Try the ex-date first; if prices are missing, use the nearest common trading day for the whole group. Record the dates used for the future peek."""
    event_day_splits = defaultdict(lambda: Fraction(1))
    groups = defaultdict(list)
    # A true split converts shares received that day, while a same-day stock dividend belongs only to shares held at the previous close.
    received_splits = defaultdict(lambda: Fraction(1))
    for event in reversed(market.actions):
        if _is_split(event):
            event_day_splits[event.permno] *= Fraction(event.value)
            if event.kind == "SPLIT":
                received_splits[event.permno] *= Fraction(event.value)
        elif isinstance(event, SpinOff):
            ratio = event.terms.effective_ratio
            quantity = ratio * received_splits[event.terms.child] if ratio is not None else None
            groups[event.permno].append((event, quantity))

    pending = {}
    for parent, legs in groups.items():
        children = {event.terms.child for event, _ in legs}
        for event, _ in legs:
            if missing := event.joint_children - children:
                raise ValueError(f"{event.reference}: incomplete joint spin-off, missing child PERMNO {sorted(missing)}.")
        if all(q is not None and q > 0 for _, q in legs):
            pending[parent] = (legs, children | {parent})
    if not pending:
        return {}

    allocations = {}
    subsequent_splits = defaultdict(lambda: Fraction(1))
    for day, session in _valuation_sessions(ex_date, market, history):
        if day > ex_date:
            for event in session.actions:
                if _is_split(event):
                    subsequent_splits[event.permno] *= Fraction(event.value)
        for parent, (legs, securities) in list(pending.items()):
            # All securities in a group use same-day prices, then convert back to ex-date closing share units.
            closes = {}
            for p in securities:
                price = session.regular_close(p)
                if price is not None:
                    closes[p] = Fraction(price) * subsequent_splits[p]
            if len(closes) < len(securities):
                continue
            remaining = event_day_splits[parent] * closes[parent]
            values = [(event, q * closes[event.terms.child]) for event, q in legs]
            total = remaining + sum(value for _, value in values)
            for event, value in values:  # Add children back in reverse to get the remaining value before each posting step.
                remaining += value
                allocations[event.reference] = SpinOffBasis(day, value / total, value / remaining)
            del pending[parent]
        if not pending:
            break
    return allocations
