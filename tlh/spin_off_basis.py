"""按除权后首个共同可用的 raw CLOSE 分摊免税分拆的原始 lot basis。"""

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
    """先试 ex-date，有缺价时整组取最近的共同交易日。显式记录 future peek 用到的日期。"""
    event_day_splits = defaultdict(lambda: Fraction(1))
    groups = defaultdict(list)
    # 真正的 split 会换算当天新收到的股数，同日股票股息只属于前收盘已经持有的股份。
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
            # 同一组所有证券使用同日行情，再换回 ex-date 收盘股数单位。
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
            for event, value in values:  # 逆序加回子股，得到每一步过账前的剩余市值。
                remaining += value
                allocations[event.reference] = SpinOffBasis(day, value / total, value / remaining)
            del pending[parent]
        if not pending:
            break
    return allocations
