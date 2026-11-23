"""The strategy reads a previous-close snapshot and returns target weights and the lot selection for sell orders."""

from collections.abc import Callable, Container, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal
from functools import cached_property
from fractions import Fraction
from types import MappingProxyType

from .ledger import Ledger
from .shares import sum_shares


@dataclass(frozen=True, slots=True)
class SegmentView:
    id: int                   # For reconciling tax records. Orders use the outer LotView.id.
    source_lot_id: int
    acquired: date
    holding_start: date
    shares: Decimal
    basis: Decimal | None


@dataclass(frozen=True, slots=True)
class LotView:
    id: int
    permno: int
    acquired: date | None
    holding_start: date | None
    shares: Decimal
    basis: Decimal | None       # Remaining total adjusted basis of the lot.
    segments: tuple[SegmentView, ...] = ()


@dataclass(frozen=True, slots=True)
class WashLotView:
    id: int
    permno: int
    acquired: date
    holding_start: date
    shares: Decimal
    basis: Decimal | None
    origin: int | None
    wash_used: bool
    exact_shares: Fraction | None
    bound_lot_id: int | None
    source_lot_id: int | None
    prior_periods: tuple


@dataclass(frozen=True, slots=True)
class WashLossView:
    sale_day: date
    sold_lot: int
    permno: int
    origin: int | None
    shares: Fraction
    remaining: Decimal
    holding_start: date
    periods: tuple


@dataclass(frozen=True)
class CloseSnapshot:
    day: date                  # Signal day whose valuation is complete.
    cash: Decimal
    nav: Decimal | None
    closes: Mapping[int, Decimal]
    lots: tuple[LotView, ...]
    pending_lots: frozenset[int]
    trade_day: date | None = None  # Known execution date, without that day's prices.
    blocked: frozenset[int] = frozenset()  # PERMNOs banned from purchase in the current pre-delisting exit window.
    carried_close_dates: Mapping[int, date] = field(default_factory=lambda: MappingProxyType({}))
    scheduled_tax_flow: Decimal = Decimal(0)  # Known amount to be credited (or debited) before the open on the trade date.
    prelisting_marks: Mapping[int, Decimal] = field(default_factory=lambda: MappingProxyType({}))
    wash_lots: tuple[WashLotView, ...] = ()
    wash_losses: tuple[WashLossView, ...] = ()
    wash_state_available: bool = False

    @property
    def investable_nav(self) -> Decimal | None:
        return self.nav + self.scheduled_tax_flow if self.nav is not None else None

    @property
    def available_cash(self) -> Decimal:
        return self.cash + self.scheduled_tax_flow

    @cached_property
    def lots_by_permno(self) -> Mapping[int, tuple[LotView, ...]]:
        grouped: dict[int, list[LotView]] = {}
        for lot in self.lots:
            grouped.setdefault(lot.permno, []).append(lot)
        return MappingProxyType({p: tuple(lots) for p, lots in grouped.items()})

    def eligible_lots(self, permno: int) -> tuple[LotView, ...]:
        return tuple(lot for lot in self.lots_by_permno.get(permno, ())
                     if lot.basis is not None and lot.id not in self.pending_lots)

    @classmethod
    def from_ledger(cls, day: date, book: Ledger, closes: Mapping[int, Decimal],
                    nav: Decimal | None, pending_lots: set[int], *, trade_day: date | None = None,
                    blocked: frozenset[int] = frozenset(), carried_close_dates: Mapping[int, date] | None = None,
                    scheduled_tax_flow: Decimal = Decimal(0)
                    ) -> "CloseSnapshot":
        lots = []
        blocked_lots = set(pending_lots)
        for i, parts in book.trading_lots().items():
            lot = parts[0]
            if lot.bound_lot_id is None:
                lots.append(LotView(i, lot.permno, lot.acquired, lot.holding_start, lot.shares, lot.basis))
                continue
            segments = tuple(SegmentView(part.id, part.source_id, part.acquired, part.holding_start, part.shares, part.basis)
                             for part in parts)
            basis = None if any(part.basis is None for part in parts) else sum_shares(part.basis for part in parts)
            lots.append(LotView(i, lot.permno, None, None, sum_shares(part.shares for part in parts), basis, segments))
            if any(part.id in pending_lots for part in parts):
                blocked_lots.add(i)
        interim = {p: grant.price for p, grant in book.prelistings.items() if day < grant.first_trade}
        return cls(day, book.cash, nav, MappingProxyType({**closes, **interim}), tuple(lots), frozenset(blocked_lots),
                   trade_day, blocked, MappingProxyType(dict(carried_close_dates or {})), scheduled_tax_flow,
                   MappingProxyType(interim),
                   tuple(WashLotView(l.id,l.permno,l.acquired,l.holding_start,l.shares,l.basis,
                                     l.origin,l.wash_used,l.exact_shares,l.bound_lot_id,l.source_lot_id,
                                     l.prior_periods) for l in book.lots.values()),
                   tuple(WashLossView(x.sale.day,x.sale.lot_id,x.sale.permno,x.origin,x.shares,
                                      x.remaining,x.sale.holding_start,x.periods)
                         for losses in book.open_losses.values() for x in losses), True)


# Arguments are the read-only snapshot, the security and the fixed net sell quantity; returns {lot_id: shares}.
LotRelief = Callable[[CloseSnapshot, int, Decimal], Mapping[int, Decimal]]


@dataclass(frozen=True)
class Target:
    weights: dict[int, Decimal]
    rebalance: bool = True
    sell_lots: dict[int, dict[int, Decimal]] = field(default_factory=dict)
    lot_relief: LotRelief | None = None

    def active_weights(self, day: date) -> dict[int, Decimal]:
        active = {}
        for p, value in self.weights.items():
            weight = Decimal(value)
            if not weight.is_finite() or not 0 <= weight <= 1:
                raise ValueError(f"{day} {p}: invalid long-only target weight {weight}.")
            if weight:
                active[p] = weight
        if sum(active.values()) > 1:
            raise ValueError(f"{day}: total target weight exceeds 1.")
        return active


Strategy = Callable[[CloseSnapshot], Target | None]


def index_strategy(strategy: Strategy, members_at_signal: Callable[[CloseSnapshot], Container[int]]) -> Strategy:
    """Keep only constituent targets at rebalance. The strategy supplies the membership list, which must be known at the signal close."""
    def decide(state: CloseSnapshot) -> Target | None:
        target = strategy(state)
        if target is None or not target.rebalance:
            return None
        members = members_at_signal(state)
        # The ledger sells all holdings not in the targets and writes the lots actually delivered to the replay.
        return replace(target, weights={p: w for p, w in target.weights.items() if p in members},
                       sell_lots={p: lots for p, lots in target.sell_lots.items() if p in members})

    return decide
