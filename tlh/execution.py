"""Lock net orders at the previous close; post only at actual prices on the trade date."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .ledger import Ledger
from .shares import fraction, sum_shares
from .strategy import CloseSnapshot, LotRelief
from .tax import ZERO


Fee = Callable[[date, int, Decimal, Decimal], Decimal]
Sizer = Callable[[Decimal, Decimal, Decimal], Decimal]


def proportional_commission(rate: Decimal) -> Fee:
    """Charge on actual notional; rate is a decimal, e.g. 0.001 means 10 bps."""
    def commission(_day: date, _permno: int, shares: Decimal, price: Decimal) -> Decimal:
        return shares * price * rate

    return commission


DEFAULT_COMMISSION = proportional_commission(Decimal("0.0005"))  # 5 bp each for buys and sells.


def floor_shares(nav: Decimal, weight: Decimal, close: Decimal) -> Decimal:
    """Round down per security; unallocated cash stays in the cash account."""
    return (nav * weight) // close


@dataclass(frozen=True)
class Order:
    permno: int
    quantity: Decimal       # Net buy quantity is positive and net sell negative, in signal-day shares.
    lots: Mapping[int, Decimal]  # Per-lot allocation for sell orders; empty for buy orders.

    def checked(self, day: date) -> "Order":
        quantity = Decimal(self.quantity)
        if not quantity.is_finite() or not quantity or fraction(quantity):
            raise ValueError(f"{day} {self.permno}: a net order requires a nonzero whole quantity.")
        allocations = {i: Decimal(q) for i, q in self.lots.items()}
        if (any(not q.is_finite() or q <= 0 for q in allocations.values())
                or sum_shares(allocations.values()) != max(-quantity, ZERO)):
            raise ValueError(f"{day} {self.permno}: lot allocation does not match the net order of {quantity} shares.")
        return Order(self.permno, quantity, allocations)


def lock_orders(state: CloseSnapshot, weights: Mapping[int, Decimal], size: Sizer,
                lot_relief: LotRelief | None, blocked: set[int],
                sell_lots: Mapping[int, Mapping[int, Decimal]] | None = None) -> tuple[Order, ...]:
    """Subtract actual holdings from target holdings on the same previous-close ledger; does not take trade-date prices."""
    by_name = state.lots_by_permno
    orders = []
    explicit = dict(sell_lots or {})
    nav = state.investable_nav
    for p in sorted((weights.keys() | by_name.keys()) - blocked):
        held = sum_shares(lot.shares for lot in by_name.get(p, ()))
        weight = weights.get(p, ZERO)
        if weight and (nav is None or state.closes.get(p, ZERO) <= 0):
            raise ValueError(f"{state.day} {p}: cannot lock shares without signal NAV and a closing price.")
        target = size(nav, weight, state.closes[p]) if weight else ZERO
        if target < 0 or target != target.to_integral_value():
            raise ValueError(f"{state.day} {p}: target must be a nonnegative whole-share quantity.")
        delta = target - held
        if delta == 0:
            continue
        if delta != delta.to_integral_value():
            raise ValueError(f"{state.day} {p}: unresolved fractional entitlement; review before trading.")
        allocations = {}
        if delta < 0:
            allocations = select_lots(state, p, -delta, lot_relief, explicit.pop(p, None))
        orders.append(Order(p, delta, allocations))
    if explicit:
        raise ValueError(f"{state.day}: lot instructions without an active net sale: {sorted(explicit)}.")
    return tuple(orders)


def select_lots(state: CloseSnapshot, p: int, quantity: Decimal,
                 lot_relief: LotRelief | None, explicit: Mapping[int, Decimal] | None) -> dict[int, Decimal]:
    available = {lot.id: lot.shares for lot in state.eligible_lots(p)}
    eligible = sum_shares(available.values())
    if quantity > eligible:
        raise ValueError(f"{state.day} {p}: requested {quantity} shares, only {eligible} have verified lots.")
    if explicit is not None:
        selected = explicit
    elif quantity == eligible:
        selected = available
    elif len(available) == 1:
        selected = {next(iter(available)): quantity}
    elif lot_relief is not None:
        selected = lot_relief(state, p, quantity)
    else:
        raise ValueError(f"{state.day} {p}: specify lot relief for a partial sale across multiple lots.")
    allocations = {i: Decimal(q) for i, q in selected.items()}
    if (any(not q.is_finite() or not 0 < q <= available.get(i, ZERO) for i, q in allocations.items()) or
            sum_shares(allocations.values()) != quantity):
        raise ValueError(f"{state.day} {p}: lot allocation does not match the requested sale.")
    if any(fraction(q) > fraction(available[i]) for i, q in allocations.items()):
        raise ValueError(f"{state.day} {p}: active lot allocation would split a whole share.")
    return allocations


def buy_shares(book: Ledger, day: date, orders: Sequence[Order], prices: Mapping[int, Decimal],
               fee: Fee, price_source: str, *, price_days: Mapping[int, date] | None = None) -> None:
    """Check the cost of all fillable buy orders together; raise on insufficient cash rather than letting security order decide which buys fill."""
    fills = []
    required = ZERO
    for order in orders:
        p, q = order.permno, order.quantity
        price = prices[p]
        commission = fee(day, p, q, price)
        if commission < 0:
            raise ValueError(f"{day} {p}: negative commission.")
        required += q * price + commission
        fills.append((p, q, price, commission))
    if required > book.cash:
        raise ValueError(f"{day}: purchases including commission cost {required}, available cash {book.cash}.")
    for p, q, price, commission in fills:
        book.buy(day, p, q, price, commission, price_source=price_source,
                 price_day=price_days.get(p) if price_days else day)
