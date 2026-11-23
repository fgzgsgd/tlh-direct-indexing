"""Allocate a stock's net sell quantity to tax lots in the priority order given by the strategy."""

from collections.abc import Callable, Iterable
from decimal import Decimal, localcontext

from .strategy import CloseSnapshot, LotView
from .shares import fraction, sum_shares
from .tax import LT_RATE, ST_RATE, is_long_term


def take_shares(ordered_lots: Iterable[LotView], shares_to_sell: Decimal) -> dict[int, Decimal]:
    """Take as much of the current lot as the order allows while leaving deliverable quantities for the remaining lots."""
    shares_to_sell = Decimal(shares_to_sell)
    if shares_to_sell <= 0:
        raise ValueError("Supply a positive number of shares to sell.")
    lots = list(ordered_lots)
    if len({lot.id for lot in lots}) != len(lots):
        raise ValueError("Duplicate lot in sale order.")
    tails = [fraction(lot.shares) for lot in lots]
    remaining_tail = sum_shares(tails)
    available = sum_shares(lot.shares for lot in lots)
    if shares_to_sell > available:
        raise ValueError(f"Eligible lots are short by {shares_to_sell - available} shares.")
    if fraction(shares_to_sell) > remaining_tail:
        raise ValueError("Eligible lots cannot deliver the requested fractional quantity.")

    selected = {}
    remaining = shares_to_sell
    with localcontext() as ctx:
        # Use the finest actual share unit so rounding does not erase the last deliverable fraction.
        unit = min(remaining.as_tuple().exponent, remaining_tail.as_tuple().exponent)
        ctx.prec = max(ctx.prec, available.adjusted() - unit + 2)
        for lot, tail in zip(lots, tails):
            remaining_tail -= tail
            whole = min(remaining // 1, lot.shares // 1)
            quantity = whole + min(remaining - whole, tail)
            left = remaining - quantity
            if fraction(left) > remaining_tail:
                # Only whole shares remain deliverable. Round the remainder up to the next integer, then compute how much this lot can sell.
                left = left // 1 + 1
                quantity = remaining - left
            if quantity:
                selected[lot.id] = quantity
            remaining = left
            if not remaining:
                break
    return selected


def _ranked(state: CloseSnapshot, permno: int, shares_to_sell: Decimal,
            score: Callable[[LotView], Decimal]) -> dict[int, Decimal]:
    def rank(lot):
        acquired = min(s.acquired for s in lot.segments) if lot.segments else lot.acquired
        return score(lot), acquired, lot.id

    lots = sorted(state.eligible_lots(permno), key=rank)
    return take_shares(lots, shares_to_sell)


def hifo(state: CloseSnapshot, permno: int, shares_to_sell: Decimal) -> dict[int, Decimal]:
    """Highest In, First Out: sell the lots with the highest adjusted basis per share first."""
    return _ranked(state, permno, shares_to_sell, lambda lot: -(lot.basis / lot.shares))


def ltfo(state: CloseSnapshot, permno: int, shares_to_sell: Decimal) -> dict[int, Decimal]:
    """Least Tax, First Out: sell first the lots with the lowest expected tax per share at the signal-day price."""
    close = state.closes.get(permno)
    if close is None or close <= 0 or state.trade_day is None:
        raise ValueError(f"{state.day} {permno}: LTFO needs the signal close and scheduled sale date.")

    def tax_per_share(lot: LotView) -> Decimal:
        # One share can have both ST and LT segments; tax is computed per segment, then converted to tax per share.
        tax = Decimal(0)
        for part in lot.segments or (lot,):
            rate = LT_RATE if is_long_term(part.holding_start, state.trade_day) else ST_RATE
            tax += (close * part.shares - part.basis) * rate
        return tax / lot.shares

    return _ranked(state, permno, shares_to_sell, tax_per_share)
