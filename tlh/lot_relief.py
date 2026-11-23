"""按策略给出的优先顺序，将一只股票的净卖量分配到 tax lots。"""

from collections.abc import Callable, Iterable
from decimal import Decimal, localcontext

from .strategy import CloseSnapshot, LotView
from .shares import fraction, sum_shares
from .tax import LT_RATE, ST_RATE, is_long_term


def take_shares(ordered_lots: Iterable[LotView], shares_to_sell: Decimal) -> dict[int, Decimal]:
    """按排序尽量多取当前 lot，同时给其余 lots 留下可交割的卖量。"""
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
        # 使用最细的实际份额单位，避免舍入把能交割的最后一点零股抹去。
        unit = min(remaining.as_tuple().exponent, remaining_tail.as_tuple().exponent)
        ctx.prec = max(ctx.prec, available.adjusted() - unit + 2)
        for lot, tail in zip(lots, tails):
            remaining_tail -= tail
            whole = min(remaining // 1, lot.shares // 1)
            quantity = whole + min(remaining - whole, tail)
            left = remaining - quantity
            if fraction(left) > remaining_tail:
                # 后面只够交整股。把余单提高到下一个整数，再算本批能卖多少。
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
    """Highest In, First Out：每股 adjusted basis 最高的先卖。"""
    return _ranked(state, permno, shares_to_sell, lambda lot: -(lot.basis / lot.shares))


def ltfo(state: CloseSnapshot, permno: int, shares_to_sell: Decimal) -> dict[int, Decimal]:
    """Least Tax, First Out：按信号日价格估算，每股预计产生的税负最低者先卖。"""
    close = state.closes.get(permno)
    if close is None or close <= 0 or state.trade_day is None:
        raise ValueError(f"{state.day} {permno}: LTFO needs the signal close and scheduled sale date.")

    def tax_per_share(lot: LotView) -> Decimal:
        # 一股里可以同时有 ST 和 LT 两段，税额先按各段计算，再换成每股税负。
        tax = Decimal(0)
        for part in lot.segments or (lot,):
            rate = LT_RATE if is_long_term(part.holding_start, state.trade_day) else ST_RATE
            tax += (close * part.shares - part.basis) * rate
        return tax / lot.shares

    return _ranked(state, permno, shares_to_sell, tax_per_share)
