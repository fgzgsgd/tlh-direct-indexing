"""单个应税账户按 PERMNO 和股数匹配 wash sale。匹配只改变税账。"""

from collections import deque
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING

from .shares import as_decimal, share_increments, sum_shares

if TYPE_CHECKING:
    from .ledger import Ledger, Lot, Sale


WINDOW = timedelta(days=30)
Periods = tuple[tuple[date, date], ...]  # 每段不计买入日，计卖出日。


@dataclass
class OpenLoss:
    sale: "Sale"
    origin: int | None
    shares: Fraction      # 与当前交易单位一致，拆股时精确换算。
    remaining: Decimal    # 尚可递延的美元损失，拆股时不变。
    periods: Periods
    unverified_splits: tuple[str, ...] = ()


@dataclass(frozen=True)
class WashMatch:
    sale_day: date
    sold_lot: int
    replacement_lot: int
    acquired: date
    permno: int
    shares: Decimal
    disallowed_loss: Decimal
    unverified_splits: tuple[str, ...] = ()


def _before(periods: Periods, acquired: date) -> Periods:
    """只承接新买入日之前实际持有的日期，合并重叠期间，保留空仓间隔。"""
    merged = []
    for start, end in sorted(periods):
        end = min(end, acquired)
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def _portion(book: "Ledger", lot: "Lot", units: Fraction) -> "Lot":
    if units == lot.units:
        return lot
    quantity = next(share_increments([units]))
    basis = None if lot.basis is None else as_decimal(Fraction(lot.basis) * units / lot.units)
    part = replace(lot, id=book.next_lot_id(), source_lot_id=lot.source_id,
                   shares=quantity, exact_shares=units, basis=basis)
    lot.exact_shares = lot.units - units
    lot.shares = sum_shares((lot.shares, quantity.copy_negate()))
    if basis is not None:
        lot.basis -= basis
    book.lots[part.id] = part
    return part


def _apply_match(book: "Ledger", loss: OpenLoss, matched: "Lot") -> None:
    units = matched.units
    amount = loss.remaining if units == loss.shares else as_decimal(Fraction(loss.remaining) * units / loss.shares)
    matched.wash_used = True
    if matched.basis is not None:
        matched.basis += amount
    matched.prior_periods = _before(matched.prior_periods + loss.periods, matched.acquired)
    days = sum((end - start).days for start, end in matched.prior_periods)
    matched.holding_start = matched.acquired - timedelta(days=days)
    loss.sale.disallowed_loss += amount
    loss.shares -= units
    loss.remaining -= amount
    book.wash_matches.append(WashMatch(loss.sale.day, loss.sale.lot_id, matched.id,
                                       matched.acquired, matched.permno, matched.shares, amount, loss.unverified_splits))


def _match(book: "Ledger", loss: OpenLoss, lot: "Lot") -> list["Lot"]:
    """匹配整股和尾数，返回这笔买入还可匹配的部分。"""
    units = min(loss.shares, lot.units)
    whole, tail = divmod(units, 1)
    if tail and lot.bound_lot_id is None and whole < int(lot.units):
        if whole:
            _apply_match(book, loss, _portion(book, lot, Fraction(whole)))
        # 当前这一股保留原 ID，下一笔亏损可接着匹配它的余额。其余整股仍独立。
        remaining = lot.units - 1
        untouched = [_portion(book, lot, remaining)] if remaining else []
        lot.bound_lot_id = lot.id
        _apply_match(book, loss, _portion(book, lot, tail))
        return [lot, *untouched]
    _apply_match(book, loss, _portion(book, lot, units))
    return [] if lot.wash_used else [lot]


def on_sale(book: "Ledger", sale: "Sale", source: "Lot") -> None:
    if not sale.wash_sale_candidate:
        return
    periods = source.prior_periods + ((source.acquired, sale.day),)
    loss = OpenLoss(sale, source.origin, sale.exact_shares or Fraction(sale.shares), -sale.gain, periods)
    # 本次卖出的各批已全部扣除，只检查仍持有、且没有匹配过的独立买入。
    candidates = (lot for lot in book.lots.values() if lot.permno == sale.permno
                  and lot.origin is not None and lot.origin != source.origin and not lot.wash_used
                  and sale.day - WINDOW <= lot.acquired <= sale.day)
    for lot in sorted(candidates, key=lambda l: (l.acquired, l.origin, l.id)):
        _match(book, loss, lot)
        if not loss.shares:
            return
    book.open_losses.setdefault(sale.permno, []).append(loss)


def on_buy(book: "Ledger", lot: "Lot") -> None:
    pending = book.open_losses.get(lot.permno)
    if not pending:
        return
    available = deque([lot])
    for loss in pending:
        if not available:
            break
        if lot.acquired - WINDOW <= loss.sale.day <= lot.acquired:
            while loss.shares and available:
                remainder = _match(book, loss, available.popleft())
                available.extendleft(reversed(remainder))
    remaining = [loss for loss in pending if loss.shares and loss.sale.day >= lot.acquired - WINDOW]
    if remaining:
        book.open_losses[lot.permno] = remaining
    else:
        del book.open_losses[lot.permno]
