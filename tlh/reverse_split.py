"""合股时先保留各批整股，再给必须跨批组成的新股绑定交易 ID。"""

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING

from .shares import apportion, as_decimal, share_increments

if TYPE_CHECKING:
    from .ledger import Ledger


@dataclass(frozen=True)
class SplitPiece:
    old_segment: int
    old_trading_lot: int
    segment: int
    trading_lot: int
    source_lot: int
    acquired: date
    holding_start: date
    old_fraction: Fraction     # 原税务明细的多少转入本段，用于同日其他前收盘权益。
    exact_shares: Fraction
    shares: Decimal
    basis: Decimal | None
    cash_in_lieu: bool


@dataclass(frozen=True)
class SplitDesignation:
    """有逐股识别记录时，以新股为单位，列出每股的来源和最终 CIL 的来源。"""
    whole_shares: Sequence[Mapping[int, Fraction]]
    cash_in_lieu: Mapping[int, Fraction]


@dataclass(frozen=True)
class _Allotment:
    sources: dict[int, Fraction]     # 以新股计，各旧交易 lot 贡献的份额。
    retain_id: int | None = None
    cash_in_lieu: bool = False


def _take(pool: deque, size: Fraction) -> dict[int, Fraction]:
    parts = {}
    while size:
        lot_id, available = pool.popleft()
        amount = min(size, available)
        parts[lot_id] = amount
        size -= amount
        if available > amount:
            pool.appendleft((lot_id, available - amount))
    return parts


def _partitions(items):
    """保持给定的持有期顺序，枚举唯一尾数位于第几股之前。"""
    total = sum(q for _, q in items)
    whole_count, tail = int(total), total % 1
    positions = range(whole_count + 1) if tail else (whole_count,)
    for position in positions:
        pool = deque(items)
        whole, remainder = [], {}
        for index in range(whole_count + 1):
            if tail and index == position:
                remainder = _take(pool, tail)
            if index < whole_count:
                whole.append(_take(pool, Fraction(1)))
        yield whole, remainder


def _identified(tails: dict[int, Fraction], designation: SplitDesignation, reference: str):
    whole = [{i: Fraction(q) for i, q in group.items()} for group in designation.whole_shares]
    cash = {i: Fraction(q) for i, q in designation.cash_in_lieu.items()}
    used = defaultdict(Fraction)
    for part in [*whole, cash]:
        for i, q in part.items():
            if q <= 0:
                raise ValueError(f"{reference}: invalid designated segment size.")
            used[i] += q
    if (dict(used) != tails or any(sum(part.values()) != 1 for part in whole)
            or sum(cash.values()) != sum(tails.values()) % 1):
        raise ValueError(f"{reference}: designated shares do not match the split entitlements.")
    return whole, cash


def _remainders(tails: dict[int, Fraction], groups, designation: SplitDesignation | None, reference: str):
    """先最大化同日同价组成的整股数，再比较持有期跨度，同分时 CIL 优先取最早来源。"""
    if designation is not None:
        return _identified(tails, designation, reference)
    if sum(tails.values()) < 1:
        return [], tails
    dates = {i: (min(part.holding_start for part in groups[i]), max(part.holding_start for part in groups[i]))
             for i in tails}
    earliest = sorted(tails, key=lambda i: (min((part.acquired, part.source_id) for part in groups[i]), i))
    # 相同取得日、成交价和持有期间的份额可互换。确定 CIL 后才将其落实到具体原始 lot。
    equivalent = defaultdict(list)
    for i in earliest:
        prices = {(part.acquired, part.purchase_price) for part in groups[i]}
        key = next(iter(prices)) if len(prices) == 1 else None
        purchase = key if key is not None and key[1] is not None else i
        equivalent[purchase, dates[i]].append((i, tails[i]))
    cohorts, by_purchase = {}, defaultdict(list)
    for (purchase, _), members in equivalent.items():
        representative = members[0][0]
        cohorts[representative] = members
        by_purchase[purchase].append((representative, sum(q for _, q in members)))

    def ordered(items):
        return sorted(items, key=lambda pair: (dates[pair[0]], pair[0]))

    def span(parts):
        return (max(dates[i][1] for i in parts) - min(dates[i][0] for i in parts)).days

    # 只有同日同价却因 wash sale 等原因有不同持有期时，一个购买组才有多种非等价的尾数。
    options = [list(_partitions(ordered(bucket))) if len(bucket) > 1 else [next(_partitions(bucket))]
               for bucket in by_purchase.values()]
    best_key, best = None, None
    for selection in product(*options):
        homogeneous = [share for whole, _ in selection for share in whole]
        rest = ordered((i, q) for _, remainder in selection for i, q in remainder.items())
        for mixed, tail in _partitions(rest):
            pools = {i: deque(members) for i, members in cohorts.items()}

            def resolve(parts):
                return {source: qty for i, size in parts.items() for source, qty in _take(pools[i], size).items()}

            cash = resolve(tail)
            shares = homogeneous + mixed
            key = (sum(span(s) for s in shares) + (span(tail) if tail else 0),
                   tuple(-cash.get(i, 0) for i in earliest))
            if best_key is None or key < best_key:
                best_key, best = key, ([resolve(s) for s in shares], cash)
    return best


def bind_split(book: "Ledger", permno: int, factor: Fraction, reference: str,
               designation: SplitDesignation | None, old_sizes: Mapping[int, Fraction] | None = None) -> tuple[SplitPiece, ...]:
    groups = book.trading_lots(permno)
    if old_sizes is None:
        old_sizes = {part.id: Fraction(part.shares) for parts in groups.values() for part in parts}
    held = {i: sum(old_sizes[part.id] for part in parts) for i, parts in groups.items()}
    entitlements = {i: q * factor for i, q in held.items()}
    tails = {i: q % 1 for i, q in entitlements.items() if q % 1}
    # 无须跨批凑股、各份额均可精确表示时，沿用原 lot；尾数由该 lot 直接结算。
    if (designation is None and sum(tails.values()) < 1 and all(len(parts) == 1 for parts in groups.values())
            and all(Fraction(as_decimal(q)) == q for q in entitlements.values())):
        for i, parts in groups.items():
            parts[0].shares = as_decimal(entitlements[i])
        return ()
    whole, cash = _remainders(tails, groups, designation, reference)
    outputs = [_Allotment({i: Fraction(int(q))}, retain_id=i) for i, q in entitlements.items() if q >= 1]
    outputs.extend(_Allotment(parts) for parts in whole)
    if cash:
        outputs.append(_Allotment(cash, cash_in_lieu=True))

    # 按原始明细分配 basis；按每个新交易单位分配股数。非终止小数只在记入 Decimal 时舍入。
    destinations = defaultdict(list)
    for index, output in enumerate(outputs):
        for source, shares in output.sources.items():
            destinations[source].append((index, shares))
    by_output = defaultdict(list)
    for source, parts in groups.items():
        targets = destinations[source]
        for lot in parts:
            bases = ([None] * len(targets) if lot.basis is None else
                     apportion(lot.basis, [q for _, q in targets]))
            for part_number, ((output_index, shares), basis) in enumerate(zip(targets, bases)):
                row = replace(lot, id=lot.id if part_number == 0 else book.next_lot_id(),
                              source_lot_id=lot.source_id, basis=basis)
                contribution = shares * old_sizes[lot.id] / held[source]
                by_output[output_index].append((row, lot, shares / entitlements[source], contribution))

    pieces = []
    totals = (sum(output.sources.values()) for output in outputs)
    for index, (output, total) in enumerate(zip(outputs, share_increments(totals))):
        rows = by_output[index]
        if output.retain_id is not None:
            bound = groups[output.retain_id][0].bound_lot_id
        else:
            bound = book.next_lot_id() if len(rows) > 1 else None
        quantities = apportion(total, [q for _, _, _, q in rows])
        for (row, previous, weight, exact), quantity in zip(rows, quantities):
            row.shares, row.bound_lot_id = quantity, bound
            book.lots[row.id] = row
            pieces.append(SplitPiece(previous.id, previous.trading_id, row.id, row.trading_id, row.source_id,
                                     row.acquired, row.holding_start, weight, exact, quantity, row.basis, output.cash_in_lieu))
    return tuple(pieces)
