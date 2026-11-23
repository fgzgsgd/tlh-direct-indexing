"""股数的有限小数精确相加、换算。避免逐批分摊后把整股误判为零股。"""

from collections.abc import Iterable
from decimal import Decimal, ROUND_DOWN, localcontext
from fractions import Fraction


def sum_shares(quantities: Iterable[Decimal]) -> Decimal:
    values = list(quantities)
    if not values:
        return Decimal(0)
    if len(values) == 1:
        return values[0]
    # 整数部分留出加总所需位数，小数部分保留所有 lots 的最小计量单位。
    high = max(q.adjusted() for q in values)
    low = min(q.as_tuple().exponent for q in values)
    with localcontext() as ctx:
        ctx.prec = max(ctx.prec, high - low + 1 + len(str(len(values))))
        return sum(values, Decimal(0))


def as_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def share_increments(amounts: Iterable[Fraction], *, starting: Fraction = Fraction(0)) -> Iterable[Decimal]:
    """先按证券累加精确权益，再以相邻累计数之差记入各 lot，完整的一股始终按 1 入账。"""
    def balance(value: Fraction) -> Decimal:
        whole, tail = divmod(value.numerator, value.denominator)
        if not tail:
            return Decimal(whole)
        with localcontext() as ctx:
            ctx.rounding = ROUND_DOWN
            # 分开整数和小数，保证分配 1 整股时不受已有零股影响。有限小数保留所有位。
            ctx.prec = max(ctx.prec, len(str(tail)) + value.denominator.bit_length())
            remainder = Decimal(tail) / Decimal(value.denominator)
        return sum_shares((Decimal(whole), remainder))

    exact, previous = starting, balance(starting)
    for size in amounts:
        exact += size
        current = balance(exact)
        yield sum_shares((current, previous.copy_negate()))
        previous = current


def fraction(shares: Decimal) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = max(ctx.prec, len(shares.as_tuple().digits), shares.adjusted() + 1)
        return shares % 1


def apportion(total: Decimal, weights: Iterable[Fraction]) -> list[Decimal]:
    """按权重分摊股数或金额。最后一段承接不可表示的小数尾差，总量精确守恒。"""
    weights = list(weights)
    if len(weights) == 1:
        return [total]
    denominator = sum(weights)
    with localcontext() as ctx:
        ctx.rounding = ROUND_DOWN
        parts = [as_decimal(Fraction(total) * weight / denominator) for weight in weights[:-1]]
    parts.append(sum_shares([total, *(part.copy_negate() for part in parts)]))
    return parts
