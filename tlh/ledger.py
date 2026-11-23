"""用交易价格和实际股数记账。每次卖出都能追到具体 tax lot。"""

from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from fractions import Fraction

from .reverse_split import SplitDesignation, SplitPiece, bind_split
from .tax import DistributionIncome, PropertyIncome, TaxYear, ZERO, is_long_term
from .shares import apportion, as_decimal, share_increments, sum_shares
from .wash import WINDOW, OpenLoss, Periods, WashMatch, on_buy, on_sale


@dataclass
class Lot:
    id: int
    permno: int
    acquired: date
    holding_start: date
    shares: Decimal
    basis: Decimal | None       # 剩余批次的总成本；待确认的 basis 保留为空。
    origin: int | None = None   # 一笔真实买入的原始编号。非应税被动分配不构成新买入。
    wash_used: bool = False
    prior_periods: Periods = ()
    bound_lot_id: int | None = None  # 同一整股的税务 segments 共用一个交易 ID。
    source_lot_id: int | None = None
    purchase_price: Decimal | None = None
    exact_shares: Fraction | None = None  # 非终止小数的税务份额可跨日精确配对。

    @property
    def units(self) -> Fraction:
        return self.exact_shares if self.exact_shares is not None else Fraction(self.shares)

    @property
    def trading_id(self) -> int:
        return self.bound_lot_id if self.bound_lot_id is not None else self.id

    @property
    def source_id(self) -> int:
        return self.source_lot_id if self.source_lot_id is not None else self.id


@dataclass
class Sale:
    day: date
    lot_id: int
    permno: int
    shares: Decimal
    proceeds: Decimal           # 已扣与这部分股数对应的卖出费用。
    basis: Decimal | None
    holding_start: date
    reason: str
    disallowed_loss: Decimal = ZERO
    event_id: str | None = None
    trading_lot_id: int | None = None
    source_lot_id: int | None = None
    exact_shares: Fraction | None = None  # 处置的精确税务份额。

    @property
    def gain(self) -> Decimal | None:
        return None if self.basis is None else self.proceeds - self.basis

    @property
    def taxable_gain(self) -> Decimal | None:
        gain = self.gain
        return None if gain is None else gain + self.disallowed_loss

    @property
    def wash_sale_candidate(self) -> bool:
        gain = self.gain
        return self.reason != "CASH_IN_LIEU" and gain is not None and gain < 0

    @property
    def is_long_term(self) -> bool:
        return is_long_term(self.holding_start, self.day)


@dataclass(frozen=True)
class Entry:
    day: date
    kind: str
    permno: int | None  # 税款划转属于账户整体。
    shares: Decimal
    cash_change: Decimal
    reference: str
    fee: Decimal = ZERO
    price: Decimal | None = None
    price_source: str | None = None
    price_day: date | None = None  # 用前值模拟成交时与实际记账日不同。


@dataclass(frozen=True)
class Prelisting:
    first_trade: date
    price: Decimal  # 发行人事后公布的 FMV，仅供上市前估值。
    reference: str
    fractions: tuple[Lot, ...] = ()  # 待上市结算的 CIL，保留各来源的 basis。

    @property
    def pending_shares(self) -> Decimal:
        return sum_shares(lot.shares for lot in self.fractions)


class Ledger:
    def __init__(self, initial_cash: Decimal, *,
                 split_designations: Mapping[str, SplitDesignation] | None = None):
        self.cash = initial_cash
        self.split_designations = dict(split_designations or {})
        self.lots: dict[int, Lot] = {}
        self.prelistings: dict[int, Prelisting] = {}
        self.sales: list[Sale] = []
        self.entries: list[Entry] = []
        self.distribution_income: list[DistributionIncome | PropertyIncome] = []
        self.symbol_history: dict[int, list[tuple[date, str]]] = {}
        self._next_id = 1
        self._events: set[tuple[int, str]] = set()
        self.wash_matches: list[WashMatch] = []
        self.open_losses: dict[int, list[OpenLoss]] = {}

    def next_lot_id(self) -> int:
        number = self._next_id
        self._next_id += 1
        return number

    def post_cash(self, amount: Decimal) -> None:
        """按已入账流水的全部小数位累加，避免 CIL 与大额现金相加时再次舍入。"""
        self.cash = sum_shares((self.cash, amount))

    @contextmanager
    def transaction(self):
        """失败时撤回当天过账。历史流水仅截断，当天可能回写的旧亏损单独恢复。"""
        cash, next_id = self.cash, self._next_id
        lots = {i: copy(lot) for i, lot in self.lots.items()}
        prelistings = self.prelistings.copy()
        pending = {p: [copy(loss) for loss in losses] for p, losses in self.open_losses.items()}
        old_losses = [(loss.sale, loss.sale.disallowed_loss) for losses in pending.values() for loss in losses]
        events, symbols = self._events.copy(), {p: list(v) for p, v in self.symbol_history.items()}
        incomes = len(self.distribution_income)
        entries, sales, matches = len(self.entries), len(self.sales), len(self.wash_matches)
        try:
            yield
        except Exception:
            self.cash, self._next_id, self.lots, self.open_losses = cash, next_id, lots, pending
            self.prelistings = prelistings
            self._events, self.symbol_history = events, symbols
            del self.distribution_income[incomes:]
            del self.entries[entries:], self.sales[sales:], self.wash_matches[matches:]
            for sale, disallowed in old_losses:
                sale.disallowed_loss = disallowed
            raise

    def register_event(self, permno: int, event_id: str) -> None:
        key = permno, event_id
        if key in self._events:
            raise ValueError(f"Corporate action already booked: {key}")
        self._events.add(key)

    def holdings(self) -> dict[int, Decimal]:
        positions: dict[int, list[Decimal]] = {}
        for lot in self.lots.values():
            positions.setdefault(lot.permno, []).append(lot.shares)
        return {p: sum_shares(q) for p, q in positions.items()}

    def trading_lots(self, permno: int | None = None) -> dict[int, list[Lot]]:
        """每个可交易 ID 对应一批可一起交割的税务明细，账本只持有一份股数和成本。"""
        groups: dict[int, list[Lot]] = {}
        for lot in self.lots.values():
            if permno is None or lot.permno == permno:
                groups.setdefault(lot.trading_id, []).append(lot)
        return groups

    def trading_positions(self, permno: int) -> dict[int, Decimal]:
        return {i: sum_shares(lot.shares for lot in parts) for i, parts in self.trading_lots(permno).items()}

    def entitled_lots(self) -> dict[int, dict[int, Decimal]]:
        """按 ex-date 前的股数固定领取资格。同日拆股或收到的新证券不能改变旧股资格。"""
        by_security: dict[int, dict[int, Decimal]] = {}
        for lot in self.lots.values():
            by_security.setdefault(lot.permno, {})[lot.id] = lot.shares
        return by_security

    def total_basis(self) -> Decimal | None:
        lots = [*self.lots.values(), *(lot for grant in self.prelistings.values() for lot in grant.fractions)]
        if any(lot.basis is None for lot in lots):
            return None
        return sum_shares(lot.basis for lot in lots)

    def receive(self, day: date, permno: int, shares: Decimal, basis: Decimal | None,
                holding_start: date, reference: str, *, prior_periods: Periods | None = None) -> Lot:
        """按已核实的条款登记被动取得的股票，例如 spin-off 分得的子公司股票。"""
        if shares <= 0 or (basis is not None and basis < 0):
            raise ValueError("Invalid share entitlement or basis.")
        periods = ((holding_start, day),) if prior_periods is None else prior_periods
        lot = Lot(self.next_lot_id(), permno, day, holding_start, shares, basis, prior_periods=periods)
        self.lots[lot.id] = lot
        self.entries.append(Entry(day, "RECEIVED_SHARES", permno, shares, ZERO, reference))
        return lot

    def buy(self, day: date, permno: int, shares: Decimal, price: Decimal, fee: Decimal,
            *, price_source: str | None = None, price_day: date | None = None) -> Lot:
        if shares <= 0 or shares != shares.to_integral_value() or price <= 0 or fee < 0:
            raise ValueError("Purchases require whole shares, a positive price and a nonnegative fee.")
        basis = shares * price + fee
        if basis > self.cash:
            raise ValueError(f"{day}: purchase including commission exceeds available cash.")
        lot_id = self.next_lot_id()
        lot = Lot(lot_id, permno, day, day, shares, basis, origin=lot_id, purchase_price=price)
        self.lots[lot.id] = lot
        self.post_cash(basis.copy_negate())
        self.entries.append(Entry(day, "BUY", permno, shares, -basis, str(lot.id), fee, price, price_source,
                                  price_day or day))
        on_buy(self, lot)
        return lot

    def sell(self, day: date, permno: int, allocations: Mapping[int, Decimal | Fraction], price: Decimal,
             fees: dict[int, Decimal], reason: str, *, event_id: str | None = None,
             price_source: str | None = None, price_day: date | None = None) -> list[Sale]:
        """调用方明确指定每个 lot 的股数和费用；先校验整单，再一次性记账。"""
        if reason not in {"TRADE", "PRE_DELISTING_EXIT", "CASH_ACQUISITION", "CASH_IN_LIEU", "LIQUIDATION"}:
            raise ValueError(f"Unknown sale reason: {reason}")
        if not allocations or allocations.keys() != fees.keys() or price < 0:
            raise ValueError("A sale requires lot allocations, corresponding fees, and a price.")
        if reason in {"TRADE", "PRE_DELISTING_EXIT"} and price == 0:
            raise ValueError("A market execution requires a positive raw price.")
        if reason == "TRADE" and sum(map(Fraction, allocations.values())) % 1:
            raise ValueError("An active sale requires a whole number of shares per security.")
        groups = self.trading_lots(permno)
        if missing := allocations.keys() - groups.keys():
            raise ValueError(f"Unknown trading lot {min(missing)} for {permno}. Use the trading lot ID for a bound segment.")
        is_cil = reason == "CASH_IN_LIEU"
        weights = {i: [lot.units for lot in groups[i]] for i in allocations}
        # CIL 可能卖出 1/3 股。先确定留下的整股，再将其投影到账本的 Decimal 股数。
        retained = share_increments(sum(weights[i]) - Fraction(q) for i, q in allocations.items())
        sales = []
        sources = {}
        segment_fees = {}
        partial_cil = None
        for trading_id, requested in allocations.items():
            parts = groups[trading_id]
            sizes_before = weights[trading_id]
            held = sum(sizes_before)
            quantity = Fraction(requested)
            # 卖完一批的账面尾数时，留下的整股数唯一，用它还原精确卖量。
            remaining = Fraction(sum_shares(lot.shares for lot in parts)) - quantity
            if not is_cil and remaining.denominator == 1:
                quantity = held - remaining
            if not 0 < quantity <= held or fees[trading_id] < 0 or any(lot.acquired > day for lot in parts):
                raise ValueError(f"Invalid sale of lot {trading_id}.")
            if reason == "TRADE" and quantity < held and quantity % 1 > held % 1:
                raise ValueError(f"Active sale would split a whole share in lot {trading_id}.")
            sold_fraction = quantity / held
            if is_cil:
                keep = apportion(next(retained), sizes_before)
                sizes = [sum_shares((lot.shares, left.copy_negate())) for lot, left in zip(parts, keep)]
            else:
                booked = requested if isinstance(requested, Decimal) else as_decimal(Fraction(requested))
                sizes = [lot.shares for lot in parts] if quantity == held else apportion(booked, sizes_before)
            # 先按可交易 ID 校验整股，再把同一次成交归到各自的税务明细。
            for lot, size, units_before, charge in zip(parts, sizes, sizes_before, apportion(fees[trading_id], sizes_before)):
                segment_fees[lot.id] = charge
                sources[lot.id] = copy(lot)
                basis = lot.basis
                if basis is not None and sold_fraction != 1:
                    basis = as_decimal(Fraction(basis) * sold_fraction)
                sold_units = units_before * sold_fraction
                proceeds = as_decimal(sold_units * Fraction(price)) - charge
                exact = sold_units if is_cil or lot.exact_shares is not None else None
                sales.append(Sale(day, lot.id, permno, size, proceeds, basis,
                                  lot.holding_start, reason, event_id=event_id, trading_lot_id=trading_id,
                                  source_lot_id=lot.source_id, exact_shares=exact))
                if is_cil and sold_fraction < 1 and lot.basis is not None:
                    partial_cil = sales[-1]

        # 券商对整笔交易收付款。若逐段金额确有尾差，部分 CIL 的回款与 basis 同额调整。
        net = as_decimal(sum(map(Fraction, allocations.values())) * Fraction(price)) - sum(fees.values(), ZERO)
        residual = sum_shares((net, *(s.proceeds.copy_negate() for s in sales)))
        if residual:
            last = partial_cil or sales[-1]
            last.proceeds = sum_shares((last.proceeds, residual))
            if partial_cil:
                last.basis = sum_shares((last.basis, residual))
        for sale in sales:
            lot = self.lots[sale.lot_id]
            if sale.exact_shares is not None:
                lot.exact_shares = lot.units - sale.exact_shares
            lot.shares = sum_shares((lot.shares, sale.shares.copy_negate()))
            if sale.basis is not None:
                lot.basis = sum_shares((lot.basis, sale.basis.copy_negate()))
            if not lot.shares:
                del self.lots[lot.id]
            self.entries.append(Entry(day, reason, permno, sale.shares.copy_negate(), sale.proceeds,
                                      str(lot.trading_id), segment_fees[lot.id], price, price_source, price_day or day))
        self.post_cash(net)
        self.sales.extend(sales)
        for sale in sorted(sales, key=lambda s: (sources[s.lot_id].acquired, s.lot_id)):
            on_sale(self, sale, sources[sale.lot_id])
        return sales

    def cash_distribution(self, day: date, permno: int, eligible_shares: Decimal,
                          amount_per_share: Decimal, event_id: str, kind: str) -> None:
        if eligible_shares < 0 or amount_per_share < 0:
            raise ValueError("A cash dividend needs nonnegative entitlement and amount.")
        self.register_event(permno, event_id)
        gross = eligible_shares * amount_per_share
        self.post_cash(gross)
        self.entries.append(Entry(day, kind, permno, eligible_shares, gross, event_id))

    def split(self, day: date, permno: int, new_per_old: Decimal | Fraction,
              event_id: str, *, exact_shares: dict[int, Fraction] | None = None,
              unverified_ratio: bool = False,
              dividend_entitlements: Mapping[int, Decimal] | None = None) -> tuple[SplitPiece, ...]:
        if new_per_old <= 0:
            raise ValueError("Split ratio must be new shares per old share and positive.")
        if (permno, event_id) in self._events:
            raise ValueError(f"Corporate action already booked: {(permno, event_id)}")
        groups = self.trading_lots(permno)
        before = sum_shares(l.shares for parts in groups.values() for l in parts)
        exact_shares = {} if exact_shares is None else exact_shares
        factor = Fraction(new_per_old)
        old_sizes = {l.id: exact_shares.get(l.id, l.units) for parts in groups.values() for l in parts}
        new_sizes = {i: size * factor for i, size in old_sizes.items()}
        if dividend_entitlements is not None:
            # 股票股息只发给前收盘的持仓。同日从其他股类分来的新股没有这次领取资格。
            for i, size in old_sizes.items():
                eligible = dividend_entitlements.get(i, ZERO)
                if eligible != self.lots[i].shares:
                    new_sizes[i] = size + Fraction(eligible) * (factor - 1)
        pieces = ()
        if new_per_old < 1:
            pieces = bind_split(self, permno, factor, event_id, self.split_designations.get(event_id), old_sizes)
        else:
            totals = (sum(new_sizes[l.id] for l in parts) for parts in groups.values())
            for parts, total in zip(groups.values(), share_increments(totals)):
                for lot, size in zip(parts, apportion(total, [new_sizes[l.id] for l in parts])):
                    lot.shares = size
        if pieces:
            new_sizes = {part.segment: part.exact_shares for part in pieces}
            after = sum_shares(part.shares for part in pieces)
        else:
            after = sum_shares(lot.shares for parts in groups.values() for lot in parts)
        exact_shares.update(new_sizes)
        for i, units in new_sizes.items():
            self.lots[i].exact_shares = units
        self._events.add((permno, event_id))
        for loss in self.open_losses.get(permno, ()):
            if loss.sale.day <= day <= loss.sale.day + WINDOW:
                loss.shares *= factor
                if unverified_ratio:
                    loss.unverified_splits += (event_id,)
        change = sum_shares((after, before.copy_negate()))
        self.entries.append(Entry(day, "SPLIT", permno, change, ZERO, event_id))
        return pieces

    def return_of_capital(self, day: date, permno: int, eligible_lots: Mapping[int, Decimal],
                          amount_per_share: Decimal, event_id: str) -> None:
        if amount_per_share < 0:
            raise ValueError("ROC amount per share must be nonnegative.")
        self.register_event(permno, event_id)
        for lot_id, shares in eligible_lots.items():
            lot = self.lots[lot_id]
            payment = shares * amount_per_share
            self.post_cash(payment)
            self.entries.append(Entry(day, "ROC", permno, ZERO, payment, event_id))
            if lot.basis is None:
                self.sales.append(Sale(day, lot.id, permno, ZERO, payment,
                                       None, lot.holding_start, "ROC_BASIS_PENDING", event_id=event_id,
                                       trading_lot_id=lot.trading_id, source_lot_id=lot.source_id))
                continue
            reduction = min(lot.basis, payment)
            lot.basis -= reduction
            # 超过本批次剩余 basis 的部分是 capital gain，实际股数不变。
            if payment > reduction:
                self.sales.append(Sale(day, lot.id, permno, ZERO, payment - reduction,
                                       ZERO, lot.holding_start, "ROC_EXCESS", event_id=event_id,
                                       trading_lot_id=lot.trading_id, source_lot_id=lot.source_id))

    def nav(self, closes: Mapping[int, Decimal]) -> Decimal | None:
        positions = self.holdings()
        if self.prelistings:
            closes = {**closes, **{p: g.price for p, g in self.prelistings.items()}}
        if any(closes.get(permno, ZERO) <= 0 for permno in positions):
            return None
        receivable = sum((p.pending_shares * p.price for p in self.prelistings.values()), ZERO)
        return self.cash + sum((q * closes[p] for p, q in positions.items()), ZERO) + receivable

    def tax_year(self, year: int, *, basis_verified: bool, wash_reviewed: bool | None = None) -> TaxYear:
        """兼容旧的 wash_reviewed 参数。应税损益按实际配对，定稿日期由 settle_years 判断。"""
        sales = [s for s in self.sales if s.day.year == year]
        known = [s for s in sales if s.basis is not None]
        st = sum((s.taxable_gain for s in known if not s.is_long_term), ZERO)
        lt = sum((s.taxable_gain for s in known if s.is_long_term), ZERO)
        dividends = sum((e.cash_change for e in self.entries
                         if e.day.year == year and e.kind == "CASH_DIVIDEND"), ZERO)
        dividends += sum((i.gross for i in self.distribution_income if i.day.year == year), ZERO)
        last_loss = max((s.day for s in known if s.wash_sale_candidate), default=None)
        verified = basis_verified and len(known) == len(sales)
        return TaxYear(year, st, lt, dividends, last_loss, verified)
