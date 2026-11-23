"""已核实的事件由明确的经济条款驱动。复杂事件可以传入有来源的逐案处理。"""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from fractions import Fraction
from typing import TypeVar

from .ledger import Entry, Ledger, Prelisting, Sale
from .reverse_split import SplitPiece
from .shares import apportion, as_decimal, share_increments, sum_shares
from .tax import DistributionIncome, PropertyIncome, ZERO

Entitlements = Mapping[int, Mapping[int, Decimal]]
Source = TypeVar("Source")


def _cil_allocations(quantities: Mapping[Source, Fraction]) -> dict[Source, Fraction]:
    """先按证券保留整股，再按各来源的零股份额分摊实际 CIL。"""
    tails = {i: tail for i, q in quantities.items() if (tail := q % 1)}
    total = sum(tails.values())
    remainder = total % 1
    return {i: q * remainder / total for i, q in tails.items()} if remainder else {}


def fractional_sales(book: Ledger, permno: int) -> dict[int, Fraction]:
    return _cil_allocations({i: sum(lot.units for lot in parts)
                             for i, parts in book.trading_lots(permno).items()})


@dataclass(frozen=True)
class ActionResult:
    reference: str
    permno: int
    status: str
    detail: str
    affected_lots: tuple[int, ...] = ()
    received_lots: tuple[int, ...] = ()  # 本次分配产生的 lot IDs。
    split_pieces: tuple[SplitPiece, ...] = ()
    ratio_source: str | None = None
    ratio_needs_review: bool = False
    split_ratio: Fraction | None = None
    reported_shares_per_parent: Decimal | Fraction | None = None
    effective_shares_per_parent: Fraction | None = None
    wash_loss_days: tuple[date, ...] = ()  # 未核实的比例换算了哪些在窗亏损。
    close_basis_fraction: Fraction | None = None  # 子股分到分拆前 lot 总 basis 的比例。
    basis_price_day: date | None = None


@dataclass(frozen=True)
class StockCashPayment:
    """发行人公布的未作选择结果。股票对价和实际到账现金分别可核对。"""
    reference: str
    permno: int
    cash_per_share: Decimal
    stock_value_per_share: Decimal  # 股票对价，单位为美元 / 原股。
    conversion_price: Decimal   # 发行人计股价格。
    tax_fractions: tuple[Decimal, Decimal, Decimal, Decimal]  # Ordinary、qualified、capital gain、§1250。

    def entitlements(self, eligible: Decimal) -> tuple[Decimal, Decimal, Fraction]:
        stock_value = eligible * self.stock_value_per_share
        return eligible * self.cash_per_share, stock_value, Fraction(stock_value) / Fraction(self.conversion_price)

    def post(self, day: date, account: Ledger, entitled: Entitlements) -> ActionResult:
        account.register_event(self.permno, self.reference)
        eligible = sum_shares(entitled.get(self.permno, {}).values())
        if not eligible:
            return ActionResult(self.reference, self.permno, "RESOLVED", "No dividend entitlement")
        cash, stock_value, units = self.entitlements(eligible)
        gross = cash + stock_value
        account.post_cash(cash)
        account.entries.append(Entry(day, "DIVIDEND_CASH_PAYMENT", self.permno, eligible, cash, self.reference))
        amounts = (gross * f for f in self.tax_fractions)
        account.distribution_income.append(DistributionIncome(day, self.reference, gross, *amounts))
        shares = next(share_increments([units]))
        # 被动收到的分配使用新 lot，只有主动买入才登记为 wash-sale replacement。
        new = account.receive(day, self.permno, shares, stock_value, day, self.reference)
        new.exact_shares = units
        new.purchase_price = self.conversion_price
        return ActionResult(self.reference, self.permno, "RESOLVED", "Taxable cash-and-stock dividend",
                            received_lots=(new.id,))


@dataclass(frozen=True)
class PropertyDividend:
    """应税配股可交付股票，或按事件条款以 FMV 一次折成现金股息。"""
    reference: str
    permno: int
    spin_off_child: int
    ratio: Fraction
    child_fmv: Decimal
    dividend_per_parent: Decimal
    source: str
    first_trade: date | None = None
    cash_at_fmv: bool = False

    def post(self, day: date, account: Ledger, entitled: Entitlements, *,
             reserved_cil: Fraction | None = None) -> ActionResult:
        shares = sum_shares(entitled.get(self.permno, {}).values())
        child = self.spin_off_child
        delayed = self.first_trade is not None and day < self.first_trade
        if delayed:
            # 首个常规交易日也约束策略独立发出的子股订单。
            grant = account.prelistings.setdefault(child, Prelisting(self.first_trade, self.child_fmv, self.reference))
        if self.cash_at_fmv:
            cash_per_parent = as_decimal(self.ratio * Fraction(self.child_fmv))
            account.cash_distribution(day, self.permno, shares, cash_per_parent, self.reference, "CASH_DIVIDEND")
            return ActionResult(self.reference, self.permno, "RESOLVED", "Cash dividend at issuer FMV",
                                effective_shares_per_parent=self.ratio)

        account.register_event(self.permno, self.reference)
        units = Fraction(shares) * self.ratio
        tail = units % 1 if reserved_cil is None else reserved_cil

        def receive(quantity, prior):
            size = next(share_increments([quantity], starting=prior))
            basis = as_decimal(quantity * Fraction(self.child_fmv))
            lot = account.receive(day, child, size, basis, day, self.reference)
            lot.exact_shares, lot.purchase_price = quantity, self.child_fmv
            return lot

        received = ()
        kept = units - tail if delayed else units
        if kept:
            held = sum(lot.units for lot in account.lots.values() if lot.permno == child)
            received = (receive(kept, held).id,)
        if delayed and tail:
            claim = receive(tail, sum(lot.units for lot in grant.fractions))
            del account.lots[claim.id]  # 上市前单列 CIL，避免计入可交割的股数。
            ref = f"{grant.reference}|{self.reference}" if grant.fractions else self.reference
            account.prelistings[child] = replace(grant, reference=ref, fractions=(*grant.fractions, claim))
        if shares:
            account.distribution_income.append(PropertyIncome(day, self.reference, shares * self.dividend_per_parent))
        return ActionResult(self.reference, self.permno, "RESOLVED", f"Taxable stock distribution ({self.source})",
                            received_lots=received, effective_shares_per_parent=self.ratio)


@dataclass(frozen=True)
class SpinOffTerms:
    child: int
    shares_per_parent: Fraction | Decimal  # 如每 3 股配 1 股，传 Fraction(1, 3)。
    child_basis_fraction: Decimal | None  # 发行人示例，仅供核对。
    source: str
    ratio_is_approximate: bool | None = None  # None 自动识别，True 仅有约数，False 表示原值已核实。
    exchange_fmv: Decimal | None = None  # 紧随分拆的 §367(a) 换股，每股最终 ADS 的发行人 FMV。

    @property
    def effective_ratio(self) -> Fraction | None:
        raw = self.shares_per_parent
        if not isinstance(raw, Decimal) or self.ratio_is_approximate is False:
            return Fraction(raw)  # 发行人写明的分子和分母直接使用。
        literal = Fraction(raw)
        if raw.as_tuple().exponent <= -6 and raw >= 0:
            scale = 1_000_000
            prefix = int(literal * scale)
            if not prefix:
                return None  # 六位全零可对应无限多个分母。
            # 截断或四舍五入能得到 prefix，当且仅当 prefix-0.5 ≤ scale/n < prefix+1。
            min_n = scale // (prefix + 1) + 1
            max_n = (2 * scale) // (2 * prefix - 1)
            if min_n == max_n:
                return Fraction(1, min_n)
            if min_n < max_n:
                return None
        return None if self.ratio_is_approximate else literal


def _spin_off_receipts(account: Ledger, eligible: Mapping[int, Decimal], child: int, ratio: Fraction,
                       exact_shares: Mapping[int, Fraction]):
    """先算子公司证券的精确累计权益，再把每个交易 lot 的份额分给其税务明细。"""
    if not eligible:
        return
    prior = sum((exact_shares.get(lot.id, lot.units) for lot in account.lots.values()
                 if lot.permno == child), Fraction(0))
    groups = {}
    for i, shares in eligible.items():
        parent = account.lots[i]
        groups.setdefault(parent.trading_id, []).append((parent, Fraction(shares) * ratio))
    totals = (sum(size for _, size in parts) for parts in groups.values())
    for parts, quantity in zip(groups.values(), share_increments(totals, starting=prior)):
        sizes = apportion(quantity, [size for _, size in parts])
        for (parent, exact), size in zip(parts, sizes):
            yield parent, exact, size


@dataclass(frozen=True)
class SpinOffBasis:
    price_day: date
    of_original: Fraction
    of_remaining: Fraction  # 同时分出多个子股时，按过账顺序从剩余母股 basis 划出。


@dataclass(frozen=True)
class SpinOff:
    reference: str
    permno: int
    terms: SpinOffTerms
    joint_children: frozenset[int] = frozenset()  # 发行人确认属于同一笔分拆的全部子股。

    @property
    def spin_off_child(self) -> int:
        return self.terms.child

    def post(self, day: date, account: Ledger, entitled: Entitlements, *,
             exact_shares: dict[int, Fraction] | None = None,
             allocation: SpinOffBasis | None = None) -> ActionResult:
        terms = self.terms
        ratio = terms.effective_ratio
        if (not terms.source or terms.child <= 0 or terms.child == self.permno
                or terms.shares_per_parent < 0 or (ratio is not None and ratio <= 0)
                or (terms.child_basis_fraction is not None and not 0 <= terms.child_basis_fraction <= 1)):
            raise ValueError(f"{self.reference}: invalid spin-off terms or missing source.")
        if ratio is None:
            detail = f"SPIN_OFF {terms.child}: no unique 1/n for reported shares per parent {terms.shares_per_parent}"
            review = CaseReview(self.reference, self.permno, "ECONOMIC_PENDING", detail, None)
            return replace(review.post(day, account, entitled), ratio_source=terms.source,
                           reported_shares_per_parent=terms.shares_per_parent)
        account.register_event(self.permno, self.reference)
        pending = []
        received_ids = []
        child_groups = {}
        exact_shares = {} if exact_shares is None else exact_shares
        for parent, exact, size in _spin_off_receipts(account, entitled.get(self.permno, {}), terms.child, ratio, exact_shares):
            basis = (as_decimal(Fraction(parent.basis) * allocation.of_remaining)
                     if parent.basis is not None and allocation is not None else None)
            parent.basis = None if basis is None else sum_shares((parent.basis, basis.copy_negate()))
            periods = parent.prior_periods + ((parent.acquired, day),)
            child = account.receive(day, terms.child, size, basis, parent.holding_start,
                                    self.reference, prior_periods=periods)
            if terms.exchange_fmv is not None and basis is not None:
                # 每个来源单独确认 gain。浮亏留在 ADS 的转入 basis，不与其他来源的 gain 净抵。
                value = as_decimal(exact * Fraction(terms.exchange_fmv))
                if value > basis:
                    account.sales.append(Sale(day, parent.id, self.permno, ZERO, value, basis, parent.holding_start,
                                              "SECTION_367_GAIN", event_id=self.reference,
                                              trading_lot_id=parent.trading_id, source_lot_id=parent.source_id))
                    child.basis = value
            child.exact_shares = exact
            if parent.bound_lot_id is not None:
                if parent.trading_id not in child_groups:
                    child_groups[parent.trading_id] = account.next_lot_id()
                child.bound_lot_id = child_groups[parent.trading_id]
            exact_shares[child.id] = exact
            received_ids.append(child.id)
            if basis is None:
                pending.extend((parent.id, child.id))
        kind = "SPIN_OFF_AND_EXCHANGE" if terms.exchange_fmv is not None else "SPIN_OFF"
        return ActionResult(self.reference, self.permno, "BASIS_PENDING" if pending else "RESOLVED",
                            f"{kind} {terms.child} ({terms.source})",
                            affected_lots=tuple(pending), received_lots=tuple(received_ids), ratio_source=terms.source,
                            reported_shares_per_parent=terms.shares_per_parent, effective_shares_per_parent=ratio,
                            close_basis_fraction=allocation.of_original if allocation else None,
                            basis_price_day=allocation.price_day if allocation else None)


@dataclass(frozen=True)
class CorporateAction:
    reference: str             # 数据源事件 ID 或可追溯的人工 override
    permno: int
    kind: str
    value: Decimal | Fraction | str   # 精确的非终止拆股比例可用 Fraction(new, old)。
    ratio_source: str | None = None
    unverified_ratio: bool = False

    def post(self, day: date, account: Ledger, entitled: Entitlements, *,
             exact_shares: dict[int, Fraction] | None = None) -> ActionResult:
        p, ref = self.permno, self.reference
        eligible_lots = entitled.get(p, {})
        if self.kind == "RIGHTS_WAIVED":
            account.register_event(p, ref)
            account.entries.append(Entry(day, self.kind, p, ZERO, ZERO, ref))
        elif self.kind == "TICKER_CHANGE":
            account.register_event(p, ref)
            account.symbol_history.setdefault(p, []).append((day, str(self.value)))
            account.entries.append(Entry(day, self.kind, p, ZERO, ZERO, ref))
        elif self.kind not in {"CASH_DIVIDEND", "UNCLASSIFIED_CASH", "SPLIT", "NONTAXABLE_STOCK_DIVIDEND",
                               "ROC", "CASH_ACQUISITION"}:
            return CaseReview(ref, p, "ECONOMIC_PENDING", self.kind, None).post(day, account, entitled)
        elif self.kind in {"SPLIT", "NONTAXABLE_STOCK_DIVIDEND"} and isinstance(self.value, (Decimal, Fraction)):
            review = self.unverified_ratio and bool(account.trading_lots(p))
            stock_dividend = eligible_lots if self.kind == "NONTAXABLE_STOCK_DIVIDEND" else None
            pieces = account.split(day, p, self.value, ref, exact_shares=exact_shares,
                                   unverified_ratio=self.unverified_ratio, dividend_entitlements=stock_dividend)
            loss_days = {loss.sale.day for loss in account.open_losses.get(p, ()) if ref in loss.unverified_splits}
            return ActionResult(ref, p, "RESOLVED", self.kind, split_pieces=pieces,
                                ratio_source=self.ratio_source, ratio_needs_review=review, split_ratio=Fraction(self.value),
                                wash_loss_days=tuple(sorted(loss_days)))
        elif not isinstance(self.value, Decimal):
            raise ValueError(f"{ref}: expected a numeric event term.")
        elif self.kind in {"CASH_DIVIDEND", "UNCLASSIFIED_CASH"}:
            eligible = sum(eligible_lots.values(), ZERO)
            account.cash_distribution(day, p, eligible, self.value, ref, self.kind)
            if self.kind == "UNCLASSIFIED_CASH" and eligible:
                lots = [account.lots[i] for i in eligible_lots if i in account.lots]
                for lot in lots:
                    lot.basis = None
                return ActionResult(ref, p, "BASIS_PENDING", self.kind, tuple(lot.id for lot in lots))
        elif self.kind == "ROC":
            account.return_of_capital(day, p, eligible_lots, self.value, ref)
        elif self.kind == "CASH_ACQUISITION":
            if self.value < 0:
                raise ValueError(f"{ref}: negative acquisition consideration.")
            positions = account.trading_positions(p)
            account.register_event(p, ref)
            if positions:
                account.sell(day, p, positions, self.value, dict.fromkeys(positions, ZERO), self.kind, event_id=ref)
            account.entries.append(Entry(day, self.kind + "_EVENT", p, ZERO, ZERO, ref))
        return ActionResult(ref, p, "RESOLVED", self.kind)


@dataclass(frozen=True)
class CaseReview:
    reference: str
    permno: int
    status: str
    detail: str
    post_known: Callable[[Ledger, date], tuple[int, ...]] | None
    spin_off_child: int | None = None  # CRSP 已标明分拆但子股代码缺失时为 0。

    def post(self, day: date, account: Ledger, entitled: Entitlements) -> ActionResult:
        if self.status not in {"RESOLVED", "BASIS_PENDING", "ECONOMIC_PENDING"}:
            raise ValueError(f"{self.reference}: unknown status {self.status}.")
        positions = account.holdings()
        if not positions.get(self.permno) and not entitled.get(self.permno):
            return ActionResult(self.reference, self.permno, "RESOLVED", "No holding or entitlement")
        if self.status != "ECONOMIC_PENDING" and self.post_known is None:
            raise ValueError(f"{self.reference}: specify the known economic posting.")
        account.register_event(self.permno, self.reference)
        affected = (self.post_known(account, day) if self.post_known is not None else
                    tuple(lot.id for lot in account.lots.values() if lot.permno == self.permno))
        if self.status == "BASIS_PENDING":
            # 人工复核可点名整股，也可点名需要核实的一段成本。
            groups = account.trading_lots()
            segments = {lot.id for i in affected for lot in (groups[i] if i in groups else [account.lots[i]])}
            for i in segments:
                account.lots[i].basis = None
            affected = tuple(sorted(segments))
        return ActionResult(self.reference, self.permno, self.status, self.detail, affected)


AccountAction = CorporateAction | SpinOff | PropertyDividend | StockCashPayment | CaseReview


def prelisting_cil(day: date, actions: Sequence[AccountAction], entitled: Entitlements) -> dict[str, Fraction]:
    """同日多种母股配发同一未上市子股时，各自只预留合计后应付现的尾数。"""
    grants = defaultdict(dict)
    for event in actions:
        if isinstance(event, PropertyDividend) and not event.cash_at_fmv and event.first_trade and day < event.first_trade:
            shares = sum_shares(entitled.get(event.permno, {}).values())
            grants[event.spin_off_child][event.reference] = Fraction(shares) * event.ratio
    return {ref: q for group in grants.values() for ref, q in _cil_allocations(group).items()}
