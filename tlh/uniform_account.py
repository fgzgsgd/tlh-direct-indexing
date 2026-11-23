"""在同一个账户上逐期运行策略或重放已有指令，按实际成交和收盘持股估值。"""

from bisect import bisect_left, bisect_right
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from decimal import Decimal
from math import prod
from pathlib import Path

from .actions import (AccountAction, ActionResult, CorporateAction, PropertyDividend,
                      SpinOff, fractional_sales, prelisting_cil)
from .execution import DEFAULT_COMMISSION, Fee, Order, Sizer, buy_shares, floor_shares, lock_orders
from .ledger import Entry, Ledger, Prelisting, Sale
from .market import Delisting, StockDay, read_close_halts
from .positions import read_weights
from .reverse_split import SplitDesignation
from .shares import apportion, fraction, sum_shares
from .spin_off_basis import close_allocations
from .strategy import CloseSnapshot, LotRelief, Strategy, Target
from .tax import ZERO, Scenario, Statement, TaxSettlement, settle_years
from .wash import WINDOW

INITIAL_CASH = Decimal("1000000")


def _valid_price(price: Decimal | None) -> bool:
    return price is not None and Decimal(price).is_finite() and price > 0


@dataclass(frozen=True, slots=True)
class Rebalance:
    signal_day: date
    trade_day: date
    next_trade_day: date

    def upcoming_delistings(self, delistings: Iterable[Delisting]) -> dict[int, list[Delisting]]:
        """包含未持有的证券。此处有意使用事后获知的退市日期。"""
        upcoming: dict[int, list[Delisting]] = {}
        for event in delistings:
            if self.trade_day < event.day <= self.next_trade_day:
                upcoming.setdefault(event.permno, []).append(event)
        return upcoming


def rebalances(sessions: Sequence[date], first_trade: date, end: date) -> Iterator[Rebalance]:
    """输入真实交易日历，包括首笔信号日以及末次调仓的完整前瞻窗口。"""
    if any(a >= b for a, b in zip(sessions, sessions[1:])):
        raise ValueError("Trading sessions must be unique and sorted.")
    index = bisect_left(sessions, first_trade)
    if not (0 < index < len(sessions) and sessions[index] == first_trade <= end):
        raise ValueError("Supply a first trade session, its previous session, and a valid end date.")
    while sessions[index] <= end:
        following = bisect_left(sessions, sessions[index] + timedelta(days=31))
        if following == len(sessions):
            raise ValueError(f"Calendar does not cover the next rebalance after {sessions[index]}.")
        yield Rebalance(sessions[index - 1], sessions[index], sessions[following])
        index = following


@dataclass(frozen=True, slots=True)
class _CloseMark:
    price: Decimal
    source_day: date


@dataclass(frozen=True)
class Session:
    """raw_closes 只存真实收盘价，成交补价另保留原报价日。"""

    execution_prices: Mapping[int, Decimal]
    raw_closes: Mapping[int, Decimal]
    actions: Sequence[AccountAction]
    price_source: str = "CLOSE"
    raw_opens: Mapping[int, Decimal] = field(default_factory=dict)
    tradable: Collection[int] | None = None  # 交易状态独立于旧实验的 VWAP 覆盖。
    halted: frozenset[int] = frozenset()
    _carried: Mapping[int, _CloseMark] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        prices = self.execution_prices
        if self.price_source == "CLOSE":
            prices = {p: px for p, px in prices.items() if px == self.raw_closes.get(p)}
            prices.update((p, mark.price) for p, mark in self._carried.items())
        eligible = self.tradable if self.tradable is not None else prices
        executable = {p: px for p, px in prices.items()
                      if p in eligible and p not in self.halted and _valid_price(px)}
        object.__setattr__(self, "execution_prices", executable)

    def with_execution_closes(self, day: date, marks: Mapping[int, _CloseMark],
                              confirmed_halts: frozenset[int]) -> "Session":
        halted = self.halted | confirmed_halts
        carried = {}
        if self.price_source == "CLOSE" and self.tradable is not None:
            carried = {p: marks[p] for p in self.tradable if p not in halted and p in marks
                       and marks[p].source_day < day}
        if halted == self.halted and carried == self._carried:
            return self
        return replace(self, halted=halted, _carried=carried)

    def price_day(self, day: date, permno: int) -> date:
        return self._carried[permno].source_day if permno in self._carried else day

    def regular_close(self, permno: int) -> Decimal | None:
        tradable = self.tradable
        if tradable is None:
            tradable = self.execution_prices if self.price_source == "CLOSE" else self.raw_closes
        price = self.raw_closes.get(permno)
        if permno not in self.halted and permno in tradable and _valid_price(price):
            return price
        return None

    def price(self, day: date, permno: int, reason: str, source: str | None = None) -> Decimal:
        if source is None:
            source = self.price_source
        if permno in self.halted:
            raise ValueError(f"{day} {permno} {reason}: trading halted at the close.")
        value = self.execution_prices.get(permno)
        if self.price_source != source or not _valid_price(value):
            raise ValueError(f"{day} {permno} {reason}: missing executable {source}.")
        return value


def _closing_marks(previous: dict[int, _CloseMark], day: date, market: Session,
                   expired: Sequence[Delisting]) -> dict[int, _CloseMark]:
    """延用同一证券最近的实际收盘。除权后等新的报价，避免把除权前价格乘上新股数。"""
    marks = previous.copy()
    for event in expired:
        marks.pop(event.permno, None)
    for event in market.actions:
        if not isinstance(event, CorporateAction) or event.kind != "TICKER_CHANGE":
            marks.pop(event.permno, None)
            if child := getattr(event, "spin_off_child", None):
                marks.pop(child, None)
    for p, price in market.raw_closes.items():
        if _valid_price(price):
            marks[p] = _CloseMark(price, day)
    return marks


def sessions_from_crsp(rows: Iterable[StockDay], *, tradable: Mapping[date, set[int]],
                       actions_by_day: Mapping[date, Sequence[AccountAction]],
                       regular_session_vwaps: Mapping[date, Mapping[int, Decimal]] | None = None,
                       close_example: bool = False
                       ) -> dict[date, Session]:
    """用可交易的 DlyClose 成交。显式传入 VWAP 表可重放先前的对照实验。"""
    if close_example and regular_session_vwaps is not None:
        raise ValueError("The CLOSE replay and VWAP table select different execution prices.")
    if tradable.keys() != actions_by_day.keys():
        raise ValueError("Trading status and event coverage must describe the same sessions.")
    closes: dict[date, dict[int, Decimal]] = {d: {} for d in tradable}
    opens: dict[date, dict[int, Decimal]] = {d: {} for d in tradable}
    eligible: dict[date, set[int]] = {d: set() for d in tradable}
    official_halts = read_close_halts()
    halts = {d: set(official_halts.get(d, ())) for d in tradable}
    seen: set[tuple[date, int]] = set()
    for row in rows:
        d, p = row.day, row.permno
        if d not in closes:
            continue
        if (d, p) in seen:
            raise ValueError(f"Duplicate CRSP security-date: {p}, {d}")
        seen.add((d, p))
        if row.trading_status in {"H", "S"}:
            halts[d].add(p)
        if _valid_price(row.raw_close):
            closes[d][p] = row.raw_close
        if p in tradable[d] and (row.trading_status == "A" or (not row.trading_status and p in closes[d])):
            eligible[d].add(p)  # 缺价格的 CRSP 行必须明确标为 Active 才可回补成交。
        if row.raw_open is not None and row.raw_open > 0:
            opens[d][p] = row.raw_open
    prices = closes if regular_session_vwaps is None else regular_session_vwaps
    source = "CLOSE" if regular_session_vwaps is None else "VWAP"
    allowed = eligible if regular_session_vwaps is None else tradable
    return {d: Session(prices.get(d, {}), closes[d], actions_by_day[d], source, opens[d],
                       allowed[d], frozenset(halts[d])) for d in closes}


@dataclass(frozen=True)
class Valuation:
    day: date
    cash: Decimal
    nav: Decimal | None
    holdings: dict[int, Decimal]
    total_basis: Decimal | None
    basis_verified: bool
    daily_return: Decimal | None = None
    market_values: dict[int, Decimal] = field(default_factory=dict)
    pending_entitlements: dict[int, Decimal] = field(default_factory=dict)
    execution_source: str = "CLOSE"
    carried_close_dates: dict[int, date] = field(default_factory=dict)
    tax_cash_flow: Decimal = ZERO  # 当日开盘前实际入账的净额。
    prelisting_marks: dict[int, Decimal] = field(default_factory=dict)  # look-ahead FMV，仅供上市前估值。

    @property
    def actual_weights(self) -> dict[int, Decimal] | None:
        return {p: mv / self.nav for p, mv in self.market_values.items()} if self.nav else None

    @property
    def stock_market_value(self) -> Decimal | None:
        return None if self.nav is None else self.nav - self.cash


@dataclass(frozen=True)
class YearEndWealth:
    as_of: date
    pre_tax: Decimal | None
    after_tax: Decimal | None
    tax: Statement
    unverified_split_ratios: tuple[str, ...] = ()  # 仍影响本期或累计税务结果的 CRSP 事件 ID。
    tax_amount_known: bool = True


@dataclass(frozen=True)
class ExitRecord:
    day: date
    permno: int
    events: tuple[Delisting, ...]
    shares: Decimal
    execution_price: Decimal | None
    weight_before_fill: Decimal | None
    price_source: str
    sales: tuple[Sale, ...]
    price_day: date | None = None


@dataclass
class Run:
    ledger: Ledger
    history: list[Valuation]
    actions: list[tuple[date, ActionResult]]
    exceptions: list[tuple[date, int, str]]
    exits: list[ExitRecord]
    plans: dict[date, tuple[Order, ...]]
    observed_through: date
    targets: dict[date, Target] = field(default_factory=dict)  # 生效日 → 权重及锁定的卖出分配。
    last_prices: Mapping[int, Decimal] = field(default_factory=dict)
    fee: Fee = DEFAULT_COMMISSION
    last_halts: frozenset[int] = frozenset()
    tax_scenario: Scenario = Scenario.EXTERNAL_GAINS
    tax_settlements: list[TaxSettlement] = field(default_factory=list)

    def pre_tax_twr(self, start: date | None = None, end: date | None = None) -> Decimal | None:
        """按收益归属日（含首尾）复合 daily returns。缺一日收益时，整段 TWR 留空。"""
        returns = [v.daily_return for v in self.history
                   if (start is None or start <= v.day) and (end is None or v.day <= end)]
        if not returns or None in returns:
            return None
        return prod((1 + r for r in returns), start=Decimal(1)) - 1

    def _report_scenario(self, scenario: Scenario | None) -> Scenario:
        if scenario is not None and scenario != self.tax_scenario and any(s.scenario_dependent for s in self.tax_settlements):
            raise ValueError(f"Tax cash flows differ. Run a separate backtest with tax_scenario={scenario.value}.")
        return scenario or self.tax_scenario

    def annual_wealth(self, scenario: Scenario | None = None, *, wash_reviewed_years: set[int] | None = None
                      ) -> list[YearEndWealth]:
        return self._wealth(self.ledger, self.history, self._report_scenario(scenario))

    def liquidated_wealth(self, scenario: Scenario | None = None, *, prices: Mapping[int, Decimal] | None = None,
                          fee: Fee | None = None, wash_reviewed_years: set[int] | None = None) -> YearEndWealth:
        """以结束交易日的同源成交价在独立账本上清算。旧接口传入的价格须与行情一致。"""
        scenario = self._report_scenario(scenario)
        fee = self.fee if fee is None else fee
        book = deepcopy(self.ledger)
        last = self.history[-1]
        if any(p.fractions for p in book.prelistings.values()):
            raise ValueError(f"{last.day} LIQUIDATION: unlisted CIL entitlement awaits its first regular close.")
        for p in book.holdings():
            if p in self.last_halts:
                raise ValueError(f"{last.day} {p} LIQUIDATION: trading halted at the close.")
            if not _valid_price(self.last_prices.get(p)) or (prices is not None and not _valid_price(prices.get(p))):
                raise ValueError(f"{last.day} {p} LIQUIDATION: missing executable {last.execution_source}.")
            if prices is not None and prices[p] != self.last_prices[p]:
                raise ValueError(f"{last.day} {p}: liquidation price differs from the session {last.execution_source}.")
        prices = self.last_prices
        if last.nav is None:
            unavailable = replace(last, nav=None, basis_verified=False)
            return self._wealth(book, [*self.history[:-1], unavailable], scenario)[-1]
        for p in book.holdings():
            lots = book.trading_positions(p)
            fees = {i: fee(last.day, p, q, prices[p]) for i, q in lots.items()}
            price_day = last.carried_close_dates.get(p, last.day) if last.execution_source == "CLOSE" else last.day
            book.sell(last.day, p, lots, prices[p], fees, "LIQUIDATION", price_source=last.execution_source,
                      price_day=price_day)
        terminal = replace(last, cash=book.cash, nav=book.cash, holdings={}, total_basis=ZERO, market_values={})
        return self._wealth(book, [*self.history[:-1], terminal], scenario)[-1]

    def _wealth(self, book: Ledger, history: Sequence[Valuation], scenario: Scenario,
                as_of: date | None = None) -> list[YearEndWealth]:
        as_of = as_of or self.observed_through
        year_ends = {v.day.year: v for v in history}
        unverified = [(d.year, event.reference) for d, event in self.actions if event.ratio_needs_review]
        # 空仓时的合股只影响仍待配对的股数。窗口结束后，只有实际用过该比例的配对影响原亏损年。
        unverified.extend((match.sale_day.year, ref) for match in book.wash_matches for ref in match.unverified_splits)
        for losses in book.open_losses.values():
            for loss in losses:
                if as_of < loss.sale.day + WINDOW:
                    unverified.extend((loss.sale.day.year, ref) for ref in loss.unverified_splits)
        incomes = [book.tax_year(y, basis_verified=v.basis_verified)
                   for y, v in year_ends.items()]
        reports = []
        verified = True
        paid = iter(self.tax_settlements)
        settlement = next(paid, None)
        settled_flow = ZERO
        for income, tax in zip(incomes, settle_years(incomes, scenario, as_of)):
            valuation = year_ends[income.year]
            while settlement and settlement.day <= valuation.day:
                settled_flow += settlement.cash_flow
                settlement = next(paid, None)
            ratios = tuple(dict.fromkeys(ref for year, ref in unverified if year <= income.year))
            if ratios:
                tax = replace(tax, final=False)  # 仍列按 CRSP 原值模拟的税额与财富。
            verified = verified and income.basis_verified
            after = (tax.after_tax_wealth(valuation.nav, settled_flow)
                     if valuation.nav is not None and verified else None)
            reports.append(YearEndWealth(valuation.day, valuation.nav, after, tax, ratios, verified))
        return reports


class AccountEngine:
    """唯一能过账的账户。研究端读 close_snapshot，文件重放走相同的两个步骤。"""

    def __init__(self, initial_cash: Decimal = INITIAL_CASH, *, start: date, fee: Fee = DEFAULT_COMMISSION,
                 size: Sizer = floor_shares, lot_relief: LotRelief | None = None,
                 price_source: str = "CLOSE", close_example: bool = False,
                 split_designations: Mapping[str, SplitDesignation] | None = None,
                 basis_days: Mapping[date, Session] | None = None, tax_scenario: Scenario = Scenario.EXTERNAL_GAINS):
        self.result = Run(Ledger(initial_cash, split_designations=split_designations), [], [], [], [], {}, start,
                          fee=fee, tax_scenario=tax_scenario)
        self._tax_review_end: date | None = None
        self.fee, self.size, self.lot_relief = fee, size, lot_relief
        self._basis_days = basis_days or {}  # 只用于免税分拆的 CLOSE 取价。
        self.blocked: set[int] = set()
        self.price_source = "CLOSE" if close_example else price_source
        if self.price_source not in {"CLOSE", "VWAP"}:
            raise ValueError(f"Unknown execution price source: {self.price_source}.")
        self.last_weights: dict[int, Decimal] | None = None
        self.last_nav: Decimal | None = initial_cash
        self._marks: dict[int, _CloseMark] = {}
        self._halts = read_close_halts()
        self.basis_pending: set[int] = set()
        self.unsettled_lots: set[int] = set()
        self.economic_pending = self.tax_pending = False

    def _opening_tax(self, day: date) -> tuple[list[TaxSettlement], date | None]:
        """在已有账本上预览一月划款或观察期后的补差。调用本身不改余额。"""
        result = self.result
        if not result.history or (day.year == result.observed_through.year
                                  and (self._tax_review_end is None or day <= self._tax_review_end)):
            return [], self._tax_review_end
        prior = [v for v in result.history if v.day.year < day.year]
        as_of = day - timedelta(days=1)
        scenario = result.tax_scenario
        alternative = Scenario.LIMITED_USE if scenario == Scenario.EXTERNAL_GAINS else Scenario.EXTERNAL_GAINS
        reports = result._wealth(result.ledger, prior, scenario, as_of)
        comparison = result._wealth(result.ledger, prior, alternative, as_of)
        previous = {s.tax_year: s for s in result.tax_settlements}
        due = []
        next_review = min((r.tax.review_end for r in reports if r.tax.review_end >= day), default=None)
        for report, other in zip(reports, comparison):
            tax = report.tax
            old = previous.get(tax.year)
            if not report.tax_amount_known:
                refs = [r.reference for d, r in result.actions if d.year <= tax.year and r.status != "RESOLVED"]
                raise ValueError(f"{day} TAX_SETTLEMENT for {tax.year}: incomplete tax basis or consideration. "
                                 f"Review the source events before funding: {', '.join(dict.fromkeys(refs))}.")
            # 资金在该税年的 wash-sale 窗口结束后结清；报告仍可保留估算比例的来源标记。
            window_closed = as_of >= tax.review_end
            if old and (old.final or not window_closed):
                continue
            flow = (old.net_tax if old else ZERO) - tax.net_tax
            due.append(TaxSettlement(day, tax.year, tax.net_tax, flow, window_closed, tax.net_tax != other.tax.net_tax))
        return due, next_review

    def close_snapshot(self, day: date, closes: Mapping[int, Decimal], *, trade_day: date | None = None,
                       blocked: frozenset[int] = frozenset()) -> CloseSnapshot:
        marks = self._marks if day == self.result.observed_through else {}
        prices = {p: m.price for p, m in marks.items()}
        observed = {p: px for p, px in closes.items() if _valid_price(px)}
        prices.update(observed)
        carried = {p: m.source_day for p, m in marks.items() if p not in observed and m.source_day < day}
        pending = self.basis_pending | self.unsettled_lots
        due, _ = self._opening_tax(trade_day) if trade_day and day == self.result.observed_through else ([], None)
        return CloseSnapshot.from_ledger(day, self.result.ledger, prices, self.last_nav, pending,
                                         trade_day=trade_day, blocked=blocked, carried_close_dates=carried,
                                         scheduled_tax_flow=sum((s.cash_flow for s in due), ZERO))

    def prepare(self, signal_day: date, closes: Mapping[int, Decimal], trade_day: date,
                target: Target, blocked: set[int], *, snapshot: CloseSnapshot | None = None) -> tuple[Order, ...]:
        weights = target.active_weights(trade_day)
        if not self.result.history and signal_day == self.result.observed_through:
            # 逐日 API 的第一笔锁单也提供了建仓前的实际收盘价。
            self._marks.update({p: _CloseMark(px, signal_day) for p, px in closes.items() if _valid_price(px)})
        if not target.rebalance:
            if self.last_weights is None and weights:
                raise ValueError(f"{trade_day}: initial construction must be marked rebalance=1.")
            if target.sell_lots:
                raise ValueError(f"{trade_day}: sell_lots requires rebalance=1.")
            if weights != (self.last_weights or {}):
                raise ValueError(f"{trade_day}: target weights changed on a day marked rebalance=0.")
            self.result.plans.pop(signal_day, None)
            self.result.targets[trade_day] = Target(weights, False)
            return ()
        if snapshot is None:
            snapshot = self.close_snapshot(signal_day, closes, trade_day=trade_day, blocked=frozenset(blocked))
        selector = target.lot_relief if target.lot_relief is not None else self.lot_relief
        orders = lock_orders(snapshot, weights, self.size, selector, blocked, target.sell_lots)
        self.result.plans[signal_day] = orders
        locked = {o.permno: dict(o.lots) for o in orders if o.quantity < 0}
        self.result.targets[trade_day] = Target(weights, True, locked)
        self.blocked = set(blocked)
        return orders

    def advance(self, day: date, market: Session, orders: Sequence[Order],
                forced_exits: Mapping[int, list[Delisting]], expired: Sequence[Delisting]) -> None:
        result = self.result
        if day <= result.observed_through:
            raise ValueError(f"{day}: session was already booked or is out of order.")
        if len({order.permno for order in orders}) != len(orders):
            raise ValueError(f"{day}: supply one net order per PERMNO.")
        orders = [order.checked(day) for order in orders]
        # A split changes the units of the previously locked order. Cancel all
        # discretionary orders for that security; corporate actions still post.
        split_names = {e.permno for e in market.actions
                       if isinstance(e, CorporateAction) and e.kind == "SPLIT"}
        cancelled = [order for order in orders if order.permno in split_names]
        orders = [order for order in orders if order.permno not in split_names]
        book = result.ledger
        unlisted = {p for p, grant in book.prelistings.items() if day < grant.first_trade}
        unlisted.update(e.spin_off_child for e in market.actions if isinstance(e, PropertyDividend)
                        and e.first_trade and day < e.first_trade)
        if unlisted:
            intents = {o.permno for o in orders} | (forced_exits.keys() & book.holdings().keys())
            if affected := unlisted & intents:
                raise ValueError(f"{day}: orders for PERMNO {sorted(affected)} precede regular-way trading.")
            market = replace(market, raw_closes={p: v for p, v in market.raw_closes.items() if p not in unlisted},
                             halted=market.halted | unlisted)
        marks = _closing_marks(self._marks, day, market, expired)
        for p in unlisted | book.prelistings.keys():
            if not _valid_price(market.raw_closes.get(p)):
                marks.pop(p, None)  # 临时估值和 when-issued 行情均不能成为上市后的前值成交价。
        market = market.with_execution_closes(day, marks, self._halts.get(day, frozenset()))
        listed = [p for p, grant in book.prelistings.items() if day >= grant.first_trade]
        due, next_review = self._opening_tax(day)
        flow = sum((s.cash_flow for s in due), ZERO)
        if not (market.actions or orders or forced_exits or expired or due or listed):
            self._mark_close(day, market, marks)
            self._tax_review_end = next_review
            return
        pending, unsettled = self.basis_pending.copy(), self.unsettled_lots.copy()
        economic, tax = self.economic_pending, self.tax_pending
        actions, exits, exceptions = len(result.actions), len(result.exits), len(result.exceptions)
        with result.ledger.transaction():
            try:
                for order in cancelled:
                    held = book.holdings().get(order.permno, ZERO)
                    side = "BUY" if order.quantity > 0 else "SALE"
                    result.exceptions.append((day, order.permno,
                                              f"SKIPPED_SPLIT_{side} {abs(order.quantity):f}/{held:f} pre-split shares"))
                for p in listed:
                    grant = book.prelistings.pop(p)
                    if grant.fractions:
                        price = market.regular_close(p)
                        if day != grant.first_trade or price is None:
                            raise ValueError(f"{day} {p} CASH_IN_LIEU: missing first regular CLOSE on {grant.first_trade}.")
                        lots = {lot.id: lot.units for lot in grant.fractions}
                        book.lots.update((lot.id, replace(lot)) for lot in grant.fractions)  # 失败后仍可重试。
                        book.sell(day, p, lots, price, dict.fromkeys(lots, ZERO), "CASH_IN_LIEU",
                                  event_id=grant.reference, price_source="CLOSE", price_day=day)
                orders = self._post_actions(day, market, expired, orders, forced_exits)
                self._force_exits(day, market, forced_exits, marks)
                self._execute(day, market, orders, due)
                self._mark_close(day, market, marks, flow)
            except Exception:
                self.basis_pending, self.unsettled_lots = pending, unsettled
                self.economic_pending, self.tax_pending = economic, tax
                del result.actions[actions:], result.exits[exits:], result.exceptions[exceptions:]
                raise
        result.tax_settlements.extend(due)
        self._tax_review_end = next_review

    def _post_actions(self, day: date, market: Session, expired: Sequence[Delisting],
                      orders: Sequence[Order], forced_exits: Mapping[int, list[Delisting]]) -> Sequence[Order]:
        result, book = self.result, self.result.ledger
        entitled = book.entitled_lots() if market.actions else {}
        deferred_cil = prelisting_cil(day, market.actions, entitled)
        share_events: dict[int, list[ActionResult]] = {}
        order_conflicts = {p: ["pre-delisting exit window"] for p in forced_exits}
        for event in expired:
            order_conflicts.setdefault(event.permno, []).append(f"delisted: {event.day}")
        allocations = close_allocations(day, market, self._basis_days)
        for event in market.actions:
            if isinstance(event, SpinOff):
                record = event.post(day, book, entitled, allocation=allocations.get(event.reference))
            elif isinstance(event, PropertyDividend):
                record = event.post(day, book, entitled, reserved_cil=deferred_cil.get(event.reference, 0))
            else:
                record = event.post(day, book, entitled)
            result.actions.append((day, record))
            # 已核实的分拆先分配 basis，母股原单仍按相同股数和原 lots 交割。
            child = getattr(event, "spin_off_child", None)
            if child is not None and (record.status != "RESOLVED" or record.effective_shares_per_parent is None):
                order_conflicts.setdefault(event.permno, []).append(f"spin-off parent: {event.reference}")
                if record.status == "ECONOMIC_PENDING" and entitled.get(event.permno):
                    reason = f"unknown spin-off entitlement: {event.reference} (parent PERMNO {event.permno})"
                    # 子股身份也未知时，无法从当日订单中排除任何一只证券。
                    affected = (child,) if child else (order.permno for order in orders)
                    for permno in affected:
                        order_conflicts.setdefault(permno, []).append(reason)
            if record.status != "RESOLVED":
                self.basis_pending.update(record.affected_lots)
                self.tax_pending = True
                if record.status == "ECONOMIC_PENDING":
                    self.economic_pending = True
                    order_conflicts.setdefault(event.permno, []).append(f"unresolved terms: {record.reference}")
            if record.ratio_needs_review or record.wash_loss_days:
                result.exceptions.append((day, event.permno,
                                          f"UNVERIFIED_SPLIT_RATIO {record.reference} raw ratio {event.value}"))
            if isinstance(event, CorporateAction) and record.status == "RESOLVED":
                if event.kind in {"SPLIT", "NONTAXABLE_STOCK_DIVIDEND"}:
                    p = event.permno
                    if event.kind == "NONTAXABLE_STOCK_DIVIDEND":
                        order_conflicts.setdefault(p, []).append(f"{event.kind}: {event.reference}")
                    if record.split_pieces:
                        self._carry_entitlements(record, entitled.setdefault(p, {}))
                    share_events.setdefault(p, []).append(record)
                elif event.kind == "CASH_ACQUISITION":
                    order_conflicts.setdefault(event.permno, []).append(f"cash acquisition: {event.reference}")
            if record.received_lots:
                receiver = book.lots[record.received_lots[0]].permno
                share_events.setdefault(receiver, []).append(record)

        # 当日 ex-date 分配先按前收盘资格入账，零股结算在主动订单之前。
        if orders and (order_conflicts or share_events):
            _check_locked_orders(day, book, orders, order_conflicts, share_events)
        for p, records in share_events.items():
            self._settle_cil(day, market, p, "|".join(record.reference for record in records))

        self.unsettled_lots.intersection_update(book.lots)
        for event in expired:
            held = tuple(l.id for l in book.lots.values() if l.permno == event.permno and l.acquired <= event.day)
            if held:
                self.unsettled_lots.update(held)
                result.actions.append((day, ActionResult(f"Delisting/{event.permno}/{event.day}", event.permno,
                                        "ECONOMIC_PENDING", f"Unsettled delisting: {event.reason}", held)))
        return orders

    def _settle_cil(self, day: date, market: Session, permno: int, reference: str) -> None:
        book = self.result.ledger
        lots = fractional_sales(book, permno)
        if lots:
            price = market.price(day, permno, "CASH_IN_LIEU", self.price_source)
            book.sell(day, permno, lots, price, dict.fromkeys(lots, ZERO), "CASH_IN_LIEU",
                      event_id=reference, price_source=market.price_source, price_day=market.price_day(day, permno))

    def _carry_entitlements(self, record: ActionResult, entitled: dict[int, Decimal]) -> None:
        """旧 segment 分到多股时，同日预先锁定的 ROC、spin-off 领取资格随原股转移。"""
        descendants = {}
        for piece in record.split_pieces:
            descendants.setdefault(piece.old_segment, []).append(piece)
        for old, pieces in descendants.items():
            if old in entitled:
                portions = apportion(entitled.pop(old), [piece.old_fraction for piece in pieces])
                entitled.update((piece.segment, q) for piece, q in zip(pieces, portions))
            for pending in (self.basis_pending, self.unsettled_lots):
                if old in pending or pieces[0].old_trading_lot in pending:
                    pending.discard(old)
                    pending.update(piece.segment for piece in pieces)

    def _force_exits(self, day: date, market: Session, forced_exits: Mapping[int, list[Delisting]],
                     marks: Mapping[int, _CloseMark]) -> None:
        result, book = self.result, self.result.ledger
        if not forced_exits:
            return
        positions = book.holdings()
        prices = ({p: marks[p].price for p in positions if p in marks} if market.price_source == "CLOSE"
                  else market.execution_prices)
        trade_nav = None if self.economic_pending or self.unsettled_lots else book.nav(prices)
        for p in sorted(forced_exits.keys() & positions.keys()):
            price = market.execution_prices.get(p)
            weight = positions[p] * price / trade_nav if price and trade_nav else None
            lots = book.trading_positions(p)
            sales = self._sell(day, market, p, lots, "PRE_DELISTING_EXIT")
            result.exits.append(ExitRecord(day, p, tuple(forced_exits[p]), positions[p], price,
                                          weight, market.price_source, tuple(sales), market.price_day(day, p)))

    def _execute(self, day: date, market: Session, orders: Sequence[Order],
                 settlements: Sequence[TaxSettlement] = ()) -> None:
        book = self.result.ledger
        buys = []
        for order in orders:
            p = order.permno
            if order.quantity < 0:
                lots = order.lots
                available = book.trading_positions(p)
                for lot_id, qty in lots.items():
                    if qty > available.get(lot_id, ZERO):
                        raise ValueError(f"{day} LOCKED_ORDER_REVIEW: PERMNO {p} lot {lot_id} needs {qty} shares, "
                                         f"available {available.get(lot_id, ZERO)}.")
                groups = book.trading_lots(p)
                pending = self.basis_pending | self.unsettled_lots
                if any(i in pending or any(lot.id in pending or lot.basis is None for lot in groups[i])
                       for i in lots):
                    raise ValueError(f"{day} {p}: selected lots have unresolved basis or delisting terms.")
                self._sell(day, market, p, lots, "TRADE")
            else:
                market.price(day, p, "BUY", self.price_source)
                buys.append(order)
        # All scheduled tax cash flows settle after original sales and before buys.
        # No additional sale, order resizing, or tax reserve is introduced.
        flow = sum((s.cash_flow for s in settlements), ZERO)
        if flow < -book.cash:
            years = ", ".join(str(s.tax_year) for s in settlements if s.cash_flow < 0)
            raise ValueError(f"{day} TAX_SETTLEMENT for {years} after original sales: "
                             f"required {-flow}, available cash {book.cash}.")
        for settlement in settlements:
            if settlement.cash_flow:
                book.post_cash(settlement.cash_flow)
                book.entries.append(Entry(day, "TAX_SETTLEMENT", None, ZERO, settlement.cash_flow,
                                          str(settlement.tax_year)))
        buy_shares(book, day, buys, market.execution_prices, self.fee, market.price_source,
                   price_days={o.permno: market.price_day(day, o.permno) for o in buys})

    def _mark_close(self, day: date, market: Session, marks: dict[int, _CloseMark], tax_flow: Decimal = ZERO) -> None:
        result, book = self.result, self.result.ledger
        positions = book.holdings()
        # 人工事件也必须遵守证券级整数股约束。
        fractions = {p: tail for p, q in positions.items() if (tail := fraction(q))}
        if fractions:
            raise ValueError(f"{day}: corporate action left unsettled fractional shares: {fractions}.")
        prices = {p: marks[p].price for p in positions if p in marks}
        carried = {p: marks[p].source_day for p in prices if marks[p].source_day < day}
        prelisting = {p: g.price for p, g in book.prelistings.items() if p in positions or g.fractions}
        prices.update(prelisting)
        nav = None if (self.economic_pending or self.unsettled_lots) else book.nav(prices)
        values = {p: q * prices[p] for p, q in positions.items() if p in prices}
        claims = {p: g.pending_shares for p, g in book.prelistings.items() if g.fractions}
        for p, shares in claims.items():
            values[p] = values.get(p, ZERO) + shares * prelisting[p]
        opening_nav = self.last_nav + tax_flow if self.last_nav is not None else None
        daily_return = nav / opening_nav - 1 if nav is not None and opening_nav else None
        result.history.append(Valuation(day, book.cash, nav, positions, book.total_basis(),
                                        not (self.tax_pending or self.unsettled_lots),
                                        daily_return, values, claims, self.price_source, carried, tax_flow, prelisting))
        target = result.targets.get(day)
        if target is not None:
            self.last_weights = target.weights
            if target.rebalance:
                for p in sorted(self.blocked & target.weights.keys()):
                    result.exceptions.append((day, p, "BUY: blocked by pre-delisting convention"))
        self.last_nav = nav
        self._marks = marks  # 当天成功过账后才推进，失败重试仍从上一个收盘开始。
        result.observed_through = day
        result.last_prices = dict(market.execution_prices) if market.price_source == self.price_source else {}
        result.last_halts = market.halted

    def _sell(self, day: date, market: Session, p: int, lots: Mapping[int, Decimal],
              reason: str) -> list[Sale]:
        price = market.price(day, p, reason, self.price_source)
        fees = {i: self.fee(day, p, q, price) for i, q in lots.items()}
        return self.result.ledger.sell(day, p, dict(lots), price, fees, reason, price_source=market.price_source,
                                       price_day=market.price_day(day, p))


def _check_locked_orders(day: date, book: Ledger, orders: Sequence[Order],
                         direct: Mapping[int, list[str]], share_events: Mapping[int, list[ActionResult]]) -> None:
    conflicts = []
    for order in orders:
        p = order.permno
        reasons = list(direct.get(p, ()))
        receipts = [record for record in share_events.get(p, ()) if record.received_lots and record.permno != p]
        if receipts and not reasons:
            received = sum(book.lots[lot_id].units for record in receipts
                           for lot_id in record.received_lots if lot_id in book.lots)
            # 直接冲突已足以暂停。其余子股订单只比较有无分股时能保留的整股数。
            total = sum(lot.units for lot in book.lots.values() if lot.permno == p)
            retained = int(total) - int(total - received)
            if retained:
                events = ", ".join(f"{r.reference} (parent PERMNO {r.permno})" for r in receipts)
                reasons.append(f"received={received}, additional whole shares={retained} | {events}")
        if not reasons:
            continue
        side = "BUY" if order.quantity > 0 else "SELL"
        lots = "|".join(f"{lot_id}:{qty:f}" for lot_id, qty in order.lots.items())
        allocation = f" sell_lots={lots}" if lots else ""
        conflicts.append(f"  PERMNO {p} {side} {abs(order.quantity):f} shares{allocation} | " + " | ".join(reasons))
    if conflicts:
        raise ValueError(f"{day} LOCKED_ORDER_REVIEW: review the locked orders below.\n" + "\n".join(conflicts))


def _delisting_windows(sessions: Sequence[date], schedule: set[date], first: date, end: date,
                       next_rebalance: date | None, known_through: date) -> dict[date, Rebalance]:
    flagged = sorted(d for d in schedule if first <= d <= end)
    tail = min((d for d in schedule if d > end), default=next_rebalance)
    if next_rebalance is not None and tail != next_rebalance:
        raise ValueError("The supplied next rebalance date conflicts with the schedule.")
    if tail is None or tail <= end or tail not in sessions:
        raise ValueError("Pre-delisting needs the scheduled rebalance session after the sample.")
    if known_through < tail:
        raise ValueError("Delisting coverage ends before the last required pre-delisting window.")
    return {trade: Rebalance(sessions[bisect_left(sessions, trade) - 1], trade, following)
            for trade, following in zip(flagged, [*flagged[1:], tail])}


def run(*, sessions: Sequence[date], first_trade: date, end: date,
        days: Mapping[date, Session], delistings: Sequence[Delisting],
        delistings_known_through: date, initial_cash: Decimal = INITIAL_CASH, fee: Fee = DEFAULT_COMMISSION,
        size: Sizer = floor_shares,
        lot_relief: LotRelief | None = None, pre_delisting: bool = True,
        next_rebalance: date | None = None, weights_file: Path | None = None,
        strategy: Strategy | None = None, rebalance_days: Iterable[date] | None = None,
        exit_days: Iterable[date] | None = None, close_example: bool = False,
        split_designations: Mapping[str, SplitDesignation] | None = None,
        tax_scenario: Scenario = Scenario.EXTERNAL_GAINS, engine_factory=AccountEngine) -> Run:
    """逐期调用策略或重放权重文件，两种入口共用同一套计划、执行与记账逻辑。"""
    if any(a >= b for a, b in zip(sessions, sessions[1:])):
        raise ValueError("Trading sessions must be unique and sorted.")
    start = bisect_left(sessions, first_trade)
    if not 0 < start < len(sessions) or sessions[start] != first_trade or end < first_trade:
        raise ValueError("Supply a valid first trade date and its preceding close.")
    if (strategy is None) == (weights_file is None):
        raise ValueError("Supply either a strategy or a weight file.")
    if strategy is None:
        if rebalance_days is not None:
            raise ValueError("In replay mode the weight file specifies rebalance dates.")
        targets = read_weights(weights_file)
        schedule = {d for d, target in targets.items() if target.rebalance}
    else:
        targets = None
        schedule = set(rebalance_days or ())
        if rebalance_days is not None and first_trade not in schedule:
            raise ValueError(f"{first_trade}: initial construction must be on the rebalance schedule.")
    exit_schedule = schedule if exit_days is None else set(exit_days)
    if pre_delisting and first_trade not in exit_schedule:
        raise ValueError("Pre-delisting requires a known exit schedule beginning on first_trade.")
    calendar = set(sessions)
    input_days = targets.keys() if targets is not None else schedule
    if invalid := {d for d in (set(input_days) | exit_schedule) if first_trade <= d <= end and d not in calendar}:
        raise ValueError(f"Target dates outside the trading calendar: {sorted(invalid)}")
    by_day = (_delisting_windows(sessions, exit_schedule, first_trade, end, next_rebalance, delistings_known_through)
              if pre_delisting else {})
    after: dict[date, list[Delisting]] = {}
    on_date: dict[date, set[int]] = {}
    for event in delistings:
        on_date.setdefault(event.day, set()).add(event.permno)
        following = bisect_right(sessions, event.day)
        if following < len(sessions):
            after.setdefault(sessions[following], []).append(event)
    # 行情入口已选定整次运行的价格源；缺某日价格时不切换。
    engine = engine_factory(initial_cash, fee=fee, size=size, lot_relief=lot_relief, start=sessions[start - 1],
                           price_source=days[first_trade].price_source, close_example=close_example,
                           split_designations=split_designations, basis_days=days, tax_scenario=tax_scenario)
    # 建仓前只预热收盘行情；账户仍从现金开始，历史分配不入账。
    for prior in sessions[:start]:
        if prior in days:
            engine._marks = _closing_marks(engine._marks, prior, days[prior], after.get(prior, ()))
            for event in days[prior].actions:
                if isinstance(event, PropertyDividend) and event.first_trade and first_trade <= event.first_trade:
                    engine.result.ledger.prelistings[event.spin_off_child] = Prelisting(
                        event.first_trade, event.child_fmv, event.reference)
    blocked: dict[int, list[Delisting]] = {}
    for i in range(start, bisect_right(sessions, end)):
        signal, day = sessions[i - 1], sessions[i]
        window = by_day.get(day)
        if window is not None:
            blocked = window.upcoming_delistings(delistings)
        banned = set(blocked)
        if pre_delisting:
            banned.update(on_date.get(day, ()))  # D 恰为本期 T 时，上期已退出，本日继续禁买。
        closes = days[signal].raw_closes
        snapshot = None
        if targets is not None:
            if day not in targets:
                raise ValueError(f"{day}: missing daily target weights; carry the last file weights forward.")
            target = targets[day]
        elif rebalance_days is None or day in schedule:
            snapshot = engine.close_snapshot(signal, closes, trade_day=day, blocked=frozenset(banned))
            target = strategy(snapshot)
            if target is None:
                target = Target(engine.last_weights or {}, False)
        else:
            target = Target(engine.last_weights, False)
        orders = engine.prepare(signal, closes, day, target, banned, snapshot=snapshot)
        engine.advance(day, days[day], orders, blocked if window else {}, after.get(day, ()))
    return engine.result
