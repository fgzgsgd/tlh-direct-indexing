"""Monthly, long-only SPY holdings replication. All signals are previous-month closes."""

from calendar import monthrange
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

from .lot_relief import hifo
from .market import crsp_table
from .strategy import CloseSnapshot, LotRelief, Target


def monthly_signals(sessions: Sequence[date], first_trade: date, end: date) -> dict[date, date]:
    """Map each first trading day to the immediately preceding month's last session."""
    if any(a >= b for a, b in zip(sessions, sessions[1:])):
        raise ValueError("Supply a sorted trading calendar with unique dates.")
    signals = {trade: signal for signal, trade in zip(sessions, sessions[1:])
               if first_trade <= trade <= end and (signal.year, signal.month) != (trade.year, trade.month)}
    if first_trade not in signals:
        raise ValueError(f"{first_trade}: need the first session of the month and the preceding close.")
    return signals


def stock_targets(stock_amounts: Mapping[int, Decimal]) -> dict[int, Decimal]:
    """Normalize the equity sleeve, allocating net cash and other net assets pro rata."""
    if not stock_amounts or any(not x.is_finite() or x <= 0 for x in stock_amounts.values()):
        raise ValueError("A snapshot requires positive finite equity amounts.")
    total = sum(stock_amounts.values())
    unit = Decimal("1e-18")
    weights = {p: (value / total).quantize(unit, rounding=ROUND_DOWN) for p, value in stock_amounts.items()}
    largest = max(stock_amounts, key=stock_amounts.__getitem__)
    weights[largest] += 1 - sum(weights.values())
    return weights


def read_monthly_holdings(path: Path, signals: Mapping[date, date]) -> dict[date, dict[int, Decimal]]:
    """Read full SPY reports: as_of,kind,permno,amount. Amounts share one unit per snapshot.

    kind is equity, cash or other_net_assets. Only audited non-equity rows belong
    to the latter two categories. Weekend calendar month-end labels are accepted.
    """
    expected = {(d.year, d.month): d for d in signals.values()}
    dates: dict[date, date] = {}
    stocks: dict[date, dict[int, Decimal]] = {}
    with crsp_table(path) as rows:
        if not {"as_of", "kind", "permno", "amount"}.issubset(rows.fieldnames):
            raise ValueError(f"{path}: expected as_of,kind,permno,amount.")
        for row in rows:
            label = date.fromisoformat(row["as_of"])
            signal = expected.get((label.year, label.month))
            if signal is None:
                continue
            calendar_end = date(label.year, label.month, monthrange(label.year, label.month)[1])
            if label not in (signal, calendar_end) or dates.setdefault(signal, label) != label:
                raise ValueError(f"{label}: expected one month-end report for {signal}.")
            amount = Decimal(row["amount"])
            if not amount.is_finite():
                raise ValueError(f"{label}: invalid reported holding amount.")
            kind, code = row["kind"].strip(), row["permno"].strip()
            if kind in {"cash", "other_net_assets"} and not code:
                continue
            if kind != "equity" or not code or amount < 0:
                raise ValueError(f"{label}: unclassified position or invalid long equity {row}.")
            p = int(code)
            equity = stocks.setdefault(signal, {})
            if p in equity:
                raise ValueError(f"{label}: duplicate PERMNO {p}.")
            equity[p] = amount
    missing = sorted(set(expected.values()) - stocks.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} required monthly SPY reports: " + ", ".join(map(str, missing)))
    return {d: stock_targets({p: v for p, v in values.items() if v}) for d, values in stocks.items()}


class SpyReplication:
    def __init__(self, signals: Mapping[date, date], weights: Mapping[date, dict[int, Decimal]],
                 corporate_actions: Mapping[date, Mapping[int, int | None]] | None = None,
                 *, lot_relief: LotRelief = hifo):
        self.signals = signals
        self.weights = weights
        self.corporate_actions = corporate_actions or {}
        self.lot_relief = lot_relief
        self.held_orders = []

    def __call__(self, state: CloseSnapshot) -> Target:
        if state.trade_day is None or self.signals.get(state.trade_day) != state.day:
            raise ValueError(f"{state.trade_day}: no scheduled month-end signal from {state.day}.")
        weights = self.weights[state.day]
        restricted = {p for p, parent in self.corporate_actions.get(state.trade_day, {}).items()
                      if parent is None or parent in state.lots_by_permno}
        restricted.update(state.prelisting_marks)
        if not restricted:
            return Target(weights, lot_relief=self.lot_relief)
        nav = state.investable_nav
        fixed = {}
        changed = False
        for p in sorted(restricted - state.blocked):
            held = sum(lot.shares for lot in state.lots_by_permno.get(p, ()))
            desired = weights.get(p, Decimal(0))
            if not (desired or held):
                continue
            if nav is None or nav <= 0 or (price := state.closes.get(p)) is None:
                raise ValueError(f"{state.day} {p}: missing price or NAV for corporate-action order review.")
            planned = int(nav * desired / price)
            if planned == held:
                fixed[p] = desired
                continue
            # Round up only when Decimal division would make the whole-share sizer lose one share.
            hold_weight = held * price / nav
            while int(nav * hold_weight / price) < held:
                hold_weight = hold_weight.next_plus()
            fixed[p] = hold_weight
            changed = True
            self.held_orders.append((state.trade_day, p, held, planned, desired))
        if not changed:
            return Target(weights, lot_relief=self.lot_relief)
        flexible = {p: v for p, v in weights.items() if p not in fixed}
        available = 1 - sum(fixed.values())
        if sum(flexible.values()) > available:
            flexible = {p: (w * available).quantize(Decimal("1e-18"), rounding=ROUND_DOWN)
                        for p, w in stock_targets(flexible).items()}
        return Target({**flexible, **fixed}, lot_relief=self.lot_relief)
