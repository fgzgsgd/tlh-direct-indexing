import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from tlh.actions import CorporateAction
from tlh.backtest import AccountEngine, Session, run
from tlh.execution import proportional_commission
from tlh.lot_relief import hifo, ltfo, take_shares
from tlh.positions import read_weights, write_weights
from tlh.strategy import CloseSnapshot, LotView, Target, index_strategy
from tlh.tax import Scenario


D = Decimal
day = date.fromisoformat


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.dates = list(map(day, ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]))
        self.signal, self.first, self.second, self.hold, self.sell = self.dates
        self.days = {d: Session({10: D(px)}, {10: D(px)}, [])
                     for d, px in zip(self.dates, [10, 10, 20, 20, 15])}
        self.inputs = dict(sessions=self.dates, first_trade=self.first, end=self.sell, days=self.days,
                           initial_cash=D(100), fee=lambda *_: D(0), pre_delisting=False,
                           delistings=[], delistings_known_through=self.sell)
        self.rebalances = [self.first, self.second, self.sell]

    def test_removed_member_stays_until_rebalance_then_all_lots_are_sold_and_replay_matches(self):
        def allocate(state):
            if state.trade_day == self.hold:
                return None
            return Target({10: D(".6") if state.trade_day != self.first else D(".3")})

        strategy = index_strategy(allocate, lambda state: {10} if state.day < self.second else set())
        result = run(strategy=strategy, rebalance_days=self.dates[1:], **self.inputs)
        self.assertEqual([v.holdings for v in result.history], [{10: 3}, {10: 6}, {10: 6}, {}])
        self.assertEqual(result.targets[self.sell].sell_lots, {10: {1: 3, 2: 3}})
        self.assertEqual(result.history[-1].cash, 100)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            write_weights(path, result.targets)
            replay = run(weights_file=path, **self.inputs)
            self.assertEqual(replay.history, result.history)

    def test_strategy_reads_previous_fills_and_explicit_lots_override_policy(self):
        seen = []

        def decide(state):
            seen.append(state)
            if not state.lots:
                return Target({10: D(".3")})
            if len(state.lots) == 1:
                return Target({10: D(".5")})
            oldest, newest = state.lots
            self.assertEqual((state.day, state.trade_day, state.closes[10], state.cash, state.nav),
                             (self.hold, self.sell, 20, 30, 130))
            self.assertEqual([(lot.shares, lot.basis) for lot in (oldest, newest)], [(3, 30), (2, 40)])
            return Target({10: D(".4")}, sell_lots={10: {oldest.id: D(3)}})

        result = run(strategy=decide, rebalance_days=self.rebalances, lot_relief=lambda *_: self.fail(), **self.inputs)
        self.assertEqual([s.day for s in seen], [self.signal, self.first, self.hold])
        self.assertEqual(seen[0].lots, ())
        self.assertEqual(seen[-1].lots[0].shares, 3)  # After the sale, the signal-day snapshot is still complete.
        with self.assertRaises(TypeError):
            seen[-1].closes[10] = D(15)
        self.assertEqual(result.targets[self.hold], Target({10: D(".5")}, False))
        self.assertEqual(result.targets[self.sell].sell_lots, {10: {1: 3}})
        sale, = result.ledger.sales
        self.assertEqual((sale.lot_id, sale.shares, sale.gain), (1, 3, 15))
        final = result.history[-1]
        self.assertEqual((final.holdings, final.cash, final.nav, final.total_basis), ({10: 2}, 75, 105, 40))

    def test_custom_selector_receives_locked_net_sale(self):
        def newest_first(state, permno, quantity):
            older, newer = state.lots_by_permno[permno]
            self.assertEqual((state.closes[permno], quantity), (20, 3))
            return {newer.id: newer.shares, older.id: quantity - newer.shares}

        weights = dict(zip(self.rebalances, map(D, [".3", ".5", ".4"])))
        result = run(strategy=lambda s: Target({10: weights[s.trade_day]}, lot_relief=newest_first),
                     rebalance_days=self.rebalances, **self.inputs)
        self.assertEqual([(s.lot_id, s.gain) for s in result.ledger.sales], [(2, -10), (1, 5)])
        final = result.history[-1]
        self.assertEqual((final.cash, final.nav, final.total_basis), (75, 105, 30))
        self.assertEqual(result.ledger.sales[0].disallowed_loss, 10)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            write_weights(path, result.targets)
            self.assertIn("2:2|1:1", path.read_text())
            replay = run(weights_file=path, **self.inputs)
        self.assertEqual((replay.targets, replay.plans, replay.ledger.entries, replay.ledger.sales, replay.history),
                         (result.targets, result.plans, result.ledger.entries, result.ledger.sales, result.history))
        for scenario in Scenario:
            self.assertEqual(replay.annual_wealth(scenario, wash_reviewed_years={2024}),
                             result.annual_wealth(scenario, wash_reviewed_years={2024}))

    def test_missing_trade_price_stops_before_next_strategy_decision(self):
        self.days[self.first] = Session({}, {10: D(10)}, [])
        seen = []

        def decide(state):
            seen.append((state.cash, len(state.lots)))
            return Target({10: D(".3")})

        with self.assertRaisesRegex(ValueError, "BUY: missing executable CLOSE"):
            run(strategy=decide, rebalance_days=self.rebalances[:2], **self.inputs)
        self.assertEqual(seen, [(100, 0)])

    def test_next_decision_sees_split_lots_at_adjusted_share_basis(self):
        self.days[self.hold] = Session({}, {10: D(10)}, [CorporateAction("s", 10, "SPLIT", D(2))])

        def decide(state):
            if state.trade_day != self.sell:
                return Target({10: D(".3") if not state.lots else D(".5")})
            older, newer = state.eligible_lots(10)
            self.assertEqual([(lot.shares, lot.basis / lot.shares) for lot in (older, newer)], [(6, 5), (4, 10)])
            # A strategy can use the generic helper to allocate its own lot priority to a 5-share net sell order.
            return Target({10: D(".4")}, sell_lots={10: take_shares([newer, older], D(5))})

        result = run(strategy=decide, rebalance_days=self.rebalances, **self.inputs)
        self.assertEqual(result.targets[self.sell].sell_lots, {10: {2: 4, 1: 1}})
        self.assertEqual([(s.lot_id, s.basis, s.gain) for s in result.ledger.sales], [(2, 40, 20), (1, 5, 10)])

    def test_invalid_explicit_instructions_fail_before_booking(self):
        engine = AccountEngine(D(100), start=self.signal, fee=lambda *_: D(0))
        book = engine.result.ledger
        first = book.buy(self.first, 10, D(3), D(10), D(0))
        second = book.buy(self.second, 10, D(2), D(20), D(0))
        foreign = book.buy(self.second, 20, D(1), D(10), D(0))
        engine.basis_pending.add(second.id)
        closes = {10: D(20), 20: D(10)}
        engine.last_nav = book.nav(closes)
        for selection in ({foreign.id: D(3)}, {first.id: D(2)}, {first.id: D(4)}, {second.id: D(1), first.id: D(2)}):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "lot allocation"):
                engine.prepare(self.hold, closes, self.sell,
                               Target({10: D(".4"), 20: D(".1")}, sell_lots={10: selection}), set())
        with self.assertRaisesRegex(ValueError, "without an active net sale"):
            engine.prepare(self.hold, closes, self.sell,
                           Target({10: D("1")}, sell_lots={10: {first.id: D(1)}}), set())
        self.assertFalse(book.sales)
        self.assertFalse(engine.result.plans)

    def test_hifo_and_ltfo_use_signal_basis_and_trade_date_holding_period(self):
        dates = list(map(day, ["2023-01-03", "2023-01-04", "2024-01-03", "2024-01-04", "2024-01-05"]))
        _, first, _, second, sale_day = dates
        weights = {first: D(".2"), second: D(".42"), sale_day: D(".25")}
        market = {d: Session({10: D(fill)}, {10: D(close)}, [], "VWAP")
                  for d, fill, close in zip(dates, [100, 112, 100, 108, 105], [100, 100, 100, 100, 103])}
        # The trade date just passes the old lot's one-year anniversary. Expected per-share tax at signal prices is -2.856 and -3.264 for the two lots.
        for policy, lot_id, gain, is_long in [(hifo, 1, -7, True), (ltfo, 2, -3, False)]:
            with self.subTest(policy=policy.__name__):
                result = run(strategy=lambda s: Target({10: weights[s.trade_day]}, lot_relief=policy),
                             rebalance_days=weights, sessions=dates, first_trade=first, end=sale_day, days=market,
                             initial_cash=D(500), fee=lambda *_: D(0), pre_delisting=False,
                             delistings=[], delistings_known_through=sale_day)
                sale, = result.ledger.sales
                self.assertEqual((sale.lot_id, sale.gain, sale.is_long_term), (lot_id, gain, is_long))
                self.assertEqual((result.history[-1].cash, result.history[-1].nav), (385, 488))

    def test_equal_scores_use_acquisition_date_then_id_only_among_eligible_lots(self):
        def lot(i, acquired, basis):
            return LotView(i, 10, day(acquired), self.first, D(1), basis)

        lots = (lot(3, "2024-01-04", D(10)), lot(9, "2024-01-03", D(10)), lot(4, "2024-01-03", D(10)),
                lot(1, "2024-01-03", D(100)), lot(2, "2024-01-03", None))
        state = CloseSnapshot(self.hold, D(0), D(50), {10: D(10)}, lots, frozenset({1}), self.sell)
        for policy in (hifo, ltfo):
            self.assertEqual(list(policy(state, 10, D(2)).items()), [(4, 1), (9, 1)])
            with self.assertRaisesRegex(ValueError, "short by 1"):
                policy(state, 10, D(4))

    def test_ranked_sale_preserves_priority_among_deliverable_allocations(self):
        def lot(i, shares, price):
            shares = D(shares)
            return LotView(i, 10, self.first, self.first, shares, shares * price)

        cases = [
            ([lot(1, ".5", 30), lot(2, ".5", 20), lot(3, 1, 10)], {2}, 1, {3: 1}),
            ([lot(1, ".75", 30), lot(2, 1, 20), lot(3, ".5", 10)], set(), 1, {1: D(".75"), 3: D(".25")}),
            ([lot(1, "1.5", 30), lot(2, 1, 20)], set(), 2, {1: 1, 2: 1}),
        ]
        for lots, pending, quantity, expected in cases:
            state = CloseSnapshot(self.hold, D(0), D(100), {10: D(10)}, tuple(lots), frozenset(pending), self.sell)
            for rule in (hifo, ltfo):
                with self.subTest(rule=rule.__name__, expected=expected):
                    self.assertEqual(rule(state, 10, D(quantity)), expected)

    def test_mix_explicit_lots_and_hifo_on_different_securities(self):
        for d, session in list(self.days.items()):
            self.days[d] = Session({**session.execution_prices, 20: session.execution_prices[10]},
                                   {**session.raw_closes, 20: session.raw_closes[10]}, [])
        self.inputs["initial_cash"] = D(200)
        self.inputs["fee"] = proportional_commission(D(".001"))
        half_weights = {self.first: D(".15"), self.second: D(".26"), self.sell: D(".20")}

        def decide(state):
            weight = half_weights[state.trade_day]
            explicit = {10: {state.lots_by_permno[10][0].id: 3}} if state.trade_day == self.sell else {}
            return Target({10: weight, 20: weight}, lot_relief=hifo, sell_lots=explicit)

        result = run(strategy=decide, rebalance_days=self.rebalances, **self.inputs)
        self.assertEqual(result.targets[self.sell].sell_lots, {10: {1: 3}, 20: {4: 2, 2: 1}})
        self.assertEqual([s.gain for s in result.ledger.sales], [D("14.925"), D("-10.07"), D("4.975")])
        # Two sell orders fill for $45 each, one selling one lot and the other two lots.
        trades = [e for e in result.ledger.entries if e.day == self.sell and e.kind == "TRADE"]
        for permno in (10, 20):
            self.assertEqual(sum(e.fee for e in trades if e.permno == permno), D(".045"))

    def test_csv_preserves_cash_only_days_and_zero_weight_sales(self):
        weights = {self.first: {}, self.second: {10: D(".5")}, self.sell: {}}
        last = day("2024-01-09")
        self.dates.append(last)
        self.days[last] = Session({}, {}, [])
        self.inputs["end"] = last
        result = run(strategy=lambda s: Target(weights[s.trade_day]), rebalance_days=self.rebalances, **self.inputs)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            write_weights(path, result.targets)
            content = path.read_text()
            self.assertIn(f"{self.first},,0,1,", content)
            self.assertIn(f"{self.sell},10,0,1,1:5", content)
            self.assertIn(f"{last},,0,0,", content)
            self.assertEqual(read_weights(path), result.targets)
            replay = run(weights_file=path, **self.inputs)
            self.assertEqual((replay.ledger.sales, replay.history), (result.ledger.sales, result.history))
            path.write_text(content.replace("1:5", "1:3|1:2"))
            with self.assertRaisesRegex(ValueError, "Duplicate lot"):
                read_weights(path)

    def test_strategy_and_replay_cancel_the_same_locked_split_order(self):
        weights = dict(zip(self.rebalances, map(D, [".3", ".5", ".4"])))
        split = CorporateAction("s", 10, "SPLIT", D(2))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            source = dict(strategy=lambda s: Target({10: weights[s.trade_day]}, lot_relief=hifo),
                          rebalance_days=self.rebalances)
            result = run(**source, **self.inputs)
            write_weights(path, result.targets)
            self.days[self.sell] = Session({10: D(8)}, {10: D(8)}, [split], raw_opens={10: D(13)})
            live, replay = [run(**inputs, **self.inputs) for inputs in (source, {"weights_file": path})]
            self.assertEqual((live.history, live.exceptions, live.ledger.sales),
                             (replay.history, replay.exceptions, replay.ledger.sales))
            self.assertEqual(live.history[-1].holdings, {10: 10})
            self.assertEqual(live.exceptions, [(self.sell, 10, 'SKIPPED_SPLIT_SALE 3/5 pre-split shares')])
            self.days[self.sell] = Session({}, {10: D(8)}, [])
            with self.assertRaisesRegex(ValueError, "TRADE: missing executable CLOSE"):
                run(weights_file=path, **self.inputs)

    def test_split_sale_at_five_percent_is_skipped_in_strategy_and_replay(self):
        ex = self.second
        prices = {10: D(10), 20: D(10)}
        after = {10: D(80), 20: D(10)}
        split = CorporateAction("eight-to-one", 10, "SPLIT", D(".125"))
        days = {self.signal: Session(prices, prices, []), self.first: Session(prices, prices, []),
                ex: Session(after, after, [split])}
        inputs = dict(self.inputs, days=days, end=ex, initial_cash=D(1000))
        weights = {self.first: {10: D(".20"), 20: D(".40")}, ex: {10: D(".19"), 20: D(".39")}}
        result = run(strategy=lambda s: Target(weights[s.trade_day]), rebalance_days=weights, **inputs)
        # Each stock locks in a one-share sale. The split on 1/20 is skipped and the ordinary rebalance on 1/40 fills.
        self.assertEqual(result.targets[ex].sell_lots, {10: {1: 1}, 20: {2: 1}})
        self.assertEqual(result.exceptions, [(ex, 10, "SKIPPED_SPLIT_SALE 1/20 pre-split shares")])
        self.assertEqual([(s.permno, s.reason, s.shares, s.proceeds) for s in result.ledger.sales],
                         [(10, "CASH_IN_LIEU", D(".5"), 40), (20, "TRADE", 1, 10)])
        close = result.history[-1]
        self.assertEqual((close.holdings, close.cash, close.total_basis, close.nav), ({10: 2, 20: 39}, 450, 550, 1000))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            write_weights(path, result.targets)
            replay = run(weights_file=path, **inputs)
        self.assertEqual((replay.exceptions, replay.ledger.sales, replay.history),
                         (result.exceptions, result.ledger.sales, result.history))
