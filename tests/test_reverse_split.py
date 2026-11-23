import unittest
from datetime import date
from decimal import Decimal
from fractions import Fraction
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

from tlh.actions import CorporateAction
from tlh.backtest import AccountEngine, Session, run
from tlh.crsp_events import ALCOA_REVERSE_SPLIT, SplitTerms, read_distributions
from tlh.execution import Order, proportional_commission
from tlh.lot_relief import hifo, ltfo
from tlh.positions import write_weights
from tlh.reverse_split import SplitDesignation
from tlh.strategy import Target
from tlh.tax import Scenario


D = Decimal
day = date.fromisoformat
P = 24643
EX = day("2016-10-06")
NEXT = day("2016-10-07")
NO_FEE = proportional_commission(D(0))


class ReverseSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "data/crsp/distributions_2015_to_2026_09.csv"
        cls.events = read_distributions(path)
        cls.alcoa = next(e for e in cls.events[EX] if e.reference == ALCOA_REVERSE_SPLIT)

    def account(self, first=2, second=2, designations=None):
        capital = D(max(1000, first * 10 + second * 16))
        engine = AccountEngine(capital, start=day("2016-10-05"), fee=NO_FEE, split_designations=designations)
        book = engine.result.ledger
        book.buy(day("2014-01-02"), P, D(first), D(10), D(0))
        if second:
            book.buy(day("2016-08-01"), P, D(second), D(16), D(0))
        return engine, book

    def settle(self, engine, event, close=36):
        market = Session({P: D(close)}, {P: D(close)}, [event])
        engine.advance(EX, market, (), {}, ())
        return engine.close_snapshot(EX, {P: D(close)}, trade_day=NEXT)

    def test_single_source_and_exact_alcoa_ratio(self):
        engine, book = self.account()
        state = self.settle(engine, CorporateAction("half", P, "SPLIT", D(".5")))
        self.assertEqual([(l.id, l.shares, l.basis, l.segments) for l in state.lots],
                         [(1, 1, 20, ()), (2, 1, 32, ())])
        self.assertFalse(book.sales)
        for n in (3, 3000):
            engine, book = self.account(n, 0)
            self.settle(engine, self.alcoa)
            self.assertEqual((book.holdings(), book.total_basis(), book.sales), ({P: n // 3}, n * 10, []))
            self.assertFalse(engine.result.exceptions)

    def test_break_even_cil_leaves_the_replacement_for_the_real_loss(self):
        engine = AccountEngine(D(1000), start=day("2016-10-05"), fee=NO_FEE)
        book = engine.result.ledger
        old = book.buy(day("2014-01-02"), P, D(4), D(100), D(0))
        self.settle(engine, self.alcoa, close=300)
        cil, = book.sales
        self.assertEqual((cil.proceeds, cil.basis, cil.gain, book.cash, book.total_basis()), (100, 100, 0, 700, 300))
        engine.advance(NEXT, Session({P: D(300)}, {P: D(300)}, []), [Order(P, 1, {})], {}, ())
        self.assertFalse(book.wash_matches)
        lower = {P: D(150)}
        engine.advance(day("2016-10-10"), Session(lower, lower, []), [Order(P, -1, {old.id: 1})], {}, ())
        sale = book.sales[-1]
        self.assertEqual((sale.gain, sale.disallowed_loss, sale.taxable_gain, book.total_basis()), (-150, 150, 0, 450))

    def test_third_share_cil_loss_stays_realized_after_forward_split_and_repurchase(self):
        engine, book = self.account(4, 0)
        self.settle(engine, self.alcoa, close=27)
        sale, = book.sales
        quotes = {P: D(9)}
        market = Session(quotes, quotes, [CorporateAction("triple", P, "SPLIT", D(3))])
        engine.advance(NEXT, market, (), {}, ())
        purchase = book.buy(day("2016-10-10"), P, D(1), D(9), D(0))
        self.assertEqual((sale.exact_shares, sale.taxable_gain, sale.disallowed_loss, book.wash_matches, purchase.basis),
                         (Fraction(1, 3), -1, 0, [], 9))

    def test_oldest_cil_and_bound_sale_keep_separate_tax_character(self):
        engine, book = self.account()
        engine.fee = proportional_commission(D(".001"))
        state = self.settle(engine, self.alcoa)
        combined, = state.lots
        self.assertNotIn(combined.id, (1, 2))
        self.assertEqual((combined.shares, combined.basis, book.total_basis(), book.cash), (1, 42, 42, 960))
        self.assertEqual([(s.source_lot_id, s.basis) for s in combined.segments], [(1, 10), (2, 32)])
        cil, = book.sales
        self.assertEqual((cil.source_lot_id, cil.basis, cil.proceeds, cil.gain, cil.is_long_term),
                         (1, 10, 12, 2, True))
        self.assertEqual([e.fee for e in book.entries if e.kind == "CASH_IN_LIEU"], [0])
        self.assertEqual(sum(e.kind == "BUY" for e in book.entries), 2)
        for selector in (hifo, ltfo):
            self.assertEqual(selector(state, P, D(1)), {combined.id: 1})
        engine.advance(NEXT, Session({P: D(45)}, {P: D(45)}, ()),
                       (Order(P, D(-1), {combined.id: D(1)}),), {}, ())
        a, b = book.sales[1:]
        self.assertEqual([(s.trading_lot_id, s.source_lot_id, s.basis, s.gain, s.is_long_term) for s in (a, b)],
                         [(combined.id, 1, 10, D("4.985"), True), (combined.id, 2, 32, D("-2.030"), False)])
        self.assertEqual((book.holdings(), book.cash), ({}, D("1004.955")))

    def test_identified_shares_override_default_and_bad_record_rolls_back(self):
        identified = SplitDesignation([{1: Fraction(2, 3), 2: Fraction(1, 3)}], {2: Fraction(1, 3)})
        engine, book = self.account(designations={ALCOA_REVERSE_SPLIT: identified})
        state = self.settle(engine, self.alcoa)
        self.assertEqual((state.lots[0].basis, book.sales[0].source_lot_id, book.sales[0].basis), (36, 2, 16))
        bad = SplitDesignation([{1: 1}], {2: Fraction(1, 3)})
        engine, book = self.account(designations={ALCOA_REVERSE_SPLIT: bad})
        with self.assertRaisesRegex(ValueError, "designated shares"):
            self.settle(engine, self.alcoa)
        self.assertEqual((book.holdings(), book.total_basis(), book.sales), ({P: 4}, 52, []))
        book.split_designations[ALCOA_REVERSE_SPLIT] = identified
        self.assertEqual(self.settle(engine, self.alcoa).lots, state.lots)

    def test_strategy_csv_replay_and_liquidation_use_one_bound_id(self):
        sessions = [day("2014-12-31"), day("2015-01-02"), day("2016-08-01"), EX, NEXT]
        prices = [10, 10, 16, 36, 45]
        markets = {d: Session({P: D(px)}, {P: D(px)}, [self.alcoa] if d == EX else [])
                   for d, px in zip(sessions, prices)}
        settings = dict(sessions=sessions, first_trade=sessions[1], end=NEXT, days=markets, delistings=[],
                        delistings_known_through=NEXT, initial_cash=D(1000), fee=NO_FEE, pre_delisting=False)
        # 第二次下单基于前收盘 10 美元，把目标从 2 股调至 4 股。
        def decide(state):
            if state.trade_day == NEXT:
                lot, = state.lots
                return Target({}, sell_lots={P: {lot.id: D(1)}})
            return Target({P: D(".02") if state.trade_day == sessions[1] else D(".04")})

        result = run(**settings, strategy=decide, rebalance_days=[sessions[1], sessions[2], NEXT])
        group_id, = result.targets[NEXT].sell_lots[P]
        self.assertNotIn(group_id, (1, 2))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "weights.csv"
            write_weights(path, result.targets)
            replay = run(**settings, weights_file=path)
        self.assertEqual((replay.history, replay.plans, replay.ledger.sales, replay.actions),
                         (result.history, result.plans, result.ledger.sales, result.actions))
        for scenario in Scenario:
            self.assertEqual(replay.annual_wealth(scenario), result.annual_wealth(scenario))
        engine, book = self.account()
        self.settle(engine, self.alcoa)
        liquidated = engine.result.liquidated_wealth(Scenario.EXTERNAL_GAINS, fee=NO_FEE)
        self.assertEqual((liquidated.pre_tax, book.holdings()), (996, {P: 1}))

    def test_report_unverified_crsp_ratio_only_when_held(self):
        ex = day("2017-01-04")
        ref = "CRSP/StkDistributions/14338/2017-01-04/3"
        raw = next(e for e in self.events[ex] if e.reference == ref)
        # 用模拟的后续核实条款重跑同一账户，验证标记和金额都按新账本重算。
        path = Path(__file__).resolve().parents[1] / "data/crsp/distributions_2015_to_2026_09.csv"
        updated = read_distributions(path, split_terms={ref: SplitTerms(Fraction(1, 3), "test override")})
        corrected = next(e for e in updated[ex] if e.reference == ref)
        for event, held in [(raw, True), (raw, False), (corrected, True)]:
            engine = AccountEngine(D(100), start=day("2016-12-29"), fee=NO_FEE)
            if held:
                engine.result.ledger.buy(day("2016-05-02"), 14338, D(3), D(10), D(0))
            before = {14338: D(10)}
            engine.advance(day("2016-12-30"), Session(before, before, []), (), {}, ())
            quotes = {14338: D(36)}
            engine.advance(ex, Session(quotes, quotes, [event]), (), {}, ())
            record = engine.result.actions[0][1]
            needs_review = event is raw and held
            self.assertEqual((record.ratio_needs_review, bool(engine.result.exceptions)), (needs_review, needs_review))
            self.assertEqual(record.ratio_source, "CRSP DisFacShr" if event is raw else "test override")
            self.assertEqual(record.split_ratio, Fraction(D(".333333")) if event is raw else Fraction(1, 3))
            engine.advance(day("2017-12-29"), Session(quotes, quotes, []), (), {}, ())
            next_year = day("2018-01-02")
            cash = engine.result.ledger.cash
            flow = D("-2.447985312") if needs_review else 0
            engine.advance(next_year, Session(quotes, quotes, []), (), {}, ())
            mark = engine.result.history[-1]
            self.assertEqual((mark.cash, mark.tax_cash_flow, mark.daily_return), (cash + flow, flow, 0))
            self.assertTrue(engine.result.tax_settlements[-1].final)
            engine.advance(day("2018-12-31"), Session(quotes, quotes, []), (), {}, ())
            for scenario in Scenario:
                prior, tax_year, *following = engine.result.annual_wealth(scenario)
                self.assertEqual((prior.tax.final, prior.unverified_split_ratios), (True, ()))
                terminal = engine.result.liquidated_wealth(scenario, fee=NO_FEE)
                for report in (tax_year, *following, terminal):
                    self.assertEqual((report.tax.final, report.unverified_split_ratios),
                                     (not needs_review, (ref,) if needs_review else ()))
                    paid = sum(s.cash_flow for s in engine.result.tax_settlements if s.day <= report.as_of)
                    self.assertEqual(report.after_tax, report.pre_tax - (report.tax.cumulative_tax + paid))
                if needs_review:
                    self.assertEqual((tax_year.pre_tax, tax_year.tax.net_tax, tax_year.after_tax),
                                     (D("105.999964"), D("2.447985312"), D("103.551978688")))
                elif held:
                    self.assertEqual((tax_year.pre_tax, tax_year.tax.net_tax, tax_year.after_tax), (106, 0, 106))

    def test_flat_account_carries_split_provenance_to_the_original_loss_year(self):
        ex, sold, bought = map(day, ("2017-01-04", "2016-12-30", "2017-01-05"))
        p = 14338
        raw = next(e for e in self.events[ex] if e.reference.endswith("14338/2017-01-04/3"))
        verified = CorporateAction(raw.reference, p, "SPLIT", Fraction(1, 3), "comparison only")
        for event, scenario in product((raw, verified), Scenario):
            engine = AccountEngine(D(2000), start=day("2016-12-29"), fee=NO_FEE, tax_scenario=scenario)
            book = engine.result.ledger
            lot = book.buy(day("2016-12-29"), p, D(6), D(20), D(0))
            quotes = {p: D(10)}
            engine.advance(sold, Session(quotes, quotes, []), [Order(p, -6, {lot.id: 6})], {}, ())
            self.assertEqual(book.holdings(), {})
            quotes = {p: D(30)}
            engine.advance(ex, Session(quotes, quotes, [event]), (), {}, ())
            pending, _ = engine.result.annual_wealth()
            self.assertEqual((pending.tax.final, pending.unverified_split_ratios),
                             (False, (raw.reference,) if event is raw else ()))
            self.assertEqual((book.wash_matches, book.sales[0].disallowed_loss), ([], 0))
            engine.advance(bought, Session(quotes, quotes, []), [Order(p, 1, {})], {}, ())
            true_up = day("2017-01-30")  # 12-30 卖出后第 31 天，也是下一交易日。
            engine.advance(true_up, Session(quotes, quotes, []), (), {}, ())
            engine.advance(day("2017-12-29"), Session(quotes, quotes, []), (), {}, ())
            engine.advance(day("2018-01-31"), Session(quotes, quotes, []), (), {}, ())
            expected = D(60) / D("1.999998") if event is raw else D(30)
            self.assertEqual(book.sales[0].disallowed_loss, expected)
            match, = book.wash_matches
            self.assertEqual((match.sale_day, match.acquired, match.permno, match.unverified_splits),
                             (sold, bought, p, (raw.reference,) if event is raw else ()))
            prior, following, _ = engine.result.annual_wealth()
            initial, correction = [s for s in engine.result.tax_settlements if s.tax_year == sold.year]
            self.assertEqual((initial.cash_flow, initial.final), (D("24.48") if scenario is Scenario.EXTERNAL_GAINS else 0, False))
            self.assertEqual((correction.day, correction.cash_flow, correction.final),
                             (true_up, -prior.tax.net_tax - initial.cash_flow, True))
            self.assertEqual(engine.result.pre_tax_twr(bought, true_up), 0)
            terminal = engine.result.liquidated_wealth(fee=NO_FEE)
            for report in (prior, following, terminal):
                self.assertEqual(report.unverified_split_ratios, (raw.reference,) if event is raw else ())
                self.assertIsNotNone(report.after_tax)
            self.assertEqual((prior.tax.final, following.tax.final), (event is verified, event is verified))

    def test_flat_account_with_no_repurchase_tracks_only_splits_inside_the_window(self):
        ex = day("2017-01-04")
        p = 14338
        raw = next(e for e in self.events[ex] if e.reference.endswith("14338/2017-01-04/3"))
        for (sold, in_window), scenario in product([(day("2016-12-28"), True), (day("2016-11-30"), False)], Scenario):
            engine = AccountEngine(D(200), start=day("2016-11-01"), fee=NO_FEE, tax_scenario=scenario)
            book = engine.result.ledger
            lot = book.buy(day("2016-11-01"), p, D(6), D(20), D(0))
            px = {p: D(10)}
            engine.advance(sold, Session(px, px, []), [Order(p, -6, {lot.id: 6})], {}, ())
            engine.advance(ex, Session({}, {}, [raw]), (), {}, ())
            pending, _ = engine.result.annual_wealth()
            self.assertEqual((pending.tax.final, pending.unverified_split_ratios),
                             (not in_window, (raw.reference,) if in_window else ()))
            # 窗口内买另一 PERMNO；到期后才买回原股。两笔均不能递延原亏损。
            quotes = {16498: D(5), p: D(30)}
            engine.advance(day("2017-01-05"), Session(quotes, quotes, []), [Order(16498, 1, {})], {}, ())
            engine.advance(day("2017-01-26"), Session(quotes, quotes, []), (), {}, ())
            self.assertEqual(engine.result.annual_wealth()[0].tax.final, not in_window)
            engine.advance(day("2017-01-27"), Session(quotes, quotes, []), (), {}, ())  # 12-28 的第 30 天。
            prior = engine.result.annual_wealth()[0]
            self.assertEqual((prior.tax.final, prior.unverified_split_ratios), (True, ()))
            self.assertEqual(engine.result.liquidated_wealth(fee=NO_FEE).unverified_split_ratios, ())
            engine.advance(day("2017-01-31"), Session(quotes, quotes, []), [Order(p, 1, {})], {}, ())
            prior = engine.result.annual_wealth()[0]
            tax = D("-24.48") if scenario is Scenario.EXTERNAL_GAINS else D(0)
            self.assertEqual((book.wash_matches, prior.tax.net_tax, prior.tax.final), ([], tax, True))
            self.assertEqual(engine.result.actions[0][1].wash_loss_days, (sold,) if in_window else ())

    def test_span_precedes_oldest_cil_and_equal_dates_use_original_id(self):
        def allocations(dates, ratio):
            engine = AccountEngine(D(1000), start=day("2016-10-05"), fee=NO_FEE)
            for d in dates:
                engine.result.ledger.buy(day(d), P, D(1), D(10), D(0))
            state = self.settle(engine, CorporateAction("test", P, "SPLIT", ratio))
            return [s.source_lot_id for s in engine.result.ledger.sales], [
                s.source_lot_id for s in state.lots[0].segments]

        self.assertEqual(allocations(["2015-01-02", "2015-01-05", "2016-08-01"], D(".5")),
                         ([3], [1, 2]))
        self.assertEqual(allocations(["2015-01-02"] * 3, D(".5")), ([1], [2, 3]))

    def test_same_date_and_price_first_and_one_pending_segment_freezes_whole(self):
        engine = AccountEngine(D(1000), start=day("2016-10-05"), fee=NO_FEE)
        book = engine.result.ledger
        for price in (10, 16, 10, 16):
            book.buy(day("2015-01-02"), P, D(1), D(price), D(0))
        engine.basis_pending.add(1)
        state = self.settle(engine, CorporateAction("half", P, "SPLIT", D(".5")))
        self.assertEqual([[s.source_lot_id for s in lot.segments] for lot in state.lots], [[1, 3], [2, 4]])
        blocked, tradable = state.lots
        self.assertEqual(state.eligible_lots(P), (tradable,))
        with self.assertRaisesRegex(ValueError, "unresolved basis"):
            engine.advance(NEXT, Session({P: D(30)}, {P: D(30)}, []),
                           (Order(P, D(-1), {blocked.id: D(1)}),), {}, ())
        self.assertEqual((book.holdings(), book.total_basis()), ({P: 2}, 52))

    def test_hifo_ltfo_and_partial_wash_keep_the_whole_share_bound(self):
        engine, book = self.account()
        self.settle(engine, self.alcoa)
        other = book.buy(EX, P, D(1), D("43.50"), D(0))
        state = engine.close_snapshot(EX, {P: D(45)}, trade_day=NEXT)
        group, = [l for l in state.lots if l.segments]
        self.assertEqual(hifo(state, P, D(1)), {other.id: 1})
        # 整股按段预估为 5 × .238 – 2 × .408 = .374，另一批预计是 1.50 × .408 = .612。
        self.assertEqual(ltfo(state, P, D(1)), {group.id: 1})

        engine = AccountEngine(D(1000), start=day("2016-10-05"), fee=NO_FEE)
        book = engine.result.ledger
        book.buy(day("2014-01-02"), P, D(1), D(30), D(0))
        book.buy(day("2016-09-20"), P, D(1), D(10), D(0))
        group, = self.settle(engine, CorporateAction("half", P, "SPLIT", D(".5"))).lots
        independent = book.buy(EX, P, D(1), D(80), D(0))
        engine.advance(NEXT, Session({P: D(40)}, {P: D(40)}, []),
                       (Order(P, D(-1), {independent.id: D(1)}),), {}, ())
        match, = book.wash_matches
        self.assertEqual((match.shares, match.disallowed_loss, book.lots[match.replacement_lot].trading_id),
                         (D(".5"), 20, group.id))
        adjusted, = engine.close_snapshot(NEXT, {P: D(40)}).lots
        self.assertEqual((adjusted.id, adjusted.shares, adjusted.basis), (group.id, 1, 60))

    def test_forward_split_sells_whole_units_and_roc_uses_pre_split_entitlements(self):
        engine, book = self.account(1, 1)
        group, = self.settle(engine, CorporateAction("half", P, "SPLIT", D(".5"))).lots
        split = CorporateAction("double", P, "SPLIT", D(2))
        quotes = {P: D(15)}
        engine.advance(NEXT, Session(quotes, quotes, [split]), (), {}, ())
        sale_day = day("2016-10-10")
        engine.advance(sale_day, Session(quotes, quotes, []), (Order(P, D(-1), {group.id: D(1)}),), {}, ())
        remaining, = engine.close_snapshot(sale_day, quotes).lots
        self.assertEqual((remaining.id, remaining.shares, remaining.basis), (group.id, 1, 13))
        self.assertEqual([(s.shares, s.basis) for s in remaining.segments], [(D(".5"), 5), (D(".5"), 8)])

        engine, book = self.account()
        roc = CorporateAction("same-day ROC", P, "ROC", D(1))
        quotes = {P: D(36)}
        engine.advance(EX, Session(quotes, quotes, [self.alcoa, roc]), (), {}, ())
        self.assertEqual((book.cash, book.total_basis(), book.sales[0].basis), (964, 39, 9))

    def test_overnight_split_cancels_sales_independently_of_locked_lot_selection(self):
        previous = day("2016-10-05")
        market = Session({P: D(20)}, {P: D(20)}, [CorporateAction("half", P, "SPLIT", D(".5"))],
                         raw_opens={P: D(20)})
        for selected in ((1, 2), (1, 3)):
            engine = AccountEngine(D(400), start=previous, fee=NO_FEE)
            book = engine.result.ledger
            for d in ["2015-01-02", "2015-01-05", "2016-08-01", "2016-08-02"]:
                book.buy(day(d), P, D(1), D(10), D(0))
            target = Target({P: D(".05")}, sell_lots={P: dict.fromkeys(selected, D(1))})
            orders = engine.prepare(previous, {P: D(10)}, EX, target, set())
            engine.advance(EX, market, orders, {}, ())
            self.assertEqual((book.holdings(), book.sales, book.total_basis(), book.cash), ({P: 2}, [], 40, 360))
            self.assertEqual(engine.result.exceptions, [(EX, P, 'SKIPPED_SPLIT_SALE 2/4 pre-split shares')])
            self.assertEqual(engine.result.plans[previous], orders)
