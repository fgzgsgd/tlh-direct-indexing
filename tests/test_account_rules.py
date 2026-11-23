import unittest
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from tlh.actions import CaseReview, CorporateAction, SpinOff, SpinOffTerms, StockCashPayment
from tlh.backtest import AccountEngine, Session, run, sessions_from_crsp
from tlh.crsp_events import read_distributions
from tlh.execution import Order, proportional_commission
from tlh.ledger import Ledger
from tlh.lot_relief import hifo, ltfo
from tlh.market import Delisting, StockDay, read_benchmark_weights
from tlh.positions import write_weights
from tlh.strategy import Target
from tlh.tax import Scenario, settle_years


D = Decimal
day = date.fromisoformat
no_fee = proportional_commission(D(0))


class AccountRules(unittest.TestCase):
    def test_known_cash_and_stock_dividend_is_spendable_on_ex_date(self):
        before, ex = day("2024-05-01"), day("2024-05-02")
        engine = AccountEngine(D(200), start=day("2024-04-30"), fee=no_fee)
        book = engine.result.ledger
        book.buy(day("2024-01-02"), 10, D(10), D(20), D(0))
        engine.advance(before, Session({10: D(20)}, {10: D(20)}, []), (), {}, ())
        # Each old share gets $1 cash and $2.50 of stock at a $20 conversion price. Ten shares receive 1.25 shares.
        event = StockCashPayment("declared terms", 10, D(1), D("2.50"), D(20), (D(1), D(0), D(0), D(0)))
        prices = {10: D(16), 20: D(12)}
        engine.advance(ex, Session(prices, prices, [event]), [Order(20, 1, {})], {}, ())
        close = engine.result.history[-1]
        self.assertEqual((close.cash, close.holdings, close.nav, close.daily_return), (2, {10: 11, 20: 1}, 190, D("-.05")))
        cil, = book.sales
        self.assertEqual((cil.day, cil.proceeds, cil.basis, cil.gain), (ex, 4, 5, -1))
        self.assertEqual(book.tax_year(2024, basis_verified=True).taxable_dividends, 35)

    def test_reported_spin_off_ratios_use_the_first_six_places(self):
        ex = day("2024-03-01")
        for reported, n in [("0.333333", 3), ("0.166666", 6), ("0.166667", 6),
                            ("0.14285799", 7), ("0.058824", 17), ("0.007812", 128),
                            ("0.007813", 128), ("0.000999", 1001)]:
            with self.subTest(reported=reported):
                engine = AccountEngine(D(10000), start=ex-timedelta(days=1), fee=no_fee)
                book = engine.result.ledger
                book.buy(day("2023-01-03"), 10, D(n), D(5), D(0))
                terms = SpinOffTerms(20, D(reported), D(".2"), "reported terms")
                quotes = {10: D(4), 20: D(n)}
                engine.advance(ex, Session(quotes, quotes, [SpinOff("grant", 10, terms)]), (), {}, ())
                self.assertEqual((book.holdings(), book.sales, book.cash, engine.last_nav),
                                 ({10: n, 20: 1}, [], 10000 - 5*n, 10000))
                record, = [record for _, record in engine.result.actions]
                self.assertEqual((record.reported_shares_per_parent, record.effective_shares_per_parent,
                                  record.ratio_source), (D(reported), Fraction(1, n), terms.source))

    def test_ambiguous_or_unmatched_approximate_spin_off_waits_for_terms(self):
        ex = day("2024-03-01")
        for reported, approximate in [("0.000100", None), ("0.000100", True), ("0.000000", None),
                                      ("0.0000009", True), ("0.333334", True), ("0.17", True)]:
            with self.subTest(reported=reported, approximate=approximate):
                engine = AccountEngine(D(1000), start=ex-timedelta(days=1), fee=no_fee)
                book = engine.result.ledger
                parent = book.buy(day("2023-01-03"), 10, D(6), D(10), D(0))
                terms = SpinOffTerms(20, D(reported), D(".2"), "reported terms", ratio_is_approximate=approximate)
                quotes = {10: D(9), 20: D(6)}
                engine.advance(ex, Session(quotes, quotes, [SpinOff("grant", 10, terms)]), (), {}, ())
                record = engine.result.actions[0][1]
                self.assertEqual((record.status, record.reported_shares_per_parent, record.effective_shares_per_parent),
                                 ("ECONOMIC_PENDING", D(reported), None))
                self.assertEqual((book.holdings(), book.cash, parent.basis, book.sales, engine.last_nav),
                                 ({10: 6}, 940, 60, [], None))

    def test_exact_spin_off_terms_keep_large_denominators_and_literal_decimals(self):
        for raw, exact in [(Fraction(1, 1_000_000), Fraction(1, 1_000_000)),
                           (Fraction(333333, 1000000), Fraction(333333, 1000000)),
                           (D("0.076543"), Fraction(76543, 1000000)), (D(".25"), Fraction(1, 4))]:
            self.assertEqual(SpinOffTerms(20, raw, D(".2"), "exact terms").effective_ratio, exact)
        for raw in (D("0.333333"), D("0.000100")):
            terms = SpinOffTerms(20, raw, D(".2"), "issuer", ratio_is_approximate=False)
            self.assertEqual(terms.effective_ratio, Fraction(raw))

    def test_spin_off_thirds_settle_once_per_child(self):
        ex = day("2024-01-03")
        for parents in ((10,), (10, 10, 10), (10, 11, 12)):
            with self.subTest(parents=parents):
                engine = AccountEngine(D(1000), start=day("2024-01-02"), fee=no_fee)
                book = engine.result.ledger
                for p in parents:
                    book.buy(day("2023-01-03"), p, D(3 // len(parents)), D(10), D(0))
                terms = SpinOffTerms(20, Fraction(1, 3), D(".1"), "issuer")
                events = [SpinOff(str(p), p, terms) for p in sorted(set(parents))]
                quotes = {**dict.fromkeys(parents, D(9)), 20: D(3)}
                market = Session(quotes, quotes, events)
                with self.assertRaisesRegex(ValueError, "additional whole shares=1"):
                    engine.advance(ex, market, [Order(20, 1, {})], {}, ())
                self.assertEqual((book.cash, book.total_basis(), engine.result.actions), (970, 30, []))
                engine.advance(ex, market, (), {}, ())
                self.assertEqual((book.holdings()[20], book.cash, book.sales, book.total_basis()), (1, 970, [], 30))
                self.assertEqual(sum(l.basis for l in book.lots.values() if l.permno == 20), 3)
                quotes[20] = D(6)
                engine.advance(day("2024-01-04"), Session(quotes, quotes, []), (), {}, ())
                self.assertEqual(engine.last_nav, 1003)

    def test_proportional_cil_preserves_break_even_and_real_subcent_losses(self):
        ex = day("2024-03-01")
        for close in (D(12), D("11.999999")):
            engine = AccountEngine(D(1000), start=ex-timedelta(days=1), fee=no_fee)
            book = engine.result.ledger
            for shares in (4, 3):
                book.buy(day("2023-01-03"), 10, D(shares), D(10), D(0))
            grant = SpinOff("sixth", 10, SpinOffTerms(20, Fraction(1, 6), D(".2"), "issuer"))
            prices = {10: D(8), 20: close}
            engine.advance(ex, Session(prices, prices, [grant]), (), {}, ())
            self.assertEqual(book.holdings()[20], 1)
            self.assertTrue(all(s.gain == 0 if close == 12 else s.gain < 0 for s in book.sales))

    def test_two_cil_losses_keep_the_next_purchase_at_its_fill_basis(self):
        first = day("2024-05-01")
        engine = AccountEngine(D(200), start=first - timedelta(days=1), fee=no_fee)
        book = engine.result.ledger
        prices = {10: D(10), 11: D(20), 20: D(30)}
        for p in (10, 11):
            book.buy(day("2024-01-02"), p, D(1), D(60), D(0))
        for offset, p in enumerate((10, 11)):
            ratio = Fraction(offset + 1, 3)
            event = SpinOff(str(p), p, SpinOffTerms(20, ratio, D(".5"), "test"))
            engine.advance(first + timedelta(days=offset), Session(prices, prices, [event]), (), {}, ())
        engine.advance(first + timedelta(days=2), Session(prices, prices, []), [Order(20, 1, {})], {}, ())
        parts = [lot for lot in book.lots.values() if lot.permno == 20]
        self.assertEqual([s.taxable_gain for s in book.sales], [-20, -10])
        self.assertEqual((book.wash_matches, len(parts), parts[0].basis, book.holdings()[20]), ([], 1, 30, 1))

    def test_wash_sale_keeps_the_two_halves_of_each_repurchase_together(self):
        engine = AccountEngine(D(1000), start=day("2024-01-02"), fee=no_fee)
        book = engine.result.ledger
        for bought, price in [("2022-01-03", 60), ("2022-01-04", 10), ("2023-01-03", 60), ("2023-01-04", 10)]:
            book.buy(day(bought), 10, D(1), D(price), D(0))
        prices = {10: D(80)}
        market = Session(prices, prices, [])
        engine.advance(day("2024-01-03"), Session(prices, prices, [CorporateAction("half", 10, "SPLIT", D(".5"))]),
                       (), {}, ())
        old = list(book.trading_positions(10))
        for lot_id, sold, bought in zip(old, ("2024-01-04", "2024-03-04"), ("2024-01-05", "2024-03-05")):
            engine.advance(day(sold), market, [Order(10, D(-1), {lot_id: D(1)})], {}, ())
            engine.advance(day(bought), market, [Order(10, D(1), {})], {}, ())
        signal, sale_day = day("2024-04-09"), day("2024-04-10")
        state = engine.close_snapshot(signal, prices, trade_day=sale_day)
        self.assertEqual([(l.shares, l.basis) for l in state.lots], [(1, 100), (1, 100)])
        weight = {10: D(100) / state.nav}  # A close of $80 locks in keeping one share.
        halves = {lot.segments[-1].id: D(".5") for lot in state.lots}
        with self.assertRaises(ValueError):
            engine.prepare(signal, prices, sale_day, Target(weight, sell_lots={10: halves}), set())
        for rule in (hifo, ltfo):
            orders = engine.prepare(signal, prices, sale_day, Target(weight, lot_relief=rule), set())
            self.assertEqual(list(orders[0].lots.values()), [1])
        previous_sales = len(book.sales)
        engine.advance(sale_day, market, orders, {}, ())
        sales = book.sales[previous_sales:]
        self.assertEqual((len({s.source_lot_id for s in sales}), sum(s.taxable_gain for s in sales), book.total_basis()),
                         (1, -20, 100))

    def test_wash_matches_only_bind_the_single_partially_matched_share(self):
        sold = day("2024-05-01")
        for offset in (-1, 1):
            book = Ledger(D(1000))
            parent_date = day("2022-01-03")
            a = book.receive(parent_date, 10, D("1.5"), D(150), parent_date, "distribution A")
            b = book.receive(parent_date, 10, D(".5"), D(30), parent_date, "distribution B")
            if offset < 0:
                book.buy(sold + timedelta(days=offset), 10, D(4), D(80), D(0))
            losses = book.sell(sold, 10, {a.id: D("1.5"), b.id: D(".5")}, D(80), {a.id: D(0), b.id: D(0)}, "TRADE")
            if offset > 0:
                book.buy(sold + timedelta(days=offset), 10, D(4), D(80), D(0))
            lots = book.trading_lots(10)
            with self.subTest(offset=offset):
                self.assertEqual(sorted((sum(s.shares for s in parts), sum(s.basis for s in parts)) for parts in lots.values()),
                                 [(1, 90), (1, 100), (2, 160)])
                self.assertEqual((losses[0].disallowed_loss, sum(m.shares for m in book.wash_matches)), (30, D("1.5")))

    def test_cil_and_later_hifo_sale_replay_from_one_weights_file(self):
        signal, first, ex_a, ex_b, sale, following = map(day, ["2024-01-02", "2024-01-03", "2024-01-04",
                                                               "2024-03-04", "2024-04-10", "2024-05-13"])
        sessions = [signal, first, ex_a, ex_b, sale, following]
        markets = {}
        for d in sessions:
            prices = {10: D(60 if d >= ex_a else 150), 11: D(60 if d >= ex_b else 150), 20: D(80)}
            parent = {ex_a: 10, ex_b: 11}.get(d)
            events = [SpinOff(str(parent), parent, SpinOffTerms(20, D(".5"), D(".6"), "issuer"))] if parent else []
            markets[d] = Session(prices, prices, events)
        child_targets = {first: 0, ex_a: 1, ex_b: 2, sale: 1}
        def decide(state):
            quantities = {10: 1, 11: 1, 20: child_targets[state.trade_day]}
            weights = {p: (D(q) + D(".01")) * state.closes[p] / state.nav for p, q in quantities.items() if q}
            return Target(weights, lot_relief=hifo)

        settings = dict(sessions=sessions, first_trade=first, end=sale, days=markets, delistings=[],
                        next_rebalance=following, delistings_known_through=following, fee=no_fee, initial_cash=D(1000))
        live = run(**settings, strategy=decide, rebalance_days=child_targets)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "weights.csv"
            write_weights(path, live.targets)
            replay = run(**settings, weights_file=path)
        self.assertEqual(list(live.targets[sale].sell_lots[20].values()), [1])
        self.assertEqual((live.history, live.ledger.sales, live.ledger.wash_matches),
                         (replay.history, replay.ledger.sales, replay.ledger.wash_matches))
        self.assertEqual([s.taxable_gain for s in replay.ledger.sales], [-20, -20, 0])
        self.assertEqual((replay.history[-1].cash, replay.history[-1].nav, replay.ledger.total_basis()), (700, 900, 260))

    def test_successive_fractional_losses_complete_the_same_replacement_share(self):
        sold = day("2024-05-01")
        old = day("2022-01-03")
        for offset in (-1, 2):
            book = Ledger(D(1000))
            grants = [(book.receive(old, 10, D(".5"), D(60), old, ref),
                       book.receive(old, 10, D(".5"), D(40), old, ref)) for ref in ("A", "B")]
            if offset < 0:
                book.buy(sold + timedelta(days=offset), 10, D(2), D(80), D(0))
            for i, pair in enumerate(grants):
                lots = {lot.id: D(".5") for lot in pair}
                book.sell(sold + timedelta(days=i), 10, lots, D(80), dict.fromkeys(lots, D(0)), "TRADE")
            if offset > 0:
                book.buy(sold + timedelta(days=offset), 10, D(2), D(80), D(0))
            groups = book.trading_lots(10).values()
            self.assertEqual(sorted((sum(l.shares for l in g), sum(l.basis for l in g)) for g in groups), [(1, 80), (1, 120)])
            self.assertEqual(len({book.lots[m.replacement_lot].trading_id for m in book.wash_matches}), 1)

    def test_child_share_units_survive_a_split_later_on_the_ex_date(self):
        ex = day("2024-03-01")
        for count, ratio, factor in [(1, Fraction(1, 3), D(3)), (3, Fraction(2, 3), D(".5"))]:
            engine = AccountEngine(D(1000), start=ex - timedelta(days=1), fee=no_fee)
            book = engine.result.ledger
            for i in range(count):
                book.buy(ex - timedelta(days=i+5), 10, D(1), D(10+i), D(0))
            basis = book.total_basis()
            events = [SpinOff("grant", 10, SpinOffTerms(20, ratio, D(".1"), "issuer")),
                      CorporateAction("child split", 20, "SPLIT", factor)]
            prices = {10: D(9), 20: D(10)}
            engine.advance(ex, Session(prices, prices, events), (), {}, ())
            self.assertEqual((book.holdings(), book.total_basis(), book.sales), ({10: count, 20: 1}, basis, []))

    def test_a_passive_fraction_from_a_bound_parent_can_deliver_with_other_entitlements(self):
        ex = day("2024-03-01")
        engine = AccountEngine(D(1000), start=ex-timedelta(days=2), fee=no_fee)
        book = engine.result.ledger
        for i, p in enumerate([10, 10, 11]):
            book.buy(ex-timedelta(days=90-i), p, D(1), D(15+i), D(0))
        prices = {10: D(20), 11: D(40), 20: D(15)}
        engine.advance(ex-timedelta(days=1), Session(prices, prices, [CorporateAction("half", 10, "SPLIT", D(".5"))]), (), {}, ())
        events = [SpinOff("A", 10, SpinOffTerms(20, Fraction(1, 3), D(".2"), "A terms")),
                  SpinOff("B", 11, SpinOffTerms(20, Fraction(2, 3), D(".2"), "B terms"))]
        engine.advance(ex, Session(prices, prices, events), (), {}, ())
        sale_day = ex + timedelta(days=3)
        lots = hifo(engine.close_snapshot(ex, prices, trade_day=sale_day), 20, D(1))
        engine.advance(sale_day, Session(prices, prices, []), [Order(20, D(-1), lots)], {}, ())
        self.assertEqual((book.holdings(), sum(s.proceeds for s in book.sales), sum(s.basis for s in book.sales)),
                         ({10: 1, 11: 1}, 15, D("9.6")))

    def test_manual_order_quantity_must_agree_with_sale_allocation(self):
        signal, trade = day("2024-01-02"), day("2024-01-03")
        engine = AccountEngine(D(100), start=signal, fee=no_fee)
        book = engine.result.ledger
        old = book.buy(signal, 10, D(3), D(10), D(0))
        prices = {10: D(10)}
        orders = (Order(10, D(-1), {old.id: D(2)}),)
        with self.assertRaisesRegex(ValueError, "allocation"):
            engine.advance(trade, Session(prices, prices, []), orders, {}, ())
        with self.assertRaisesRegex(ValueError, "one net order"):
            engine.advance(trade, Session(prices, prices, []), [Order(10, 1, {}), Order(10, 1, {})], {}, ())
        self.assertEqual((book.holdings(), book.cash, book.sales, engine.result.observed_through), ({10: 3}, 70, [], signal))
        engine.advance(trade, Session(prices, prices, []), [Order(10, -1, {old.id: 1})], {}, ())
        self.assertEqual((book.holdings(), book.cash), ({10: 2}, 80))

    def test_replanning_cash_shortfall_books_only_the_final_buy_block(self):
        signal, trade = day("2024-01-02"), day("2024-01-03")
        engine = AccountEngine(D(100), start=signal, fee=no_fee)
        prices = {10: D(10), 20: D(10), 30: D(10)}
        target = Target({10: D(".9"), 20: D(".05"), 30: D(".05")})
        orders = engine.prepare(signal, prices, trade, target, {20, 30})
        expensive = {**prices, 10: D(20)}
        with self.assertRaisesRegex(ValueError, "available cash 100"):
            engine.advance(trade, Session(expensive, expensive, []), orders, {}, ())
        orders = engine.prepare(signal, prices, trade, Target({10: D(".9"), 20: D(".1")}), {20})
        engine.advance(trade, Session(prices, prices, []), orders, {}, ())
        self.assertEqual(engine.result.exceptions, [(trade, 20, "BUY: blocked by pre-delisting convention")])
        tomorrow = trade + timedelta(days=1)
        orders = engine.prepare(trade, prices, tomorrow, Target({10: D(".1"), 30: D(".9")}), set())
        expensive = {**prices, 30: D(50)}
        with self.assertRaisesRegex(ValueError, "available cash"):
            engine.advance(tomorrow, Session(expensive, expensive, []), orders, {}, ())
        carried = Target({10: D(".9"), 20: D(".1")}, False)
        orders = engine.prepare(trade, prices, tomorrow, carried, set())
        engine.advance(tomorrow, Session(prices, prices, []), orders, {}, ())
        self.assertEqual((engine.result.targets[tomorrow], engine.result.ledger.holdings()), (carried, {10: 9}))
        self.assertNotIn(trade, engine.result.plans)

    def test_formal_entry_points_use_only_tradable_raw_closes(self):
        t = day("2024-01-03")
        rows = [StockDay(10, t, D(".12"), D(18), D(20)),
                StockDay(11, t, None, D(12), None), StockDay(12, t, None, D(8), D(9))]
        market = sessions_from_crsp(rows, tradable={t: {10, 11}}, actions_by_day={t: []})[t]
        engine = AccountEngine(D(100), start=t-timedelta(days=1), fee=no_fee)
        for p in (11, 12):
            with self.assertRaisesRegex(ValueError, f"{t} {p} BUY: missing executable CLOSE"):
                engine.advance(t, market, (Order(p, D(1), {}),), {}, ())
            self.assertEqual((engine.result.ledger.cash, engine.result.history), (100, []))
        for unavailable in (Session({10: D(18)}, {10: D(20)}, []),
                            Session({10: D(20)}, {10: D(20)}, [], tradable=set())):
            with self.assertRaisesRegex(ValueError, f"{t} 10 BUY: missing executable CLOSE"):
                engine.advance(t, unavailable, (Order(10, D(1), {}),), {}, ())
        engine.advance(t, market, (Order(10, D(1), {}),), {}, ())
        fill, = engine.result.ledger.entries
        self.assertEqual((fill.price_source, fill.price, engine.result.ledger.cash, engine.last_nav),
                         ("CLOSE", 20, 80, 100))

    def test_cil_aggregates_tax_lots_and_charges_no_commission(self):
        t = day("2024-02-05")
        engine = AccountEngine(D(1000), start=t-timedelta(days=1), fee=proportional_commission(D(".01")))
        book = engine.result.ledger
        for qty in (1, 2, 3):
            book.buy(day("2023-01-03"), 10, D(qty), D(10), D(0))
        split = CorporateAction("r", 10, "SPLIT", D(".25"))
        engine.advance(t, Session({10: D(40)}, {10: D(40)}, [split]), (), {}, ())
        # 1.5 shares are delivered in total. One whole share is kept and only 0.5 share is sold; all three lots have the same per-share basis.
        self.assertEqual(book.holdings(), {10: 1})
        self.assertEqual((book.cash, sum(s.shares for s in book.sales), sum(s.proceeds for s in book.sales)),
                         (960, D(".5"), 20))
        self.assertEqual(sum(s.basis for s in book.sales), 20)
        self.assertTrue(all(e.fee == 0 for e in book.entries if e.kind == "CASH_IN_LIEU"))
        self.assertEqual(engine.result.history[-1].nav, 1000)

    def test_existing_half_shares_can_be_combined_without_splitting_an_intact_share(self):
        t = day("2024-03-04")
        engine = AccountEngine(D(1000), start=t-timedelta(days=1), fee=no_fee)
        book = engine.result.ledger
        for price in (30, 10):
            book.buy(day("2023-01-03"), 10, D(1), D(price), D(0))
        split = CorporateAction("two halves", 10, "SPLIT", D(".5"))
        engine.advance(t, Session({}, {10: D(40)}, [split]), (), {}, ())
        self.assertEqual((book.holdings(), book.sales), ({10: 1}, []))
        book.buy(t, 10, D(1), D(40), D(0))
        state = engine.close_snapshot(t, {10: D(40)})
        combined, ordinary = state.eligible_lots(10)
        self.assertEqual(hifo(state, 10, D(1)), {combined.id: D(1)})
        self.assertEqual([s.source_lot_id for s in combined.segments], [1, 2])
        with self.assertRaisesRegex(ValueError, "bound segment"):
            book.sell(t, 10, {1: D(".5"), ordinary.id: D(".5")}, D(40), {1: D(0), ordinary.id: D(0)}, "TRADE")

    def test_consecutive_splits_keep_exact_whole_security_positions(self):
        first = day("2024-01-02")
        engine = AccountEngine(D(10000), start=first, fee=no_fee)
        for q in (13, 27, 41, 53):
            engine.result.ledger.buy(first, 11, D(q), D(10), D(0))
        for n, (ratio, expected) in enumerate([(".45", 60), (".625", 37), ("2", 74), (".45", 33), ("4", 132)], 1):
            event = CorporateAction(str(n), 11, "SPLIT", D(ratio))
            prices = {} if ratio in {"2", "4"} else {11: D(10)}
            engine.advance(first + timedelta(days=n), Session(prices, {11: D(10)}, [event]), (), {}, ())
            self.assertEqual(engine.result.ledger.holdings(), {11: expected})

    def test_split_on_a_rebalance_day_cannot_create_an_extra_order(self):
        signal, ex = day("2024-03-04"), day("2024-03-05")
        engine = AccountEngine(D(100), start=signal, fee=no_fee)
        engine.result.ledger.buy(signal, 10, D(2), D(20), D(0))
        target = Target({10: D(".4"), 11: D(".1")})
        orders = engine.prepare(signal, {10: D(20), 11: D(10)}, ex, target, set())
        self.assertEqual([(o.permno, o.quantity) for o in orders], [(11, 1)])
        closes = {10: D(10), 11: D(10)}
        split = CorporateAction("double", 10, "SPLIT", D(2))
        engine.advance(ex, Session(closes, closes, [split], raw_opens={10: D(8)}), orders, {}, ())
        book = engine.result.ledger
        self.assertEqual((book.holdings(), book.cash, book.total_basis(), engine.last_nav), ({10: 4, 11: 1}, 50, 50, 100))

    def test_roc_sale_uses_the_locked_lot_and_reduced_basis(self):
        signal, ex = day("2024-03-04"), day("2024-03-05")
        engine = AccountEngine(D(100), start=signal, fee=no_fee)
        book = engine.result.ledger
        lot = book.buy(signal, 10, D(1), D(100), D(0))
        prices = {10: D(95)}
        roc = CorporateAction("return of capital", 10, "ROC", D(10))
        engine.advance(ex, Session(prices, prices, [roc]), [Order(10, -1, {lot.id: 1})], {}, ())
        sale, = book.sales
        self.assertEqual((sale.lot_id, sale.basis, sale.gain, book.cash, engine.last_nav), (lot.id, 90, 5, 105, 105))

    def test_split_sale_is_cancelled_and_keeps_the_locked_plan(self):
        t = day("2024-03-04")
        engine = AccountEngine(D(100), start=t-timedelta(days=1), fee=no_fee, lot_relief=hifo)
        book = engine.result.ledger
        for dt in ("2023-01-03", "2023-02-06"):
            book.buy(day(dt), 10, D(4), D(10), D(0))
        orders = engine.prepare(t-timedelta(days=1), {10: D(10)}, t, Target({10: D(".6")}), set())
        split = CorporateAction("r", 10, "SPLIT", D(2))
        engine.advance(t, Session({10: D(5)}, {10: D(5)}, [split], raw_opens={10: D(6)}), orders, {}, ())
        self.assertEqual((book.holdings(), book.sales, book.cash), ({10: 16}, [], 20))
        self.assertEqual(engine.result.exceptions, [(t, 10, 'SKIPPED_SPLIT_SALE 2/8 pre-split shares')])
        self.assertEqual(len(engine.result.actions), 1)
        self.assertEqual(engine.result.plans[t-timedelta(days=1)], orders)

    def test_stock_dividend_alone_still_requires_locked_order_review(self):
        t = day("2024-03-04")
        dividend = CorporateAction("same stock", 10, "NONTAXABLE_STOCK_DIVIDEND", D("1.1"))
        for held, events, reason in [(20, [dividend], "NONTAXABLE_STOCK_DIVIDEND: same stock")]:
            with self.subTest(held=held, reason=reason):
                engine = AccountEngine(D(200), start=t-timedelta(days=1), fee=no_fee)
                book = engine.result.ledger
                lot = book.buy(day("2023-01-03"), 10, D(held), D(10), D(0))
                with self.assertRaisesRegex(ValueError, reason):
                    engine.advance(t, Session({10: D(5)}, {10: D(5)}, events), [Order(10, -1, {lot.id: 1})], {}, ())
                self.assertEqual((book.holdings(), book.sales, engine.result.exceptions), ({10: held}, [], []))

    def test_cross_year_loss_matches_by_share_at_day_30_and_not_day_31(self):
        book = Ledger(D(2000))
        old = book.buy(day("2023-12-01"), 11, D(6), D(100), D(0))
        sold = day("2024-12-31")
        loss, = book.sell(sold, 11, {old.id: D(6)}, D(90), {old.id: D(0)}, "TRADE")
        book.buy(sold, 99, D(2), D(70), D(0))
        book.buy(sold + timedelta(days=30), 11, D(4), D(80), D(0))
        book.buy(sold + timedelta(days=31), 11, D(2), D(80), D(0))
        matched, = book.wash_matches
        replacement = book.lots[matched.replacement_lot]
        self.assertEqual((loss.gain, loss.disallowed_loss, loss.taxable_gain, replacement.basis), (-60, 40, -20, 360))
        self.assertEqual(replacement.holding_start, old.acquired + timedelta(days=30))
        income = book.tax_year(2024, basis_verified=True)
        tax, = settle_years([income], Scenario.EXTERNAL_GAINS, sold + timedelta(days=30))
        self.assertEqual((income.long_term, tax.net_tax, tax.final), (-20, D("-4.76"), True))

    def test_lookback_allocates_only_unused_shares_and_respects_same_purchase(self):
        book = Ledger(D(1000))
        a = book.buy(day("2024-01-02"), 11, D(2), D(100), D(0))
        b = book.buy(day("2024-02-01"), 11, D(3), D(80), D(0))
        sale, = book.sell(day("2024-03-02"), 11, {a.id: D(2)}, D(90), {a.id: D(0)}, "TRADE")
        matched = book.lots[book.wash_matches[0].replacement_lot]
        self.assertEqual((sale.disallowed_loss, matched.shares, matched.basis, matched.holding_start),
                         (20, 2, 180, a.acquired))
        self.assertEqual((b.shares, b.basis), (1, 80))
        follow, = book.sell(day("2024-03-03"), 11, {b.id: D(1)}, D(70), {b.id: D(0)}, "TRADE")
        self.assertEqual((follow.disallowed_loss, len(book.wash_matches)), (0, 1))

    def test_multiple_losses_share_one_repurchase_in_sale_order(self):
        book = Ledger(D(1000))
        a = book.buy(day("2024-01-02"), 11, D(2), D(100), D(0))
        b = book.buy(day("2024-01-10"), 11, D(3), D(110), D(0))
        newer, older = book.sell(day("2024-04-01"), 11, {b.id: D(3), a.id: D(2)}, D(90),
                                 {a.id: D(0), b.id: D(0)}, "TRADE")
        book.split(day("2024-04-02"), 11, D(2), "split")
        book.buy(day("2024-04-03"), 11, D(5), D(43), D(0))
        self.assertEqual([(m.sold_lot, m.shares) for m in book.wash_matches], [(a.id, 4), (b.id, 1)])
        self.assertEqual((older.disallowed_loss, newer.disallowed_loss, book.total_basis()), (20, 10, 245))

    def test_reverse_split_cil_loss_repurchased_16_days_later(self):
        t = day("2024-02-01")
        engine = AccountEngine(D(1000), start=t, fee=no_fee)
        book = engine.result.ledger
        lot = book.buy(t, 11, D(3), D(100), D(0))
        event_day = t + timedelta(days=15)
        event = CorporateAction("split", 11, "SPLIT", D(".5"))
        engine.advance(event_day, Session({11: D(180)}, {11: D(180)}, [event]), (), {}, ())
        cil, = book.sales
        self.assertEqual((book.cash, book.holdings(), cil.proceeds, cil.basis, cil.taxable_gain),
                         (790, {11: 1}, 90, 100, -10))
        self.assertEqual(engine.result.history[-1].nav, 970)
        engine.advance(t + timedelta(days=31), Session({11: D(170)}, {11: D(170)}, []),
                       (Order(11, D(1), {}),), {}, ())
        purchase, = [l for l in book.lots.values() if l.acquired == t + timedelta(days=31)]
        self.assertEqual((cil.lot_id, book.wash_matches, cil.disallowed_loss, cil.taxable_gain), (lot.id, [], 0, -10))
        self.assertEqual((purchase.basis, book.cash, book.holdings()), (170, 620, {11: 2}))
        self.assertEqual(engine.result.history[-1].nav, 960)

    def test_december_cil_loss_alone_does_not_extend_the_wash_window(self):
        signal, ex = day("2024-12-30"), day("2024-12-31")
        engine = AccountEngine(D(1000), start=signal, fee=no_fee)
        book = engine.result.ledger
        book.buy(signal, 11, D(3), D(100), D(0))
        price = {11: D(180)}
        engine.advance(ex, Session(price, price, [CorporateAction("half", 11, "SPLIT", D(".5"))]), (), {}, ())
        income = book.tax_year(2024, basis_verified=True)
        self.assertEqual((income.short_term, income.last_loss_day, book.wash_matches, book.open_losses), (-10, None, [], {}))
        external, = engine.result.annual_wealth(Scenario.EXTERNAL_GAINS)
        limited, = engine.result.annual_wealth(Scenario.LIMITED_USE)
        self.assertEqual((external.tax.final, external.tax.net_tax, external.after_tax), (True, D("-4.08"), D("974.08")))
        self.assertEqual((limited.tax.final, limited.tax.st_carryforward, limited.after_tax), (True, 10, 970))

    def test_cash_shortfall_rolls_back_events_sales_and_wash_then_can_retry(self):
        sold = day("2024-03-04")
        engine = AccountEngine(D(100), fee=no_fee, start=sold - timedelta(days=1))
        book = engine.result.ledger
        lot = book.buy(day("2024-01-02"), 10, D(10), D(10), D(0))
        event = CorporateAction("div", 10, "CASH_DIVIDEND", D(".1"))
        orders = (Order(10, -D(1), {lot.id: D(1)}), Order(20, D(1), {}))
        def session(price):
            prices = {10: D(9), 20: D(price)}
            return Session(prices, prices, [event])
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "cost 11, available cash 10"):
                engine.advance(sold, session(11), orders, {}, ())
            self.assertEqual((book.cash, book.holdings(), book.sales, len(book.entries)), (0, {10: 10}, [], 1))
            self.assertEqual((engine.result.actions, engine.result.history, book.open_losses.get(10)), ([], [], None))
        engine.advance(sold, session(10), orders, {}, ())
        self.assertEqual((book.cash, book.holdings(), len(book.sales)), (0, {10: 9, 20: 1}, 1))
        with self.assertRaisesRegex(ValueError, "already booked"):
            engine.advance(sold, session(10), orders, {}, ())

    def test_daily_strategy_and_known_exit_dates_are_independent(self):
        dates = list(map(day, ["2023-12-29", "2024-01-02", "2024-01-03", "2024-02-02", "2024-03-04", "2024-04-04"]))
        signal, first, extra, exit_day, delist_day, following = dates
        prices = {11: D(100), 22: D(50)}
        markets = {d: Session(prices, prices, []) for d in dates}
        seen = []
        def decide(s):
            seen.append((s.trade_day, s.blocked))
            if s.trade_day == extra:
                return Target({11: D(".4"), 22: D(".2")})
            return Target({11: D(".5")})
        result = run(strategy=decide, sessions=dates, first_trade=first, end=delist_day, days=markets,
                     fee=no_fee, exit_days=[first, exit_day, delist_day, following],
                     delistings=[Delisting(11, delist_day, "MER")], delistings_known_through=following)
        self.assertEqual([x[0] for x in seen], dates[1:-1])
        self.assertEqual([v.holdings for v in result.history],
                         [{11: 5000}, {11: 4000, 22: 4000}, {}, {}])
        self.assertEqual((result.exits[0].day, result.exits[0].shares), (exit_day, 4000))
        self.assertEqual(seen[-1][1], {11})

    def test_spin_off_and_child_cil_preserve_basis_and_use_child_price(self):
        ex = day("2024-05-01")
        later = {10: D(30), 20: D(15)}
        engine = AccountEngine(D(100), fee=no_fee, start=ex-timedelta(days=1),
                               basis_days={ex+timedelta(days=1): Session(later, later, [])})
        book = engine.result.ledger
        parent = book.buy(day("2022-01-03"), 10, D(3), D(30), D(0))
        event = SpinOff("sp-1", 10, SpinOffTerms(20, D(".5"), D(".2"), "synthetic issuer terms"))
        for fill in (Session({10: D(24)}, {10: D(24)}, [event]),
                     Session({20: D(12)}, {10: D(24)}, [event], "VWAP")):
            with self.assertRaisesRegex(ValueError, "20 CASH_IN_LIEU: missing executable CLOSE"):
                engine.advance(ex, fill, (), {}, ())
            self.assertEqual((book.cash, book.total_basis(), book.holdings()), (10, 90, {10: 3}))
        quotes = {10: D(24), 20: D(12)}
        engine.advance(ex, Session(quotes, quotes, [event]), (), {}, ())
        cil, = book.sales
        self.assertEqual((book.cash, book.holdings(), book.total_basis(), cil.basis, cil.gain), (16, {10: 3, 20: 1}, 84, 6, 0))
        self.assertEqual((cil.is_long_term, cil.holding_start, engine.result.history[-1].nav), (True, parent.acquired, 100))

    def test_spin_off_uses_the_first_common_future_close_for_basis_only_in_strategy_and_replay(self):
        signal, first, ex, split_day, joint_day, later = map(
            day, ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"))
        grant = SpinOff("half", 10, SpinOffTerms(20, Fraction(1, 2), D(".7"), "issuer example"))
        split = CorporateAction("split", 20, "SPLIT", D(2))
        data = [(signal, {10: D(100)}, []), (first, {10: D(100)}, []), (ex, {10: D(90)}, [grant]),
                (split_day, {20: D(5)}, [split]), (joint_day, {10: D(60), 20: D(20)}, []),
                (later, {10: D(90), 20: D(1)}, [])]
        markets = {d: Session(prices, prices, events) for d, prices, events in data}
        settings = dict(sessions=list(markets), first_trade=first, end=ex, days=markets, delistings=[],
                        pre_delisting=False, delistings_known_through=ex, fee=no_fee, initial_cash=D(1000))
        live = run(**settings, strategy=lambda s: Target({10: D(".2")}), rebalance_days=[first])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            write_weights(path, live.targets)
            replay = run(**settings, weights_file=path)
        for result in (live, replay):
            record = result.actions[0][1]
            self.assertEqual((record.basis_price_day, record.close_basis_fraction), (joint_day, Fraction(1, 4)))
            self.assertEqual({lot.permno: lot.basis for lot in result.ledger.lots.values()}, {10: 150, 20: 50})
            # One share arrives on the event date and cash stays at 800. Future prices and splits are used only to estimate basis.
            self.assertEqual((result.history[-1].holdings, result.history[-1].cash, result.history[-1].nav),
                             ({10: 2, 20: 1}, 800, None))

    def test_future_basis_and_ex_date_cil_proceeds_use_their_own_prices(self):
        before, ex, next_day = map(day, ("2024-05-01", "2024-05-02", "2024-05-03"))
        later = {10: D(40), 20: D(20)}
        engine = AccountEngine(D(100), start=before, fee=no_fee, basis_days={next_day: Session(later, later, [])})
        book = engine.result.ledger
        book.buy(before, 10, D(1), D(100), D(0))
        grant = SpinOff("half", 10, SpinOffTerms(20, Fraction(1, 2), None, "issuer"))
        child_only = {20: D(10)}
        engine.advance(ex, Session(child_only, child_only, [grant]), (), {}, ())
        cil, = book.sales
        self.assertEqual((cil.day, cil.proceeds, cil.basis, cil.gain, book.cash), (ex, 5, 20, -15, 5))
        self.assertEqual((book.total_basis(), engine.last_nav, engine.result.actions[0][1].basis_price_day),
                         (80, None, next_day))

    def test_spin_off_reports_all_conflicting_orders_and_allows_a_reviewed_retry(self):
        signal, trade = day("2024-04-30"), day("2024-05-01")
        quotes = {10: D(20), 20: D(10), 30: D(10), 40: D(10)}
        engine = AccountEngine(D(1000), start=signal-timedelta(days=1), fee=no_fee)
        book = engine.result.ledger
        initial = (Order(10, D(20), {}), Order(30, D(4), {}))
        engine.advance(signal, Session(quotes, quotes, []), initial, {}, ())
        events = [SpinOff("spin-A", 10, SpinOffTerms(20, D(".5"), D(".25"), "issuer A")),
                  SpinOff("spin-B", 10, SpinOffTerms(30, D(".1"), D(".1"), "issuer B")),
                  SpinOff("unheld-parent", 99, SpinOffTerms(40, D(1), D(".1"), "issuer C"))]
        ex_quotes = {**quotes, 10: D(14)}
        market = Session(ex_quotes, ex_quotes, events)
        target = Target({10: D(".2"), 20: D(".1"), 30: D(".02"), 40: D(".01")})
        orders = engine.prepare(signal, quotes, trade, target, set())
        with self.assertRaisesRegex(ValueError, "LOCKED_ORDER_REVIEW") as failure:
            engine.advance(trade, market, orders, {}, ())
        message = str(failure.exception)
        self.assertIn(str(trade), message)
        self.assertNotIn("PERMNO 10 SELL", message)
        self.assertIn("PERMNO 20 BUY 10", message)
        self.assertIn("PERMNO 30 SELL 2", message)
        self.assertIn("additional whole shares=10", message)
        self.assertIn("parent PERMNO 10", message)
        self.assertIn("sell_lots=2:2", message)
        self.assertIn("spin-A", message)
        self.assertIn("spin-B", message)
        self.assertNotIn("unheld-parent", message)
        self.assertNotIn("PERMNO 40", message)
        self.assertEqual((book.cash, book.holdings(), book.total_basis()), (560, {10: 20, 30: 4}, 440))
        self.assertEqual((len(book.entries), book.sales, engine.result.actions), (2, [], []))
        self.assertEqual((engine.result.observed_through, len(engine.result.history)), (signal, 1))

        # Retry after cancelling the two child-share orders; the original parent sell order still settles.
        reviewed = Target({10: D(".2"), 30: D(".04"), 40: D(".01")})
        orders = engine.prepare(signal, quotes, trade, reviewed, set())
        engine.advance(trade, market, orders, {}, ())
        self.assertEqual((book.holdings(), book.cash, engine.last_nav),
                         ({10: 10, 20: 10, 30: 6, 40: 1}, 690, 1000))
        sale, = book.sales
        self.assertEqual((sale.lot_id, sale.shares, sale.proceeds, sale.basis), (1, 10, 140, 140))

    def test_parent_buy_after_spin_off_uses_the_locked_shares_and_gets_no_distribution(self):
        signal, trade = day("2024-04-30"), day("2024-05-01")
        engine = AccountEngine(D(200), start=signal, fee=no_fee)
        book = engine.result.ledger
        original = book.buy(signal, 10, D(2), D(50), D(0))
        event = SpinOff("grant", 10, SpinOffTerms(20, Fraction(1), None, "issuer"))
        quotes = {10: D(35), 20: D(15)}
        engine.advance(trade, Session(quotes, quotes, [event]), [Order(10, 2, {})], {}, ())
        self.assertEqual((book.holdings(), book.cash, original.basis), ({10: 4, 20: 2}, 30, 70))
        bought, = [lot for lot in book.lots.values() if lot.permno == 10 and lot.acquired == trade]
        self.assertEqual((bought.shares, bought.basis, book.total_basis()), (2, 70, 170))

    def test_pending_spin_off_reports_a_parent_buy_without_an_entitlement(self):
        ex = day("2015-01-16")
        ref = "unpublished terms"
        event = CaseReview(ref, 28388, "ECONOMIC_PENDING", "Unknown share ratio", None, spin_off_child=15142)
        engine = AccountEngine(D(100), start=day("2015-01-15"), fee=no_fee)
        prices = {28388: D(10)}
        with self.assertRaisesRegex(ValueError, f"PERMNO 28388 BUY 1 shares .*spin-off parent: {ref}"):
            engine.advance(ex, Session(prices, prices, [event]), [Order(28388, 1, {})], {}, ())
        self.assertEqual((engine.result.ledger.cash, engine.result.ledger.holdings(), engine.result.actions),
                         (100, {}, []))
        child = 15142
        child_prices = {28388: D(10), child: D(5)}
        market = Session(child_prices, child_prices, [event])
        engine.advance(ex, market, [Order(child, 1, {})], {}, ())
        self.assertEqual(engine.result.ledger.holdings(), {child: 1})

        entitled = AccountEngine(D(100), start=day("2015-01-15"), fee=no_fee)
        entitled.result.ledger.buy(day("2015-01-15"), 28388, D(1), D(10), D(0))
        with self.assertRaisesRegex(ValueError, f"PERMNO {child} BUY 1 shares .*unknown spin-off entitlement: {ref}"):
            entitled.advance(ex, market, [Order(child, 1, {})], {}, ())
        self.assertEqual((entitled.result.ledger.holdings(), entitled.result.actions), ({28388: 1}, []))
        unidentified = CaseReview("missing child", 28388, "ECONOMIC_PENDING", "Spin-off terms missing", None,
                                  spin_off_child=0)
        with self.assertRaisesRegex(ValueError, "PERMNO 99 BUY 1 shares .*unknown spin-off entitlement: missing child"):
            entitled.advance(ex, Session(child_prices, child_prices, [unidentified]), [Order(99, 1, {})], {}, ())

    def test_cash_acquisition_cannot_silently_replace_a_locked_sale(self):
        signal, ex = day("2024-03-04"), day("2024-03-05")
        engine = AccountEngine(D(100), start=signal, fee=no_fee)
        book = engine.result.ledger
        original = book.buy(signal, 10, D(2), D(15), D(0))
        quote = {10: D(20)}
        market = Session(quote, quote, [CorporateAction("cash-merger", 10, "CASH_ACQUISITION", D(20))])
        with self.assertRaisesRegex(ValueError, "PERMNO 10 SELL 1 shares .*cash acquisition: cash-merger"):
            engine.advance(ex, market, [Order(10, -1, {original.id: 1})], {}, ())
        self.assertEqual((book.cash, book.holdings(), book.total_basis(), book.sales), (70, {10: 2}, 30, []))
        engine.advance(ex, market, (), {}, ())
        self.assertEqual((book.cash, book.holdings(), book.sales[0].gain), (110, {}, 10))

    def test_unresolved_entitlement_and_pre_delisting_conflict_roll_back_together(self):
        signal, ex = day("2024-03-04"), day("2024-03-05")
        engine = AccountEngine(D(100), start=signal, fee=no_fee)
        book = engine.result.ledger
        old = book.buy(signal, 10, D(1), D(20), D(0))
        pending = CaseReview("unsettled-merger", 10, "ECONOMIC_PENDING", "Unknown consideration", None)
        forced = {11: [Delisting(11, day("2024-03-12"), "MER")]}
        prices = {10: D(20), 11: D(10)}
        with self.assertRaisesRegex(ValueError, "LOCKED_ORDER_REVIEW") as failure:
            engine.advance(ex, Session(prices, prices, [pending]), [Order(10, -1, {old.id: 1}), Order(11, 1, {})],
                           forced, ())
        self.assertIn("PERMNO 10 SELL 1 shares sell_lots=1:1 | unresolved terms: unsettled-merger", str(failure.exception))
        self.assertIn("PERMNO 11 BUY 1 shares | pre-delisting exit window", str(failure.exception))
        self.assertEqual((book.holdings(), book.cash, engine.economic_pending, engine.result.actions), ({10: 1}, 80, False, []))

    def test_cil_only_receipt_keeps_original_buy_in_strategy_and_csv(self):
        calendar = list(map(day, ["2024-04-29", "2024-04-30", "2024-05-01"]))
        signal, first, ex = calendar
        targets = {first: Target({10: D(".12")}), ex: Target({10: D(".12"), 20: D(".1")})}
        prices = {10: D(120), 20: D(10)}
        markets = {d: Session(prices, prices, []) for d in (signal, first)}
        ex_prices = {10: D(95), 20: D(10)}
        event = SpinOff("half", 10, SpinOffTerms(20, D(".5"), D(".06"), "issuer"))
        markets[ex] = Session(ex_prices, ex_prices, [event])
        settings = dict(sessions=calendar, first_trade=first, end=ex, days=markets, initial_cash=D(1000),
                        fee=no_fee, delistings=[], delistings_known_through=ex, pre_delisting=False)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            write_weights(path, targets)
            for source in ({"strategy": lambda state: targets[state.trade_day]}, {"weights_file": path}):
                with self.subTest(source=next(iter(source))):
                    result = run(**settings, **source)
                    end = result.history[-1]
                    self.assertEqual((end.holdings, end.cash, end.nav), ({10: 1, 20: 10}, 785, 980))
                    cil, buy = [e for e in result.ledger.entries if e.day == ex and e.price_source == "CLOSE"]
                    self.assertEqual((cil.kind, cil.shares, cil.cash_change, cil.fee), ("CASH_IN_LIEU", D("-.5"), 5, 0))
                    self.assertEqual((buy.kind, buy.shares, buy.cash_change), ("BUY", 10, -100))
                    sale, = result.ledger.sales
                    self.assertEqual((sale.basis, sale.gain, sale.disallowed_loss, sale.taxable_gain), (6, -1, 0, -1))

    def test_cil_and_locked_full_lot_sale_post_separately_in_replay(self):
        signal, first, ex, next_exit = map(day, ("2024-01-02", "2024-01-03", "2024-01-04", "2024-02-05"))
        before, after = {10: D(100), 20: D(20)}, {10: D(40), 20: D(20)}
        grant = SpinOff("grant", 10, SpinOffTerms(20, D(".5"), D(".2"), "issuer"))
        markets = {signal: Session(before, before, []), first: Session(before, before, []),
                   ex: Session(after, after, [grant])}
        targets = {first: Target({10: D(".1"), 20: D(".04")}), ex: Target({10: D(".1")})}
        settings = dict(sessions=[signal, first, ex, next_exit], first_trade=first, end=ex, days=markets,
                        delistings=[], delistings_known_through=next_exit, exit_days=[first],
                        next_rebalance=next_exit, fee=no_fee, initial_cash=D(1000))
        live = run(**settings, strategy=lambda s: targets[s.trade_day], rebalance_days=targets)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            write_weights(path, live.targets)
            replay = run(**settings, weights_file=path)
        self.assertEqual(live.targets[ex].sell_lots[20], {2: 2})
        for result in (live, replay):
            book = result.ledger
            self.assertEqual((book.holdings(), book.cash, result.history[-1].nav), ({10: 1}, 910, 950))
            self.assertEqual([(s.reason, s.taxable_gain) for s in book.sales], [("CASH_IN_LIEU", -10), ("TRADE", 0)])
            self.assertEqual(book.wash_matches, [])
        self.assertEqual(live.ledger.sales, replay.ledger.sales)

    def test_cil_keeps_the_original_lot_deliverable_for_a_partial_sale(self):
        signal, ex = day("2024-01-03"), day("2024-01-04")
        engine = AccountEngine(D(1000), start=signal, fee=no_fee)
        book = engine.result.ledger
        book.buy(signal, 10, D(1), D(100), D(0))
        old = book.buy(signal, 20, D(2), D(20), D(0))
        event = SpinOff("grant", 10, SpinOffTerms(20, D(".5"), D(".2"), "issuer"))
        prices = {10: D(40), 20: D(20)}
        engine.advance(ex, Session(prices, prices, [event]), [Order(20, -1, {old.id: 1})], {}, ())
        self.assertEqual((book.holdings(), book.cash, book.lots[old.id].basis, book.wash_matches),
                         ({10: 1, 20: 1}, 890, 20, []))
        self.assertEqual([(s.reason, s.taxable_gain) for s in book.sales], [("CASH_IN_LIEU", -10), ("TRADE", 0)])

    def test_two_half_share_receipts_for_one_child_report_one_order(self):
        signal, trade = day("2024-04-30"), day("2024-05-01")
        engine = AccountEngine(D(100), start=signal, fee=no_fee)
        for parent in (10, 11):
            engine.result.ledger.buy(signal, parent, D(1), D(20), D(0))
        terms = SpinOffTerms(20, D(".5"), D(".25"), "issuer")
        events = [SpinOff("first-half", 10, terms), SpinOff("second-half", 11, terms)]
        prices = {10: D(15), 11: D(15), 20: D(10)}
        with self.assertRaisesRegex(ValueError, "LOCKED_ORDER_REVIEW") as failure:
            engine.advance(trade, Session(prices, prices, events), [Order(20, D(2), {})], {}, ())
        message = str(failure.exception)
        self.assertEqual(message.count("PERMNO 20 BUY 2"), 1)
        self.assertIn("additional whole shares=1", message)
        self.assertIn("first-half", message)
        self.assertIn("second-half", message)

    def test_real_rights_and_spin_off_requirements(self):
        distributions = Path(__file__).resolve().parents[1] / "data/crsp/distributions_2015_to_2026_09.csv"
        key = "CRSP/StkDistributions/28388/2015-01-16/1"
        terms = SpinOffTerms(15142, D(".25"), None, "test terms only")
        events = read_distributions(distributions, spin_off_terms={key: terms})
        spin, = [e for e in events[day("2015-01-16")] if e.reference == key]
        book = Ledger(D(1000))
        book.buy(day("2015-01-02"), 28388, D(4), D(100), D(0))
        rec = spin.post(day("2015-01-16"), book, book.entitled_lots())
        self.assertEqual((rec.status, book.holdings(), book.total_basis()), ("BASIS_PENDING", {28388: 4, 15142: 1}, None))
        rights = [e for e in events[day("2019-11-25")] if e.permno == 81696]
        self.assertIn("RIGHTS_WAIVED", [e.kind for e in rights])
        cash = next(e for e in events[day("2018-01-30")] if e.reference.endswith("/58819/2018-01-30/2"))
        book.buy(day("2018-01-25"), 58819, D(2), D(100), D(0))
        cash.post(day("2018-01-30"), book, book.entitled_lots())
        self.assertEqual(book.tax_year(2018, basis_verified=True).taxable_dividends, D(".001"))

    def test_weight_reader_preserves_observations_and_rejects_missing_values(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark.csv"
            path.write_text("date,permno,weight\n2024-01-02,10,0.003007\n")
            weights = read_benchmark_weights(path)
            self.assertEqual(weights, {day("2024-01-02"): {10: D(".003007")}})
            with self.assertRaises(KeyError):
                _ = weights[day("2024-01-03")]
            path.write_text("date,permno,weight\n2024-01-02,10,\n")
            with self.assertRaisesRegex(ValueError, "missing benchmark weight"):
                read_benchmark_weights(path)
