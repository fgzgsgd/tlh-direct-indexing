import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from tlh.actions import CaseReview, CorporateAction
from tlh.backtest import AccountEngine, Delisting, Session, rebalances, run, sessions_from_crsp
from tlh.execution import Order, floor_shares, proportional_commission
from tlh.ledger import Ledger
from tlh.market import StockDay
from tlh.positions import Target, write_weights
from tlh.tax import Scenario, TaxYear, settle_years


D = Decimal
day = date.fromisoformat


class BacktestTests(unittest.TestCase):
    def test_pre_tax_twr_compounds_daily_returns_and_keeps_a_gap_visible(self):
        prior, up, down, gap = map(day, ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"))
        engine = AccountEngine(D(100), start=prior)
        engine.result.ledger.buy(prior, 10, D(1), D(100), D(0))
        for d, price in ((up, 110), (down, 99)):
            quotes = {10: D(price)}
            engine.advance(d, Session(quotes, quotes, ()), (), {}, ())
        self.assertEqual(engine.result.pre_tax_twr(), D("-.01"))
        self.assertEqual(engine.result.pre_tax_twr(down, down), D("-.10"))
        # A newly received security has no usable price, so this day's return cannot be skipped.
        engine.result.ledger.receive(down, 20, D(1), D(0), down, "grant")
        engine.advance(gap, Session({}, {}, ()), (), {}, ())
        self.assertIsNone(engine.result.pre_tax_twr())
        self.assertEqual(engine.result.pre_tax_twr(up, down), D("-.01"))

    def test_halt_uses_last_close_until_the_stock_reopens(self):
        dates = list(map(day, ("2020-11-04", "2020-11-05", "2020-11-06", "2020-11-09")))
        signal, bought, halt, reopened = dates
        p = 76841  # BIIB
        initial, before, after = D("355.63"), D("328.90"), D("236.26")
        days = {d: Session({p: px}, {p: px}, []) for d, px in
                ((signal, initial), (bought, before), (reopened, after))}
        days[halt] = Session({}, {}, [])
        result = self.replay(dates, days, {d: {p: "1"} for d in dates[1:]},
                             initial=initial, pre_delisting=False)
        final = initial - before + after  # One BIIB share plus $26.73 of cash left after the initial build.
        self.assertEqual([v.stock_market_value for v in result.history], [before, before, after])
        self.assertEqual([v.nav for v in result.history], [initial, initial, final])
        self.assertEqual([v.daily_return for v in result.history], [0, 0, final / initial - 1])
        self.assertEqual([v.carried_close_dates for v in result.history], [{}, {p: bought}, {}])
        self.assertEqual([entry.kind for entry in result.ledger.entries], ["BUY"])

    def test_consecutive_missing_closes_do_not_freeze_other_holdings_or_use_future_prices(self):
        start, first, gap, second_gap, resumed = map(day, ("2024-01-02", "2024-01-03", "2024-01-04",
                                                            "2024-01-05", "2024-01-08"))
        engine = AccountEngine(D(100), start=start, fee=proportional_commission(D(0)))
        for p in (10, 20):
            engine.result.ledger.buy(start, p, D(1), D(50), D(0))
        for d, quotes in ((first, {10: D(50)}), (gap, {20: D(55)}),
                          (second_gap, {20: D(57)}), (resumed, {10: D(45), 20: D(58)})):
            engine.advance(d, Session(quotes, quotes, []), (), {}, ())
        first_mark, a, b, c = engine.result.history
        self.assertIsNone(first_mark.nav)  # Security 20 has no close yet; the purchase cost cannot stand in for a historical quote.
        self.assertEqual((a.nav, b.nav, c.nav), (105, 107, 103))
        self.assertEqual((a.carried_close_dates, b.carried_close_dates, c.carried_close_dates),
                         ({10: first}, {10: first}, {}))

    def test_same_prior_quote_can_size_and_fill_an_active_order_in_strategy_and_replay(self):
        dates = list(map(day, ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")))
        prior, signal, buy, sell = dates
        rows = [StockDay(10, d, None, None, px, trading_status=status) for d, px, status in
                ((prior, D(10), "A"), (signal, None, "H"), (buy, None, "A"), (sell, None, "A"))]
        markets = sessions_from_crsp(rows, tradable={d: {10} for d in dates}, actions_by_day={d: [] for d in dates})
        inputs = dict(sessions=dates, first_trade=buy, end=sell, days=markets, delistings=[],
                      delistings_known_through=sell, pre_delisting=False, initial_cash=D(100))

        def strategy(state):
            self.assertEqual((state.closes[10], state.carried_close_dates[10]), (10, prior))
            return Target({10: D(".5")} if state.trade_day == buy else {})

        result = run(strategy=strategy, **inputs)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            write_weights(path, result.targets)
            replay = run(weights_file=path, **inputs)
        self.assertEqual((replay.history, replay.ledger.entries), (result.history, result.ledger.entries))
        self.assertEqual([(e.kind, e.day, e.price_day, e.fee) for e in result.ledger.entries],
                         [("BUY", buy, prior, D(".025")), ("TRADE", sell, prior, D(".025"))])
        self.assertEqual((result.history[0].total_basis, result.history[0].nav, result.history[1].nav,
                          result.ledger.sales[0].gain), (D("50.025"), D("99.975"), D("99.95"), D("-.05")))
        holding = run(strategy=strategy, **(inputs | {"end": buy}))
        self.assertEqual(holding.liquidated_wealth(Scenario.EXTERNAL_GAINS).pre_tax, D("99.95"))
        self.assertIsNone(markets[buy].regular_close(10))
        self.assertNotIn(10, markets[buy].raw_closes)

    def test_a_confirmed_halt_blocks_sales_and_liquidation_but_allows_a_valuation(self):
        prior, halt = day("2020-11-05"), day("2020-11-06")
        p = 76841
        engine = AccountEngine(D(100), start=prior)
        lot = engine.result.ledger.buy(prior, p, D(1), D(100), D(0))
        order, = engine.prepare(prior, {p: D(100)}, halt, Target({}), set())
        missing = StockDay(p, halt, None, None, None, trading_status="A")
        market = sessions_from_crsp([missing], tradable={halt: {p}}, actions_by_day={halt: []})[halt]
        with self.assertRaisesRegex(ValueError, f"{halt} {p} TRADE: trading halted"):
            engine.advance(halt, market, [order], {}, ())
        self.assertEqual((engine.result.history, engine.result.ledger.holdings()), ([], {p: 1}))
        engine.advance(halt, market, (), {}, ())
        self.assertEqual((engine.last_nav, engine.result.history[0].carried_close_dates), (100, {p: prior}))
        with self.assertRaisesRegex(ValueError, f"{halt} {p} LIQUIDATION: trading halted"):
            engine.result.liquidated_wealth(Scenario.EXTERNAL_GAINS)
        self.assertEqual(engine.result.ledger.lots[lot.id].basis, 100)

    def test_missing_close_requires_active_status_and_never_uses_a_future_price(self):
        prior, trade, future = map(day, ("2024-01-02", "2024-01-03", "2024-01-04"))
        for status in ("", "X", "D", "H", "S"):
            with self.subTest(status=status):
                row = StockDay(10, trade, None, None, None, trading_status=status)
                market = sessions_from_crsp([row], tradable={trade: {10, 20}}, actions_by_day={trade: []})[trade]
                engine = AccountEngine(D(100), start=prior)
                orders = engine.prepare(prior, {10: D(10)}, trade, Target({10: D(".5")}), set())
                with self.assertRaisesRegex(ValueError, "trading halted|missing executable CLOSE"):
                    engine.advance(trade, market, orders, {}, ())
                self.assertNotIn(20, market.tradable)  # A security with no rows at all has no Active record either.
        later_quote = Session({10: D(90)}, {10: D(90)}, [])
        engine = AccountEngine(D(100), start=prior, basis_days={future: later_quote})
        with self.assertRaisesRegex(ValueError, "missing executable CLOSE"):
            engine.advance(trade, Session({}, {}, [], tradable={10}), [Order(10, D(1), {})], {}, ())

    def test_forced_exit_uses_the_prior_quote_and_values_the_other_halted_holding(self):
        prior, trade, last = map(day, ("2024-01-02", "2024-01-03", "2024-01-19"))
        engine = AccountEngine(D(100), start=prior, fee=proportional_commission(D(0)))
        prices = {10: D(40), 20: D(60)}
        for p, px in prices.items():
            engine.result.ledger.buy(prior, p, D(1), px, D(0))
        engine.prepare(prior, prices, trade, Target({10: D(".4"), 20: D(".6")}), set())
        market = Session({}, {}, [], tradable={10, 20}, halted=frozenset({20}))
        with self.assertRaisesRegex(ValueError, f"{trade} 20 PRE_DELISTING_EXIT: trading halted"):
            engine.advance(trade, market, (), {20: [Delisting(20, last, "merger")]}, ())
        engine.advance(trade, market, (), {10: [Delisting(10, last, "merger")]}, ())
        exit_record, = engine.result.exits
        self.assertEqual((exit_record.price_day, exit_record.execution_price, exit_record.weight_before_fill),
                         (prior, 40, D(".4")))
        self.assertEqual((engine.result.ledger.cash, engine.last_nav), (40, 100))

    def test_five_basis_points_on_each_side_and_on_liquidation(self):
        dates = list(map(day, ("2024-01-02", "2024-01-03", "2024-01-04", "2024-02-05")))
        signal, buy, sell, following = dates
        prices = {10: D(100)}
        days = dict.fromkeys(dates, Session(prices, prices, ()))

        def strategy(snapshot):
            return Target({10: D(1)} if snapshot.day == signal else {})

        args = dict(sessions=dates, first_trade=buy, days=days, delistings=[], initial_cash=D(10005),
                    delistings_known_through=following, strategy=strategy, rebalance_days=[buy, sell, following])
        holding = run(end=buy, **args)
        self.assertEqual((holding.ledger.cash, holding.ledger.total_basis()), (0, 10005))
        self.assertEqual(holding.liquidated_wealth(Scenario.EXTERNAL_GAINS).pre_tax, 9995)
        closed = run(end=sell, **args)
        self.assertEqual([e.fee for e in closed.ledger.entries], [5, 5])
        self.assertEqual((closed.ledger.cash, closed.ledger.sales[0].gain), (9995, -10))

    def replay(self, dates, markets, weights, *, initial=100, fee=lambda *_: D(0), flagged=None, **kwargs):
        if flagged is None:
            flagged, previous = [], None
            for d, row in weights.items():
                positive = {p: D(w) for p, w in row.items() if D(w)}
                if positive != previous:
                    flagged.append(d)
                previous = positive
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            path.write_text("date,permno,target_weight,rebalance\n" +
                            "".join(f"{d},{p},{w},{int(d in flagged)}\n"
                                    for d, row in weights.items() for p, w in row.items()))
            return run(sessions=dates, first_trade=dates[1], end=max(weights), days=markets, weights_file=path,
                       initial_cash=D(initial), fee=fee, size=floor_shares,
                       delistings=kwargs.pop("delistings", []), delistings_known_through=dates[-1], next_rebalance=dates[-1], **kwargs)

    def test_passive_split_preserves_entitlement_and_forced_delisting_exit(self):
        dates = list(map(day, ["2023-12-29", "2024-01-02", "2024-01-03",
                               "2024-02-01", "2024-02-02", "2024-03-04"]))
        signal, first, split, ex_date, second, following = dates
        rows = [StockDay(11, d, D("0.90"), D(99), D(p)) for d, p in zip(dates, [100, 100, 200, 196, 198])]
        events = {d: [] for d in dates[:-1]}
        events[split] = [CorporateAction("split-1", 11, "SPLIT", D("0.5"))]
        events[ex_date] = [CorporateAction("div-1", 11, "CASH_DIVIDEND", D(4))]
        markets = sessions_from_crsp(rows, tradable={d: {11} for d in events}, actions_by_day=events)
        weights = {d: {11: ".3"} for d in (first, split, ex_date)}
        weights[second] = {11: "0", 22: ".1"}
        result = self.replay(dates, markets, weights, initial=1000,
                             delistings=[Delisting(11, following, "merger"), Delisting(22, following, "delist")])
        self.assertEqual(set(result.plans), {signal, ex_date})
        self.assertEqual([v.nav for v in result.history], [1000, 1000, 1000, 1002])
        self.assertEqual(result.history[1].holdings, {11: 1})
        self.assertEqual(result.history[1].pending_entitlements, {})
        self.assertEqual(result.history[2].cash, 804)
        cil, sale = result.ledger.sales
        self.assertEqual((cil.shares, cil.proceeds, cil.basis, cil.gain, cil.reason),
                         (D(".5"), 100, 100, 0, "CASH_IN_LIEU"))
        self.assertEqual((sale.shares, sale.gain, sale.reason), (1, -2, "PRE_DELISTING_EXIT"))
        fills = [(e.kind, e.price, e.price_source) for e in result.ledger.entries if e.price is not None]
        self.assertEqual(fills, [("BUY", 100, "CLOSE"), ("CASH_IN_LIEU", 200, "CLOSE"),
                                 ("PRE_DELISTING_EXIT", 198, "CLOSE")])
        self.assertIn((second, 22, "BUY: blocked by pre-delisting convention"), result.exceptions)
        self.assertEqual(result.exits[0].weight_before_fill, D(198)/D(1002))
        wealth, = result.annual_wealth(Scenario.EXTERNAL_GAINS)
        self.assertEqual(wealth.after_tax, D("1001.864"))
        self.assertFalse(wealth.tax.final)

    def test_delisting_window_skips_weekend_and_includes_upper_boundary(self):
        dates = list(map(day, ["2024-07-31", "2024-08-01", "2024-08-30", "2024-09-03"]))
        window, = rebalances(dates, dates[1], dates[2])
        self.assertEqual(window.next_trade_day, dates[-1])
        self.assertEqual(set(window.upcoming_delistings([Delisting(11, dates[1], "a"),
                                                         Delisting(12, dates[-1], "b")])), {12})

    def test_same_targets_drift_then_trade_at_close_with_prior_close_quantity(self):
        dates = list(map(day, ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
        sig, first, hold, sell = dates
        closes = [10, 12, 20, 19]
        rows = [StockDay(10, d, D(".99"), D(9), D(p)) for d, p in zip(dates, closes)]
        markets = sessions_from_crsp(rows, tradable={d: {10} for d in dates}, actions_by_day={d: [] for d in dates})
        result = self.replay(dates, markets, {first: {10: ".5"}, hold: {10: ".500", 20: "0"},
                                             sell: {10: ".25"}}, fee=proportional_commission(D(".001")), pre_delisting=False)
        opening, holding, final = result.history
        self.assertEqual((opening.holdings, opening.cash, opening.nav, opening.total_basis),
                         ({10: 5}, D("39.94"), D("99.94"), D("60.06")))
        self.assertEqual((holding.holdings, holding.cash, holding.nav, holding.actual_weights[10]),
                         ({10: 5}, D("39.94"), D("139.94"), D(100) / D("139.94")))
        self.assertEqual(holding.daily_return, D("139.94") / D("99.94") - 1)
        self.assertEqual(set(result.plans), {sig, hold})
        sale, = result.ledger.sales
        self.assertEqual((sale.shares, sale.proceeds, sale.basis, sale.gain),
                         (4, D("75.924"), D("48.048"), D("27.876")))
        self.assertEqual((final.holdings, final.cash, final.nav, final.execution_source),
                         ({10: 1}, D("115.864"), D("134.864"), "CLOSE"))

    def test_cash_shortfall_rejects_entire_buy_batch(self):
        sig, first = day("2024-01-02"), day("2024-01-03")
        engine = AccountEngine(D(100), fee=proportional_commission(D(".001")), start=sig)
        orders = engine.prepare(sig, {10: D(10), 20: D(10)}, first, Target({10: D(".5"), 20: D(".5")}, True), set())
        with self.assertRaisesRegex(ValueError, r"100\.1.*100"):
            prices = {10: D(10), 20: D(10)}
            engine.advance(first, Session(prices, prices, []), orders, {}, ())
        self.assertEqual((engine.result.ledger.cash, engine.result.ledger.holdings()), (100, {}))

    def test_same_target_requires_an_explicit_instruction_to_rebalance(self):
        sig, first, next_day = dates = list(map(day, ["2024-01-02", "2024-01-03", "2024-01-04"]))
        markets = {sig: Session({}, {10: D(10)}, []),
                   first: Session({10: D(10)}, {10: D(20)}, [], "VWAP"),
                   next_day: Session({10: D(20)}, {10: D(20)}, [], "VWAP")}
        weights = {first: {10: ".50"}, next_day: {10: ".5"}}
        carry = self.replay(dates, markets, weights, pre_delisting=False)
        rebalance = self.replay(dates, markets, weights, flagged=[first, next_day], pre_delisting=False)
        self.assertEqual((carry.history[-1].holdings, rebalance.history[-1].holdings), ({10: 5}, {10: 3}))
        self.assertEqual((carry.history[-1].nav, rebalance.history[-1].nav), (150, 150))
        with self.assertRaisesRegex(ValueError, "initial construction"):
            self.replay(dates, markets, weights, flagged=[], pre_delisting=False)
        with self.assertRaisesRegex(ValueError, "changed on a day marked rebalance=0"):
            self.replay(dates, markets, {first: {10: ".5"}, next_day: {10: ".4"}},
                        flagged=[first], pre_delisting=False)

    def test_vwap_feed_does_not_substitute_open_close_or_quotes_for_a_missing_fill(self):
        sig, first = dates = [day("2024-01-02"), day("2024-01-03")]
        rows = [StockDay(10, d, None, D(8), D(10)) for d in dates]
        markets = sessions_from_crsp(rows, tradable={d: {10} for d in dates}, actions_by_day={d: [] for d in dates},
                                     regular_session_vwaps={})
        with self.assertRaisesRegex(ValueError, f"{first} 10 BUY: missing executable VWAP"):
            self.replay(dates, markets, {first: {10: "1"}}, pre_delisting=False)

    def test_lot_selection_locked_on_signal_date(self):
        dates = list(map(day, ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
        sig, a, b, c = dates
        markets = {d: Session({10: D(x)}, {10: D(x)}, []) for d, x in zip(dates, [10, 10, 20, 15])}
        weights = {a: {10: ".3"}, b: {10: ".5"}, c: {10: ".4"}}

        def select(snapshot, p, q):
            self.assertEqual((snapshot.day, snapshot.closes[p], q), (b, 20, 3))
            newer, older = sorted(snapshot.lots, key=lambda l: l.basis / l.shares, reverse=True)
            return {newer.id: D(2), older.id: D(1)}

        with self.assertRaisesRegex(ValueError, "specify lot relief"):
            self.replay(dates, markets, weights, pre_delisting=False)
        result = self.replay(dates, markets, weights, pre_delisting=False, lot_relief=select)
        self.assertEqual([(s.lot_id, s.gain) for s in result.ledger.sales], [(2, -10), (1, 5)])
        self.assertEqual((result.history[-1].holdings, result.history[-1].cash), ({10: 2}, 75))
        liquidation = result.liquidated_wealth(Scenario.EXTERNAL_GAINS, fee=proportional_commission(D(".001")))
        self.assertEqual((liquidation.pre_tax, result.ledger.cash), (D("104.97"), 75))
        self.assertEqual(result.liquidated_wealth(Scenario.EXTERNAL_GAINS).pre_tax, 105)  # This run uses zero commission.
        with self.assertRaisesRegex(ValueError, "liquidation price differs from the session CLOSE"):
            result.liquidated_wealth(Scenario.EXTERNAL_GAINS, prices={10: D(14)}, fee=proportional_commission(D(".001")))

    def test_no_rounding_per_lot_on_reverse_split(self):
        book = Ledger(D(60))
        for d in (day("2024-01-02"), day("2024-01-03")):
            book.buy(d, 10, D(1), D(20), D(0))
        book.split(day("2024-01-04"), 10, D(".5"), "r")
        self.assertEqual((book.holdings(), book.total_basis()), ({10: 1}, 40))
        self.assertEqual([l.shares for l in book.lots.values()], [D(".5"), D(".5")])

    def test_split_cancels_a_locked_buy_with_or_without_an_open_price(self):
        sig, t = dates = [day("2024-01-02"), day("2024-01-03")]
        action = CorporateAction("s", 10, "SPLIT", D(".5"))
        markets = {sig: Session({}, {10: D(10)}, []),
                   t: Session({10: D(21)}, {10: D(21)}, [action], raw_opens={10: D(18)})}
        for shares, opening in [(4, {10: D(18)}), (3, {})]:
            markets[t] = Session({10: D(21)}, {10: D(21)}, [action], raw_opens=opening)
            result = self.replay(dates, markets, {t: {10: D(shares)/10}}, pre_delisting=False)
            self.assertEqual(result.ledger.holdings(), {})
            self.assertEqual(result.ledger.cash, 100)
            self.assertEqual(result.exceptions, [(t, 10, f'SKIPPED_SPLIT_BUY {shares}/0 pre-split shares')])

    def test_basis_partial_sale_dividend_idempotence_and_roc(self):
        ledger = Ledger(D(1000))
        initial, later = day("2023-02-10"), day("2024-02-12")
        lot = ledger.buy(initial, 10, D(10), D(10), D(1))
        sale, = ledger.sell(later, 10, {lot.id: D(4)}, D(15), {lot.id: D("0.4")}, "TRADE")
        self.assertEqual((sale.gain, sale.is_long_term, lot.basis), (D("19.2"), True, D("60.6")))
        ledger.cash_distribution(later, 10, D(6), D(1), "div", "CASH_DIVIDEND")
        with self.assertRaises(ValueError):
            ledger.cash_distribution(later, 10, D(6), D(1), "div", "CASH_DIVIDEND")
        ledger.return_of_capital(later, 10, ledger.entitled_lots()[10], D(11), "roc")
        self.assertEqual(ledger.holdings(), {10: D(6)})
        self.assertEqual((lot.basis, ledger.sales[-1].gain), (0, D("5.4")))
        self.assertEqual(ledger.tax_year(2024, basis_verified=True, wash_reviewed=True).taxable_dividends, 6)

    def test_pending_economics_keep_position_and_invalidate_nav(self):
        dates = list(map(day, ["2024-01-02", "2024-01-03", "2024-01-04", "2024-02-05"]))
        sig, first, event_day, following = dates
        events = [CaseReview("ca-1", 10, "ECONOMIC_PENDING", "Merger consideration unknown", None)]
        markets = {sig: Session({}, {10: D(50)}, []), first: Session({10: D(50)}, {10: D(50)}, []),
                   event_day: Session({}, {}, events)}
        result = self.replay(dates, markets, {d: {10: ".5"} for d in (first, event_day)})
        self.assertIsNone(result.history[-1].nav)
        self.assertFalse(result.history[-1].basis_verified)
        self.assertEqual(result.ledger.holdings(), {10: D(1)})

    def test_missing_forced_exit_price_stops_run_and_cash_acquisition_still_settles(self):
        dates = list(map(day, ("2024-01-02", "2024-01-03", "2024-02-02", "2024-02-05",
                               "2024-02-07", "2024-02-08", "2024-02-09", "2024-03-07")))
        signal, first, next_signal, second, last_price, pending, payment, following = dates
        markets = {d: Session({10: D(10)}, {10: D(10)}, []) for d in (signal, first, next_signal)}
        markets[second] = Session({}, {10: D(9)}, [])
        markets[last_price] = Session({}, {10: D(8)}, [])
        markets[pending] = Session({}, {}, [])
        markets[payment] = Session({}, {}, [CorporateAction("terms-10", 10, "CASH_ACQUISITION", D("8.5"))])
        weights = {d: {10: "1"} for d in dates[1:-1]}
        events = [Delisting(10, last_price, "MER")]
        with self.assertRaisesRegex(ValueError, f"{second} 10 PRE_DELISTING_EXIT: missing executable CLOSE"):
            self.replay(dates, markets, weights, flagged=[first, second], delistings=events)
        result = self.replay(dates, markets, weights, flagged=[first], delistings=events, pre_delisting=False)
        self.assertEqual([v.nav for v in result.history[-3:]], [80, None, 85])
        self.assertIsNone(result.history[-1].daily_return)  # One close is missing, so a two-day change cannot be treated as a daily return.
        self.assertEqual(result.actions[-2][1].status, "ECONOMIC_PENDING")
        self.assertEqual(result.ledger.sales[0].gain, -15)
        self.assertEqual(result.ledger.holdings(), {})

    def test_unclassified_cash_preserves_nav_and_freezes_affected_basis(self):
        ledger = Ledger(D(100))
        first, ex_date = day("2024-01-03"), day("2024-01-04")
        lot = ledger.buy(first, 10, D(1), D(100), D(0))
        entitled = ledger.entitled_lots()
        cash = CorporateAction("dis-1", 10, "UNCLASSIFIED_CASH", D(4))
        result = cash.post(ex_date, ledger, entitled)
        self.assertEqual((result.status, result.affected_lots, lot.basis), ("BASIS_PENDING", (lot.id,), None))
        self.assertEqual((ledger.cash, ledger.nav({10: D(96)})), (4, 100))
        unresolved = CorporateAction("unknown-1", 10, "MIXED_MERGER", "Unconfirmed terms")
        self.assertEqual(unresolved.post(ex_date, ledger, entitled).status, "ECONOMIC_PENDING")

    def test_external_tax_account_carryforward_and_year_end(self):
        one = TaxYear(2024, D(-50), D(10), D(4), day("2024-12-31"), True)
        two = TaxYear(2025, D(10), D(40), D(0), None, True)
        ext, = settle_years([one], Scenario.EXTERNAL_GAINS, day("2025-01-29"))
        self.assertEqual(ext.net_tax, D("-17.068"))
        self.assertFalse(ext.final)
        closed, = settle_years([one], Scenario.EXTERNAL_GAINS, day("2025-01-30"))
        self.assertTrue(closed.final)
        self.assertEqual(closed.after_tax_wealth(D(1000)), D("1017.068"))
        a, b = settle_years([one, two], Scenario.LIMITED_USE, day("2025-12-31"))
        self.assertEqual((a.st_carryforward, a.net_tax), (40, D("0.952")))
        self.assertEqual((b.st_carryforward, b.net_tax), (0, D("2.38")))

    def test_tax_year_closes_on_the_last_session_and_preserves_the_wash_deadline(self):
        def report(year, as_of, loss=None, verified=True):
            income = TaxYear(year, D(0), D(0), D(100), day(loss) if loss else None, verified)
            return next(settle_years([income], Scenario.EXTERNAL_GAINS, day(as_of)))

        for year, last in [(2021, "2021-12-31"), (2022, "2022-12-30"), (2023, "2023-12-29")]:
            with self.subTest(year=year):
                self.assertTrue(report(year, last).final)
        self.assertFalse(report(2023, "2023-12-28").final)
        self.assertFalse(report(2026, "2026-09-30").final)  # The last day of local market data is still mid-year.
        self.assertFalse(report(2023, "2023-12-29", verified=False).final)
        self.assertFalse(report(2023, "2023-12-29", loss="2023-11-30").final)  # Day 30 is a Saturday.
        self.assertTrue(report(2023, "2024-01-02", loss="2023-11-30").final)
        self.assertFalse(report(2023, "2024-01-26", loss="2023-12-29").final)
        self.assertTrue(report(2023, "2024-01-29", loss="2023-12-29").final)

    def test_year_end_hold_and_liquidation_use_the_same_calendar(self):
        signal, bought, year_end, following = dates = list(map(day, ("2023-12-27", "2023-12-28",
                                                                      "2023-12-29", "2024-01-02")))
        quotes = {10: D(10)}
        markets = {d: Session(quotes, quotes, ()) for d in (signal, bought)}
        ex_price = {10: D(9)}
        markets[year_end] = Session(ex_price, ex_price, [CorporateAction("div", 10, "CASH_DIVIDEND", D(1))])
        result = run(sessions=dates, first_trade=bought, end=year_end, days=markets, initial_cash=D(2000),
                     strategy=lambda _: Target({10: D(".5")}), rebalance_days=[bought, following],
                     delistings=[], delistings_known_through=following, fee=proportional_commission(D(0)))
        for scenario in Scenario:
            holding, = result.annual_wealth(scenario)
            self.assertEqual((holding.as_of, holding.tax.net_tax, holding.after_tax, holding.tax.final),
                             (year_end, D("23.80"), D("1976.20"), True))
            # Assume liquidation creates a $100 loss on 12/29. Knowing next year's first trading day is not the same as having observed next year's purchases.
            self.assertFalse(result.liquidated_wealth(scenario).tax.final)
        self.assertEqual(result.ledger.sales, [])
