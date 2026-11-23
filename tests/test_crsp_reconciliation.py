"""从项目的真实 CIZ 导出逐表复核持仓、现金股息和拆股。"""

import csv
import unittest
from dataclasses import replace
from datetime import date
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.prepare_daily_prices import merge_daily
from tlh.actions import PropertyDividend, SpinOff, SpinOffTerms
from tlh.backtest import AccountEngine, Session, run, sessions_from_crsp
from tlh.crsp_events import AIMCO_PAYMENT, AIMCO_REVERSE_SPLIT, ALCOA_SPIN_OFF, read_distributions
from tlh.execution import Order, proportional_commission
from tlh.ledger import Ledger
from tlh.market import StockDay, nyse_trading_days, read_crsp_daily, read_delistings, read_membership
from tlh.positions import write_weights
from tlh.strategy import Target
from tlh.tax import Scenario


DATA = Path(__file__).resolve().parents[1] / "data" / "crsp"
D = Decimal


class CRSPReconciliation(unittest.TestCase):
    def test_grail_basis_matches_issuer_and_settles_across_year_end(self):
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        ex, parent, child = date(2024, 6, 25), 88446, 25143
        before = date(2024, 6, 24)
        engine = AccountEngine(D(3000), start=before)
        book = engine.result.ledger
        book.buy(before, parent, D(60), D(50), D(0))
        prices = {parent: D("109.84"), child: D(17)}
        engine.advance(ex, Session(prices, prices, events[ex]), (), {}, ())
        self.assertEqual(book.holdings(), {parent: 60, child: 10})
        self.assertEqual({lot.permno: round(lot.basis, 2) for lot in book.lots.values()},
                         {parent: D("2924.56"), child: D("75.44")})
        self.assertEqual((book.total_basis(), book.cash, book.sales), (3000, 0, []))
        for day in (date(2024, 12, 31), date(2025, 1, 2)):
            engine.advance(day, Session(prices, prices, []), (), {}, ())
        self.assertEqual(engine.result.history[-1].nav, D("6760.40"))

    def test_foreign_stock_grant_posts_all_entitled_shares_as_one_cash_dividend(self):
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        before, ex, parent = date(2021, 2, 12), date(2021, 2, 16), 16538
        engine = AccountEngine(D(260), start=before)
        book = engine.result.ledger
        book.buy(before, parent, D(26), D(10), D(0))
        engine.advance(ex, Session({parent: D(8)}, {parent: D(8)}, events[ex]), (), {}, ())
        self.assertEqual((book.holdings(), book.total_basis(), book.sales), ({parent: 26}, 260, []))
        self.assertEqual(book.cash, D("79.948024"))
        self.assertEqual(book.tax_year(2021, basis_verified=True).taxable_dividends, book.cash)
        self.assertEqual(engine.result.annual_wealth()[0].tax.net_tax, book.cash * D(".238"))

    def test_foreign_cash_dividends_post_gross_usd_on_ex_date(self):
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv",
                                    DATA / "spy_supplemental_distributions.csv")
        calendar = nyse_trading_days()
        # CRSP 的前收盘、ex-date 收盘和每股美元股息。
        cases = (
            (76932, date(2017, 2, 13), "43.16", "43.18", "0.445440"),
            (76932, date(2017, 5, 11), "41.34", "40.03", "0.445420"),
            (76932, date(2017, 8, 11), "39.90", "39.47", "0.479550"),
            (20745, date(2021, 10, 28), "150.87", "151.72", "1.697780"),
            (20745, date(2022, 10, 27), "112.56", "109.48", "2.069490"),
        )
        for permno, ex, prior_close, close, per_share in cases:
            with self.subTest(permno=permno, ex_date=ex):
                before = calendar[calendar.index(ex) - 1]
                basis, gross = 100 * D(prior_close), 100 * D(per_share)
                engine = AccountEngine(basis, start=before)
                book = engine.result.ledger
                book.buy(before, permno, D(100), D(prior_close), D(0))
                prices = {permno: D(close)}
                engine.advance(ex, Session(prices, prices, events[ex]), (), {}, ())
                mark, = engine.result.history
                self.assertEqual((mark.cash, mark.total_basis, mark.nav), (gross, basis, 100 * D(close) + gross))
                self.assertEqual(book.tax_year(ex.year, basis_verified=True).taxable_dividends, gross)
                for scenario in Scenario:
                    report, = engine.result.annual_wealth(scenario)
                    self.assertEqual(report.tax.net_tax, gross * D(".238"))

    def test_price_merge_discards_only_full_row_duplicates(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, supplement, output = (folder / name for name in ("base.csv", "extra.csv", "prices.csv.gz"))
            header = "PERMNO,DlyCalDt,DlyClose,DlyVol\n"
            quote = "7,2024-03-14,100,230\n"
            source.write_text(header + quote * 2)
            supplement.write_text(header + "5,2024-03-14,75,400\n")
            self.assertEqual(merge_daily([source, supplement], output), (2, 1))
            original = output.read_bytes()
            rows = list(read_crsp_daily(output, require_prices=True))
            self.assertEqual([(r.permno, r.raw_close) for r in rows], [(5, 75), (7, 100)])
            supplement.write_text(header + quote.replace(",230", ",231"))
            with self.assertRaisesRegex(ValueError, "Conflicting CRSP daily rows"):
                merge_daily([source, supplement], output)
            self.assertEqual(output.read_bytes(), original)

    def test_gilead_cash_dividend_keeps_basis_and_posts_tax_once(self):
        p, before, ex = 77274, date(2024, 3, 13), date(2024, 3, 14)
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        batches = (batch for day, batch in events.items() if date(2015, 9, 14) <= day <= ex)
        kinds = [event.kind for batch in batches for event in batch if event.permno == p]
        self.assertEqual(kinds, ["CASH_DIVIDEND"] * 35)
        engine = AccountEngine(D(7594), start=before)
        engine.result.ledger.buy(before, p, D(100), D("75.94"), D(0))
        closes = {p: D("74.21")}
        engine.advance(ex, Session(closes, closes, events[ex]), (), {}, ())
        mark, = engine.result.history
        self.assertEqual((mark.cash, mark.total_basis, mark.nav, mark.basis_verified), (77, 7594, 7498, True))
        for scenario in Scenario:
            report, = engine.result.annual_wealth(scenario)
            self.assertEqual((report.tax.net_tax, report.after_tax), (D("18.326"), D("7479.674")))

    def test_sl_green_election_is_modeled_as_two_cash_dividends(self):
        p, before, ex = 85265, date(2020, 12, 11), date(2020, 12, 14)
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")[ex]
        engine = AccountEngine(D(5500), start=before)
        book = engine.result.ledger
        lot = book.buy(before, p, D(100), D(55), D(0))
        prices = {p: D("58.35")}
        engine.advance(ex, Session(prices, prices, events), (), {}, ())
        self.assertEqual([e.cash_change for e in book.entries if e.permno == p and e.kind == "CASH_DIVIDEND"],
                         [D("30.33"), D("169.67")])
        self.assertEqual((book.cash, book.holdings(), list(book.lots), lot.basis), (200, {p: 100}, [lot.id], 5500))
        self.assertEqual(engine.result.annual_wealth()[0].tax.net_tax, D("47.60"))

    def test_carrier_and_otis_both_allocate_from_each_original_utc_lot(self):
        day, parent, carrier, otis = date(2020, 4, 3), 17830, 19285, 19286
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        actions = [e for e in events[day] if e.permno == parent]
        early, late = date(2018, 1, 2), date(2020, 2, 3)
        engine = AccountEngine(D(1000), start=late, fee=proportional_commission(D(0)))
        book = engine.result.ledger
        book.buy(early, parent, D(4), D(100), D(0))
        book.buy(late, parent, D(6), D(100), D(0))
        prices = {parent: D(55), carrier: D(19), otis: D(52)}
        engine.advance(day, Session(prices, prices, actions), (), {}, ())
        self.assertEqual(book.holdings(), {parent: 10, carrier: 10, otis: 5})
        for acquired, original in [(early, D(400)), (late, D(600))]:
            basis = {lot.permno: lot.basis for lot in book.lots.values() if lot.holding_start == acquired}
            self.assertEqual(basis, {parent: original * D(".55"), carrier: original * D(".19"),
                                     otis: original * D(".26")})
        self.assertEqual((book.sales, book.cash), ([], 0))

    def test_corteva_entitlement_then_exact_dupont_reverse_split(self):
        day, before, parent, child = date(2019, 6, 3), date(2019, 5, 31), 16851, 18592
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        spin, split = [e for e in events[day] if e.permno == parent]
        self.assertEqual((spin.terms.effective_ratio, split.value), (Fraction(1, 3), Fraction(1, 3)))
        engine = AccountEngine(D(300), start=before, fee=proportional_commission(D(0)))
        book = engine.result.ledger
        book.buy(before, parent, D(3), D(100), D(0))
        prices = {parent: D(75), child: D(25)}
        engine.advance(day, Session(prices, prices, [spin, split]), (), {}, ())
        self.assertEqual(book.holdings(), {parent: 1, child: 1})
        self.assertEqual({lot.permno: lot.basis for lot in book.lots.values()},
                         {parent: D(225), child: D(75)})
        self.assertEqual((book.cash, book.sales, engine.result.exceptions), (0, [], []))

    def test_joint_spin_off_requires_a_regular_close_for_every_leg(self):
        day, parent, carrier, otis = date(2020, 4, 3), 17830, 19285, 19286
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        legs = [event for event in events[day] if event.permno == parent]
        prices = {parent: D(55), carrier: D(19), otis: D(52)}
        before = date(2020, 4, 2)
        engine = AccountEngine(D(200), start=before, fee=proportional_commission(D(0)))
        book = engine.result.ledger
        book.buy(before, parent, D(2), D(100), D(0))
        with self.assertRaisesRegex(ValueError, "incomplete joint spin-off.*19286"):
            engine.advance(day, Session(prices, prices, legs[:1]), (), {}, ())
        # Otis 仅有估值报价。两个子股都按条款到账，全部原始 basis 等待共同收盘行情。
        active = {parent: prices[parent], carrier: prices[carrier]}
        engine.advance(day, Session(active, prices, legs), (), {}, ())
        self.assertEqual(book.holdings(), {parent: 2, carrier: 2, otis: 1})
        self.assertTrue(all(lot.basis is None for lot in book.lots.values()))
        self.assertEqual([(r.status, r.close_basis_fraction) for _, r in engine.result.actions],
                         [("BASIS_PENDING", None), ("BASIS_PENDING", None)])

    def test_viatris_final_ratio_preserves_the_actual_cil(self):
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        day, before = date(2020, 11, 17), date(2020, 11, 16)
        spin, = [e for e in events[day] if e.reference == "CRSP/StkDistributions/21936/2020-11-17/1"]
        p, child = spin.permno, spin.terms.child
        engine = AccountEngine(D(200000), start=before, fee=proportional_commission(D(0)))
        book = engine.result.ledger
        book.buy(before, p, D(3393), D(40), D(0))
        prices = {p: D(40), child: D(15)}
        engine.advance(day, Session(prices, prices, [spin]), (), {}, ())
        sale, = book.sales
        self.assertEqual(book.holdings()[child], 420)
        self.assertEqual((sale.reason, sale.exact_shares, sale.proceeds),
                         ("CASH_IN_LIEU", Fraction("0.9996564657"), D("14.9948469855")))
        self.assertEqual(book.total_basis() + sale.basis, D(3393 * 40))

    def test_catalog_distinguishes_share_entitlement_and_us_tax_treatment(self):
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        for day, parent, child, shares, expected_status in [
            (date(2017, 1, 30), 27043, 16543, 2, "BASIS_PENDING"),  # 五股 Varian 应收两股 Varex。
            (date(2017, 9, 1), 15707, 16850, D(".68663055"), "BASIS_PENDING"),  # 直接 post 未提供分拆行情。
        ]:
            with self.subTest(parent=parent):
                book = Ledger(D(500))
                book.buy(date(2016, 1, 4), parent, D(5), D(100), D(0))
                action, = [e for e in events[day] if e.permno == parent and e.spin_off_child == child]
                result = action.post(day, book, book.entitled_lots())
                self.assertEqual((result.status, book.holdings().get(child, 0)), (expected_status, shares))
                if expected_status == "BASIS_PENDING":
                    self.assertTrue(all(lot.basis is None for lot in book.lots.values()))
        sandisk, = [e for e in events[date(2025, 2, 24)] if e.permno == 66384]
        self.assertEqual((sandisk.terms.effective_ratio, sandisk.terms.child_basis_fraction),
                         (Fraction(1, 3), D(".248390")))

    def test_hpe_exchange_recognizes_each_gain_and_preserves_each_loss_in_basis(self):
        day = date(2017, 9, 1)
        p, child = 15707, 16850
        event, = [e for e in read_distributions(DATA / "distributions_2015_to_2026_09.csv")[day] if e.permno == p]
        engine = AccountEngine(D(2000), start=date(2017, 8, 31))
        book = engine.result.ledger
        # 两个来源各 10 股，一批有利得，一批有亏损。税务 FMV 与 CLOSE 独立。
        low = book.buy(date(2016, 2, 2), p, D(10), D(10), D(0))
        high = book.buy(date(2017, 3, 2), p, D(10), D(40), D(0))
        closes = {p: D("14.31"), child: D("28.81")}
        engine.advance(day, Session(closes, closes, [event]), (), {}, ())
        report = engine.result.actions[0][1]
        ratio = Fraction("0.13732611")
        weight = ratio * Fraction("28.81") / (Fraction("14.31") + ratio * Fraction("28.81"))
        self.assertEqual((report.status, report.close_basis_fraction), ("RESOLVED", weight))
        gains = [s for s in book.sales if s.reason == "SECTION_367_GAIN"]
        gain, = gains
        self.assertEqual((gain.lot_id, gain.is_long_term, gain.proceeds),
                         (low.id, True, D("1.3732611") * D("29.34")))
        self.assertAlmostEqual(gain.basis, 100 - low.basis, places=20)
        self.assertGreater(400 - high.basis, gain.proceeds)
        cil = [s for s in book.sales if s.reason == "CASH_IN_LIEU"]
        self.assertEqual({s.holding_start for s in cil}, {low.holding_start, high.holding_start})
        self.assertEqual((book.holdings()[child], book.cash - D(1500)),
                         (2, D(".7465222") * D("28.81")))
        # 留仓与零股合计的 basis 只增加该批实际认列的 gain。
        self.assertAlmostEqual(book.total_basis() + sum(s.basis for s in cil), 500 + gain.gain, places=20)

    def test_property_dividend_and_reclassified_organon_use_their_own_basis_rules(self):
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        ex = date(2019, 2, 26)
        ge, wab = 12060, 81677
        event, = [e for e in events[ex] if e.permno == ge]
        self.assertIsInstance(event, PropertyDividend)
        engine = AccountEngine(D(2000), start=date(2019, 2, 25))
        book = engine.result.ledger
        parent = book.buy(date(2018, 1, 2), ge, D(200), D(10), D(0))
        quotes = {ge: D("10"), wab: D("73.96")}
        engine.advance(ex, Session(quotes, quotes, [event]), (), {}, ())
        cil, = book.sales
        child, = [l for l in book.lots.values() if l.permno == wab]
        self.assertEqual((parent.basis, child.shares, child.basis, child.holding_start), (2000, 1, D("78.06"), ex))
        self.assertEqual((cil.shares, cil.basis, book.cash), (D(".0742"), D(".0742") * D("78.06"), D(".0742") * quotes[wab]))
        self.assertEqual(book.tax_year(2019, basis_verified=True).taxable_dividends, D("1.0742") * D("78.06"))

        ex = date(2021, 6, 3)
        mrk, ogn = 22752, 21124
        action, = [e for e in events[ex] if e.permno == mrk]
        self.assertIsInstance(action, SpinOff)
        engine = AccountEngine(D(1000), start=date(2021, 6, 2))
        book = engine.result.ledger
        book.buy(date(2020, 1, 2), mrk, D(10), D(100), D(0))
        quotes = {mrk: D("73.91"), ogn: D("37")}
        engine.advance(ex, Session(quotes, quotes, [action]), (), {}, ())
        report = engine.result.actions[0][1]
        self.assertEqual((report.status, report.close_basis_fraction), ("RESOLVED", Fraction(370, 7761)))
        self.assertEqual((book.holdings(), book.cash, book.total_basis()), ({mrk: 10, ogn: 1}, 0, 1000))

    def test_taxable_stock_dividends_leave_parent_basis_and_cash_intact(self):
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        for ex, parent, child, count, child_basis, taxable in [
            (date(2016, 10, 31), 45356, 16342, 10, "45.51", "45.51"),
            (date(2016, 11, 1), 67598, 16346, 5, "30.85", "30.85"),
            (date(2020, 12, 15), 80711, 20191, 1, "38.0711", "38.0711"),
            (date(2021, 11, 15), 81045, 22264, 10, "20.6272", "20.597123"),
        ]:
            event, = [e for e in events[ex] if isinstance(e, PropertyDividend) and e.permno == parent]
            engine = AccountEngine(D(count), start=date(2016, 1, 4))
            book = engine.result.ledger
            source = book.buy(date(2016, 1, 4), parent, D(count), D(1), D(0))
            quotes = {parent: D(30), child: D(20)}
            engine.advance(ex, Session(quotes, quotes, [event]), (), {}, ())
            grant, = [l for l in book.lots.values() if l.permno == child]
            self.assertEqual((grant.shares, grant.basis, grant.holding_start, source.basis, book.cash),
                             (1, D(child_basis), ex, count, 0))
            self.assertEqual(engine.result.actions[0][1].status, "RESOLVED")
            tax = book.tax_year(ex.year, basis_verified=True)
            self.assertEqual((tax.taxable_dividends, tax.long_term, tax.short_term), (D(taxable), 0, 0))
            self.assertEqual(engine.result.annual_wealth()[0].tax.net_tax, D(taxable) * D(".238"))

    def test_lennar_stock_dividend_only_uses_overnight_class_a_and_b(self):
        prior, ex = date(2017, 11, 8), date(2017, 11, 9)
        a, b = 52708, 89731
        closes = {a: D("58.93"), b: D("49.88")}
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")[ex]
        actions = [event for event in events if event.permno in (a, b)]
        fraction = Fraction(closes[b]) / Fraction(50 * closes[a] + closes[b])
        for old_b in (0, 50):
            capital = D(50 + old_b) * 100
            engine = AccountEngine(capital, start=prior)
            book = engine.result.ledger
            book.buy(prior, a, D(50), D(100), D(0))
            if old_b:
                book.buy(prior, b, D(old_b), D(100), D(0))
            engine.advance(ex, Session(closes, closes, actions), (), {}, ())
            self.assertEqual(book.holdings(), {a: 50, b: 1 + old_b + old_b // 50})
            self.assertEqual((book.sales, book.cash, book.total_basis()), ([], 0, capital))
            self.assertEqual(engine.result.actions[0][1].close_basis_fraction, fraction)

    def test_stock_settled_grant_keeps_fixed_nav_until_first_regular_close_and_cil(self):
        ex, interim, first = date(2025, 1, 21), date(2025, 2, 5), date(2025, 2, 7)
        parent, child = 52708, 26039
        event, = [e for e in read_distributions(DATA / "distributions_2015_to_2026_09.csv")[ex]
                  if isinstance(e, PropertyDividend) and e.permno == parent]
        event = replace(event, cash_at_fmv=False)  # 同一组发行条款用于复核实物交付选项。
        for count in (1, 2, 3):
            signal = date(2025, 1, 17)
            engine = AccountEngine(D(100 * count), start=signal)
            book = engine.result.ledger
            book.buy(signal, parent, D(count), D(100), D(0))
            quotes = {parent: D("88.255"), child: D(90)}  # 模拟 when-issued 价，应忽略。
            for d, events in [(ex, [event]), (interim, [])]:
                engine.advance(d, Session(quotes, quotes, events), (), {}, ())
                mark = engine.result.history[-1]
                self.assertEqual((mark.nav, mark.cash, mark.daily_return, mark.prelisting_marks),
                                 (count * 100, 0, 0, {child: D("23.49")}))
                self.assertEqual(mark.pending_entitlements, {child: D(".5")} if count % 2 else {})
                self.assertEqual((mark.holdings.get(child, 0), book.total_basis()),
                                 (count // 2, count * D("111.745")))
                self.assertEqual(engine.close_snapshot(d, quotes).closes[child], D("23.49"))
                self.assertNotIn(child, engine.result.last_prices)
            with self.assertRaisesRegex(ValueError, "precede regular-way trading"):
                engine.advance(date(2025, 2, 6), Session(quotes, quotes, []), [Order(child, 1, {})], {}, ())
            if count % 2:
                with self.assertRaisesRegex(ValueError, "missing first regular CLOSE"):
                    engine.advance(first, Session({parent: D(90)}, {parent: D(90)}, []), (), {}, ())
                self.assertEqual((book.prelistings[child].pending_shares, book.cash), (D(".5"), 0))
            actual = {parent: D("88.255"), child: D(24)}  # 手算上市日 CLOSE。
            engine.advance(first, Session(actual, actual, []), (), {}, ())
            mark = engine.result.history[-1]
            self.assertEqual((mark.prelisting_marks, mark.pending_entitlements, mark.cash), ({}, {}, 12 * (count % 2)))
            self.assertEqual(mark.nav, count * (D("88.255") + 12))
            self.assertEqual([s.taxable_gain for s in book.sales], [D(".255")] if count % 2 else [])
            self.assertEqual(book.tax_year(2025, basis_verified=True).taxable_dividends, count * D("11.745"))

        # 上市日才开始的账户没有获配资格，行情预热同样不能拿 when-issued 价成交。
        wi = {child: D(90)}
        days = {ex: Session(wi, wi, [event]), interim: Session(wi, wi, []),
                first: Session({}, {}, [], tradable={child})}
        with self.assertRaisesRegex(ValueError, "missing executable CLOSE"):
            run(sessions=list(days), first_trade=first, end=first, days=days, delistings=(),
                delistings_known_through=first, pre_delisting=False, initial_cash=D(100),
                strategy=lambda snapshot: Target({child: D(".5")}), rebalance_days=[first])

    def test_millrose_is_a_cash_dividend_on_ex_date_for_both_lennar_classes(self):
        prior, ex, listed = date(2025, 1, 17), date(2025, 1, 21), date(2025, 2, 7)
        a, b = 52708, 89731
        actions = [e for e in read_distributions(DATA / "distributions_2015_to_2026_09.csv")[ex]
                   if e.permno in (a, b)]
        prices = {a: D("133.14"), b: D("126.07")}
        for qa, qb in [(15, 0), (0, 1), (1, 1), (2, 1), (0, 0)]:
            with self.subTest(a=qa, b=qb):
                engine = AccountEngine(D(1500), start=prior)
                book = engine.result.ledger
                positions = {p: q for p, q in ((a, qa), (b, qb)) if q}
                for p, q in positions.items():
                    book.buy(prior, p, D(q), D(100), D(0))
                old_cash, old_basis = book.cash, book.total_basis()
                gross = D(qa + qb) * D("11.745")
                for d, events in [(ex, actions), (date(2025, 2, 3), []), (listed, [])]:
                    engine.advance(d, Session(prices, prices, events), (), {}, ())
                    mark = engine.result.history[-1]
                    self.assertEqual((mark.cash, mark.holdings, mark.total_basis), (old_cash + gross, positions, old_basis))
                    self.assertEqual(mark.nav, mark.cash + sum(q * prices[p] for p, q in positions.items()))
                    self.assertEqual((mark.prelisting_marks, mark.pending_entitlements, book.sales), ({}, {}, []))
                payments = [(e.day, e.permno, e.cash_change) for e in book.entries if e.kind == "CASH_DIVIDEND"]
                self.assertEqual(payments, [(ex, a, qa * D("11.745")), (ex, b, qb * D("11.745"))])
                self.assertFalse(book.distribution_income)
                for scenario in Scenario:
                    self.assertEqual(engine.result.annual_wealth(scenario)[0].tax.net_tax, gross * D(".238"))

    def test_millrose_cash_can_fund_an_ex_date_buy(self):
        prior, ex, parent = date(2025, 1, 17), date(2025, 1, 21), 52708
        event, = [e for e in read_distributions(DATA / "distributions_2015_to_2026_09.csv")[ex] if e.permno == parent]
        engine = AccountEngine(D(1500), start=prior)
        book = engine.result.ledger
        book.buy(prior, parent, D(15), D(100), D(0))
        prices = {parent: D("133.14")}
        engine.advance(ex, Session(prices, prices, [event]), [Order(parent, 1, {})], {}, ())
        # 前收盘的 15 股领取 $176.175，当天买入第 16 股花 $133.14 加 5 bp。
        self.assertEqual((book.cash, book.holdings(), book.tax_year(2025, basis_verified=True).taxable_dividends),
                         (D("42.96843"), {parent: 16}, D("176.175")))
        mrp = event.spin_off_child
        with self.assertRaisesRegex(ValueError, "precede regular-way trading"):
            engine.advance(date(2025, 2, 3), Session({mrp: D(10)}, {mrp: D(10)}, []), [Order(mrp, 1, {})], {}, ())

    def test_varex_basis_uses_raw_closes_independently_of_issuer_example_and_vwap(self):
        day, before = date(2017, 1, 30), date(2017, 1, 27)
        parent, child = 27043, 16543
        spin, = [e for e in read_distributions(DATA / "distributions_2015_to_2026_09.csv")[day]
                 if e.permno == parent]
        self.assertIsNone(spin.terms.child_basis_fraction)
        # 手算行情。5 股母股分得 2 股子股，子股市值 100，母股市值 400。
        rows = [StockDay(parent, day, None, None, D(80)), StockDay(child, day, None, None, D(50))]
        market = sessions_from_crsp(rows, tradable={day: {parent, child}}, actions_by_day={day: [spin]},
                                    regular_session_vwaps={day: {child: D(49)}})[day]
        engine = AccountEngine(D(500), start=before, fee=proportional_commission(D(0)), price_source="VWAP")
        book = engine.result.ledger
        book.buy(before, parent, D(5), D(100), D(0))
        engine.advance(day, market, (), {}, ())
        self.assertEqual({lot.permno: lot.basis for lot in book.lots.values()}, {parent: 400, child: 100})
        self.assertEqual(engine.result.actions[0][1].close_basis_fraction, Fraction(1, 5))

    def test_aimco_payout_is_taxed_once_and_only_a_purchase_replaces_a_loss(self):
        before, ex, paid, next_day = map(date.fromisoformat, ("2020-11-02", "2020-11-03", "2020-11-30", "2020-12-01"))
        p, other = 80711, 10001
        engine = AccountEngine(D(10000), start=before, fee=proportional_commission(D(0)))
        book = engine.result.ledger
        acquired = date(2020, 9, 21)
        book.buy(acquired, p, D(100), D(30), D(0))
        loss_lot = book.buy(acquired, p, D(1), D(40), D(0))
        loss, = book.sell(before, p, {loss_lot.id: D(1)}, D(30), {loss_lot.id: D(0)}, "TRADE")
        # 取计股价作手算 CLOSE，使 CIL 损益为零，单独检验股息税与原有亏损。
        px = {p: D("30.98"), other: D(10)}
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        for d in (ex, paid):
            actions = [event for event in events.get(d, ()) if event.permno == p]
            engine.advance(d, Session(px, px, actions), [Order(other, 1, {})] if d == paid else (), {}, ())
        stock, = [lot for lot in book.lots.values() if lot.permno == p and lot.origin is None]
        self.assertEqual((stock.shares, stock.basis, stock.purchase_price), (23, D("712.54"), D("30.98")))
        self.assertEqual((book.wash_matches, loss.disallowed_loss), ([], 0))
        cil, = [s for s in book.sales if s.reason == "CASH_IN_LIEU"]
        self.assertEqual((cil.gain, cil.basis, cil.proceeds), (0, D("4.536"), D("4.536")))

        engine.advance(next_day, Session(px, px, []), [Order(p, 1, {})], {}, ())
        match, = book.wash_matches
        self.assertEqual((match.permno, match.acquired, match.disallowed_loss), (p, next_day, 10))
        year = book.tax_year(2020, basis_verified=True)
        self.assertEqual((year.taxable_dividends, year.short_term, year.long_term, year.basis_verified), (820, 0, 0, True))
        for scenario in Scenario:
            report, = engine.result.annual_wealth(scenario)
            self.assertEqual((report.tax.net_tax, report.after_tax), (D("195.16"), report.pre_tax - D("195.16")))

    def test_aimco_default_election_and_reverse_split_use_issuer_terms(self):
        cash, stock_value, shares = AIMCO_PAYMENT.entitlements(D(100))
        self.assertEqual((cash, stock_value, shares), (D("102.924"), D("717.076"), Fraction(179269, 7745)))
        self.assertEqual([D(820) * f for f in AIMCO_PAYMENT.tax_fractions],
                         list(map(D, ("81.538750", "15.552120", "495.413742", "227.495388"))))
        ex = date(2020, 12, 1)
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        split, = [event for event in events[ex] if event.reference == AIMCO_REVERSE_SPLIT]
        self.assertEqual((split.value, split.unverified_ratio), (Fraction(100000, 123821), False))

    def test_aimco_immediate_dividend_and_locked_sale_replay_with_crsp_closes(self):
        signal, first, ex, sale, paid, after, next_exit = map(date.fromisoformat, (
            "2020-10-29", "2020-10-30", "2020-11-03", "2020-11-27", "2020-11-30", "2020-12-01", "2020-12-16"))
        p = 80711
        rows = []
        for row in read_crsp_daily(DATA / "sp500_member_daily_2015_2026_09.csv.gz", require_prices=True):
            if row.day > after:
                break
            if row.day >= signal and row.permno == p:
                rows.append(row)
        tradable = {row.day: {p} if row.trading_status == "A" else set() for row in rows}
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        actions = {d: [e for e in events.get(d, ()) if e.permno == p] for d in tradable}
        self.assertEqual((actions[ex], actions[paid]), ([AIMCO_PAYMENT], []))
        markets = sessions_from_crsp(rows, tradable=tradable, actions_by_day=actions)
        settings = dict(sessions=[*sorted(markets), next_exit], first_trade=first, end=after, days=markets,
                        initial_cash=D(10000), fee=proportional_commission(D(0)), delistings=[],
                        delistings_known_through=next_exit, exit_days=[first], next_rebalance=next_exit)
        for sell_day in (None, ex, sale):
            targets = {first: Target({p: D(".31")})}  # 实际买入 100 股，均有领取资格。
            if sell_day:
                targets[sell_day] = Target({})
            live = run(**settings, strategy=lambda s: targets[s.trade_day], rebalance_days=targets)
            with TemporaryDirectory() as folder:
                path = Path(folder) / "weights.csv"
                write_weights(path, live.targets)
                replay = run(**settings, weights_file=path)
            self.assertEqual((live.history, live.ledger.entries, live.ledger.sales, live.ledger.distribution_income),
                             (replay.history, replay.ledger.entries, replay.ledger.sales, replay.ledger.distribution_income))
            history = {v.day: v for v in live.history}
            self.assertEqual(history[ex].holdings[p], 23 if sell_day == ex else 123)
            self.assertEqual(history[ex].nav, D("10232.95140348612007746933505"))
            expected_paid, expected_after = {None: (123, 99), ex: (23, 18), sale: (0, 0)}[sell_day]
            self.assertEqual((history[paid].holdings.get(p, 0), history[after].holdings.get(p, 0)),
                             (expected_paid, expected_after))
            income, = live.ledger.distribution_income
            self.assertEqual((income.day, income.ordinary, income.qualified, income.capital_gain, income.section_1250),
                             (ex, D("81.538750"), D("15.552120"), D("495.413742"), D("227.495388")))
            cash, = [e for e in live.ledger.entries if e.kind == "DIVIDEND_CASH_PAYMENT"]
            cil, = [s for s in live.ledger.sales if s.day == ex and s.reason == "CASH_IN_LIEU"]
            self.assertEqual((cash.day, cash.cash_change, cil.proceeds, cil.basis, cil.exact_shares),
                             (ex, D("102.924"), D("3.947403486120077469335054874"), D("4.536"), Fraction(1134, 7745)))
            self.assertFalse(any(m.sale_day == ex and m.sold_lot == cil.lot_id for m in live.ledger.wash_matches))
            self.assertEqual((cil.disallowed_loss, cil.taxable_gain), (0, cil.proceeds - cil.basis))
            self.assertEqual(Fraction(history[ex].cash), 6810 + Fraction(cash.cash_change) + Fraction(cil.proceeds)
                             + (2696 if sell_day == ex else 0))
            self.assertEqual(history[paid].cash, history[sale].cash)
            self.assertFalse(any(e.day == paid for e in live.ledger.entries))
            if sell_day == ex:
                self.assertEqual(live.targets[ex].sell_lots[p], {1: 100})
                traded = [s for s in live.ledger.sales if s.day == ex and s.reason == "TRADE"]
                self.assertEqual((sum(s.shares for s in traded), {s.source_lot_id for s in traded}), (100, {1}))

    def test_aimco_ex_date_cash_is_spendable_and_late_buyer_has_no_entitlement(self):
        before, ex, paid = map(date.fromisoformat, ("2020-11-02", "2020-11-03", "2020-11-30"))
        p = 80711
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        claim, = [e for e in events[ex] if e.permno == p]
        fee = proportional_commission(D(0))
        ex_prices, pay_prices = {p: D("26.96"), 1: D(50)}, {p: D("30.35"), 1: D(50)}
        engine = AccountEngine(D(3212), start=before, fee=fee)
        book = engine.result.ledger
        book.buy(before, p, D(100), D("32.12"), D(0))
        ex_market = Session(ex_prices, ex_prices, [claim])
        with self.assertRaisesRegex(ValueError, "available cash 106"):
            engine.advance(ex, ex_market, [Order(1, 3, {})], {}, ())
        with self.assertRaisesRegex(ValueError, "CASH_IN_LIEU: missing executable CLOSE"):
            engine.advance(ex, Session({}, ex_prices, [claim]), (), {}, ())
        self.assertEqual((book.cash, book.holdings(), book.distribution_income, book.wash_matches),
                         (0, {p: 100}, [], []))
        engine.advance(ex, ex_market, [Order(1, 2, {})], {}, ())
        self.assertEqual((book.holdings(), len(book.distribution_income)), ({p: 123, 1: 2}, 1))

        late = AccountEngine(D(100), start=before, fee=fee)
        late.advance(ex, ex_market, [Order(p, 1, {})], {}, ())
        late.advance(paid, Session(pay_prices, pay_prices, []), (), {}, ())
        self.assertEqual((late.result.ledger.holdings(), late.result.ledger.cash, late.result.ledger.distribution_income),
                         ({p: 1}, D("73.04"), []))

    def test_alcoa_spin_off_preserves_both_securities_in_strategy_and_replay(self):
        signal, first, ex, following = map(date.fromisoformat, ("2016-10-18", "2016-10-19", "2016-11-01", "2016-11-21"))
        parent, child = 24643, 16347
        rows, tradable = [], {}
        for row in read_crsp_daily(DATA / "sp500_member_daily_2015_2026_09.csv.gz", require_prices=True):
            if row.day > ex:  # 文件按交易日排序。
                break
            if row.day >= signal and row.permno in {parent, child}:
                rows.append(row)
                market = tradable.setdefault(row.day, set())
                if row.trading_status == "A":
                    market.add(row.permno)
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        actions = {d: [e for e in events.get(d, ()) if e.permno in {parent, child}] for d in tradable}
        days = sessions_from_crsp(rows, tradable=tradable, actions_by_day=actions)
        settings = dict(sessions=[*sorted(days), following], days=days, first_trade=first, end=ex,
                        next_rebalance=following, delistings=read_delistings(DATA / "delistings_2015_to_2026_09.csv"),
                        delistings_known_through=following, initial_cash=D(1000), fee=proportional_commission(D(0)))
        weight = D("3.1") * days[signal].raw_closes[parent] / D(1000)
        live = run(**settings, strategy=lambda s: Target({parent: weight}), rebalance_days=[first])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "weights.csv"
            write_weights(path, live.targets)
            replay = run(**settings, weights_file=path)
        self.assertEqual((live.history, live.actions, live.ledger.entries, live.ledger.sales),
                         (replay.history, replay.actions, replay.ledger.entries, replay.ledger.sales))
        end = live.history[-1]
        self.assertEqual((end.holdings, end.cash, end.nav, live.ledger.sales), ({parent: 3, child: 1}, D("918.88"), D("998.64"), []))
        child_basis = D("81.12") * 23 / D("79.76")
        for result in (live, replay):
            record, = [action for _, action in result.actions if action.reference == ALCOA_SPIN_OFF]
            lots = {lot.permno: lot for lot in result.ledger.lots.values()}
            self.assertEqual((record.status, result.history[-1].basis_verified, result.ledger.total_basis()),
                             ("RESOLVED", True, D("81.12")))
            # 三股 ARNC 的收盘市值为 56.76 美元，分得的一股 AA 为 23 美元。
            self.assertEqual(record.close_basis_fraction, Fraction(23) / Fraction("79.76"))
            self.assertEqual((lots[parent].basis, lots[child].basis, lots[child].holding_start),
                             (D("81.12") - child_basis, child_basis, lots[parent].holding_start))

        # 分拆日清母股，按原 lot 的调整后成本交割，获配的 AA 留在账户。
        targets = {**live.targets, ex: Target({})}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "weights.csv"
            write_weights(path, targets)
            for source in ({"strategy": lambda s: targets[s.trade_day], "rebalance_days": [first, ex]},
                           {"weights_file": path}):
                result = run(**settings, **source)
                sale, = result.ledger.sales
                self.assertEqual((sale.permno, sale.lot_id, sale.shares, sale.basis),
                                 (parent, 1, 3, D("81.12") - child_basis))
                self.assertEqual((result.ledger.holdings(), result.ledger.cash, result.history[-1].nav),
                                 ({child: 1}, D("975.64"), D("998.64")))

    def test_issuer_terms_override_the_stock_dividend_code_but_missing_closes_leave_basis_pending(self):
        ex = date(2016, 11, 1)
        path = DATA / "distributions_2015_to_2026_09.csv"
        terms = SpinOffTerms(16347, D(".333333"), D(".2686"), "Arconic shareholder tax basis example")
        events = read_distributions(path, spin_off_terms={ALCOA_SPIN_OFF: terms})
        event, = [e for e in events[ex] if e.reference == ALCOA_SPIN_OFF]
        ledger = Ledger(D(1000))
        old = ledger.buy(date(2016, 10, 19), 24643, D(3), D("27.04"), D(0))
        result = event.post(ex, ledger, ledger.entitled_lots())
        child = ledger.lots[result.received_lots[0]]
        self.assertEqual((result.reported_shares_per_parent, result.effective_shares_per_parent, child.shares),
                         (D(".333333"), Fraction(1, 3), 1))
        self.assertEqual((result.status, ledger.total_basis(), old.basis, child.basis, child.holding_start),
                         ("BASIS_PENDING", None, None, None, old.acquired))

    def test_first_build_drift_and_msft_cash_dividend(self):
        signal, first, ordinary, ex_date = map(date.fromisoformat,
                                             ("2015-02-11", "2015-02-12", "2015-02-13", "2015-02-17"))
        permno = 10107
        members = read_membership(DATA / "sp500_membership_intervals.csv")
        self.assertTrue(any(m.permno == permno and m.start <= signal <= ex_date <= m.end
                            for m in members))
        # 此小窗口使用 benchmark 观察日期，整个样本的 NYSE 日期核对见 analysis.check_trading_calendar。
        with (DATA.parent / "market/sp500_observed_trading_dates.csv").open() as source:
            calendar = [date.fromisoformat(r["trading_date"]) for r in csv.DictReader(source)]
        study_days = {d for d in calendar if signal <= d <= ex_date}
        names = {permno, 14593}
        observed = {d: {} for d in study_days}
        for row in read_crsp_daily(DATA / "sp500_daily_2015_2025.csv", require_prices=True):
            if row.permno in names and row.day in study_days:
                observed[row.day][row.permno] = row
                if all(len(quotes) == len(names) for quotes in observed.values()):
                    break
        events = read_distributions(DATA / "distributions_2015_to_2026_09.csv")
        rows = [r for quotes in observed.values() for r in quotes.values()]
        tradable = {d: {p for p, r in quotes.items() if r.trading_status == "A"} for d, quotes in observed.items()}
        actions = {d: [e for e in events.get(d, []) if e.permno in names] for d in study_days}

        def replay(vwaps=None, *, strategy=None):
            market = sessions_from_crsp(rows, tradable=tradable, actions_by_day=actions,
                                        regular_session_vwaps=vwaps)
            source = ({"weights_file": DATA.parents[1] / "examples/initial_weights.csv"} if strategy is None
                      else {"strategy": strategy, "rebalance_days": [first]})
            return run(**source, sessions=calendar,
                       first_trade=first, end=ex_date, days=market,
                       delistings=read_delistings(DATA / "delistings_2015_to_2026_09.csv"),
                       delistings_known_through=date(2026, 9, 30), next_rebalance=date(2015, 3, 16),
                       initial_cash=D(1000), fee=proportional_commission(D(".001")))

        result = replay()
        opening, drift, ex = result.history
        self.assertEqual((opening.holdings, opening.cash, opening.nav, opening.daily_return, opening.execution_source),
                         ({10107: 12, 14593: 3}, D("102.64354"), D("999.10354"), D("-.00089646"), "CLOSE"))
        self.assertEqual([lot.basis for lot in result.ledger.lots.values()], [D("517.59708"), D("379.75938")])
        self.assertEqual([e.price_source for e in result.ledger.entries if e.kind == "BUY"], ["CLOSE", "CLOSE"])
        self.assertEqual((drift.nav, drift.holdings, drift.daily_return),
                         (D("1010.32354"), opening.holdings, D("1010.32354") / D("999.10354") - 1))
        self.assertEqual(len(result.plans), 1)
        dividend, = [e for e in result.ledger.entries if e.kind == "CASH_DIVIDEND"]
        self.assertEqual(dividend.cash_change, D(12) * D(".31"))
        self.assertEqual(ex.cash, drift.cash + dividend.cash_change)
        # 个股含息收益只用于外部核对；用昨日实际股票市值加权，现金的日收益为零。
        stock_pnl = sum(drift.market_values[p] * observed[ex_date][p].total_return for p in names)
        self.assertLessEqual(abs(stock_pnl / drift.nav - ex.daily_return), D("0.0000005"))
        self.assertEqual(result.exceptions, [])
        interactive = replay(strategy=lambda state: Target({10107: D(".55"), 14593: D(".45")}))
        self.assertEqual((interactive.history, interactive.ledger.entries), (result.history, result.ledger.entries))

        # 真实收盘与目标保持不变，只换成用来手算的假设 VWAP，不冒充历史数据。
        alternative = replay({first: {10107: D(43), 14593: D(126)}})
        bought = alternative.history[0]
        self.assertEqual((bought.holdings, bought.cash, bought.nav, bought.daily_return),
                         (opening.holdings, D("105.106"), D("1001.566"), D(".001566")))
        self.assertEqual([lot.basis for lot in alternative.ledger.lots.values()], [D("516.516"), D("378.378")])
        self.assertEqual([e.price_source for e in alternative.ledger.entries if e.kind == "BUY"], ["VWAP", "VWAP"])

    def test_apple_split_from_distribution_factor(self):
        events = read_distributions(DATA / "raw" / "distributions_base.zip")
        ex_date = date(2020, 8, 31)
        split, = [e for e in events[ex_date] if e.permno == 14593]
        self.assertEqual(split.value, 4)
        ledger = Ledger(D(400))
        lot = ledger.buy(date(2020, 8, 28), split.permno, D(1), D(400), D(0))
        split.post(ex_date, ledger, ledger.entitled_lots())
        self.assertEqual((lot.shares, lot.basis, ledger.cash), (4, 400, 0))
