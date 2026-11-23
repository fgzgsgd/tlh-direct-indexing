import unittest
from datetime import date,timedelta
from decimal import Decimal as D
from fractions import Fraction as F
from copy import deepcopy
from dataclasses import replace
import numpy as np
from tlh.ledger import Ledger
from tlh.strategy import CloseSnapshot
from tlh.optimization.wash_cost import WashAdjustedLtfoCosts,ledger_reference,Piecewise
from tlh.optimization.two_stage_ltfo import LtfoCosts

TRADE=date(2025,3,3)
SIGNAL=date(2025,2,28)
OLD=date(2023,1,2)
FEE=D('.0005')

def state(book):
    return CloseSnapshot.from_ledger(SIGNAL,book,{1:D(100)},book.cash+sum(l.shares*100 for l in book.lots.values()),set(),trade_day=TRADE)

class WashCostTests(unittest.TestCase):
    def check_trades(self,book,quantities):
        snapshot=state(book)
        original=deepcopy(book.__dict__)
        cost=WashAdjustedLtfoCosts(snapshot,(1,),fee_rate=FEE)
        for q in quantities:
            q=D(q)
            curve=cost.wash_sell_curves[0] if q<=0 else cost.wash_buy_curves[0]
            value=curve.value_and_slope(abs(float(q)))[0]
            expected=float(ledger_reference(snapshot,1,q,fee_rate=FEE))
            self.assertAlmostEqual(value,expected,places=7,msg=f'q={q}')
        self.assertEqual(book.__dict__,original)
        return cost

    def test_old_buy_side_zero_when_no_recent_losses(self):
        b=Ledger(D(10000));b.buy(OLD,1,D(10),D(120),D(0))
        s=state(b);w=np.array([(1000+500)/float(s.nav)])
        self.assertEqual(LtfoCosts(s,(1,)).evaluate(w)[0],0)
        self.assertEqual(WashAdjustedLtfoCosts(s,(1,)).evaluate(w)[0],0)

    def test_buy_matches_historical_sale_order_not_smallest_loss(self):
        b=Ledger(D(100000))
        a=b.buy(OLD,1,D(10),D(120),D(0))
        z=b.buy(OLD+timedelta(days=1),1,D(10),D(101),D(0))
        b.sell(date(2025,2,5),1,{a.id:D(10)},D(100),{a.id:D(0)},'TRADE')
        b.sell(date(2025,2,6),1,{z.id:D(10)},D(100),{z.id:D(0)},'TRADE')
        c=self.check_trades(b,[0,1,5,10,11,20,25])
        self.assertAlmostEqual(c.wash_buy_curves[0].value_and_slope(5)[0],5*20*.238)

    def test_sale_matches_specific_lots_not_reduced_quantity_ltfo(self):
        b=Ledger(D(100000))
        b.buy(OLD,1,D(50),D(110),D(0))
        b.buy(OLD+timedelta(days=1),1,D(50),D(101),D(0))
        b.buy(date(2025,2,20),1,D(50),D(100),D(0))
        self.check_trades(b,[0,-1,-49,-50,-51,-75,-99,-100,-125,-149,-150])

    def test_selling_recent_replacements_changes_available_quantity(self):
        b=Ledger(D(100000))
        b.buy(OLD,1,D(50),D(110),D(0))
        b.buy(date(2025,2,20),1,D(50),D(150),D(0))
        self.check_trades(b,[-1,-25,-49,-50,-51,-75,-100])

    def test_already_used_replacements_do_not_match_twice(self):
        b=Ledger(D(100000))
        a=b.buy(OLD,1,D(20),D(120),D(0))
        b.buy(OLD+timedelta(days=1),1,D(20),D(110),D(0))
        b.sell(date(2025,2,5),1,{a.id:D(10)},D(100),{a.id:D(0)},'TRADE')
        b.buy(date(2025,2,10),1,D(15),D(100),D(0))
        self.check_trades(b,[-1,-5,-10,-20,-30,-40,-45,1,15])

    def test_positive_gain_tax_is_not_erased(self):
        b=Ledger(D(100000))
        b.buy(OLD,1,D(10),D(90),D(0))
        b.buy(date(2025,2,20),1,D(20),D(90),D(0))
        c=self.check_trades(b,[-1,-10,-20,-30])
        self.assertGreater(c.wash_sell_curves[0].value_and_slope(10)[0],0)

    def test_30_calendar_days_inclusive_trade_date_not_signal(self):
        for distance in (30,31):
            b=Ledger(D(100000))
            b.buy(OLD,1,D(10),D(120),D(0))
            b.buy(TRADE-timedelta(days=distance),1,D(10),D(100),D(0))
            c=self.check_trades(b,[-1,-5,-10,-15,-20])
            if distance==30:self.assertAlmostEqual(c.wash_sell_curves[0].value_and_slope(5)[0],0)
            else:self.assertLess(c.wash_sell_curves[0].value_and_slope(5)[0],0)

    def test_split_segments_and_proportional_lot_sales(self):
        b=Ledger(D(100000))
        b.buy(OLD,1,D(13),D(110),D(0))
        b.buy(date(2025,2,10),1,D(8),D(120),D(0))
        b.split(date(2025,2,15),1,F(1,3),'test')
        self.check_trades(b,[-1,-2,-3,-4,-5,-6,-7])

    def test_analytic_derivative_away_from_knots(self):
        b=Ledger(D(100000))
        b.buy(OLD,1,D(20),D(120),D(0))
        b.buy(date(2025,2,10),1,D(10),D(115),D(0))
        c=WashAdjustedLtfoCosts(state(b),(1,))
        for curve in (c.wash_sell_curves[0],c.wash_buy_curves[0]):
            for a,z in zip(curve.x[:-1],curve.x[1:]):
                if z-a<1e-6:continue
                q=(a+z)/2;h=min(1e-5,(z-a)/10)
                numerical=(curve.value_and_slope(q+h)[0]-curve.value_and_slope(q-h)[0])/(2*h)
                self.assertAlmostEqual(numerical,curve.value_and_slope(q)[1],places=5)

    def test_randomized_real_ledger_comparison(self):
        rng=np.random.default_rng(179)
        for _ in range(25):
            b=Ledger(D(100000))
            for j in range(8):
                day=OLD+timedelta(days=j) if j<4 else date(2025,2,4)+timedelta(days=j)
                b.buy(day,1,D(int(rng.integers(1,15))),D(int(rng.integers(80,140))),D(0))
            held=int(sum(l.shares for l in b.lots.values()))
            quantities=sorted(set([0,held,*rng.integers(0,held+1,12)]))
            self.check_trades(b,[-int(q) for q in quantities])

if __name__=='__main__':unittest.main()
