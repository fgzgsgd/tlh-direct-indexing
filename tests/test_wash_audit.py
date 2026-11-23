import unittest
from datetime import date,timedelta
from decimal import Decimal as D
from fractions import Fraction as F
from copy import deepcopy
import numpy as np
from tlh.ledger import Ledger
from tlh.strategy import CloseSnapshot
from tlh.execution import Order
from tlh.optimization.two_stage_ltfo import ltfo_with_fees
from tlh.optimization.wash_cost import WashAdjustedLtfoCosts
from tlh.optimization.wash_audit import audit_integer_tax,continuous_allocations
from tlh.shares import sum_shares
from types import SimpleNamespace

class DeliverableAuditTests(unittest.TestCase):
    def test_continuous_allocation_keeps_exact_decimal_split_balances(self):
        lots=[SimpleNamespace(id=1,shares=D('1.33333333333333333333333333333333333')),
              SimpleNamespace(id=2,shares=D('1.66666666666666666666666666666666667'))]
        allocations=continuous_allocations(SimpleNamespace(lots=[lots]),0,D(3))
        self.assertEqual(allocations,{l.id:l.shares for l in lots})
        self.assertEqual(sum_shares(allocations.values()),D(3))

    def audit(self,book,quantities):
        original=deepcopy(book.__dict__)
        snapshot=CloseSnapshot.from_ledger(date(2025,2,28),book,{1:D(100)},
            book.cash+sum(l.shares*100 for l in book.lots.values()),set(),trade_day=date(2025,3,3))
        costs=WashAdjustedLtfoCosts(snapshot,(1,))
        reports=[]
        for q in quantities:
            allocations=ltfo_with_fees(snapshot,1,D(q),fee_rate=D('.0005'))
            reports.append(audit_integer_tax(snapshot,(1,),costs,[Order(1,-D(q),allocations)]))
        self.assertEqual(book.__dict__,original)
        self.assertLess(max(r['max_error_usd'] for r in reports),1e-7)
        return reports

    def test_fractional_tail_execution_differs_from_continuous_ltfo(self):
        book=Ledger(D(100000))
        for j,price in enumerate((150,130,110,90)):
            book.buy(date(2023,1,2)+timedelta(days=j),1,D(1),D(price),D(0))
        book.split(date(2025,2,25),1,F(5,4),'regression')
        report=self.audit(book,[2])[0]
        self.assertEqual(report['orders_with_different_deliverable_lot_allocation'],1)
        self.assertNotEqual(report['deliverable_lot_allocation_difference_usd'],0)

    def test_randomized_split_lots_with_recent_wash_replacements(self):
        rng=np.random.default_rng(29946)
        for sample in range(25):
            book=Ledger(D(100000))
            for j in range(8):
                day=date(2023,1,2)+timedelta(days=j) if j<4 else date(2025,2,10)+timedelta(days=j)
                book.buy(day,1,D(int(rng.integers(1,6))),D(int(rng.integers(90,180))),D(0))
            book.split(date(2025,2,25),1,(F(5,4),F(1,3),F(7,3))[sample%3],str(sample))
            held=int(sum(l.shares for l in book.lots.values()))
            self.audit(book,sorted(set([1,held,*map(int,rng.integers(1,held+1,12))])))

if __name__=='__main__':unittest.main()
