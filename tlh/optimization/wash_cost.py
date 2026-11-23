"""Exact piecewise-affine extension of fixed-LTFO trades with known past wash state.

No future transactions are assumed. Curves are compiled once per signal; candidate
evaluation never mutates the account. Shares are continuous in the optimizer.
The independent reference below uses the actual Decimal/Fraction wash ledger.
"""
from bisect import bisect_right
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal as D
from fractions import Fraction
from collections import defaultdict

import numpy as np

from tlh.optimization.two_stage_ltfo import LtfoCosts, ltfo_with_fees
from tlh.tax import ST_RATE, LT_RATE, is_long_term
from tlh.shares import as_decimal


@dataclass(frozen=True)
class Piecewise:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        if len(self.x)<2 or len(self.x)!=len(self.y) or np.any(np.diff(self.x)<=0):
            raise ValueError('Invalid piecewise-affine curve')
        object.__setattr__(self,'xs',self.x.tolist())
        slopes=np.diff(self.y)/np.diff(self.x)
        object.__setattr__(self,'slopes',slopes.tolist())
        object.__setattr__(self,'intercepts',(self.y[:-1]-slopes*self.x[:-1]).tolist())

    def value_and_slope(self,q):
        if q < -1e-8 or q > self.x[-1]+1e-7:
            raise ValueError(f'Trade outside compiled domain: {q}, {self.x[-1]}')
        q=max(0.,min(q,self.x[-1]))
        j=min(bisect_right(self.xs,q)-1,len(self.slopes)-1)
        return self.slopes[j]*q+self.intercepts[j],self.slopes[j]

    @classmethod
    def constant(cls,limit,value=0.):
        return cls(np.array([0.,limit]),np.array([value,value],dtype=float))

    def scaled(self,k):return Piecewise(self.x,self.y*k)

    def combine(self,other,subtract=False,minimum=False):
        x=np.union1d(self.x,other.x)
        a,b=np.interp(x,self.x,self.y),np.interp(x,other.x,other.y)
        if minimum:
            delta=a-b
            crossing=np.flatnonzero(((delta[:-1]<0)&(delta[1:]>0))|((delta[:-1]>0)&(delta[1:]<0)))
            if len(crossing):
                roots=x[crossing]-delta[crossing]*(x[crossing+1]-x[crossing])/(delta[crossing+1]-delta[crossing])
                roots=roots[(roots>x[0])&(roots<x[-1])]
                x=np.union1d(x,roots)
                a,b=np.interp(x,self.x,self.y),np.interp(x,other.x,other.y)
            y=np.minimum(a,b)
        else:y=a-b if subtract else a+b
        return Piecewise(x,y)

    def __add__(self,other):return self.combine(other)
    def __sub__(self,other):return self.combine(other,subtract=True)
    def minimum(self,other):return self.combine(other,minimum=True)


class WashAdjustedLtfoCosts(LtfoCosts):
    """Same LTFO sale order; tax includes new sale disallowance and buy reversals."""
    def __init__(self,state,permnos,*,fee_rate=D('.0005')):
        if not state.wash_state_available:
            raise ValueError('Stage-one wash objective requires an account wash-state snapshot')
        super().__init__(state,permnos,fee_rate=fee_rate)
        self.state=state
        raw={l.id:l for l in state.wash_lots}
        pending=defaultdict(list)
        for loss in state.wash_losses:
            if state.trade_day-timedelta(days=30)<=loss.sale_day<=state.day and loss.shares>0:
                pending[loss.permno].append(loss)
        self.wash_sell_curves=[]
        self.wash_buy_curves=[]
        self.sale_wash_active=[]
        for i,p in enumerate(permnos):
            curve=self.curves[i]
            limit=float(self.shares[i])
            buy_limit=self.nav/self.prices[i]+1.
            losses=pending[p]
            ends=np.cumsum([float(x.shares) for x in losses])
            # Include every legal candidate buy quantity, and a flat tail once all losses match.
            domain=max(buy_limit,float(ends[-1])+1 if len(ends) else 1.)
            x=np.r_[0.,ends,domain]
            increments=[float(x.remaining)*(float(LT_RATE) if is_long_term(x.holding_start,x.sale_day) else float(ST_RATE)) for x in losses]
            y=np.r_[0.,np.cumsum(increments),sum(increments)]
            self.wash_buy_curves.append(Piecewise(x,y))
            if not limit:
                self.wash_sell_curves.append(Piecewise.constant(1.))
                self.sale_wash_active.append(False)
                continue
            base=Piecewise(np.r_[0.,curve.ends],np.r_[0.,curve.cumulative_tax])
            sold={}
            names={}
            left=0.
            for lot,right in zip(self.lots[i],curve.ends):
                xs=np.unique(np.array([0.,left,right,limit]))
                total=np.clip(xs-left,0.,right-left)
                for part in lot.segments or (lot,):
                    l=raw[part.id]
                    names[l.id]=l
                    sold[l.id]=Piecewise(xs,total*(float(part.shares)/float(lot.shares)))
                left=right
            recent=[l for l in names.values() if l.origin is not None and not l.wash_used
                    and state.trade_day-timedelta(days=30)<=l.acquired<=state.day]
            recent.sort(key=lambda l:(l.acquired,l.origin,l.id))
            available={l.id:Piecewise.constant(limit,float(l.shares))-sold[l.id] for l in recent}
            extra=Piecewise.constant(limit)
            active=False
            # The ledger removes ALL candidate sales before looking for remaining replacements.
            for l in sorted(names.values(),key=lambda l:(l.acquired,l.id)):
                per_share=float(l.basis/l.shares-state.closes[p]*(1-fee_rate))
                if per_share<=0:continue
                remaining=sold[l.id]
                rate=float(LT_RATE if is_long_term(l.holding_start,state.trade_day) else ST_RATE)
                for replacement in recent:
                    if replacement.origin==l.origin:continue
                    matched=remaining.minimum(available[replacement.id])
                    if np.max(matched.y)>1e-12:
                        active=True
                        extra=extra+matched.scaled(per_share*rate)
                        remaining=remaining-matched
                        available[replacement.id]=available[replacement.id]-matched
            self.wash_sell_curves.append(base+extra if active else base)
            self.sale_wash_active.append(active)
        self.cache_payload=dict(sells=[(c.x.tolist(),c.y.tolist()) for c in self.wash_sell_curves],
                                buys=[(c.x.tolist(),c.y.tolist()) for c in self.wash_buy_curves])

    def evaluate(self,weights):
        quantity=np.asarray(weights)*self.nav/self.prices
        delta=quantity-self.shares
        tax=0.
        tax_gradient=np.zeros(len(delta))
        for i,q in enumerate(delta):
            if q<=0:
                # Preserve the original implementation exactly where wash matching cannot apply.
                curve=self.wash_sell_curves[i] if self.sale_wash_active[i] else self.curves[i]
                value,slope=curve.value_and_slope(-q)
                tax_gradient[i]=-slope*self.nav/self.prices[i]
            else:
                value,slope=self.wash_buy_curves[i].value_and_slope(q)
                tax_gradient[i]=slope*self.nav/self.prices[i]
            tax+=value
        fees=float(self.fee_rate*np.abs(delta)@self.prices)
        fee_gradient=self.fee_rate*self.nav*np.sign(delta)
        return float(tax),fees,tax_gradient,fee_gradient


def ledger_reference(state,permno,delta,*,fee_rate=D('.0005')):
    """Independent real-ledger reference for one net integer trade, at signal prices.

Only the affected security is copied. Funding is irrelevant to its incremental tax.
Past pending losses retain their original holding periods and FIFO match order.
"""
    from tlh.ledger import Ledger,Lot,Sale
    from tlh.wash import OpenLoss
    book=Ledger(D('1e30'))
    for l in state.wash_lots:
        if l.permno!=permno:continue
        book.lots[l.id]=Lot(l.id,l.permno,l.acquired,l.holding_start,l.shares,l.basis,l.origin,
                           l.wash_used,l.prior_periods,l.bound_lot_id,l.source_lot_id,
                           exact_shares=l.exact_shares)
    book._next_id=max([0,*[l.id for l in state.wash_lots],*[l.sold_lot for l in state.wash_losses]])+1
    for l in state.wash_losses:
        if l.permno!=permno:continue
        sale=Sale(l.sale_day,l.sold_lot,l.permno,as_decimal(l.shares),D(0),l.remaining,l.holding_start,'TRADE')
        book.sales.append(sale)
        book.open_losses.setdefault(permno,[]).append(OpenLoss(sale,l.origin,l.shares,l.remaining,l.periods))
    old=len(book.sales)
    if delta>0:
        price=state.closes[permno]
        book.buy(state.trade_day,permno,delta,price,delta*price*fee_rate)
    elif delta<0:
        allocations=ltfo_with_fees(state,permno,-delta,fee_rate=fee_rate)
        book.sell(state.trade_day,permno,allocations,state.closes[permno],
                  {k:q*state.closes[permno]*fee_rate for k,q in allocations.items()},'TRADE')
    tax=sum((s.disallowed_loss*(LT_RATE if s.is_long_term else ST_RATE) for s in book.sales[:old]),D(0))
    tax+=sum((s.taxable_gain*(LT_RATE if s.is_long_term else ST_RATE) for s in book.sales[old:]),D(0))
    return tax
