"""Audit continuous LTFO allocations and deliverable orders separately.

The existing take_shares rule preserves whole shares inside each trading lot.
After splits its allocations need not equal the continuous LTFO relaxation.
Both allocation sets must reconcile independently; their difference is reported.
"""
from datetime import timedelta
from decimal import Decimal as D
from fractions import Fraction as F

from tlh.tax import ST_RATE, LT_RATE, is_long_term
from tlh.shares import as_decimal, sum_shares
from tlh.optimization.wash_cost import ledger_reference


def continuous_allocations(costs,index,quantity):
    allocations={}
    for lot in costs.lots[index]:
        take=min(quantity,lot.shares)
        if take>0:allocations[lot.id]=take
        quantity=sum_shares((quantity,take.copy_negate()))
        if not quantity:break
    if quantity:raise ValueError('Continuous sale exceeds available lots')
    return allocations


def allocation_tax(state,permno,allocations,fee_rate):
    """Independent scalar Fraction calculation, not the compiled PWL operations."""
    raw=[l for l in state.wash_lots if l.permno==permno]
    groups={}
    for l in raw:groups.setdefault(l.bound_lot_id if l.bound_lot_id is not None else l.id,[]).append(l)
    sold={l.id:F(0) for l in raw}
    for key,q in allocations.items():
        total=sum((l.exact_shares or F(l.shares) for l in groups[key]),F(0))
        for l in groups[key]:sold[l.id]=F(q)*(l.exact_shares or F(l.shares))/total
    recent=sorted((l for l in raw if l.origin is not None and not l.wash_used
                   and state.trade_day-timedelta(days=30)<=l.acquired<=state.day),
                  key=lambda l:(l.acquired,l.origin,l.id))
    available={l.id:(l.exact_shares or F(l.shares))-sold[l.id] for l in recent}
    tax=F(0)
    for l in sorted(raw,key=lambda l:(l.acquired,l.id)):
        units=l.exact_shares or F(l.shares)
        gain=F(state.closes[permno])*(1-F(fee_rate))-F(l.basis)/units
        rate=F(LT_RATE if is_long_term(l.holding_start,state.trade_day) else ST_RATE)
        remaining=sold[l.id]
        tax+=gain*remaining*rate
        if gain>=0:continue
        for replacement in recent:
            if replacement.origin==l.origin:continue
            matched=min(remaining,available[replacement.id])
            tax-=gain*matched*rate
            remaining-=matched;available[replacement.id]-=matched
    return as_decimal(tax)


def allocated_ledger_tax(state,permno,allocations,fee_rate,*,continuous=False):
    """Real wash ledger with exactly the specified allocation, on a private copy.

LIQUIDATION permits fractional hypothetical lot sales for the continuous audit;
its wash/tax accounting is identical to TRADE. Actual orders always use TRADE.
No simulated trade is executed in the backtest account.
"""
    from tlh.ledger import Ledger,Lot
    book=Ledger(D('1e30'))
    for l in state.wash_lots:
        if l.permno!=permno:continue
        book.lots[l.id]=Lot(l.id,l.permno,l.acquired,l.holding_start,l.shares,l.basis,l.origin,
            l.wash_used,l.prior_periods,l.bound_lot_id,l.source_lot_id,exact_shares=l.exact_shares)
    book._next_id=max([0,*[l.id for l in state.wash_lots]])+1
    book.sell(state.trade_day,permno,allocations,state.closes[permno],
              {k:q*state.closes[permno]*fee_rate for k,q in allocations.items()},
              'LIQUIDATION' if continuous else 'TRADE')
    return sum((s.taxable_gain*(LT_RATE if s.is_long_term else ST_RATE) for s in book.sales),D(0))


def audit_integer_tax(state,permnos,costs,orders):
    total=0.;errors=[];model_errors=[];allocation_difference=0.;different=0
    fee=D(str(costs.fee_rate))
    for order in orders:
        i=permnos.index(order.permno);q=order.quantity
        if q>0:
            forecast=costs.wash_buy_curves[i].value_and_slope(float(q))[0]
            actual=float(ledger_reference(state,order.permno,q,fee_rate=fee))
            errors.append(abs(forecast-actual));model_errors.append(abs(forecast-actual))
        else:
            continuous=continuous_allocations(costs,i,q.copy_abs())
            model=float(allocated_ledger_tax(state,order.permno,continuous,fee,continuous=True))
            curve=costs.wash_sell_curves[i].value_and_slope(float(-q))[0]
            model_errors.append(abs(curve-model))
            forecast=float(allocation_tax(state,order.permno,order.lots,fee))
            actual=float(allocated_ledger_tax(state,order.permno,order.lots,fee))
            errors.append(abs(forecast-actual))
            allocation_difference+=actual-model
            different+=continuous!=order.lots
        if max(errors[-1],model_errors[-1])>.0001:
            raise ValueError(f'Wash audit disagreement {state.trade_day} {order.permno}: '
                             f'execution_error={errors[-1]}, continuous_error={model_errors[-1]}')
        total+=actual
    return dict(independently_checked_orders=len(orders),max_error_usd=max([*errors,*model_errors],default=0.),
        max_execution_error_usd=max(errors,default=0.),max_continuous_error_usd=max(model_errors,default=0.),
        estimated_integer_incremental_tax_usd=total,
        deliverable_lot_allocation_difference_usd=allocation_difference,
        orders_with_different_deliverable_lot_allocation=different)
