"""Option A: wash-adjusted det200 first stage; original convex LTFO second stage."""
import hashlib
import json
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path
from functools import partial
from time import perf_counter
import cloudpickle
import cvxpy as cp
import numpy as np

from tlh.optimization.two_stage_ltfo import (TwoStageLtfo,LtfoCosts,solve_lots,target_decimals,ltfo_with_fees)
from tlh.optimization.problem import PortfolioProblem
from tlh.optimization.wash_cost import WashAdjustedLtfoCosts
from tlh.optimization.wash_audit import audit_integer_tax
from tlh.optimization.deterministic200 import candidates
from tlh.strategy import Target
from tlh.execution import lock_orders,floor_shares


class WashAwareDet200(TwoStageLtfo):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.max_iterations!=200:raise ValueError('This strategy requires 200 iterations per start')
        if self.solver_cache:
            self.audit_base=self.solver_cache.parent
            self.solver_cache=self.audit_base/'wash_det200_candidates'
        else:self.audit_base=None

    def __call__(self,state):
        if self.signals.get(state.trade_day)!=state.day:raise ValueError('Unexpected signal/trade date')
        started=perf_counter()
        p=PortfolioProblem.build(state,self.benchmarks[state.day],self.risk_model,
                                 gamma=self.gamma,te_limit=self.te_limit,fee_rate=self.fee_rate)
        p=replace(p,costs=LtfoCosts(state,p.permnos,fee_rate=self.fee_rate))
        first_problem=replace(p,costs=WashAdjustedLtfoCosts(state,p.permnos,fee_rate=self.fee_rate))
        print(f'{state.trade_day}: wash-adjusted stage 1, 3 x 200, deterministic repair/pool',flush=True)
        options,attempts,key,from_cache=candidates(first_problem,cache_dir=self.solver_cache,workers=self.solver_workers)
        stage_one_seconds=perf_counter()-started
        accepted=[];rejected=[];candidate_reports=[]
        for name,v,ds in options:
            try:
                z,report=solve_lots(p,v,ds)
                target_decimals(p,z,ds)
                accepted.append((report['objective_usd'],name,v,ds,z,report))
                candidate_reports.append(dict(name=name,stage_one_wash_objective_usd=first_problem.objective(v)[0]*p.costs.nav,
                    stage_two_original_objective_usd=report['objective_usd'],status=report['status']))
            except (ValueError,cp.error.SolverError) as exc:rejected.append(dict(name=name,error=str(exc)))
        if not accepted:raise ValueError(f'No validated stage-two candidate: {rejected}')
        _,name,first,directions,vector,second=min(accepted,key=lambda x:x[0])
        weights=target_decimals(p,vector,directions)
        vector=np.array([float(weights[k]) for k in p.permnos])
        tax,fees,*_=p.costs.evaluate(vector)
        first_tax,first_fees,*_=first_problem.costs.evaluate(first)
        second_wash_tax,second_wash_fees,*_=first_problem.costs.evaluate(vector)
        execution=dict(weights);overrides={}
        for permno in sorted(state.blocked):
            if execution.get(permno,0):
                overrides[str(permno)]=dict(continuous_weight=str(weights[permno]),execution_weight='0')
                execution[permno]=D(0)
        relief=partial(ltfo_with_fees,fee_rate=self.fee_rate)
        orders=lock_orders(state,execution,floor_shares,relief,set())
        for order in orders:
            direction=directions.get(p.permnos.index(order.permno))
            if order.permno not in state.blocked and (direction=='HOLD' or direction=='BUY' and order.quantity<0 or direction=='SELL' and order.quantity>0):
                raise ValueError('Integer order reversed selected direction')
        tax_audit=audit_integer_tax(state,p.permnos,first_problem.costs,orders)
        # Also check the chosen first-stage anchor after the ordinary whole-share conversion.
        anchor={k:D(str(max(0.,v))) for k,v in zip(p.permnos,first)}
        anchor_orders=lock_orders(state,anchor,floor_shares,relief,set())
        anchor_audit=audit_integer_tax(state,p.permnos,first_problem.costs,anchor_orders)
        if self.audit_base:
            folder=self.audit_base/'monthly_inputs';folder.mkdir(parents=True,exist_ok=True)
            with (folder/f'{state.trade_day}.pkl').open('wb') as stream:
                cloudpickle.dump(dict(stage_one_problem=first_problem,stage_two_problem=p,first_weights=first,directions=directions,second_weights=vector),stream)
        converged=name=='all_fixed' or any(a['start']==name and a['success'] for a in attempts)
        record=dict(signal_date=str(state.day),trade_date=str(state.trade_day),window_start=str(p.risk.start),window_end=str(p.risk.end),
            risk_stocks=len(p.risk.permnos),shrinkage=p.risk.shrinkage,
            incomplete_history={str(k):p.risk.observations[k] for k in p.risk.incomplete},
            pre_delisting_blocked=sorted(state.blocked),forced_exits=list(p.forced_exits),available_cash_usd=float(state.available_cash),
            investable_nav_usd=p.costs.nav,estimated_tax_usd=tax,estimated_fees_usd=fees,objective_usd=self.gamma*tax+fees,
            continuous_te=float(np.sqrt(max(0.,p.variance(vector)))),annual_te_limit=self.te_limit,base_annual_te_limit=self.te_limit,
            pre_delisting_execution_overrides=overrides,execution_target_weight_sum=str(sum(execution.values())),
            selected_start=name,selected_converged=converged,accepted_without_convergence=not converged,attempts=list(attempts),
            solver_result_source='stage1_cache+stage2_computed' if from_cache else 'computed',solver_cache_key=key,
            solver_workers=self.solver_workers,solve_wall_seconds=perf_counter()-started,stage_one_wall_seconds=stage_one_seconds,
            solver_max_iterations_used=200,solver_acceptance_policy='deterministic200_pool_with_10bp_endpoint_repair',
            stage_one_weights={str(k):float(w) for k,w in zip(p.permnos,first)},current_weights={str(k):float(w) for k,w in zip(p.permnos,p.costs.current_weights)},
            loss_stock_directions={str(p.permnos[i]):d for i,d in directions.items()},stage_two=second,
            global_optimality_guaranteed=False,
            integer_orders=[dict(permno=o.permno,quantity=str(o.quantity),lots={str(l):str(q) for l,q in o.lots.items()}) for o in orders],
            wash_objective=dict(stage_one_tax_usd=first_tax,stage_one_objective_usd=self.gamma*first_tax+first_fees,
                stage_two_tax_usd=second_wash_tax,stage_two_objective_usd=self.gamma*second_wash_tax+second_wash_fees,
                incremental_tax_penalty_at_stage_two_usd=second_wash_tax-tax,
                first_stage_anchor_audit=anchor_audit,final_order_audit=tax_audit),
            pilot=dict(mode='det200_wash_option_a',candidates=len(options),winner=name,rejections=rejected,reports=candidate_reports))
        self.decisions.append(record)
        if self.on_decision:self.on_decision(record,weights)
        return Target(execution,lot_relief=relief)
