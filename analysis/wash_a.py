"""Run the isolated option-A det200 strategy from cash, with uniform settlement."""
import argparse
import json
from datetime import date
from pathlib import Path
from analysis import tax_aware
from tlh.uniform_account import AccountEngine
from tlh.optimization.wash_a_strategy import WashAwareDet200

def factory(*args,tax_after_sales=(),**kwargs):
    if tax_after_sales:raise ValueError('No per-date settlement exceptions')
    return AccountEngine(*args,**kwargs)

def run(output,start=date(2016,1,4),end=date(2025,12,31),solver_cache=None):
    tax_aware.TwoStageLtfo=WashAwareDet200
    tax_aware.ConfirmedTaxTimingEngine=factory
    original_report=tax_aware.write_strategy_report
    def report(*args,**kwargs):
        kwargs['account_name']='Two-stage LTFO det200 / stage-one wash objective (A)'
        return original_report(*args,**kwargs)
    tax_aware.write_strategy_report=report
    try:
        return tax_aware.run_experiment(output,start=start,end=end,gamma=.5,te_limit=.01,
            max_iterations=200,solver_workers=3,solver_cache=solver_cache)
    finally:
        tax_aware.write_strategy_report=original_report
        path=output/'config.json'
        if path.exists():
            config=json.loads(path.read_text())
            config.update(strategy_version='wash_option_A_det200',wash_sale_in_objective='stage_one_only',
                settlement='Original sales, scheduled tax, original buys',historical_prefix='None; new evolving account from cash',
                stage_one_candidates='det200: 3 starts x 200; feasible initial/major iterates/endpoints; <=10bp TE-violating endpoint repair; max 3 distinct directions',
                candidate_ranking='Stage-one wash-adjusted objective for direction pool; original stage-two objective for final candidate selection',
                stage_one_tax='Fixed net-fee LTFO allocation plus exact past-30-calendar-day matching disallowance and historical-loss reversals',
                stage_two_tax='Original LTFO tax, no wash adjustment',
                future_wash_policy='No assumed future buys; no new buy restriction; actual ledger handles subsequent matches',
                wash_implementation='Immutable ledger-derived view, compiled piecewise-affine curves, per-month independent ledger checks',
                extra_candidate_repair_margin_bp=10,max_direction_candidates=3)
            path.write_text(json.dumps(config,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--end',type=date.fromisoformat,default=date(2025,12,31))
    parser.add_argument('--solver-cache',type=Path)
    args=parser.parse_args()
    print(json.dumps(run(**vars(args)),indent=2))
