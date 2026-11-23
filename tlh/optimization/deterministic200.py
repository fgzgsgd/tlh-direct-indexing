"""The prior det200 direction pool, including its <=10bp endpoint repair policy."""
import hashlib
import json
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed,parallel_config
from tlh.optimization import best_feasible as bfs


def candidates(p,*,cache_dir=None,workers=3):
    key=hashlib.sha256((bfs.best_feasible_fingerprint(p,200,True)+Path(__file__).read_text()).encode()).hexdigest()
    path=Path(cache_dir)/f'{key}.json' if cache_dir else None
    if path and path.exists():
        data=json.loads(path.read_text())
        options=[(x['name'],np.array(x['weights']),{int(k):v for k,v in x['directions'].items()}) for x in data['options']]
        for (_,v,_),r in zip(options,data['options']):
            p.check(v,tolerance=1e-7)
            if abs(p.objective(v)[0]-r['objective_nav'])>1e-12:raise ValueError('Candidate cache objective mismatch')
        return options,tuple(data['attempts']),key,True
    from tlh.optimization.two_stage_ltfo import loss_directions
    free=p.free_indices
    template=np.zeros(len(p.permnos))
    for i,weight in p.fixed.items():template[i]=float(weight)
    budget=1-template.sum()
    if not len(free) or budget<=0:
        p.check(template,tolerance=1e-7)
        return [('all_fixed',template,loss_directions(p,template))],(),key,False
    benchmark=p.benchmark_vector[free]
    b=benchmark*budget/benchmark.sum() if benchmark.sum() else np.full(len(free),budget/len(free))
    c=p.costs.current_weights[free]
    c=c+(budget-c.sum())*b/budget if c.sum()<=budget else c*budget/c.sum()
    starts=[('benchmark',b),('current_plus_cash',c),('diversified',.9*b+.1*budget/len(free))]
    if workers==3:
        with parallel_config(backend='loky',n_jobs=3,inner_max_num_threads=1):
            batches=Parallel()(delayed(bfs._observed_attempt)(p,n,v,free,template,budget,200) for n,v in starts)
    elif workers==1:
        batches=[bfs._observed_attempt(p,n,v,free,template,budget,200) for n,v in starts]
    else:raise ValueError('Expected one sequential worker or three start workers')
    pairs=[item for batch in batches for item in batch]
    feasible=[(a,v) for a,v in pairs if a['feasible']]
    if not feasible:raise ValueError('No feasible stage-one candidate at 1% TE')
    best,w=bfs._best(feasible,p)
    initial=[]
    for n,x in starts:
        v=template.copy();v[free]=x
        try:p.check(v,tolerance=1e-7)
        except ValueError:continue
        initial.append(('initial_'+n,v))
    pool=[(a['start'],v) for a,v in feasible]+initial
    anchor=min(initial,key=lambda item:p.variance(item[1]))[1] if initial else w
    for a,v in pairs:
        if not (a['optimizer_endpoint'] and a['constraint_error']=='Continuous solution violates tracking-error constraint.'
                and 0<(a['continuous_te']-p.te_limit)*1e4<=10):continue
        low,high=0.,1.
        for _ in range(40):
            t=(low+high)/2
            if p.variance(anchor+t*(v-anchor))<=p.te_limit**2*(1-1e-9):low=t
            else:high=t
        repaired=anchor+low*(v-anchor)
        p.check(repaired,tolerance=1e-7)
        pool.append(('repaired_'+a['start'],repaired))
    seen=set();options=[]
    for name,v in [(best['start'],w),*sorted(pool,key=lambda z:p.objective(z[1])[0])]:
        ds=loss_directions(p,v);signature=tuple(ds.items())
        if signature not in seen:
            seen.add(signature);options.append((name,v,ds))
        if len(options)==3:break
    attempts=tuple(a for a,v in pairs)
    if path:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(options=[dict(name=n,weights=v.tolist(),directions=ds,objective_nav=p.objective(v)[0]) for n,v,ds in options],attempts=attempts))+'\n')
    return options,attempts,key,False
