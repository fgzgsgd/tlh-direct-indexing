"""Reuse successful solves only when the numerical inputs and implementation match."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile


def fingerprint(problem, max_iterations):
    digest = hashlib.sha256()
    sources = Path(__file__).parent
    for name in ("cache.py", "solver.py", "problem.py", "risk.py", "tax_cost.py"):
        digest.update(name.encode())
        digest.update((sources / name).read_bytes())

    def metadata(value):
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        digest.update(b"\0")

    def array(value):
        metadata((str(value.dtype), value.shape))
        digest.update(value.tobytes(order="C"))

    cost, risk = problem.costs, problem.risk
    metadata(dict(schema=1, machine=platform.machine(),
                  packages={name: version(name) for name in ("numpy", "scipy")},
                  permnos=problem.permnos, benchmark={p: str(w) for p, w in problem.benchmark.items()},
                  fixed={i: str(w) for i, w in problem.fixed.items()}, gamma=problem.gamma,
                  te_limit=problem.te_limit, max_iterations=max_iterations,
                  signal_date=str(problem.state.day), trade_date=str(problem.state.trade_day),
                  nav=cost.nav, fee_rate=cost.fee_rate, risk_permnos=risk.permnos,
                  window_start=str(risk.start), window_end=str(risk.end),
                  observations=risk.observations, shrinkage=risk.shrinkage))
    for value in (risk.covariance, problem.risk_indices, cost.prices, cost.shares, cost.current_weights):
        array(value)
    for curve in cost.curves:
        metadata(curve.lot_ids)
        for value in (curve.ends, curve.marginal_tax, curve.cumulative_tax):
            array(value)
    if hasattr(cost, 'cache_payload'):
        metadata(cost.cache_payload)
        digest.update((sources / 'wash_cost.py').read_bytes())
    return digest.hexdigest()


def read_entry(directory, key):
    path = Path(directory) / f"{key}.json"
    if not path.exists():
        return None
    entry = json.loads(path.read_text())
    if entry.get("fingerprint") != key:
        raise ValueError(f"Solver cache fingerprint mismatch: {path}")
    return entry


def write_entry(directory, key, solution):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entry = dict(fingerprint=key, weights=solution.weights.tolist(),
                 attempts=solution.attempts, selected_start=solution.selected_start)
    with NamedTemporaryFile(mode="w", dir=directory, suffix=".tmp", delete=False) as stream:
        draft = Path(stream.name)
        json.dump(entry, stream)  # Preserve diagnostics for unsuccessful starts as recorded by the solver.
        stream.write("\n")
    draft.replace(directory / f"{key}.json")
