"""Translate one immutable account snapshot into continuous weights and fixed trades."""

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from tlh.optimization.risk import RiskEstimate, RollingLedoitWolf
from tlh.optimization.tax_cost import TradeCosts
from tlh.strategy import CloseSnapshot


@dataclass(frozen=True)
class PortfolioProblem:
    state: CloseSnapshot
    permnos: tuple[int, ...]
    benchmark: dict[int, Decimal]
    risk: RiskEstimate
    costs: TradeCosts
    fixed: dict[int, Decimal]     # Position index -> exact continuous target weight.
    forced_exits: tuple[int, ...]
    risk_indices: np.ndarray
    gamma: float = .5
    te_limit: float = .01

    @classmethod
    def build(cls, state: CloseSnapshot, benchmark, risk_model: RollingLedoitWolf,
              *, gamma=.5, te_limit=.01, fee_rate=Decimal("0.0005")):
        if state.investable_nav is None or state.investable_nav <= 0:
            raise ValueError(f"{state.day}: optimizer needs positive, fully valued investable NAV.")
        benchmark = {p: Decimal(w) for p, w in benchmark.items()}
        if (not benchmark or any(not w.is_finite() or w < 0 for w in benchmark.values())
                or abs(sum(benchmark.values()) - 1) > Decimal("1e-15")):
            raise ValueError("Benchmark weights must be finite, nonnegative and sum to one.")
        if not np.isfinite(gamma) or gamma < 0 or not np.isfinite(te_limit) or te_limit <= 0:
            raise ValueError("Tax aversion must be nonnegative; tracking-error limit must be positive.")
        # The user defines index membership by this signal's SPY weights, not a separate index file.
        members = set(benchmark)
        universe = members | benchmark.keys() | state.lots_by_permno.keys()
        permnos = tuple(sorted(universe))
        if missing := [p for p in permnos if state.closes.get(p, Decimal(0)) <= 0]:
            raise ValueError(f"{state.day}: missing signal prices for PERMNOs {missing}.")
        positions = {p: i for i, p in enumerate(permnos)}
        risk = risk_model.estimate(state.day, members | benchmark.keys())
        forbidden = (universe - members) | (set(state.blocked) & universe)
        fixed = {positions[p]: Decimal(0) for p in forbidden}
        for p in risk.incomplete:
            weight = benchmark.get(p, Decimal(0))
            # Keep zero active risk for incomplete histories in the continuous solve.
            # A conflicting pre-delisting exit takes priority in the execution target.
            fixed[positions[p]] = weight
        forced = tuple(sorted(forbidden & state.lots_by_permno.keys()))
        costs = TradeCosts(state, permnos, fee_rate=fee_rate)
        return cls(state, permnos, benchmark, risk, costs, fixed, forced,
                   np.array([positions[p] for p in risk.permnos], dtype=int), gamma, te_limit)

    @property
    def benchmark_vector(self):
        return np.array([float(self.benchmark.get(p, 0)) for p in self.permnos])

    @property
    def free_indices(self):
        return np.array([i for i in range(len(self.permnos)) if i not in self.fixed], dtype=int)

    def variance(self, weights):
        active = (np.asarray(weights) - self.benchmark_vector)[self.risk_indices]
        return float(active @ self.risk.covariance @ active)

    def variance_gradient(self, weights):
        active = (np.asarray(weights) - self.benchmark_vector)[self.risk_indices]
        gradient = np.zeros(len(self.permnos))
        gradient[self.risk_indices] = 2 * self.risk.covariance @ active
        return gradient

    def objective(self, weights):
        tax, fees, tax_gradient, fee_gradient = self.costs.evaluate(weights)
        # Both terms are divided by the same NAV; gamma stays dimensionless.
        return ((self.gamma * tax + fees) / self.costs.nav,
                (self.gamma * tax_gradient + fee_gradient) / self.costs.nav)

    def check(self, weights, *, tolerance=1e-9):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(self.permnos),) or not np.isfinite(weights).all():
            raise ValueError("Solver returned invalid target weights.")
        if weights.min() < -tolerance or abs(weights.sum() - 1) > tolerance:
            raise ValueError("Continuous solution violates long-only or fully invested constraint.")
        if any(abs(weights[i] - float(w)) > tolerance for i, w in self.fixed.items()):
            raise ValueError("Continuous solution violates a forced exit or fixed benchmark weight.")
        if self.variance(weights) > self.te_limit ** 2 + tolerance * self.te_limit ** 2:
            raise ValueError("Continuous solution violates tracking-error constraint.")
