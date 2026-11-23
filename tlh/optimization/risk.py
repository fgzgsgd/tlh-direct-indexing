"""Point-in-time, 252-session Ledoit–Wolf risk estimates from CRSP DlyRet."""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.covariance import ledoit_wolf

from tlh.market import crsp_table


@dataclass(frozen=True)
class RiskEstimate:
    start: date
    end: date
    permnos: tuple[int, ...]       # Only securities with all 252 observations.
    covariance: np.ndarray       # Annualized; same ordering as permnos.
    observations: dict[int, int]
    shrinkage: float

    @property
    def incomplete(self) -> tuple[int, ...]:
        complete = set(self.permnos)
        return tuple(p for p in self.observations if p not in complete)

    def tracking_error(self, weights, benchmark) -> float:
        # Incomplete names must have zero active weight; they are not assigned zero risk.
        if any(abs(float(weights.get(p, 0)) - float(benchmark.get(p, 0))) > 1e-12
               for p in self.incomplete):
            raise ValueError("An incomplete-history security has nonzero active weight.")
        active = np.array([float(weights.get(p, 0)) - float(benchmark.get(p, 0)) for p in self.permnos])
        return float(np.sqrt(max(0., active @ self.covariance @ active)))


class RollingLedoitWolf:
    def __init__(self, dates, permnos, returns, *, window=252):
        self.dates = tuple(dates)
        self.permnos = tuple(permnos)
        self.returns = np.asarray(returns, dtype=float)
        self.window = window
        self.columns = {p: i for i, p in enumerate(self.permnos)}
        if self.dates != tuple(sorted(set(self.dates))) or len(self.columns) != len(self.permnos):
            raise ValueError("Risk inputs need unique securities and sorted, unique sessions.")
        if self.returns.shape != (len(self.dates), len(self.permnos)) or window < 2:
            raise ValueError("Return panel shape or lookback window is invalid.")

    @classmethod
    def from_crsp(cls, path: Path, dates, permnos, *, window=252):
        dates, permnos = tuple(dates), tuple(sorted(set(permnos)))
        rows_by_date = {day: i for i, day in enumerate(dates)}
        columns = {p: i for i, p in enumerate(permnos)}
        panel = np.full((len(dates), len(permnos)), np.nan)
        with crsp_table(path) as rows:
            for row in rows:
                p = int(row["permno"])
                if p not in columns:
                    continue
                day = date.fromisoformat(row["dlycaldt"])
                if day in rows_by_date and row["dlyret"]:
                    panel[rows_by_date[day], columns[p]] = float(row["dlyret"])
        return cls(dates, permnos, panel, window=window)

    def estimate(self, as_of: date, universe) -> RiskEstimate:
        stop = bisect_right(self.dates, as_of)
        if stop < self.window or self.dates[stop - 1] != as_of:
            raise ValueError(f"{as_of}: need {self.window} sessions ending on the signal date.")
        start = stop - self.window
        universe = tuple(sorted(set(universe)))
        # NaN remains missing. Neither pre-listing returns nor isolated gaps are filled.
        panel = np.column_stack([
            self.returns[start:stop, self.columns[p]] if p in self.columns else np.full(self.window, np.nan)
            for p in universe
        ]) if universe else np.empty((self.window, 0))
        counts = np.isfinite(panel).sum(axis=0)
        complete = counts == self.window
        names = tuple(p for p, ok in zip(universe, complete) if ok)
        if names:
            covariance, shrinkage = ledoit_wolf(panel[:, complete], assume_centered=False)
            covariance *= 252
        else:
            covariance, shrinkage = np.empty((0, 0)), 0.
        return RiskEstimate(self.dates[start], as_of, names, covariance,
                            dict(zip(universe, map(int, counts))), float(shrinkage))
