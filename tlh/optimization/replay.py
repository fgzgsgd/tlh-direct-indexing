"""Jump directly to an earlier accepted solve after validating its exact inputs."""

import json
from datetime import date
from pathlib import Path

import numpy as np

from tlh.optimization.cache import fingerprint, read_entry
from tlh.optimization.feasible_fallback import FEASIBLE_POLICY, fallback_fingerprint, validate_selected_candidate
from tlh.optimization.solver import Solution


def read_replay_decisions(path):
    if path is None:
        return {}
    records = {}
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        day = date.fromisoformat(record["trade_date"])
        if day in records:
            raise ValueError(f"Duplicate replay decision: {day}")
        if not record.get("solver_cache_key"):
            raise ValueError(f"Replay decision has no exact-input cache key: {day}")
        records[day] = record
    if not records:
        raise ValueError("Replay decisions file is empty.")
    return records


def load_cached_decision(problem, record, cache_dir):
    if cache_dir is None:
        raise ValueError("Replaying accepted decisions requires a solver cache directory.")
    if (record["signal_date"] != str(problem.state.day)
            or record["trade_date"] != str(problem.state.trade_day)):
        raise ValueError("Replay decision date does not match the account snapshot.")
    iterations = record["solver_max_iterations_used"]
    key = fingerprint(problem, iterations)
    if key != record["solver_cache_key"]:
        if record.get("solver_acceptance_policy") == FEASIBLE_POLICY:
            key = fallback_fingerprint(problem, iterations)
        if key != record["solver_cache_key"]:
            raise ValueError(f"Replay numerical inputs changed: {problem.state.trade_day}")
    entry = read_entry(cache_dir, key)
    if entry is None:
        raise ValueError(f"Missing accepted replay cache: {problem.state.trade_day} {key}")
    if (entry["selected_start"] != record["selected_start"]
            or entry["attempts"] != record["attempts"]):
        raise ValueError(f"Replay cache diagnostics do not match the accepted decision: {problem.state.trade_day}")
    weights = np.asarray(entry["weights"], dtype=float)
    problem.check(weights, tolerance=1e-7)
    validate_selected_candidate(entry, allow_nonconverged=record.get("solver_acceptance_policy") == FEASIBLE_POLICY)
    return Solution(weights, tuple(entry["attempts"]), entry["selected_start"], True, key), iterations
