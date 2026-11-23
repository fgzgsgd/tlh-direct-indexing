"""Read and save daily weights and lot-selection orders. Dates are effective trading days; shares are in signal-day units."""

import csv
from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory

from .market import crsp_table
from .strategy import Target


WEIGHT_COLUMNS = ("date", "permno", "target_weight", "rebalance")


def _read_sell_lots(cell: str) -> dict[int, Decimal]:
    lots = {}
    for item in cell.split("|"):
        lot_id, shares = item.split(":")
        lot_id = int(lot_id)
        if lot_id in lots:
            raise ValueError(f"Duplicate lot {lot_id}.")
        lots[lot_id] = Decimal(shares)
    return lots


def read_weights(path: Path) -> dict[date, Target]:
    """One full cross-section per day; positive weights that disappear must be set to zero explicitly so missing rows do not trigger sales."""
    result: dict[date, Target] = {}
    with crsp_table(path) as rows:
        if not set(WEIGHT_COLUMNS).issubset(rows.fieldnames):
            raise ValueError("Weight file requires date, permno, target_weight, rebalance.")
        for row in rows:
            day = date.fromisoformat(row["date"])
            weight = Decimal(row["target_weight"])
            flag = row["rebalance"]
            if flag not in {"0", "1"}:
                raise ValueError(f"{day}: rebalance must be 0 or 1.")
            target = result.setdefault(day, Target({}, flag == "1"))
            if target.rebalance != (flag == "1"):
                raise ValueError(f"{day}: inconsistent rebalance flag.")
            cell = row.get("sell_lots") or ""
            if not row["permno"]:  # All-cash days also have a date and a rebalance flag.
                if weight != 0 or cell:
                    raise ValueError(f"{day}: a row without PERMNO requires zero weight and no sell_lots.")
                continue
            permno = int(row["permno"])
            if permno in target.weights:
                raise ValueError(f"{day} {permno}: duplicate weight.")
            target.weights[permno] = weight
            if cell:
                try:
                    target.sell_lots[permno] = _read_sell_lots(cell)
                except (ValueError, InvalidOperation) as error:
                    raise ValueError(f"{day} {permno}: invalid sell_lots {cell!r}: {error}") from error

    previous: set[int] = set()
    for day, target in sorted(result.items()):
        if missing := previous.difference(target.weights):
            raise ValueError(f"{day}: missing carried weights; specify zero to exit {sorted(missing)}.")
        # Ignore explicit zeros when comparing; 0.5 and 0.500 are the same weight.
        active = target.active_weights(day)
        result[day] = Target(active, target.rebalance, target.sell_lots)
        previous = set(active)
    return result


def write_weights(path: Path, targets: Mapping[date, Target]) -> None:
    """Write run().targets to one CSV, including the allocations locked on the signal day for unfilled sell orders."""
    if path.suffix.lower() in {".gz", ".zip"}:
        raise ValueError("Write weights to a plain CSV file.")
    previous: set[int] = set()
    with TemporaryDirectory(dir=path.parent) as tmp:
        draft = Path(tmp) / path.name
        with draft.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow((*WEIGHT_COLUMNS, "sell_lots"))
            for day, target in sorted(targets.items()):
                if target.lot_relief is not None:
                    raise ValueError(f"{day}: export run().targets after resolving the lot selection.")
                names = previous | target.weights.keys() | target.sell_lots.keys()
                if not names:
                    writer.writerow((day, "", 0, int(target.rebalance), ""))
                for permno in sorted(names):
                    lots = target.sell_lots.get(permno, {})
                    cell = "|".join(f"{i}:{q}" for i, q in lots.items())
                    writer.writerow((day, permno, target.weights.get(permno, 0), int(target.rebalance), cell))
                previous = {p for p, weight in target.weights.items() if weight}
        draft.replace(path)
