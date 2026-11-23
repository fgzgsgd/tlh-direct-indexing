"""Net realized gains, losses and dividend tax under the research scenario to compute tax owed or saved for each tax year."""

from calendar import isleap
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum

from .market import last_trading_day_of_year
from .wash import WINDOW


ST_RATE = Decimal("0.408")
LT_RATE = DIVIDEND_RATE = Decimal("0.238")
ZERO = Decimal(0)


def is_long_term(holding_start: date, sale_day: date) -> bool:
    """Uses the lot's tax holding start; a position is LT only from the day after its one-year anniversary."""
    year = holding_start.year + 1
    day = holding_start.day
    if holding_start.month == 2 and day == 29 and not isleap(year):
        day = 28
    return sale_day > date(year, holding_start.month, day)


class Scenario(Enum):
    EXTERNAL_GAINS = "external_gains"
    LIMITED_USE = "limited_use"


@dataclass(frozen=True)
class DistributionIncome:
    """gross is used for the project's dividend tax. The issuer's four categories are kept for reconciliation."""
    day: date
    reference: str
    gross: Decimal
    ordinary: Decimal
    qualified: Decimal
    capital_gain: Decimal
    section_1250: Decimal


@dataclass(frozen=True)
class PropertyIncome:
    day: date
    reference: str
    gross: Decimal  # Taxable dividend computed on parent shares before the distribution; may differ in presentation from the new lot's FMV.


@dataclass(frozen=True)
class TaxYear:
    year: int
    short_term: Decimal
    long_term: Decimal
    taxable_dividends: Decimal
    last_loss_day: date | None
    basis_verified: bool


@dataclass(frozen=True)
class Statement:
    year: int
    net_tax: Decimal
    cumulative_tax: Decimal
    st_carryforward: Decimal
    lt_carryforward: Decimal
    final: bool
    review_end: date  # Last date that must be observed for this tax year (inclusive).

    def after_tax_wealth(self, nav: Decimal, settled_cash_flow: Decimal = ZERO) -> Decimal:
        return nav - (self.cumulative_tax + settled_cash_flow)


@dataclass(frozen=True)
class TaxSettlement:
    day: date
    tax_year: int
    net_tax: Decimal
    cash_flow: Decimal  # Refunds positive, payments negative; only the difference is transferred at finalization.
    final: bool  # The tax year's wash-sale window has ended and cash is settled under the current research convention.
    scenario_dependent: bool


def settle_years(years: Iterable[TaxYear], scenario: Scenario, observed_through: date
                 ) -> Iterator[Statement]:
    """Starting from the cash-funded build year, pass wash-sale-adjusted gains and losses by tax year. Losses and tax savings are negative."""
    st_carry = lt_carry = cumulative = ZERO
    previous_year = None
    final = True
    for income in years:
        if previous_year is not None and income.year != previous_year + 1:
            raise ValueError("Supply consecutive tax years, including years with zero activity.")
        previous_year = income.year
        st, lt = income.short_term - st_carry, income.long_term - lt_carry
        if scenario is Scenario.LIMITED_USE:
            # Net within each class first, then offset one class's net loss against the other's net gain; unused losses keep their character.
            if st < 0 < lt:
                offset = min(-st, lt)
                st, lt = st + offset, lt - offset
            elif lt < 0 < st:
                offset = min(-lt, st)
                st, lt = st - offset, lt + offset
            st_carry, lt_carry = max(-st, ZERO), max(-lt, ZERO)
            st, lt = max(st, ZERO), max(lt, ZERO)
        elif scenario is not Scenario.EXTERNAL_GAINS:
            raise ValueError(f"Unknown tax scenario: {scenario}")

        tax = st * ST_RATE + lt * LT_RATE + income.taxable_dividends * DIVIDEND_RATE
        cumulative += tax
        review_end = last_trading_day_of_year(income.year)
        if income.last_loss_day is not None:
            review_end = max(review_end, income.last_loss_day + WINDOW)
        # December losses may change with next year's purchases; this flag applies to both cumulative tax and wealth.
        final = final and income.basis_verified and observed_through >= review_end
        yield Statement(income.year, tax, cumulative, st_carry, lt_carry, final, review_end)
