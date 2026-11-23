"""按研究情景净抵已实现损益和股息税，计算每个税年的应缴税或节税。"""

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
    """使用 lot 的税务持有起点；满一周年的次日才算 LT。"""
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
    """gross 用于项目股息税。发行人的四项分类保留供对账。"""
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
    gross: Decimal  # 按获配前的母股股数计算应税股息；可与新 lot 的 FMV 有列示差异。


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
    review_end: date  # 该税年最后一个必须观察的日期（含当日）。

    def after_tax_wealth(self, nav: Decimal, settled_cash_flow: Decimal = ZERO) -> Decimal:
        return nav - (self.cumulative_tax + settled_cash_flow)


@dataclass(frozen=True)
class TaxSettlement:
    day: date
    tax_year: int
    net_tax: Decimal
    cash_flow: Decimal  # 退税为正，付款为负；定稿时只划差额。
    final: bool  # 本税年的 wash-sale 观察期已结束，按当前研究口径结清现金。
    scenario_dependent: bool


def settle_years(years: Iterable[TaxYear], scenario: Scenario, observed_through: date
                 ) -> Iterator[Statement]:
    """从现金建仓年开始，按税年传入含 wash-sale 调整的损益。亏损及节税取负。"""
    st_carry = lt_carry = cumulative = ZERO
    previous_year = None
    final = True
    for income in years:
        if previous_year is not None and income.year != previous_year + 1:
            raise ValueError("Supply consecutive tax years, including years with zero activity.")
        previous_year = income.year
        st, lt = income.short_term - st_carry, income.long_term - lt_carry
        if scenario is Scenario.LIMITED_USE:
            # 同类先汇总，再用一类的净亏损抵另一类的净利得；未用亏损保留性质。
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
        # 12 月的亏损可能因次年买入而变化；这个标记同时适用于累计税款和财富。
        final = final and income.basis_verified and observed_through >= review_end
        yield Statement(income.year, tax, cumulative, st_carry, lt_carry, final, review_end)
