"""把 CRSP 分配表的原始条款转换为账本事件，以 PERMNO、ex-date 和序号定位来源。"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from .actions import AccountAction, CaseReview, CorporateAction, PropertyDividend, SpinOff, SpinOffTerms, StockCashPayment
from .market import crsp_table
from .shares import as_decimal


@dataclass(frozen=True)
class SplitTerms:
    ratio: Fraction  # new shares / old shares
    source: str


ALCOA_REVERSE_SPLIT = "CRSP/StkDistributions/24643/2016-10-06/1"
AIMCO_REVERSE_SPLIT = "CRSP/StkDistributions/80711/2020-12-01/1"
SPIN_OFF_TABLE = Path(__file__).resolve().parents[1] / "data" / "crsp" / "spin_off_terms.csv"
SPLIT_TABLE = SPIN_OFF_TABLE.with_name("split_terms.csv")
ALCOA_SPIN_OFF = "CRSP/StkDistributions/24643/2016-11-01/1"

# 同一笔可选分配在 CRSP 有现金、股票两行。未作选择的账户采用发行人实际公布的配比。
AIMCO_CASH_ROW = "CRSP/StkDistributions/80711/2020-11-03/1"
AIMCO_STOCK_ROW = "CRSP/StkDistributions/80711/2020-11-03/2"
AIMCO_PAYMENT = StockCashPayment(
    f"{AIMCO_CASH_ROW}|{AIMCO_STOCK_ROW}", 80711, Decimal("1.02924"), Decimal("7.17076"), Decimal("30.98"),
    tuple(map(Decimal, ("0.0994375", "0.0189660", "0.6041631", "0.2774334"))))


def _issuer_spin_off(ref: str, row: dict[str, str]) -> SpinOffTerms | PropertyDividend:
    child = int(row["child_permno"])
    treatment = row["treatment"]
    sources = " | ".join(dict.fromkeys(url for url in
                         (row["share_source"], row["basis_source"], row["additional_source"]) if url))
    if treatment in {"property_dividend", "cash_dividend_at_fmv"}:
        first = date.fromisoformat(row["first_regular_date"]) if row["first_regular_date"] else None
        return PropertyDividend(ref, int(row["parent_permno"]), child, Fraction(row["shares_per_parent"]),
                                Decimal(row["child_fmv_usd"]), Decimal(row["taxable_amount_per_parent_usd"]),
                                sources, first, cash_at_fmv=treatment == "cash_dividend_at_fmv")
    if treatment not in {"spin_off", "taxable_exchange"}:
        raise ValueError(f"{ref}: unknown published treatment {treatment}.")
    basis = Decimal(row["issuer_child_basis_pct"]) / 100 if row["issuer_child_basis_pct"] else None
    exchange_fmv = Decimal(row["child_fmv_usd"]) if treatment == "taxable_exchange" else None
    return SpinOffTerms(child, Fraction(row["shares_per_parent"]), basis, sources, exchange_fmv=exchange_fmv)


def read_distributions(*paths: Path, spin_off_terms: Mapping[str, SpinOffTerms] | None = None,
                       split_terms: Mapping[str, SplitTerms] | None = None
                       ) -> dict[date, list[AccountAction]]:
    """合并 WRDS 分批导出；同一事件有冲突时指出原始键，供核对。"""
    events = {}
    for path in paths:
        with crsp_table(path) as rows:
            for row in rows:
                key = date.fromisoformat(row["disexdt"]), int(row["permno"]), int(row["disseqnbr"])
                terms = {k: v for k, v in row.items() if k.startswith("dis")}
                if key in events and events[key] != terms:
                    raise ValueError(f"Conflicting CRSP distribution rows: {key} in {path.name}")
                events[key] = terms
    by_day = defaultdict(list)
    with crsp_table(SPLIT_TABLE) as rows:
        ratios = {f"CRSP/StkDistributions/{r['permno']}/{r['ex_date']}/{r['sequence']}":
                  SplitTerms(Fraction(r["new_per_old"]), r["source"]) for r in rows}
    ratios.update(split_terms or {})
    spin_offs = spin_off_terms or {}
    with crsp_table(SPIN_OFF_TABLE) as rows:
        published = {(date.fromisoformat(r["ex_date"]), int(r["parent_permno"]), int(r["sequence"])): r for r in rows}
    joint_children = defaultdict(set)
    for r in published.values():
        if group := r["basis_group"]:
            joint_children[group].add(int(r["child_permno"]))
    for (day, p, sequence), row in sorted(events.items()):
        ref = f"CRSP/StkDistributions/{p}/{day}/{sequence}"
        if ref in (AIMCO_CASH_ROW, AIMCO_STOCK_ROW):
            cash, stock = (events.get((day, p, i)) for i in (1, 2))
            if cash is None or stock is None or cash.get("dispaydt") != stock.get("dispaydt"):
                raise ValueError(f"{ref}: paired Aimco distribution records are missing or disagree on pay date.")
            if ref == AIMCO_CASH_ROW:
                by_day[day].append(AIMCO_PAYMENT)  # ex-date 按最终配比一次交付。
            continue
        terms = spin_offs.get(ref)
        issuer = published.get((day, p, sequence))
        if issuer and issuer["treatment"] == "foreign_cash_dividend":
            # 实际交付境外上市股票时，整项权益按 ex-date 美元行情一次性记普通股息。
            if row["distype"] != "SP" or issuer["close_date"] != day.isoformat() or not issuer["child_close_usd"]:
                raise ValueError(f"{ref}: foreign share distribution requires its ex-date USD close.")
            price = Fraction(issuer["child_close_usd"])
            amount = as_decimal(Fraction(issuer["shares_per_parent"]) * price)
            by_day[day].append(CorporateAction(ref, p, "CASH_DIVIDEND", amount))
            continue
        if issuer:
            if terms is None:
                terms = _issuer_spin_off(ref, issuer)
            elif terms.child != int(issuer["child_permno"]):
                raise ValueError(f"{ref}: spin-off terms conflict with published child PERMNO.")
        event = _classify(p, ref, row, terms, ratios.get(ref))
        if issuer and (group := issuer["basis_group"]) and isinstance(event, SpinOff):
            event = replace(event, joint_children=frozenset(joint_children[group]))
        by_day[day].append(event)
    return dict(by_day)


def _classify(p: int, ref: str, r: dict[str, str], terms: SpinOffTerms | PropertyDividend | None = None,
              split: SplitTerms | None = None
              ) -> CorporateAction | CaseReview | SpinOff | PropertyDividend:
    group, detail, tax, payment = (r[k] for k in ("distype", "disdetailtype", "distaxtype", "dispaymenttype"))
    if split is not None:
        if ((group, tax, payment) != ("FRS", "N", "SS") or detail not in {"STKSPL", "REVSPL"}
                or not split.source or split.ratio <= 0):
            raise ValueError(f"{ref}: invalid verified split terms or incompatible event.")
        return CorporateAction(ref, p, "SPLIT", split.ratio, ratio_source=split.source)
    if terms is not None:
        # Alcoa 2016 的原始行为母公司 STKDIV。只有逐笔提供发行人条款才重分类。
        kind = group, detail, tax, payment
        child = terms.child if isinstance(terms, SpinOffTerms) else terms.spin_off_child
        crsp_child = int(r.get("dispermno") or 0)
        if (kind not in {("SP", "SECSO", "N", "OS"), ("SP", "SECNOD", "N", "OS"),
                         ("SP", "SECEX", "N", "OS"), ("FRS", "STKDIV", "N", "SS")}
                or crsp_child not in (0, child)):
            raise ValueError(f"{ref}: spin-off terms conflict with CRSP event or child PERMNO.")
        return SpinOff(ref, p, terms) if isinstance(terms, SpinOffTerms) else terms
    if detail == "SECRD":
        return CorporateAction(ref, p, "RIGHTS_WAIVED", Decimal(0))
    taxable_dividend = tax == "D" and detail in {"CDIV", "SDIV", "CDPSR"}
    # CRSP 的 DisDivAmt 使用 USD/share，包括以外币支付的现金股息。
    cash_payment = payment == "USD" or (payment == "FX" and taxable_dividend)
    if cash_payment and group in {"CD", "SD", "ROC"} and r["disdivamt"]:
        amount = Decimal(r["disdivamt"])
        if taxable_dividend:
            return CorporateAction(ref, p, "CASH_DIVIDEND", amount)
        if tax == "R" and detail in {"ROC", "SDROC"}:
            return CorporateAction(ref, p, "ROC", amount)
        # Gilead 这 35 笔季度现金股息缺股东税务拆分，按项目约定走普通股息税账。
        if (p == 77274 and (group, detail, tax) == ("CD", "CDIV", "N")
                and r["disordinaryflg"] == "Y" and r["disfreqtype"] == "Q"
                and "2015-09-14" <= r["disexdt"] <= "2024-03-14"):
            return CorporateAction(ref, p, "CASH_DIVIDEND", amount)
        # 可核实的现金先入账；对每个 lot 的 basis 或税务性质留给逐案核实。
        return CorporateAction(ref, p, "UNCLASSIFIED_CASH", amount)

    if group == "FRS" and payment == "SS" and tax == "N" and r["disfacshr"]:
        # 事件表记录每股新增的股数。例如 4-for-1 的 DisFacShr 是 3。
        factor = 1 + Decimal(r["disfacshr"])
        if factor > 0 and detail in {"STKSPL", "REVSPL", "STKDIV"}:
            kind = "NONTAXABLE_STOCK_DIVIDEND" if detail == "STKDIV" else "SPLIT"
            return CorporateAction(ref, p, kind, factor, ratio_source="CRSP DisFacShr", unverified_ratio=factor < 1)

    return CaseReview(ref, p, "ECONOMIC_PENDING", f"CRSP {group}/{detail}/{tax}/{payment}", None,
                      spin_off_child=int(r.get("dispermno") or 0) if detail == "SECSO" else None)
