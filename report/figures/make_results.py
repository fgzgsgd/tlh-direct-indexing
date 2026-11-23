"""Build Section 3 figures and summary numbers from the final strategy run.
Run from the repository root:  python report/figures/make_results.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/strategy"
BASE = ROOT / "results/no_tlh"
OUT = Path(__file__).resolve().parent

d = pd.read_csv(RUN / "report_daily.csv", parse_dates=["date"]).set_index("date")
p = pd.read_csv(RUN / "report_periods.csv")
b = pd.read_csv(RUN / "report_benchmarks.csv")
cmp_ = json.load(open(RUN / "comparison.json"))
nd = pd.read_csv(BASE / "daily.csv", parse_dates=["date"]).set_index("date")
na = pd.read_csv(BASE / "annual.csv")

S = (1 + d.pretax_return.astype(float)).cumprod()
I = (1 + d.sp500_tr_return.fillna(0).astype(float)).cumprod()
A = (1 + nd.replica_pretax_return.astype(float)).cumprod().reindex(S.index)

def mdd(x): return float((x / x.cummax() - 1).min())
active = d.pretax_return.astype(float) - d.sp500_tr_return.fillna(0).astype(float)
roll_te = active.rolling(252).std() * np.sqrt(252)
month_te = active.groupby(active.index.to_period("M")).std() * np.sqrt(252)
month_te.index = month_te.index.to_timestamp(how="end").normalize()
exante = d.ex_ante_te.dropna().astype(float)

# ---- annual table
years = list(range(2016, 2026))
ann = cmp_["annual"]
rows = []
for y in years:
    pr = p[p.period == str(y)].iloc[0]
    br = b[(b.period == str(y)) & (b.benchmark == "sp500_tr")].iloc[0]
    ta = next(a for a in ann if a["year"] == y)["wash_A"]["tax_alpha_bp"]
    base_pre = float(na[na.year == y].replica_pretax_return.iloc[0])
    rows.append(dict(year=y, pretax=float(pr.pretax_return), base_pre=base_pre, bench=float(br.benchmark_return),
                     alpha=float(br.pretax_alpha), tax_alpha_bp=float(ta), te=float(br.realized_te),
                     turnover=float(pr.turnover), holdings=int(pr.holdings),
                     st_loss=float(pr.st_recognized_loss), lt_loss=float(pr.lt_recognized_loss)))
annual = pd.DataFrame(rows)

m = cmp_["metrics"]
full_b = b[(b.period == "full") & (b.benchmark == "sp500_tr")].iloc[0]
summary = dict(
    s_pre=m["wash_A"]["pretax_cagr"], s_after=m["wash_A"]["aftertax_cagr"], s_wealth=m["wash_A"]["aftertax_wealth"],
    s_tax_alpha_bp=m["wash_A"]["tax_alpha_cagr_bp"], s_te=m["wash_A"]["realized_te"]["sp500_tr"],
    s_mdd=m["wash_A"]["max_drawdown"], s_fees=m["wash_A"]["fees"],
    a_pre=m["no_tlh"]["pretax_cagr"], a_after=m["no_tlh"]["aftertax_cagr"], a_wealth=m["no_tlh"]["aftertax_wealth"],
    a_te=m["no_tlh"]["realized_te"]["sp500_tr"], a_mdd=m["no_tlh"]["max_drawdown"], a_fees=m["no_tlh"]["fees"],
    i_cagr=float(full_b.benchmark_cagr), i_mdd=mdd(I), s_alpha=float(full_b.pretax_alpha_cagr),
    exante_mean=float(exante.mean()), exante_max=float(exante.max()),
    roll_te_min=float(roll_te.min()), roll_te_max=float(roll_te.max()),
    month_te_min=float(month_te.min()), month_te_max=float(month_te.max()), month_te_median=float(month_te.median()),
    month_te_share_above_1=float((month_te > 0.01).mean()),
    avg_turnover_ex2016=float(annual.turnover[annual.year > 2016].mean()),
    turnover_2016=float(annual.turnover[annual.year == 2016].iloc[0]),
    holdings_end=int(d.holdings.iloc[-1]), holdings_min=int(d.holdings.min()), holdings_start=int(d.holdings.iloc[0]),
    base_holdings_end=int(nd.stocks.iloc[-1]),
    alpha_ex2024=float(np.prod([1 + r for r in annual.pretax[annual.year != 2024]]) ** (1 / 9)
                       - np.prod([1 + r for r in annual.bench[annual.year != 2024]]) ** (1 / 9)),
)
annual.to_csv(OUT / "results_annual.csv", index=False)
json.dump(summary, open(OUT / "results_summary.json", "w"), indent=1)

# ---- style
BLUE, ORANGE, GRAY, INK, MUTED, AQUA = "#2a78d6", "#eb6834", "#8a8a85", "#0b0b0b", "#52514e", "#1baf7a"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.edgecolor": "#c9c9c4", "axes.linewidth": 0.6, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK})

def panel_title(ax, t): ax.set_title(t, pad=6)
def year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Figure 2: pre-tax cumulative log return
fig, ax1 = plt.subplots(figsize=(6.5, 2.3))
ax1.plot(S.index, np.log(S), color=BLUE, lw=1.4, label="Strategy")
ax1.plot(A.index, np.log(A), color=ORANGE, lw=1.4, label="No-TLH")
ax1.plot(I.index, np.log(I), color=GRAY, lw=1.1, ls=(0, (4, 2)), label="S&P 500 TR")
ax1.axhline(0, color="#9a9a95", lw=0.6)
ax1.set_ylabel("log cumulative return")
panel_title(ax1, "Pre-tax cumulative log return")
ax1.legend(loc="upper left", fontsize=7, handlelength=2.2)
ax1.set_xlim(S.index[0], S.index[-1]); year_axis(ax1)
fig.tight_layout()
fig.savefig(OUT / "fig_performance.pdf"); plt.close(fig)

# Figure 3: tracking error, monthly and annual
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.3))
ax1.plot(month_te.index, month_te * 100, color=BLUE, lw=1.1, marker="o", ms=2.4, label="Realized, each month")
ax1.plot(exante.index, exante * 100, color=ORANGE, lw=0, marker="o", ms=2.4, label="Ex-ante, at rebalance")
ax1.axhline(1.0, color=GRAY, lw=0.9, ls=(0, (4, 2)))
ax1.set_ylim(0, 3.6); ax1.set_ylabel("% per year")
panel_title(ax1, "Monthly tracking error vs. S&P 500 TR")
ax1.legend(loc="upper left", fontsize=7, ncol=2)
ax1.set_xlim(d.index[0], d.index[-1]); year_axis(ax1)
x = np.arange(len(annual))
ax2.bar(x, annual.te * 100, 0.62, color=BLUE, zorder=3)
ax2.axhline(1.0, color=GRAY, lw=0.9, ls=(0, (4, 2)), zorder=4)
for xi, v in zip(x, annual.te * 100):
    ax2.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3, color=MUTED)
ax2.set_xticks(x, [f"'{str(y)[2:]}" for y in annual.year])
ax2.set_ylim(0, 1.8); ax2.set_ylabel("% per year")
ax2.grid(axis="x", visible=False)
panel_title(ax2, "Annual realized tracking error")
fig.tight_layout(w_pad=2.2)
fig.savefig(OUT / "fig_tracking.pdf"); plt.close(fig)

# Figure 4: number of holdings
fig, ax = plt.subplots(figsize=(6.5, 1.8))
ax.plot(d.index, d.holdings, color=BLUE, lw=1.3, label="Strategy")
ax.plot(nd.index, nd.stocks, color=ORANGE, lw=1.3, label="No-TLH")
ax.set_ylim(0, 560); ax.set_ylabel("stocks held")
panel_title(ax, "Number of holdings")
ax.legend(loc="lower left", fontsize=7, ncol=2)
ax.set_xlim(d.index[0], d.index[-1]); year_axis(ax)
fig.tight_layout()
fig.savefig(OUT / "fig_holdings.pdf"); plt.close(fig)
# ---- Portfolio aging
dec = []
first_seen, prev_trade = {}, None
from datetime import date as _date
with open(RUN / "decisions.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        td = _date.fromisoformat(r["trade_date"])
        lots = r["stage_two"]["lot_sales"]
        saving = sum(-l["tax_per_dollar"] * l["limit_usd"] for l in lots if l["tax_per_dollar"] < 0)
        # Lot-level unrealized gain from the optimizer's per-lot tax rate:
        # tax_per_dollar = tau * (1 - c - basis/price), so gain = MV * (c + tax_per_dollar / tau).
        # tau is inferred from the lot's first appearance (acquired at the previous trade date).
        gains = losses = 0.0
        for l in lots:
            key = (l["permno"], l["lot_id"])
            acq = first_seen.setdefault(key, prev_trade or td)
            tau = 0.238 if td > _date(acq.year + 1, acq.month, min(acq.day, 28)) else 0.408
            g = l["limit_usd"] * (0.0005 + l["tax_per_dollar"] / tau)
            if g >= 0: gains += g
            else: losses -= g
        nav_ = r["investable_nav_usd"]
        dec.append(dict(date=pd.Timestamp(r["signal_date"]), saving_nav=saving / nav_,
                        loss_stocks=len(r["loss_stock_directions"]),
                        unreal_gain_nav=gains / nav_, unreal_loss_nav=losses / nav_))
        prev_trade = td
dec = pd.DataFrame(dec).set_index("date")
start_nav = [1_000_000.0] + [float(p[p.period == str(y)].nav.iloc[0]) for y in years[:-1]]
aging = annual[["year", "st_loss", "lt_loss"]].copy()
aging["start_nav"] = start_nav
aging["st_pct"] = aging.st_loss / aging.start_nav
aging["lt_pct"] = aging.lt_loss / aging.start_nav
aging["basis_nav_end"] = [float(p[p.period == str(y)].basis_nav.iloc[0]) for y in years]
aging["unreal_nav_end"] = [float(p[p.period == str(y)].unrealized_nav.iloc[0]) for y in years]
dec_y = dec.groupby(dec.index.year).agg(saving_mean=("saving_nav", "mean"), loss_stocks_median=("loss_stocks", "median"))
aging = aging.merge(dec_y, left_on="year", right_index=True)
aging.to_csv(OUT / "aging_annual.csv", index=False)

# Figure 5: annual realized losses and annual tax alpha
fig, (a1, a4) = plt.subplots(1, 2, figsize=(6.5, 2.3))
x = np.arange(len(aging)); yrs = [f"'{str(y)[2:]}" for y in aging.year]
a1.bar(x, aging.st_pct * 100, 0.62, color=BLUE, label="Short-term", zorder=3)
a1.bar(x, aging.lt_pct * 100, 0.62, bottom=aging.st_pct * 100, color=ORANGE, label="Long-term", zorder=3,
       edgecolor="white", linewidth=0.8)
a1.set_xticks(x, yrs); a1.grid(axis="x", visible=False)
a1.set_ylabel("% of start-of-year NAV"); a1.legend(loc="upper right", fontsize=7)
panel_title(a1, "Realized losses by holding period")
ta = annual.tax_alpha_bp / 100
a4.bar(x, ta, 0.62, color=BLUE, zorder=3)
a4.axhline(0, color="#9a9a95", lw=0.6, zorder=2)
for xi, v in zip(x, ta):
    a4.text(xi, v + (0.08 if v >= 0 else -0.08), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
            fontsize=6.3, color=MUTED)
a4.set_xticks(x, yrs); a4.grid(axis="x", visible=False)
a4.set_ylim(-1, 4.2); a4.set_ylabel("% per year")
panel_title(a4, "Annual tax alpha")
fig.tight_layout(w_pad=2.2)
fig.savefig(OUT / "fig_aging.pdf"); plt.close(fig)

# Figure 6: cost basis and unrealized gains (top), unrealized losses on their own scale (bottom)
fig, (a2, a5) = plt.subplots(2, 1, figsize=(6.5, 3.1), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
a2.plot(d.index, d.basis_nav.astype(float) * 100, color=BLUE, lw=1.3)
a2.plot(dec.index, dec.unreal_gain_nav * 100, color=ORANGE, lw=1.3, marker="o", ms=1.8)
a2.set_ylim(0, 105); a2.set_ylabel("% of NAV")
a2.text(pd.Timestamp("2016-07-01"), 92, "Cost basis", color=BLUE, fontsize=7.5, va="bottom")
a2.text(pd.Timestamp("2018-01-01"), 15, "Unrealized gains", color=ORANGE, fontsize=7.5, va="bottom")
panel_title(a2, "Cost basis and unrealized gains")
a5.plot(dec.index, dec.unreal_loss_nav * 100, color=BLUE, lw=1.2, marker="o", ms=1.8)
a5.set_ylim(0, 6); a5.set_ylabel("% of NAV")
panel_title(a5, "Unrealized losses")
a5.set_xlim(d.index[0], d.index[-1]); year_axis(a5)
fig.tight_layout(h_pad=0.8)
fig.savefig(OUT / "fig_basis.pdf"); plt.close(fig)
dec.to_csv(OUT / "unrealized_monthly.csv")
print(aging.round(4).to_string())
print(dec.saving_nav.describe())
print(json.dumps(summary, indent=1)); print(annual.round(4).to_string())
