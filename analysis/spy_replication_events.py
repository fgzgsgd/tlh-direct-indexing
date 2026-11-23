"""SPY 示例在正式分配表上补入 ETF 曾持有的两只证券的原始记录。"""

from tlh.crsp_events import read_distributions


def benchmark_actions(crsp):
    return read_distributions(crsp / "distributions_2015_to_2026_09.csv",
                              crsp / "spy_supplemental_distributions.csv")
