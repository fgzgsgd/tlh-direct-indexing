"""Add the raw records of the two securities SPY once held to the official distribution table for the SPY example."""

from tlh.crsp_events import read_distributions


def benchmark_actions(crsp):
    return read_distributions(crsp / "distributions_2015_to_2026_09.csv",
                              crsp / "spy_supplemental_distributions.csv")
