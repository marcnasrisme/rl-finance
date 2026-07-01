"""Build the market-state prompt shown to the model each decision day."""

from __future__ import annotations

import pandas as pd

from . import data as D

INSTRUCTIONS = """\
You are a systematic portfolio manager. Based on the market state below, \
allocate a long-only portfolio for the coming week across the listed ETFs \
and CASH (CASH earns 0%).

Rules:
- Weights must be between 0 and 1 and sum to 1.0.
- You may concentrate or diversify as you see fit.
- Think briefly (under 80 words), then output ONE JSON object mapping \
tickers to weights on the final line.

Example final line: {"SPY": 0.4, "TLT": 0.3, "GLD": 0.2, "CASH": 0.1}
"""

ASSET_DESC = {
    "SPY": "US large-cap equity",
    "QQQ": "US tech equity",
    "IWM": "US small-cap equity",
    "EFA": "Intl developed equity",
    "EEM": "Emerging mkt equity",
    "TLT": "20y+ US Treasuries",
    "IEF": "7-10y US Treasuries",
    "LQD": "IG corporate bonds",
    "HYG": "High-yield bonds",
    "GLD": "Gold",
    "DBC": "Commodities",
    "VNQ": "US REITs",
}


def build_prompt(date: pd.Timestamp, feat_row: pd.Series) -> str:
    """`feat_row` is one row of the (ticker, feature) MultiIndex frame."""
    lines = [
        INSTRUCTIONS,
        f"Date: {date:%Y-%m-%d} (month: {date:%B})",
        "",
        "Market state (returns/vol in %, distance from 52-week high in %):",
        f"{'Ticker':<7}{'Asset':<24}{'1d':>7}{'1w':>7}{'1m':>7}{'3m':>7}"
        f"{'Vol':>7}{'vs52wH':>8}",
    ]
    for t in D.UNIVERSE:
        r = feat_row[t]
        lines.append(
            f"{t:<7}{ASSET_DESC[t]:<24}"
            f"{r['r_1d'] * 100:>7.1f}{r['r_5d'] * 100:>7.1f}"
            f"{r['r_21d'] * 100:>7.1f}{r['r_63d'] * 100:>7.1f}"
            f"{r['vol_21d'] * 100:>7.1f}{r['pct_52w_high'] * 100:>8.1f}"
        )
    lines.append("")
    lines.append("Your allocation:")
    return "\n".join(lines)
