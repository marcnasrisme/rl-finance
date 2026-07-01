"""Price data download, feature computation, and RL sample construction.

Universe: diversified, liquid ETFs with history back to ~2007. Using ETFs
instead of single stocks avoids survivorship bias in the training data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

UNIVERSE = [
    "SPY",  # US large cap
    "QQQ",  # US tech / Nasdaq-100
    "IWM",  # US small cap
    "EFA",  # Developed intl equity
    "EEM",  # Emerging markets equity
    "TLT",  # 20y+ Treasuries
    "IEF",  # 7-10y Treasuries
    "LQD",  # Investment-grade credit
    "HYG",  # High-yield credit
    "GLD",  # Gold
    "DBC",  # Broad commodities
    "VNQ",  # US REITs
]
CASH = "CASH"  # earns 0%, always available

FEATURES = ["r_1d", "r_5d", "r_21d", "r_63d", "vol_21d", "pct_52w_high"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Strict time splits — never let training see validation/test dates.
SPLITS = {
    "train": ("2008-01-01", "2019-12-31"),
    "val": ("2020-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2026-12-31"),
}


def download_prices(
    tickers: list[str] = UNIVERSE,
    start: str = "2006-01-01",
    cache: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily adjusted close prices, cached to parquet."""
    cache = cache or DATA_DIR / "prices.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    prices = raw["Close"][tickers]
    # Keep only dates where every asset trades (DBC/HYG start ~2006-2007).
    prices = prices.dropna()
    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache)
    return prices


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-asset state features. Columns: MultiIndex (ticker, feature).

    Everything is computed from data available at the close of that day —
    no lookahead.
    """
    rets = prices.pct_change(fill_method=None)
    cols = {}
    for t in prices.columns:
        p = prices[t]
        cols[(t, "r_1d")] = p.pct_change(1, fill_method=None)
        cols[(t, "r_5d")] = p.pct_change(5, fill_method=None)
        cols[(t, "r_21d")] = p.pct_change(21, fill_method=None)
        cols[(t, "r_63d")] = p.pct_change(63, fill_method=None)
        cols[(t, "vol_21d")] = rets[t].rolling(21).std() * np.sqrt(252)
        cols[(t, "pct_52w_high")] = p / p.rolling(252).max() - 1
    feats = pd.DataFrame(cols)
    feats.columns = pd.MultiIndex.from_tuples(feats.columns, names=["ticker", "feature"])
    return feats


def forward_returns(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Return over the next `horizon` trading days, aligned to decision date."""
    return prices.shift(-horizon) / prices - 1


def build_samples(
    prices: pd.DataFrame,
    split: str,
    step: int = 5,
    horizon: int = 5,
) -> list[dict]:
    """One sample per rebalance date: state features + realized forward returns.

    `step=5` means weekly decisions. The forward return is the reward target;
    it is *only* used to score the model's output after generation, never
    shown in the prompt.
    """
    from .prompts import build_prompt

    feats = compute_features(prices)
    fwd = forward_returns(prices, horizon=horizon)
    start, end = SPLITS[split]
    dates = prices.loc[start:end].index

    samples = []
    for i in range(0, len(dates), step):
        d = dates[i]
        row = feats.loc[d]
        f = fwd.loc[d]
        if row.isna().any() or f.isna().any():
            continue
        samples.append(
            {
                "date": str(d.date()),
                "prompt": build_prompt(d, row),
                "fwd_returns": {t: round(float(f[t]), 6) for t in prices.columns},
            }
        )
    return samples


def write_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh]
