"""Walk-forward daily backtester with transaction costs and weight drift.

A policy is any callable (date, feat_row) -> {ticker: weight}. The LLM
policy, the benchmarks, and any future strategy all share this interface,
so every number in the comparison table is produced by the same engine.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .data import CASH, compute_features
from .metrics import summarize

Policy = Callable[[pd.Timestamp, pd.Series], dict[str, float]]


def run_backtest(
    policy: Policy,
    prices: pd.DataFrame,
    start: str,
    end: str,
    rebalance_every: int = 5,
    cost_bps: float = 10.0,
) -> dict:
    """Simulate the policy with weekly rebalancing.

    - Weights drift with returns between rebalances (no free rebalancing).
    - Transaction cost = cost_bps * sum(|weight change|) at each rebalance.
    - The policy only ever sees features computed through the decision date.
    """
    feats = compute_features(prices)
    rets = prices.pct_change(fill_method=None)
    dates = prices.loc[start:end].index

    tickers = list(prices.columns)
    holdings = {t: 0.0 for t in tickers}
    holdings[CASH] = 1.0

    daily_returns = []
    weight_log = []
    turnovers = []

    for i, d in enumerate(dates):
        # Rebalance at the close of every Nth day, using that day's features.
        if i % rebalance_every == 0 and not feats.loc[d].isna().any():
            target = policy(d, feats.loc[d])
            target = {**{t: 0.0 for t in tickers}, CASH: 0.0, **target}
            turnover = sum(abs(target[k] - holdings.get(k, 0.0)) for k in target)
            cost = cost_bps / 1e4 * turnover
            holdings = dict(target)
            turnovers.append(turnover)
        else:
            cost = 0.0

        # Next day's P&L accrues to the holdings set today.
        if i + 1 < len(dates):
            nxt = dates[i + 1]
            gross = sum(
                holdings[t] * rets.loc[nxt, t] for t in tickers if holdings.get(t)
            )
            daily_returns.append((nxt, gross - cost))
            # Drift weights with realized returns.
            grown = {t: holdings[t] * (1 + rets.loc[nxt, t]) for t in tickers}
            grown[CASH] = holdings[CASH]
            total = sum(grown.values())
            holdings = {k: v / total for k, v in grown.items()}

        weight_log.append({"date": d, **holdings})

    ret_series = pd.Series(
        [r for _, r in daily_returns],
        index=[d for d, _ in daily_returns],
        name="return",
    )
    metrics = summarize(ret_series)
    metrics["avg_turnover"] = (
        sum(turnovers) / len(turnovers) if turnovers else 0.0
    )
    return {
        "returns": ret_series,
        "equity": (1 + ret_series).cumprod(),
        "weights": pd.DataFrame(weight_log).set_index("date"),
        "metrics": metrics,
    }
