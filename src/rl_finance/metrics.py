"""Performance metrics computed from a daily-return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summarize(daily_returns: pd.Series) -> dict[str, float]:
    r = daily_returns.dropna()
    if len(r) == 0:
        return {}
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS

    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan

    downside = r[r < 0]
    sortino = (
        r.mean() / downside.std() * np.sqrt(TRADING_DAYS)
        if len(downside) > 1 and downside.std() > 0
        else np.nan
    )

    dd = equity / equity.cummax() - 1
    max_dd = dd.min()

    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "total_return": equity.iloc[-1] - 1,
    }


def format_table(results: dict[str, dict[str, float]]) -> str:
    """results: {strategy_name: metrics_dict} -> aligned text table."""
    cols = ["cagr", "vol", "sharpe", "sortino", "max_drawdown", "total_return", "avg_turnover"]
    header = f"{'strategy':<16}" + "".join(f"{c:>14}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        cells = []
        for c in cols:
            v = m.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append(f"{'-':>14}")
            elif c in ("sharpe", "sortino"):
                cells.append(f"{v:>14.2f}")
            else:
                cells.append(f"{v * 100:>13.1f}%")
        lines.append(f"{name:<16}" + "".join(cells))
    return "\n".join(lines)
