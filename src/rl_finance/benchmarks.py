"""Baseline policies the RL agent must beat out-of-sample."""

from __future__ import annotations

import pandas as pd

from .data import CASH, UNIVERSE

EQUITY = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ"]


def equal_weight(date: pd.Timestamp, feats: pd.Series) -> dict[str, float]:
    w = 1.0 / len(UNIVERSE)
    return {t: w for t in UNIVERSE}


def spy_only(date: pd.Timestamp, feats: pd.Series) -> dict[str, float]:
    return {"SPY": 1.0}


def sixty_forty(date: pd.Timestamp, feats: pd.Series) -> dict[str, float]:
    return {"SPY": 0.6, "IEF": 0.4}


def momentum(date: pd.Timestamp, feats: pd.Series) -> dict[str, float]:
    """Top-3 assets by 3-month return, but only those with positive momentum;
    unfilled slots go to cash (classic time-series momentum filter)."""
    mom = {t: feats[t]["r_63d"] for t in UNIVERSE}
    top = sorted(mom, key=mom.get, reverse=True)[:3]
    weights = {t: 1 / 3 for t in top if mom[t] > 0}
    weights[CASH] = 1.0 - sum(weights.values())
    return weights


def inverse_vol(date: pd.Timestamp, feats: pd.Series) -> dict[str, float]:
    inv = {t: 1.0 / max(feats[t]["vol_21d"], 1e-6) for t in UNIVERSE}
    total = sum(inv.values())
    return {t: v / total for t, v in inv.items()}


BENCHMARKS = {
    "spy": spy_only,
    "equal_weight": equal_weight,
    "60_40": sixty_forty,
    "momentum": momentum,
    "inverse_vol": inverse_vol,
}
