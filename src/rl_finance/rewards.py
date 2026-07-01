"""Parse model output into weights and score it against realized returns.

Reward design notes:
- We reward the *realized forward return* of the proposed portfolio, not
  similarity to a hindsight-optimal portfolio (which would be supervised
  overfitting to noise).
- GRPO normalizes advantages within a group of generations for the SAME
  prompt (same market day), so market direction cancels out and the model
  is trained on *relative* allocation quality.
- Malformed output gets a flat penalty well below any achievable market
  reward, so format compliance is learned quickly.
"""

from __future__ import annotations

import json
import math
import re

from .data import CASH, UNIVERSE

INVALID_REWARD = -1.0
RETURN_SCALE = 20.0  # 5-day log returns are ~±3%; scale into a useful range

_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_weights(text: str, tickers: list[str] | None = None) -> dict[str, float] | None:
    """Extract the last JSON object and normalize into valid weights.

    Returns None if there is no usable allocation. Weights are clipped to
    [0, 1]; if they sum to more than 1 they are rescaled; if less than 1 the
    remainder goes to CASH.
    """
    tickers = tickers or UNIVERSE
    matches = _JSON_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace("'", '"')
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    valid = set(tickers) | {CASH}
    weights: dict[str, float] = {}
    for k, v in obj.items():
        if k not in valid or not isinstance(v, (int, float)) or math.isnan(v):
            continue
        weights[k] = min(max(float(v), 0.0), 1.0)

    total = sum(weights.values())
    if total <= 0:
        return None
    if total > 1.0:
        weights = {k: v / total for k, v in weights.items()}
        total = 1.0
    weights[CASH] = weights.get(CASH, 0.0) + (1.0 - total)
    return weights


def portfolio_return(weights: dict[str, float], fwd_returns: dict[str, float]) -> float:
    """Realized forward return of the allocation (CASH earns 0)."""
    return sum(w * fwd_returns.get(t, 0.0) for t, w in weights.items() if t != CASH)


def score_completion(text: str, fwd_returns: dict[str, float]) -> float:
    """Reward for one generated completion."""
    weights = parse_weights(text)
    if weights is None:
        return INVALID_REWARD
    r = portfolio_return(weights, fwd_returns)
    return RETURN_SCALE * math.log1p(r)


def grpo_reward_func(completions, fwd_returns, **kwargs) -> list[float]:
    """TRL GRPOTrainer-compatible reward function.

    `completions` and `fwd_returns` arrive as parallel lists (extra dataset
    columns are forwarded by TRL). Handles both plain-text and chat-format
    completions.
    """
    rewards = []
    for comp, fwd in zip(completions, fwd_returns):
        if isinstance(comp, list):  # chat format: [{"role": ..., "content": ...}]
            comp = comp[-1]["content"]
        rewards.append(score_completion(comp, fwd))
    return rewards
