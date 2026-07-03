"""Phase 2 environment: the model writes an allocate() function; we grade it.

One episode = one completion (rationale + code) graded by walking the
strategy forward over a hidden, randomly-drawn multi-year window of the
training period. Reward = annualized Sharpe of the ~150 weekly net returns.

Design notes (motivated by Phase 1's findings, see README):
- Reward over a whole window, not one week -> order-of-magnitude better
  signal-to-noise per episode.
- Windows are drawn from different eras; within a GRPO group all completions
  share one window (group baseline cancels window luck), across groups the
  era varies (regime-fragile logic cannot win training on average).
- The strategy function receives an integer-indexed price frame — no dates —
  so pretraining knowledge of specific history is structurally useless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SPLITS, UNIVERSE
from .sandbox import StrategyError, clean_weights, compile_strategy, time_limit

# --- episode parameters ---------------------------------------------------------

WINDOW_DAYS = 756       # ~3 years of trading days per evaluation window
MIN_HISTORY = 252       # the strategy is guaranteed >= 1y of history at first call
REBALANCE_EVERY = 5     # weekly decisions, matching Phase 1 and the backtester
COST_BPS = 10.0         # transaction cost per unit of turnover
EPISODE_TIMEOUT = 20.0  # wall-clock budget for one full walk-forward (seconds)

INVALID_REWARD = -5.0   # crash / timeout / contract violation
REWARD_CLIP = (-3.0, 6.0)  # a 3y Sharpe outside this range is noise or an exploit

PROMPT = f"""\
You are a quantitative developer. Write a Python trading strategy for a
12-ETF universe plus CASH (CASH earns 0%).

The universe: SPY (US large-cap), QQQ (US tech), IWM (US small-cap),
EFA (intl developed), EEM (emerging markets), TLT (20y+ Treasuries),
IEF (7-10y Treasuries), LQD (IG credit), HYG (high-yield credit),
GLD (gold), DBC (commodities), VNQ (REITs).

Implement exactly this function:

    def allocate(prices: pd.DataFrame) -> dict:
        \"\"\"
        prices: daily close prices up to and including today.
                Columns = the 12 tickers above. Rows = trading days,
                oldest first, INTEGER-indexed (no dates available).
                At least {MIN_HISTORY} rows of history are guaranteed.
        Returns: dict mapping tickers to weights — long-only, each in
                 [0, 1], summing to at most 1.0. Any remainder is
                 automatically held as CASH.
        \"\"\"

How you will be scored: your function is called once a week, walking
forward through a multi-year evaluation period that you cannot see, and
the portfolio is scored on its ANNUALIZED SHARPE RATIO net of 0.1%
transaction costs per unit of turnover. The evaluation window is drawn
at random from different market eras (bull runs, crashes, rising and
falling rates) — your logic must be robust across regimes, not tuned to
one. High turnover is penalized through costs.

Rules:
- Only `np` (numpy), `pd` (pandas) and `math` are available. No imports,
  no file or network access.
- Keep it fast: well under 2 seconds per call.
- If your code raises an exception, times out, returns malformed weights,
  or violates long-only, the episode scores the minimum reward.

First briefly state your strategy rationale (3-5 lines), then give ONE
```python code block containing only the allocate function (helper
functions inside the block are fine).
"""


# --- windows ---------------------------------------------------------------------

def train_bounds(prices: pd.DataFrame) -> tuple[int, int]:
    """Positional [lo, hi) range inside which training windows must fall."""
    train_end = pd.Timestamp(SPLITS["train"][1])
    hi = int(prices.index.searchsorted(train_end, side="right"))
    return MIN_HISTORY, hi


def make_windows(
    prices: pd.DataFrame,
    n: int,
    seed: int,
    window_days: int = WINDOW_DAYS,
) -> list[tuple[int, int]]:
    """Draw n random evaluation windows (positional index pairs) within train."""
    lo, hi = train_bounds(prices)
    last_start = hi - window_days
    if last_start <= lo:
        raise ValueError("training period too short for this window length")
    rng = np.random.default_rng(seed)
    starts = rng.integers(lo, last_start, size=n)
    return [(int(s), int(s) + window_days) for s in starts]


# --- walk-forward grading ---------------------------------------------------------

def run_strategy(
    fn,
    prices: pd.DataFrame,
    start: int,
    end: int,
    rebalance_every: int = REBALANCE_EVERY,
    cost_bps: float = COST_BPS,
    timeout: float = EPISODE_TIMEOUT,
) -> pd.Series:
    """Walk fn forward over prices[start:end]; return weekly net returns.

    The function only ever receives rows [0, t] — the future is physically
    absent. Prices are re-indexed to integers so no dates leak. Raises
    StrategyError (incl. StrategyTimeout) on any contract violation.
    """
    p = prices.reset_index(drop=True)  # strip dates
    weekly_returns = []
    prev = {t: 0.0 for t in UNIVERSE}

    with time_limit(timeout):
        for t in range(start, end - rebalance_every, rebalance_every):
            try:
                raw = fn(p.iloc[: t + 1])
            except StrategyError:
                raise
            except BaseException as e:
                raise StrategyError(f"allocate() raised {type(e).__name__}: {e}") from e
            weights = clean_weights(raw, UNIVERSE)

            turnover = sum(
                abs(weights.get(k, 0.0) - prev.get(k, 0.0))
                for k in set(weights) | set(prev)
            )
            week = p.iloc[t + rebalance_every] / p.iloc[t] - 1.0
            gross = sum(w * week[tkr] for tkr, w in weights.items())
            weekly_returns.append(gross - cost_bps / 1e4 * turnover)
            prev = weights

    return pd.Series(weekly_returns)


def weekly_metrics(weekly: pd.Series) -> dict[str, float]:
    """Metrics for a weekly-return series (52 periods/year)."""
    equity = (1 + weekly).cumprod()
    years = len(weekly) / 52
    dd = equity / equity.cummax() - 1
    std = weekly.std()
    return {
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
        "vol": float(std * np.sqrt(52)),
        "sharpe": float(weekly.mean() / std * np.sqrt(52)) if std > 1e-9 else 0.0,
        "max_drawdown": float(dd.min()),
        "total_return": float(equity.iloc[-1] - 1),
    }


def score_strategy(fn, prices: pd.DataFrame, start: int, end: int) -> float:
    """One episode's reward: clipped annualized Sharpe over the window."""
    weekly = run_strategy(fn, prices, start, end)
    std = weekly.std()
    if std < 1e-9:  # all-cash / zero-risk: no reward, no penalty
        return 0.0
    sharpe = float(weekly.mean() / std * np.sqrt(52))
    return float(np.clip(sharpe, *REWARD_CLIP))


# --- TRL glue ----------------------------------------------------------------------

MIN_ERA_GAP = 504  # when an episode has 2+ windows, force starts >= 2y apart


def make_dataset_rows(
    prices: pd.DataFrame,
    n_episodes: int,
    seed: int,
    windows_per_episode: int = 1,
) -> list[dict]:
    """Rows for GRPOTrainer: identical prompt, per-row hidden window set.

    windows_per_episode > 1 enables minimax scoring (see make_code_reward):
    the episode's reward is the WORST window's Sharpe, and the windows are
    forced to come from different eras — punishing era-memorization harder
    than averaging does.
    """
    rng = np.random.default_rng(seed)
    lo, hi = train_bounds(prices)
    last_start = hi - WINDOW_DAYS
    rows = []
    for _ in range(n_episodes):
        starts: list[int] = []
        while len(starts) < windows_per_episode:
            s = int(rng.integers(lo, last_start))
            if all(abs(s - t) >= MIN_ERA_GAP for t in starts):
                starts.append(s)
        rows.append({
            "prompt": [{"role": "user", "content": PROMPT}],
            "window": [[s, s + WINDOW_DAYS] for s in starts],
        })
    return rows


def make_code_reward(prices: pd.DataFrame):
    """Reward function (closure over prices) for TRL's GRPOTrainer.

    `window` (a list of [start, end] pairs) arrives per-completion via the
    extra dataset column. With one window the reward is that window's
    clipped Sharpe; with several it is the minimum across them (minimax:
    to score well you must be robust in EVERY era you were dealt).
    """

    def code_reward(completions, window, **kwargs) -> list[float]:
        rewards = []
        for comp, wins in zip(completions, window):
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            try:
                fn = compile_strategy(text)
                rewards.append(
                    min(score_strategy(fn, prices, s, e) for s, e in wins)
                )
            except StrategyError:
                rewards.append(INVALID_REWARD)
        return rewards

    return code_reward
