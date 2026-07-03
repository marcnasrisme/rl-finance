"""Phase 2 environment tests.

The core trick: feed KNOWN strategies (equal-weight, momentum) through the
sandbox+grader as if a model had written them, and check the scores against
independently computed values.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from rl_finance.codegen import (
    EPISODE_TIMEOUT, INVALID_REWARD, PROMPT, REWARD_CLIP,
    make_code_reward, make_dataset_rows, make_windows, run_strategy,
    score_strategy, train_bounds,
)
from rl_finance.data import SPLITS, UNIVERSE, download_prices
from rl_finance.sandbox import (
    StrategyError, StrategyTimeout, clean_weights, compile_strategy,
    extract_code,
)


@pytest.fixture(scope="module")
def prices():
    return download_prices()


# --- realistic "model completions" ------------------------------------------------

EQUAL_WEIGHT = '''\
Rationale: naive diversification across all 12 assets, rebalanced weekly.

```python
def allocate(prices: pd.DataFrame) -> dict:
    tickers = list(prices.columns)
    return {t: 1.0 / len(tickers) for t in tickers}
```
'''

MOMENTUM_INVVOL = '''\
Rationale: hold positive-momentum assets, weighted by inverse volatility,
keep 10% cash as a buffer.

```python
def allocate(prices: pd.DataFrame) -> dict:
    rets = prices.pct_change().dropna()
    mom = prices.iloc[-1] / prices.iloc[-63] - 1
    vol = rets.tail(21).std() * np.sqrt(252)
    winners = mom[mom > 0].index
    if len(winners) == 0:
        return {}
    inv = 1.0 / vol[winners]
    w = 0.9 * inv / inv.sum()
    return w.to_dict()
```
'''

ALL_CASH = '''```python
def allocate(prices):
    return {}
```'''

CRASHES = '''```python
def allocate(prices):
    return {"SPY": prices["NOPE"].iloc[-1]}
```'''

INFINITE_LOOP = '''```python
def allocate(prices):
    x = 0
    while True:
        x += 1
    return {"SPY": 1.0}
```'''

TRIES_IMPORT = '''```python
import os

def allocate(prices):
    return {"SPY": 1.0}
```'''

LEVERAGED = '''```python
def allocate(prices):
    return {"SPY": 1.5}
```'''

SHORTS = '''```python
def allocate(prices):
    return {"SPY": 1.2, "TLT": -0.2}
```'''

NO_CODE = "I would buy low and sell high."


# --- sandbox ----------------------------------------------------------------------

def test_extracts_last_python_block():
    text = "reasoning...\n```python\nx = 1\n```\nmore\n" + EQUAL_WEIGHT
    assert "def allocate" in extract_code(text)


def test_extracts_bare_function_without_fences():
    text = "Here you go:\ndef allocate(prices):\n    return {}"
    assert extract_code(text).startswith("def allocate")


def test_no_code_raises():
    with pytest.raises(StrategyError):
        compile_strategy(NO_CODE)


def test_import_is_blocked():
    with pytest.raises(StrategyError):
        compile_strategy(TRIES_IMPORT)


def test_clean_weights_rejects_leverage_shorts_unknowns():
    for bad in ({"SPY": 1.5}, {"SPY": 1.2, "TLT": -0.2}, {"AAPL": 0.5}):
        with pytest.raises(StrategyError):
            clean_weights(bad, UNIVERSE)


def test_clean_weights_accepts_series_and_cash_remainder():
    w = clean_weights(pd.Series({"SPY": 0.6, "GLD": 0.2}), UNIVERSE)
    assert w == {"SPY": 0.6, "GLD": 0.2}  # remainder handled by caller as cash


# --- grading ----------------------------------------------------------------------

def test_equal_weight_score_matches_direct_computation(prices):
    fn = compile_strategy(EQUAL_WEIGHT)
    lo, hi = train_bounds(prices)
    start, end = lo, lo + 756
    weekly = run_strategy(fn, prices, start, end)

    # independent computation, no sandbox: equal-weight weekly returns w/ costs
    p = prices.reset_index(drop=True)
    expected = []
    prev_w = 0.0
    for t in range(start, end - 5, 5):
        week = p.iloc[t + 5] / p.iloc[t] - 1.0
        gross = week.mean()
        turnover = 12 * abs(1 / 12 - prev_w)
        expected.append(gross - 10 / 1e4 * turnover)
        prev_w = 1 / 12
    assert np.allclose(weekly.values, expected, atol=1e-12)


def test_momentum_strategy_runs_and_scores_sanely(prices):
    fn = compile_strategy(MOMENTUM_INVVOL)
    lo, _ = train_bounds(prices)
    s = score_strategy(fn, prices, lo, lo + 756)
    assert REWARD_CLIP[0] <= s <= REWARD_CLIP[1]


def test_all_cash_scores_zero(prices):
    fn = compile_strategy(ALL_CASH)
    assert score_strategy(fn, prices, *make_windows(prices, 1, seed=0)[0]) == 0.0


def test_crash_and_timeout_and_leverage_get_invalid_reward(prices):
    reward = make_code_reward(prices)
    win = [list(make_windows(prices, 1, seed=1)[0])]
    comps = [CRASHES, LEVERAGED, SHORTS, NO_CODE]
    out = reward(comps, [win] * len(comps))
    assert out == [INVALID_REWARD] * len(comps)


def test_infinite_loop_times_out(prices):
    fn = compile_strategy(INFINITE_LOOP)
    with pytest.raises(StrategyTimeout):
        run_strategy(fn, prices, *make_windows(prices, 1, seed=2)[0], timeout=1.0)


def test_no_dates_reach_the_strategy(prices):
    seen = {}

    def spy_fn(p):
        seen["index_type"] = type(p.index).__name__
        seen["rows"] = len(p)
        return {"SPY": 1.0}

    lo, _ = train_bounds(prices)
    run_strategy(spy_fn, prices, lo, lo + 20)
    assert seen["index_type"] == "RangeIndex"  # integers, no dates
    assert seen["rows"] >= 252                 # promised minimum history


def test_windows_stay_inside_train(prices):
    lo, hi = train_bounds(prices)
    train_end = pd.Timestamp(SPLITS["train"][1])
    for s, e in make_windows(prices, 200, seed=3):
        assert s >= lo and e <= hi
        assert prices.index[e - 1] <= train_end


def test_dataset_rows_shape(prices):
    rows = make_dataset_rows(prices, 5, seed=4)
    assert len(rows) == 5
    assert rows[0]["prompt"][0]["role"] == "user"
    assert "def allocate" in rows[0]["prompt"][0]["content"]
    assert len(rows[0]["window"]) == 1        # one window by default
    assert len(rows[0]["window"][0]) == 2     # [start, end]


def test_chat_format_completions_work(prices):
    reward = make_code_reward(prices)
    win = [list(make_windows(prices, 1, seed=5)[0])]
    out = reward([[{"role": "assistant", "content": EQUAL_WEIGHT}]], [win])
    assert REWARD_CLIP[0] <= out[0] <= REWARD_CLIP[1]


def test_minimax_windows_are_era_separated_and_score_is_min(prices):
    from rl_finance.codegen import MIN_ERA_GAP

    rows = make_dataset_rows(prices, 8, seed=6, windows_per_episode=2)
    for r in rows:
        (s1, _), (s2, _) = r["window"]
        assert abs(s1 - s2) >= MIN_ERA_GAP

    reward = make_code_reward(prices)
    wins = rows[0]["window"]
    fn_scores = [
        score_strategy(compile_strategy(EQUAL_WEIGHT), prices, s, e)
        for s, e in wins
    ]
    out = reward([EQUAL_WEIGHT], [wins])
    assert out[0] == pytest.approx(min(fn_scores))


def test_benign_imports_are_allowed(prices):
    code = '''```python
import numpy as np
import pandas as pd
from math import sqrt

def allocate(prices):
    n = len(prices.columns)
    assert sqrt(4) == 2.0
    vol = prices.pct_change().dropna().tail(21).std() * np.sqrt(252)
    return {t: 1.0 / n for t in prices.columns}
```'''
    fn = compile_strategy(code)
    lo, _ = train_bounds(prices)
    weekly = run_strategy(fn, prices, lo, lo + 30)
    assert len(weekly) > 0


def test_file_io_is_blocked_in_np_and_pd():
    for code in (
        '```python\ndef allocate(prices):\n    pd.read_parquet("data/prices.parquet")\n    return {}\n```',
        '```python\ndef allocate(prices):\n    np.load("x.npy")\n    return {}\n```',
    ):
        fn = compile_strategy(code)
        with pytest.raises(StrategyError):
            fn(pd.DataFrame({"SPY": [1.0] * 300}))
