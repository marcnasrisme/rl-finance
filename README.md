# RL Finance

Post-train an LLM into a portfolio manager with reinforcement learning (GRPO),
then backtest it against classic benchmarks.

## How it works

1. **State → prompt.** Each trading week, the model sees a compact table of
   market features (1d/1w/1m/3m returns, realized vol, distance from 52-week
   high) for 12 diversified ETFs.
2. **Action.** The model reasons briefly and outputs JSON portfolio weights
   (long-only, plus CASH).
3. **Reward.** The realized *forward 5-day log return* of that allocation.
   Malformed output gets a flat -1. We deliberately do NOT reward matching a
   hindsight-optimal portfolio — that would be supervised overfitting to noise.
4. **GRPO.** For each market day, G=8 allocations are sampled and advantages
   are computed *within the group*, so overall market direction cancels and
   the model learns relative allocation quality. No critic network needed
   (that's why GRPO over PPO — it also halves GPU memory, which matters on Colab).

## Anti-overfitting guardrails

- Strict time splits: **train 2008–2019, val 2020–2022, test 2023–2026**.
  The test split is only touched once, at the very end.
- ETF universe (no single-stock survivorship bias), CASH always available.
- Transaction costs (10 bps per unit turnover) and weight drift in the backtest.
- Benchmarks the agent must beat *risk-adjusted, out-of-sample*: SPY buy & hold,
  equal weight, 60/40, cross-sectional momentum, inverse vol.

## Layout

```
src/rl_finance/
  data.py        # yfinance download, features, sample construction, splits
  prompts.py     # market-state prompt builder
  rewards.py     # output parsing + GRPO reward function
  backtest.py    # walk-forward backtester (costs, drift)
  benchmarks.py  # baseline policies
  metrics.py     # Sharpe, Sortino, max drawdown, ...
scripts/
  prepare_data.py    # download prices, write data/{train,val,test}.jsonl
  run_benchmarks.py  # backtest baselines on all splits
notebooks/
  train_grpo_colab.ipynb  # GRPO training on Colab (A100), Unsloth + TRL
```

## Workflow

```bash
# local (no GPU needed)
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/run_benchmarks.py
```

Then push to GitHub, open `notebooks/train_grpo_colab.ipynb` in Colab
(A100 runtime), set `REPO_URL`, and run all cells. Training data (JSONL) is
committed to the repo so Colab needs no market-data download.

Model: `unsloth/Qwen3-4B-Instruct-2507` with rank-32 LoRA. On an L4/T4
runtime set `load_in_4bit=True` and lower `num_generations`.

## Expectations

The honest goal is to beat the naive baselines on **risk-adjusted** metrics
(Sharpe, max drawdown) out-of-sample — not to print money. Weekly-horizon
returns are extremely noisy; treat any val-split win as provisional until it
also holds on the untouched test split.
