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

## Phase 1 results (July 2026): two runs, two textbook findings

Both runs: Qwen3-4B + rank-32 LoRA, GRPO (8 generations/day, temp 1.0),
2 epochs over the 591 training days, evaluated on val (2020–2022) at temp 0.2.
Zero unparseable outputs in either run. Adapters archived on Drive
(`grpo_trader_lora`, `grpo_trader_lora_1b`).

| policy (val 2020–22) | CAGR | vol | Sharpe | max DD | turnover/wk |
|---|---|---|---|---|---|
| zero-shot (untrained) | 1.6% | 15.5% | 0.18 | −25.6% | 71% |
| run 1: reward = 20·log1p(fwd 5d return) | 7.2% | 25.6% | 0.40 | −32.6% | 4.8% |
| run 1b: reward = weekly Sharpe (clipped ±4) | −0.2% | 14.0% | 0.05 | −25.3% | 39% |
| SPY buy & hold | 7.2% | 25.0% | 0.40 | −33.7% | 1.3% |
| 60/40 | 3.4% | 14.9% | 0.30 | −21.1% | 2.4% |
| train-tangency portfolio (Markowitz on 2008–19) | −0.4% | 9.2% | 0.01 | −21.9% | 2.3% |

**Finding 1 — linear rewards collapse to corners.** E[20·log1p(wᵀR)] ≈ 20·wᵀμ
is ~linear in the weights, and a linear objective on a simplex is maximized at
a vertex. Run 1 duly collapsed to a static ~80/20 SPY/QQQ (its row is SPY's
row) and shed its reasoning tokens — RL deletes whatever the reward doesn't
pay for. The optimizer solved the problem we posed, not the one we meant.

**Finding 2 — single-regime Sharpe optimization overfits the regime.** Run 1b's
risk-adjusted reward worked as designed: vol halved, allocations diversified
(~57% equity / 25% bonds / 8% real / 9% cash on average) and became
state-responsive (defensive crouch in March 2020, real-asset tilt in 2022).
But the risk playbook 2008–2019 teaches — duration hedges everything — was
falsified by 2022's joint stock/bond crash, and reactive vol-timing is
whipsawed by V-recoveries. The learned policy scored ≈ the classical
train-tangency portfolio (0.05 vs 0.01): both "optimal," both wrong-footed.
Meanwhile the naive equity corner from run 1 beat both out of sample, because
the equity premium is the most regime-proof signal there is.

**Conclusion.** The GRPO machinery reliably finds the optimum of whatever
reward we write; the binding constraint is regime generalization, not
optimization. Phase 2 therefore rewards performance over multi-year windows
sampled from *different sub-periods*, so no single regime's playbook can win
training — and Phase 3 hands the model tools to reason about regime fragility
instead of fitting one history.
