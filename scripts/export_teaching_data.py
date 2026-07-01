"""Export real backtest/market data as JSON for the teaching document."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rl_finance.backtest import run_backtest
from rl_finance.benchmarks import BENCHMARKS
from rl_finance.data import DATA_DIR, SPLITS, download_prices, read_jsonl
from rl_finance.rewards import INVALID_REWARD, RETURN_SCALE, portfolio_return


def clean(v):
    return None if (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def main(out_path: Path) -> None:
    prices = download_prices()
    out = {"metrics": {}, "equity": {}}

    for split, (s, e) in SPLITS.items():
        step = 21 if split == "train" else 5
        out["metrics"][split] = {}
        curves = {}
        dates = None
        for name, pol in BENCHMARKS.items():
            bt = run_backtest(pol, prices, s, e)
            out["metrics"][split][name] = {k: clean(v) for k, v in bt["metrics"].items()}
            eq = bt["equity"].iloc[::step]
            if eq.index[-1] != bt["equity"].index[-1]:
                eq = eq._append(bt["equity"].iloc[[-1]])
            dates = [d.strftime("%Y-%m-%d") for d in eq.index]
            curves[name] = [round(float(v), 4) for v in eq.values]
        out["equity"][split] = {"dates": dates, "curves": curves}

    # Distribution of SPY forward 5-day returns in the training period —
    # this is the raw noise the reward signal lives inside.
    spy = prices.loc[slice(*SPLITS["train"]), "SPY"].pct_change(5).dropna()
    counts, edges = np.histogram(spy, bins=48, range=(-0.12, 0.12))
    out["spy5d"] = {
        "edges": [round(float(x), 4) for x in edges],
        "counts": [int(c) for c in counts],
        "mean": round(float(spy.mean()), 5),
        "std": round(float(spy.std()), 5),
    }

    # Concrete GRPO group demo on a real crisis-era day.
    train = read_jsonl(DATA_DIR / "train.jsonl")
    sample = next(s for s in train if s["date"].startswith("2008-10"))
    allocations = [
        ("All SPY", {"SPY": 1.0}),
        ("All QQQ", {"QQQ": 1.0}),
        ("Equal weight", {t: 1 / 12 for t in sample["fwd_returns"]}),
        ("60/40", {"SPY": 0.6, "IEF": 0.4}),
        ("Risk-off", {"TLT": 0.5, "GLD": 0.3, "CASH": 0.2}),
        ("All TLT", {"TLT": 1.0}),
        ("All CASH", {"CASH": 1.0}),
    ]
    rows = []
    for name, w in allocations:
        r = portfolio_return(w, sample["fwd_returns"])
        rows.append({
            "name": name,
            "portfolio_return": round(r, 5),
            "reward": round(RETURN_SCALE * math.log1p(r), 4),
        })
    rows.append({"name": "Malformed output", "portfolio_return": None,
                 "reward": INVALID_REWARD})
    rewards = np.array([r["reward"] for r in rows])
    for r, adv in zip(rows, (rewards - rewards.mean()) / rewards.std()):
        r["advantage"] = round(float(adv), 3)
    out["grpo_demo"] = {
        "date": sample["date"],
        "fwd_returns": sample["fwd_returns"],
        "rows": rows,
    }

    # A real prompt, verbatim, plus the matching next-week returns.
    p2010 = next(s for s in train if s["date"].startswith("2010-04"))
    out["sample_prompt"] = {"date": p2010["date"], "prompt": p2010["prompt"],
                            "fwd_returns": p2010["fwd_returns"]}

    out_path.write_text(json.dumps(out))
    print(f"{out_path} written: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("teaching_data.json"))
