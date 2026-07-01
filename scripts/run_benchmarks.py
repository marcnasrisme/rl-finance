"""Backtest all baseline policies on each split and print comparison tables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_finance.backtest import run_backtest
from rl_finance.benchmarks import BENCHMARKS
from rl_finance.data import SPLITS, download_prices
from rl_finance.metrics import format_table


def main() -> None:
    prices = download_prices()
    for split, (start, end) in SPLITS.items():
        results = {}
        for name, policy in BENCHMARKS.items():
            bt = run_backtest(policy, prices, start, end)
            results[name] = bt["metrics"]
        print(f"\n=== {split} ({start} -> {end}) ===")
        print(format_table(results))


if __name__ == "__main__":
    main()
