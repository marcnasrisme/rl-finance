"""Download prices and export GRPO training/eval samples as JSONL."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_finance.data import DATA_DIR, SPLITS, build_samples, download_prices, write_jsonl


def main(refresh: bool = False) -> None:
    prices = download_prices(refresh=refresh)
    print(f"Prices: {prices.shape[0]} days x {prices.shape[1]} assets "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")

    for split in SPLITS:
        samples = build_samples(prices, split)
        path = DATA_DIR / f"{split}.jsonl"
        write_jsonl(samples, path)
        print(f"{split}: {len(samples)} samples -> {path}")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
