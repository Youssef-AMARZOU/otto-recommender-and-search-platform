"""Generate a synthetic OTTO-shaped JSONL for local pipeline validation.

Not a substitute for the real dataset: it only guarantees the schema
(session / ts / prev_items / type) and chronological session order so every
pipeline stage can be exercised before the 11 GB download lands.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TYPES = ["clicks"] * 8 + ["carts"] * 1 + ["orders"] * 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sample/sample.jsonl")
    parser.add_argument("--sessions", type=int, default=50000)
    parser.add_argument("--items", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)  # noqa: S311 - seeded synthetic data generator, not cryptographic
    weights = [1.0 / (rank**0.9) for rank in range(1, args.items + 1)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ts = 1_660_000_000_000
    with out.open("w", encoding="utf-8") as handle:
        for _ in range(args.sessions):
            length = min(max(1, int(rng.expovariate(1 / 4.5))), 60)
            items = [f"item_{rng.choices(range(args.items), weights=weights)[0]:05d}" for _ in range(length)]
            types = [rng.choice(TYPES) for _ in range(length)]
            timestamps = []
            for _ in range(length):
                ts += rng.randint(1_000, 300_000)
                timestamps.append(ts)
            handle.write(
                json.dumps(
                    {
                        "session": f"synth_{rng.randint(0, 10**9):09d}",
                        "ts": timestamps,
                        "prev_items": items,
                        "type": types,
                    }
                )
                + "\n"
            )
    print(f"wrote {args.sessions} sessions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
