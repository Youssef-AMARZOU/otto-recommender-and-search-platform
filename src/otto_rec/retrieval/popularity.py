"""M2: item popularity features — all-time and trailing windows.

Popularity is the retrieval baseline and a ranking feature. Windows are
computed relative to the corpus max timestamp so they mirror what serving
would see at "now".
"""

from __future__ import annotations

import time
from pathlib import Path


from otto_rec.features.etl import connect

WINDOWS_DAYS = (7, 30, 90)


def build_popularity(
    events_parquet: str | Path,
    out_path: str | Path,
    windows_days: tuple[int, ...] = WINDOWS_DAYS,
    memory_limit: str = "6GB",
    temp_dir: str = "data/tmp",
) -> dict:
    """Write item popularity parquet with all-time and windowed type counts."""
    started = time.perf_counter()
    events_path = Path(events_parquet).resolve().as_posix()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    con = connect(memory_limit=memory_limit, temp_dir=temp_dir)
    max_ts = con.execute(f"SELECT max(ts) FROM '{events_path}'").fetchone()[0]
    print(f"[popularity] max_ts={max_ts}", flush=True)

    window_filters = ",\n            ".join(
        f"""
        count(*) FILTER (WHERE ts >= {max_ts} - {days * 86400000} AND etype = 'clicks')::UINT32 AS clicks_{days}d,
        count(*) FILTER (WHERE ts >= {max_ts} - {days * 86400000} AND etype = 'carts')::UINT32 AS carts_{days}d,
        count(*) FILTER (WHERE ts >= {max_ts} - {days * 86400000} AND etype = 'orders')::UINT32 AS orders_{days}d,
        count(*) FILTER (WHERE ts >= {max_ts} - {days * 86400000})::UINT32 AS total_{days}d"""
        for days in windows_days
    )
    con.execute(
        f"""
        COPY (
            SELECT
                item,
                count(*) FILTER (WHERE etype = 'clicks')::UINT32 AS clicks_all,
                count(*) FILTER (WHERE etype = 'carts')::UINT32 AS carts_all,
                count(*) FILTER (WHERE etype = 'orders')::UINT32 AS orders_all,
                count(*)::UINT32 AS total_all,
                {window_filters}
            FROM '{events_path}'
            GROUP BY item
        ) TO '{out.resolve().as_posix()}' (FORMAT PARQUET)
        """
    )
    n_items = con.execute(f"SELECT count(*) FROM '{out.resolve().as_posix()}'").fetchone()[0]
    con.close()
    stats = {"n_items": int(n_items), "max_ts": int(max_ts), "elapsed_s": round(time.perf_counter() - started, 1)}
    print(f"[popularity] {stats}", flush=True)
    return stats


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build item popularity parquet")
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--out", default="data/processed/popularity.parquet")
    args = parser.parse_args()
    print(json.dumps(build_popularity(args.events, args.out), indent=2))
