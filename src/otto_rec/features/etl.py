"""M1: OTTO JSONL -> columnar artifacts, streamed and out-of-core.

The raw train data is ~11 GB, far beyond comfortable in-memory pandas on a
16 GB machine, so DuckDB reads the JSONL with explicit columns and spills to
disk (data/tmp) while building:

- events.parquet      one row per interaction (session_code, pos, item, ts, etype)
- sessions_meta.parquet  one row per session (temporal split uses min_ts)
- split.json          temporal threshold: train = min_ts < threshold, val = rest

Two source schemas are auto-detected from the first record:

- parallel (competition train.jsonl): {"session", "ts"[..], "prev_items"[..], "type"[..]}
- events (kaggle.com/datasets/otto/recsys-dataset): {"session", "events": [{"aid", "ts", "type"}]}

Both normalize to the same events table; ``item`` is the aid cast to VARCHAR
in the events schema so every downstream artifact stays string-keyed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb

EVENT_TYPES = ("clicks", "carts", "orders")
MAX_OBJECT_SIZE = 33554432


def connect(memory_limit: str = "6GB", temp_dir: str = "data/tmp") -> duckdb.DuckDBPyConnection:
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{temp_dir}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def detect_schema(jsonl_path: str | Path) -> str:
    """Return 'events' (nested structs) or 'parallel' (column arrays)."""
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    if "events" in first:
        return "events"
    if "prev_items" in first:
        return "parallel"
    raise ValueError(f"unrecognized OTTO schema: keys {sorted(first)}")


def register_jsonl(con: duckdb.DuckDBPyConnection, jsonl_path: str | Path, schema: str | None = None) -> str:
    """Create the sessions_raw view; returns the schema that was used."""
    path = Path(jsonl_path).resolve().as_posix()
    schema = schema or detect_schema(jsonl_path)
    if schema == "events":
        columns = "{'session': 'BIGINT', 'events': 'STRUCT(aid BIGINT, ts BIGINT, type VARCHAR)[]'}"
    elif schema == "parallel":
        columns = (
            "{'session': 'VARCHAR', 'ts': 'BIGINT[]', 'prev_items': 'VARCHAR[]', 'type': 'VARCHAR[]'}"
        )
    else:
        raise ValueError(f"unknown schema: {schema}")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW sessions_raw AS
        SELECT * FROM read_json(
            '{path}',
            format = 'newline_delimited',
            columns = {columns},
            maximum_object_size = {MAX_OBJECT_SIZE}
        )
        """
    )
    return schema


def _events_sql(schema: str) -> str:
    if schema == "events":
        return """
        CREATE OR REPLACE TABLE events AS
        WITH numbered AS (
            SELECT row_number() OVER ()::UINT32 AS session_code, events
            FROM sessions_raw
        )
        SELECT
            session_code,
            g.pos::UINT16 AS pos,
            ev.aid::VARCHAR AS item,
            ev.ts AS ts,
            ev.type AS etype
        FROM numbered, UNNEST(events) WITH ORDINALITY AS g(ev, pos)
        """
    return """
    CREATE OR REPLACE TABLE events AS
    WITH numbered AS (
        SELECT row_number() OVER ()::UINT32 AS session_code, ts, prev_items, type
        FROM sessions_raw
    )
    SELECT
        session_code,
        i::UINT16 AS pos,
        prev_items[i] AS item,
        ts[i] AS ts,
        type[i] AS etype
    FROM numbered, UNNEST(GENERATE_SERIES(1, len(prev_items))) AS g(i)
    """


def export_artifacts(
    jsonl_path: str | Path,
    out_dir: str | Path,
    train_ratio: float = 0.8,
    memory_limit: str = "6GB",
    temp_dir: str = "data/tmp",
) -> dict:
    """Build events + session metadata + temporal split from raw JSONL."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    con = connect(memory_limit=memory_limit, temp_dir=temp_dir)
    schema = register_jsonl(con, jsonl_path)
    print(f"[etl] source schema: {schema}", flush=True)

    print("[etl] unnesting events (session_code, pos, item, ts, type) ...", flush=True)
    con.execute(_events_sql(schema))
    n_events = con.execute("SELECT count(*) FROM events").fetchone()[0]
    print(f"[etl] events: {n_events:,}", flush=True)

    print("[etl] session metadata + type counts ...", flush=True)
    con.execute(
        """
        CREATE OR REPLACE TABLE sessions_meta AS
        SELECT
            session_code,
            min(ts)::BIGINT AS min_ts,
            max(ts)::BIGINT AS max_ts,
            count(*)::UINT16 AS n_events,
            count(*) FILTER (WHERE etype = 'clicks')::UINT16 AS n_clicks,
            count(*) FILTER (WHERE etype = 'carts')::UINT16 AS n_carts,
            count(*) FILTER (WHERE etype = 'orders')::UINT16 AS n_orders
        FROM events
        GROUP BY session_code
        """
    )

    print(f"[etl] temporal split at train_ratio={train_ratio} ...", flush=True)
    threshold = con.execute(
        f"SELECT quantile_cont(min_ts, {train_ratio}) FROM sessions_meta"
    ).fetchone()[0]

    print("[etl] writing parquet ...", flush=True)
    con.execute(f"COPY events TO '{(out / 'events.parquet').as_posix()}' (FORMAT PARQUET)")
    con.execute(f"COPY sessions_meta TO '{(out / 'sessions_meta.parquet').as_posix()}' (FORMAT PARQUET)")

    stats = con.execute(
        """
        SELECT
            count(*) FILTER (WHERE min_ts < $t)::BIGINT AS n_train,
            count(*) FILTER (WHERE min_ts >= $t)::BIGINT AS n_val,
            min(min_ts)::BIGINT, max(max_ts)::BIGINT,
            quantile_cont(n_events, 0.5), max(n_events)
        FROM sessions_meta
        """,
        {"t": threshold},
    ).fetchone()
    split = {
        "schema": schema,
        "train_ratio": train_ratio,
        "threshold_min_ts": int(threshold),
        "n_train_sessions": int(stats[0]),
        "n_val_sessions": int(stats[1]),
        "min_ts": int(stats[2]),
        "max_ts": int(stats[3]),
        "median_session_len": float(stats[4]),
        "max_session_len": int(stats[5]),
        "n_events": int(n_events),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    (out / "split.json").write_text(json.dumps(split, indent=2))
    con.close()
    print(f"[etl] done in {split['elapsed_s']}s -> {out}", flush=True)
    return split


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OTTO JSONL -> parquet artifacts")
    parser.add_argument("--jsonl", default="data/raw/otto/train.jsonl")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()
    print(json.dumps(export_artifacts(args.jsonl, args.out, args.train_ratio), indent=2))
