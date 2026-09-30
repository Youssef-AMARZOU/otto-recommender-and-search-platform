"""M2: co-visitation candidate generation over events.parquet.

Directed item-to-item counts (a before b inside a session, gap-capped),
filtered to significant pairs and truncated to top-N neighbours per item.
Built bucket-by-bucket over session hashes so the self-join's build side
stays small on the full 216M-event OTTO corpus (single-shot join OOMs);
partials are merged once at the end.
"""

from __future__ import annotations

import time
from pathlib import Path


from otto_rec.features.etl import connect


def build_co_visitation(
    events_parquet: str | Path,
    out_path: str | Path,
    min_co_occurrence: int = 2,
    top_n: int = 100,
    max_gap: int = 50,
    n_buckets: int = 8,
    bucket_ids: list[int] | None = None,
    merge_partitions: int = 16,
    merge_start: int = 0,
    merge_memory_limit: str = "2GB",
    merge_threads: int = 2,
    memory_limit: str = "6GB",
    temp_dir: str = "data/tmp",
) -> dict:
    """Write neighbour parquet (i, j, weight): item i -> ranked neighbours j.

    n_buckets partitions sessions (build phase); merge_partitions partitions
    pairs by i (merge phase). Each merge partition runs as two fresh
    connections — count aggregation, then top-N window — so neither operator
    pool ever exceeds merge_memory_limit. bucket_ids limits the build to a
    subset of session buckets; merge_start skips already-written merge
    partitions (resume). The merge consumes all partials, so omitted buckets
    must already exist.
    """
    started = time.perf_counter()
    events_path = Path(events_parquet).resolve().as_posix()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    con = connect(memory_limit=memory_limit, temp_dir=temp_dir)
    con.execute(f"CREATE OR REPLACE VIEW ev AS SELECT * FROM '{events_path}'")

    active_buckets = list(range(n_buckets)) if bucket_ids is None else list(bucket_ids)
    partial_paths = [
        (out.parent / f"_covis_partial_{b}.parquet").resolve().as_posix()
        for b in range(n_buckets)
    ]
    print(
        f"[covis] pairing events (min_co={min_co_occurrence}, top_n={top_n}, "
        f"max_gap={max_gap}, buckets={active_buckets}) ...",
        flush=True,
    )
    try:
        for bucket in active_buckets:
            bucket_started = time.perf_counter()
            con.execute(
                f"""
                CREATE OR REPLACE TABLE bkt AS
                SELECT session_code, pos, item FROM ev
                WHERE hash(session_code) % {n_buckets} = {bucket}
                """
            )
            con.execute(
                f"""
                COPY (
                    SELECT a.item AS i, b.item AS j, count(*) AS c
                    FROM bkt a
                    JOIN bkt b
                      ON a.session_code = b.session_code
                     AND b.pos > a.pos
                     AND b.pos - a.pos <= {max_gap}
                    GROUP BY 1, 2
                ) TO '{partial_paths[bucket]}' (FORMAT PARQUET)
                """
            )
            n_rows = con.execute(f"SELECT count(*) FROM '{partial_paths[bucket]}'").fetchone()[0]
            print(
                f"[covis] bucket {bucket}: {n_rows} rows "
                f"in {round(time.perf_counter() - bucket_started, 1)}s",
                flush=True,
            )

        missing = [p for p in partial_paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"missing covis partials: {missing}")
        partial_rows = sum(
            int(con.execute(f"SELECT count(*) FROM '{p}'").fetchone()[0]) for p in partial_paths
        )
        partial_list = ", ".join(f"'{p}'" for p in partial_paths)
    finally:
        con.close()

    final_paths = [
        (out.parent / f"_covis_final_{p}.parquet").resolve().as_posix()
        for p in range(merge_partitions)
    ]
    # Fresh connections per phase: DuckDB keeps buffer-pool pages alive across
    # statements on one connection, and on a tight-RAM host a shared pool
    # cannot survive repeated large aggregations; splitting count aggregation
    # from the top-N window also keeps both operator states inside the limit.
    for part in range(merge_start, merge_partitions):
        merge_started = time.perf_counter()
        grouped = (out.parent / f"_covis_grouped_{part}.parquet").resolve().as_posix()
        gcon = connect(memory_limit=merge_memory_limit, temp_dir=temp_dir)
        try:
            gcon.execute(f"SET threads={merge_threads}")
            gcon.execute(
                f"""
                COPY (
                    SELECT i, j, sum(c)::BIGINT AS c
                    FROM read_parquet([{partial_list}])
                    WHERE hash(i) % {merge_partitions} = {part}
                    GROUP BY 1, 2
                    HAVING sum(c) >= {min_co_occurrence}
                ) TO '{grouped}' (FORMAT PARQUET)
                """
            )
        finally:
            gcon.close()
        wcon = connect(memory_limit=merge_memory_limit, temp_dir=temp_dir)
        try:
            wcon.execute(f"SET threads={merge_threads}")
            wcon.execute(
                f"""
                COPY (
                    SELECT i, j, c::DOUBLE AS weight
                    FROM '{grouped}'
                    QUALIFY row_number() OVER (PARTITION BY i ORDER BY c DESC, j ASC) <= {top_n}
                ) TO '{final_paths[part]}' (FORMAT PARQUET)
                """
            )
        finally:
            wcon.close()
        Path(grouped).unlink(missing_ok=True)
        print(
            f"[covis] merge {part + 1}/{merge_partitions} in "
            f"{round(time.perf_counter() - merge_started, 1)}s",
            flush=True,
        )
    missing = [p for p in final_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing covis merge outputs: {missing}")
    final_list = ", ".join(f"'{p}'" for p in final_paths)
    ccon = connect(memory_limit="1GB", temp_dir=temp_dir)
    try:
        ccon.execute(
            f"""
            COPY (
                SELECT i, j, weight FROM read_parquet([{final_list}])
            ) TO '{out.resolve().as_posix()}' (FORMAT PARQUET)
            """
        )
        row, i_distinct = ccon.execute(
            f"SELECT count(*), count(DISTINCT i) FROM '{out.resolve().as_posix()}'"
        ).fetchone()
    finally:
        ccon.close()
    for path in [
        *partial_paths,
        *final_paths,
        *(p.resolve().as_posix() for p in out.parent.glob("_covis_grouped_*.parquet")),
    ]:
        Path(path).unlink(missing_ok=True)
        Path(path).unlink(missing_ok=True)
    stats = {
        "n_pairs": int(row),
        "n_items_with_neighbours": int(i_distinct),
        "n_partial_rows": partial_rows,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    print(f"[covis] {stats}", flush=True)
    return stats


def load_neighbours(neighbours_parquet: str | Path) -> dict[str, list[tuple[str, float]]]:
    """Item -> [(neighbour, weight), ...] ordered best-first (fits in RAM for ~1M items x top-100)."""
    import pyarrow.parquet as pq

    table = pq.read_table(neighbours_parquet)
    items = table.column("i").to_pylist()
    neighbours = table.column("j").to_pylist()
    weights = table.column("weight").to_pylist()
    mapping: dict[str, list[tuple[str, float]]] = {}
    for item, nbr, weight in zip(items, neighbours, weights):
        mapping.setdefault(item, []).append((nbr, weight))
    return mapping


def retrieve(
    session_items: list[str],
    neighbours: dict[str, list[tuple[str, float]]],
    top_n: int = 100,
) -> list[tuple[str, float]]:
    """Aggregate neighbour weights over a session's items; best-first candidates."""
    scores: dict[str, float] = {}
    for item in session_items:
        for nbr, weight in neighbours.get(item, ()):  # already ranked, early items weighted by list order
            scores[nbr] = scores.get(nbr, 0.0) + weight
    for item in session_items:
        scores.pop(item, None)
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return ranked[:top_n]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build co-visitation neighbour parquet")
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--out", default="data/processed/neighbours.parquet")
    parser.add_argument("--min-co", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--max-gap", type=int, default=50)
    parser.add_argument(
        "--buckets",
        default=None,
        help="comma-separated session buckets to (re)build, e.g. '7'; '' = merge only; default: all",
    )
    parser.add_argument("--merge-partitions", type=int, default=16)
    parser.add_argument("--merge-start", type=int, default=0, help="resume merge from this partition")
    args = parser.parse_args()
    if args.buckets is None:
        bucket_ids = None
    else:
        bucket_ids = [int(b) for b in args.buckets.split(",") if b]
    print(
        json.dumps(
            build_co_visitation(
                args.events,
                args.out,
                args.min_co,
                args.top_n,
                args.max_gap,
                bucket_ids=bucket_ids,
                merge_partitions=args.merge_partitions,
                merge_start=args.merge_start,
            ),
            indent=2,
        )
    )
