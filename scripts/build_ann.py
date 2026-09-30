"""Build the FAISS index from trained two-tower exports and measure quality.

Outputs:
- ``models/two_tower/faiss.index`` + ``faiss.index.ids.json``
- ``reports/ann_quality.json``: ANN-vs-exact recall and next-item hit rates

Usage (from the project root, venv active)::

    PYTHONPATH=src python scripts/build_ann.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from otto_rec.features.etl import connect
from otto_rec.retrieval.ann import AnnIndex
from otto_rec.retrieval.two_tower import session_query_vector


def _load_val_queries(
    events_path: str, n_queries: int, seed: int
) -> list[tuple[list[str], str]]:
    """History / next-item pairs from sessions that fall after the split."""
    threshold = json.loads((Path(events_path).parent / "split.json").read_text())[
        "threshold_min_ts"
    ]
    con = connect(memory_limit="4GB", temp_dir="data/tmp")
    con.execute(f"CREATE VIEW events AS SELECT * FROM '{Path(events_path).resolve().as_posix()}'")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE val_sessions AS
        SELECT session_code FROM events
        WHERE ts >= {int(threshold)}
        GROUP BY session_code
        HAVING count(*) >= 4
        ORDER BY hash(session_code)
        LIMIT {n_queries}
        """
    )
    frame = con.execute(
        """
        SELECT e.session_code, e.pos, e.item FROM events e
        JOIN val_sessions USING (session_code)
        ORDER BY e.session_code, e.pos
        """
    ).fetch_df()
    con.close()
    pairs: list[tuple[list[str], str]] = []
    for _, group in frame.groupby("session_code", sort=False):
        items = [str(item) for item in group["item"].tolist()]
        cut = max(1, int(len(items) * 0.7))
        if cut >= len(items):
            continue
        pairs.append((items[:cut], items[cut]))
    return pairs


def _exact_topk(
    queries: np.ndarray, vectors: np.ndarray, item_ids: np.ndarray, k: int, chunk: int = 200_000
) -> list[list[str]]:
    """Brute-force cosine top-k over the full vocabulary, chunked by item."""
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0
    best_scores = np.full((len(queries), k), -np.inf, dtype=np.float32)
    best_ids = np.full((len(queries), k), -1, dtype=np.int64)
    for start in range(0, len(vectors), chunk):
        block = np.asarray(vectors[start : start + chunk], dtype=np.float32)
        block = block / norms[start : start + chunk][:, None]
        scores = queries @ block.T  # cosine (both sides normalised)
        merged_scores = np.concatenate([best_scores, scores], axis=1)
        merged_ids = np.concatenate(
            [best_ids, np.broadcast_to(np.arange(start, start + scores.shape[1]), scores.shape)],
            axis=1,
        )
        top = np.argpartition(merged_scores, -k, axis=1)[:, -k:]
        row = np.arange(len(queries))[:, None]
        order = np.argsort(merged_scores[row, top], axis=1)[:, ::-1]
        best_scores = merged_scores[row, top[row, order]]
        best_ids = merged_ids[row, top[row, order]]
    return [[str(item_ids[row]) for row in query_row] for query_row in best_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index + quality report")
    parser.add_argument("--model-dir", default="models/two_tower")
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--report", default="reports/ann_quality.json")
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-only", action="store_true", help="reuse an existing faiss.index"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    started = time.perf_counter()
    item_ids = pd.read_parquet(model_dir / "item_ids.parquet")["item"].astype(str).to_numpy()
    vectors = np.load(model_dir / "item_embeddings.npy")
    print(f"[ann] loaded {len(item_ids)} item vectors ({vectors.nbytes / 1e6:.0f} MB)", flush=True)

    if args.eval_only:
        index = AnnIndex.load(model_dir / "faiss.index")
        print(
            f"[ann] reused index: nprobe={index._index.nprobe} "  # noqa: SLF001 - diagnostic
            f"n={index.size} ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    else:
        index = AnnIndex()
        index.build(vectors, item_ids)
        index_path = index.save(model_dir / "faiss.index")
        print(
            f"[ann] index built: type={index.index_type} n={index.size} "
            f"({time.perf_counter() - started:.1f}s) -> {index_path}",
            flush=True,
        )

    pairs = _load_val_queries(args.events, args.queries, args.seed)
    print(f"[ann] evaluating on {len(pairs)} held-out sessions", flush=True)

    lookup = {item: i for i, item in enumerate(item_ids.tolist())}
    query_vectors = []
    targets = []
    for history, target in pairs:
        history_rows = [lookup[item] for item in history if item in lookup]
        if not history_rows:
            continue
        pooled = np.asarray(vectors[history_rows], dtype=np.float32).mean(axis=0)
        query_vectors.append(session_query_vector(pooled[None, :], str(model_dir)))
        targets.append(target)
    queries = np.asarray(query_vectors, dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-9)

    ann_hits = {10: 0, 50: 0, 100: 0}
    exact_hits = {10: 0, 50: 0, 100: 0}
    ann_vs_exact = {10: 0, 100: 0}
    batch = 256
    evaluated = 0
    for start in range(0, len(queries), batch):
        qbatch = queries[start : start + batch]
        tbatch = targets[start : start + batch]
        exact_lists = _exact_topk(qbatch, vectors, item_ids, k=100)
        for qrow, target, exact_list in zip(qbatch, tbatch, exact_lists):
            ann_list = [item for item, _ in index.search(qrow, k=100)]
            evaluated += 1
            for cutoff in ann_hits:
                if target in ann_list[:cutoff]:
                    ann_hits[cutoff] += 1
                if target in exact_list[:cutoff]:
                    exact_hits[cutoff] += 1
            for cutoff in ann_vs_exact:
                ann_vs_exact[cutoff] += len(set(ann_list[:cutoff]) & set(exact_list[:cutoff])) / cutoff
        print(f"[ann] {evaluated}/{len(queries)} queries scored", flush=True)

    report = {
        "n_queries": evaluated,
        "index_type": index.index_type,
        "n_items": int(index.size),
        "next_item_hit_rate": {
            f"ann@{cutoff}": round(ann_hits[cutoff] / max(evaluated, 1), 5)
            for cutoff in ann_hits
        },
        "next_item_hit_rate_exact": {
            f"exact@{cutoff}": round(exact_hits[cutoff] / max(evaluated, 1), 5)
            for cutoff in exact_hits
        },
        "ann_recall_vs_exact": {
            f"recall@{cutoff}": round(ann_vs_exact[cutoff] / max(evaluated, 1), 5)
            for cutoff in ann_vs_exact
        },
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"[ann] report -> {args.report}: {json.dumps(report)}", flush=True)


if __name__ == "__main__":
    main()
