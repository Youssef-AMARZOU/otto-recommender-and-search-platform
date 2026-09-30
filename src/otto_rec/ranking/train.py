"""M3: multi-objective LambdaMART ranker over retrieval candidates.

Training data is built out-of-core from the temporal training sessions:

- history = first 70% of each session, labels = remaining events graded by
  event type (clicks=1, carts=2, orders=3) so the model jointly optimizes
  clicks, carts and orders instead of CTR alone
- candidates = the same pool used at evaluation/serving: session
  co-visitation neighbours unioned with the per-type popular pool, so the
  model learns to discriminate among the distractors it will actually see
- features = item popularity windows, co-visitation scores vs. history,
  session context

The fitted LightGBM model (models/ranker.txt) is the scoring component of the
two-stage path at evaluation and serving time.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from otto_rec.features.etl import connect

FEATURE_COLUMNS = [
    "clicks_all", "carts_all", "orders_all", "total_all",
    "clicks_30d", "carts_30d", "orders_30d", "total_30d",
    "total_7d", "total_90d",
    "covis_score", "covis_max",
    "n_events", "n_clicks", "n_carts", "n_orders", "span_ms",
    "in_history",
]


def prepare_training_frame(
    processed_dir: str | Path,
    out_path: str | Path,
    n_sessions: int = 100_000,
    covis_top: int = 100,
    pop_pool_per_type: int = 30,
    memory_limit: str = "6GB",
    temp_dir: str = "data/tmp",
) -> dict:
    """Sample temporal train sessions and materialize candidate feature rows.

    Candidates mirror the evaluation/serving pool exactly (session
    co-visitation neighbours plus the per-type popular pool), so the model
    learns to discriminate among the same distractors it will see at eval.
    """
    processed = Path(processed_dir)
    split = json.loads((processed / "split.json").read_text())
    threshold = split["threshold_min_ts"]
    started = time.perf_counter()
    con = connect(memory_limit=memory_limit, temp_dir=temp_dir)
    con.execute(f"CREATE VIEW events AS SELECT * FROM '{(processed / 'events.parquet').resolve().as_posix()}'")
    con.execute(f"CREATE VIEW sessions_meta AS SELECT * FROM '{(processed / 'sessions_meta.parquet').resolve().as_posix()}'")
    con.execute(f"CREATE VIEW popularity AS SELECT * FROM '{(processed / 'popularity.parquet').resolve().as_posix()}'")
    con.execute(f"CREATE VIEW neighbours AS SELECT * FROM '{(processed / 'neighbours.parquet').resolve().as_posix()}'")

    print(f"[ranker] sampling {n_sessions} train sessions ...", flush=True)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE sample_sessions AS
        SELECT session_code, n_events, n_clicks, n_carts, n_orders, (max_ts - min_ts)::BIGINT AS span_ms
        FROM sessions_meta
        WHERE min_ts < {threshold} AND n_events >= 2
        ORDER BY hash(session_code)
        LIMIT {n_sessions}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE session_events AS
        SELECT e.session_code, e.pos, e.item, e.etype,
               CASE WHEN e.pos < GREATEST(1, (ss.n_events * 0.7)::INT) THEN 1 ELSE 0 END AS is_hist
        FROM events e
        JOIN sample_sessions ss USING (session_code)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE positives AS
        SELECT session_code, item,
               max(CASE etype WHEN 'clicks' THEN 1 WHEN 'carts' THEN 2 WHEN 'orders' THEN 3 END)::INT AS label
        FROM session_events
        WHERE is_hist = 0
        GROUP BY 1, 2
        """
    )
    con.execute("CREATE OR REPLACE TABLE hist AS SELECT DISTINCT session_code, item FROM session_events WHERE is_hist = 1")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE train_covis AS
        SELECT session_code, item, covis_score, covis_max
        FROM (
            SELECT h.session_code, nb.j AS item,
                   sum(nb.weight)::DOUBLE AS covis_score,
                   max(nb.weight)::DOUBLE AS covis_max
            FROM hist h
            JOIN neighbours nb ON nb.i = h.item
            GROUP BY 1, 2
        )
        QUALIFY row_number() OVER (PARTITION BY session_code ORDER BY covis_score DESC, item ASC) <= {covis_top}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE pop_pool AS
        SELECT DISTINCT item FROM (
            (SELECT item FROM popularity ORDER BY clicks_30d DESC LIMIT {pop_pool_per_type})
            UNION ALL
            (SELECT item FROM popularity ORDER BY carts_30d DESC LIMIT {pop_pool_per_type})
            UNION ALL
            (SELECT item FROM popularity ORDER BY orders_30d DESC LIMIT {pop_pool_per_type})
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE candidates AS
        SELECT session_code, item FROM train_covis
        UNION
        SELECT ss.session_code, pp.item FROM sample_sessions ss CROSS JOIN pop_pool pp
        """
    )
    print("[ranker] assembling features ...", flush=True)
    con.execute(
        f"""
        COPY (
            SELECT
                c.session_code, c.item, coalesce(pos.label, 0)::INT AS label,
                ss.n_events, ss.n_clicks, ss.n_carts, ss.n_orders, ss.span_ms,
                coalesce(p.clicks_all, 0)::UINT32 AS clicks_all,
                coalesce(p.carts_all, 0)::UINT32 AS carts_all,
                coalesce(p.orders_all, 0)::UINT32 AS orders_all,
                coalesce(p.total_all, 0)::UINT32 AS total_all,
                coalesce(p.clicks_30d, 0)::UINT32 AS clicks_30d,
                coalesce(p.carts_30d, 0)::UINT32 AS carts_30d,
                coalesce(p.orders_30d, 0)::UINT32 AS orders_30d,
                coalesce(p.total_30d, 0)::UINT32 AS total_30d,
                coalesce(p.total_7d, 0)::UINT32 AS total_7d,
                coalesce(p.total_90d, 0)::UINT32 AS total_90d,
                coalesce(cv.covis_score, 0)::DOUBLE AS covis_score,
                coalesce(cv.covis_max, 0)::DOUBLE AS covis_max,
                CASE WHEN h.session_code IS NOT NULL THEN 1 ELSE 0 END AS in_history
            FROM candidates c
            JOIN sample_sessions ss USING (session_code)
            LEFT JOIN positives pos ON pos.session_code = c.session_code AND pos.item = c.item
            LEFT JOIN popularity p ON p.item = c.item
            LEFT JOIN train_covis cv ON cv.session_code = c.session_code AND cv.item = c.item
            LEFT JOIN hist h ON h.session_code = c.session_code AND h.item = c.item
            ORDER BY session_code
        ) TO '{Path(out_path).resolve().as_posix()}' (FORMAT PARQUET)
        """
    )
    n_rows, n_pos, n_sessions_done = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE label > 0), count(DISTINCT session_code) FROM '{Path(out_path).resolve().as_posix()}'"
    ).fetchone()
    con.close()
    stats = {
        "n_rows": int(n_rows),
        "n_positives": int(n_pos),
        "n_sessions": int(n_sessions_done),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    print(f"[ranker] frame ready: {stats}", flush=True)
    return stats


def train_model(
    frame_path: str | Path,
    model_path: str | Path,
    importance_path: str | Path | None = None,
    num_boost_round: int = 400,
    early_stopping_rounds: int = 40,
    learning_rate: float = 0.05,
    num_leaves: int = 255,
    min_data_in_leaf: int = 200,
    seed: int = 42,
) -> dict:
    """Fit LightGBM LambdaMART on graded labels with session-grouped early stopping."""
    started = time.perf_counter()
    df = pq.read_table(frame_path).to_pandas()
    for column in FEATURE_COLUMNS:
        df[column] = df[column].astype(np.float32)
    df = df.sort_values("session_code", kind="stable").reset_index(drop=True)

    session_codes = df["session_code"].to_numpy()
    y = df["label"].to_numpy(dtype=np.float32)
    X = df[FEATURE_COLUMNS]

    unique_codes = np.unique(session_codes)
    valid_sessions = {int(code): int(code) % 10 == 0 for code in unique_codes}
    row_is_valid = np.array([valid_sessions[int(code)] for code in session_codes])

    def groups_for(mask: np.ndarray) -> list[int]:
        codes = session_codes[mask]
        if codes.size == 0:
            return []
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        return np.diff(np.r_[starts, codes.size]).tolist()

    train_mask = ~row_is_valid
    valid_mask = row_is_valid
    if not train_mask.any() or not valid_mask.any():
        train_mask = np.ones(len(df), dtype=bool)
        valid_mask = np.zeros(len(df), dtype=bool)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10, 20],
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "label_gain": [0, 1, 3, 7],
        "seed": seed,
        "verbosity": -1,
        "num_threads": os.cpu_count() or 8,
    }

    dtrain = lgb.Dataset(X[train_mask], label=y[train_mask], group=groups_for(train_mask), free_raw_data=False)
    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=100)]
    if valid_mask.any():
        dvalid = lgb.Dataset(X[valid_mask], label=y[valid_mask], group=groups_for(valid_mask), free_raw_data=False)
        valid_sets.append(dvalid)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))

    importance = sorted(
        zip(FEATURE_COLUMNS, booster.feature_importance(importance_type="gain")),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if importance_path is not None:
        Path(importance_path).parent.mkdir(parents=True, exist_ok=True)
        Path(importance_path).write_text(json.dumps({name: int(gain) for name, gain in importance}, indent=2))

    evaluation = {
        "best_iteration": booster.best_iteration or num_boost_round,
        "train_ndcg@10": round(float(booster.best_score["train"]["ndcg@10"]), 5),
        "train_ndcg@20": round(float(booster.best_score["train"]["ndcg@20"]), 5),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    if "valid" in booster.best_score:
        evaluation["valid_ndcg@10"] = round(float(booster.best_score["valid"]["ndcg@10"]), 5)
        evaluation["valid_ndcg@20"] = round(float(booster.best_score["valid"]["ndcg@20"]), 5)
    print(f"[ranker] trained: {evaluation}", flush=True)
    return evaluation


def score_candidates(model_path: str | Path, frame_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Score every candidate row of a feature frame; returns (scores, session_codes)."""
    booster = lgb.Booster(model_file=str(model_path))
    df = pq.read_table(frame_path, columns=["session_code", *FEATURE_COLUMNS]).to_pandas()
    for column in FEATURE_COLUMNS:
        df[column] = df[column].astype(np.float32)
    scores = booster.predict(df[FEATURE_COLUMNS])
    return scores, df["session_code"].to_numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training frame and fit the LambdaMART ranker")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--n-sessions", type=int, default=100_000)
    parser.add_argument("--model-out", default="models/ranker.txt")
    args = parser.parse_args()
    processed = Path(args.processed)
    frame_path = processed / "ranker_train_frame.parquet"
    frame_stats = prepare_training_frame(processed, frame_path, n_sessions=args.n_sessions)
    train_stats = train_model(frame_path, args.model_out, importance_path="reports/ranker_importance.json")
    print(json.dumps({"frame": frame_stats, "train": train_stats}, indent=2))
