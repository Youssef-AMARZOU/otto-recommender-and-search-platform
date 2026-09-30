"""Offline IR evaluation on the temporal holdout (M3 validation).

Three systems are compared on identical sessions:

- popularity  global top items ranked by trailing-30d counts, per event type
- covis       session candidates from the co-visitation graph, ranked by score
- two_stage   covisibility + popular pool reranked by the LambdaMART model

Ground truth = last 30% of each session's events (history = first 70%),
mirroring the training construction. Metrics come from the package's
evaluation module (Recall@K, NDCG@K, MAP@K, MRR, HitRate@K) plus OTTO's
weighted Recall@20. A per-session outcomes file is written for the A/B replay.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from otto_rec.evaluation.metrics import evaluate_sessions, otto_weighted_recall
from otto_rec.features.etl import connect
from otto_rec.ranking.train import FEATURE_COLUMNS

TYPES = ("clicks", "carts", "orders")
KS = (5, 10, 20)
METRIC_NAMES = ("recall", "precision", "hit_rate", "mrr", "map", "ndcg")
EMPTY_METRICS = {f"{name}@{k}": 0.0 for k in KS for name in METRIC_NAMES}


def build_eval_frame(
    processed_dir: Path,
    out_path: Path,
    n_eval: int,
    covis_top: int = 100,
    pop_pool_per_type: int = 30,
    memory_limit: str = "6GB",
    temp_dir: str = "data/tmp",
) -> dict:
    """Sample holdout sessions and materialize candidate feature rows."""
    split = json.loads((processed_dir / "split.json").read_text())
    threshold = split["threshold_min_ts"]
    started = time.perf_counter()
    con = connect(memory_limit=memory_limit, temp_dir=temp_dir)
    for name in ("events", "sessions_meta", "popularity", "neighbours"):
        path = (processed_dir / f"{name}.parquet").resolve().as_posix()
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM '{path}'")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE eval_sessions AS
        SELECT session_code, n_events, n_clicks, n_carts, n_orders, (max_ts - min_ts)::BIGINT AS span_ms
        FROM sessions_meta
        WHERE min_ts >= {threshold} AND n_events >= 2
        ORDER BY hash(session_code)
        LIMIT {n_eval}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE eval_events AS
        SELECT e.session_code, e.pos, e.item, e.etype,
               CASE WHEN e.pos < GREATEST(1, (ss.n_events * 0.7)::INT) THEN 1 ELSE 0 END AS is_hist
        FROM events e
        JOIN eval_sessions ss USING (session_code)
        """
    )
    con.execute("CREATE OR REPLACE TABLE eval_hist AS SELECT DISTINCT session_code, item FROM eval_events WHERE is_hist = 1")
    con.execute(
        """
        CREATE OR REPLACE TABLE eval_gt AS
        SELECT session_code, item, etype
        FROM eval_events
        WHERE is_hist = 0
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE eval_covis AS
        SELECT session_code, item, covis_score, covis_max
        FROM (
            SELECT h.session_code, nb.j AS item,
                   sum(nb.weight)::DOUBLE AS covis_score,
                   max(nb.weight)::DOUBLE AS covis_max
            FROM eval_hist h
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
        CREATE OR REPLACE TABLE eval_candidates AS
        SELECT session_code, item FROM eval_covis
        UNION
        SELECT es.session_code, pp.item FROM eval_sessions es CROSS JOIN pop_pool pp
        """
    )
    print("[eval] assembling candidate features ...", flush=True)
    con.execute(
        f"""
        COPY (
            SELECT
                c.session_code, c.item,
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
            FROM eval_candidates c
            JOIN eval_sessions ss USING (session_code)
            LEFT JOIN popularity p ON p.item = c.item
            LEFT JOIN eval_covis cv ON cv.session_code = c.session_code AND cv.item = c.item
            LEFT JOIN eval_hist h ON h.session_code = c.session_code AND h.item = c.item
            ORDER BY session_code
        ) TO '{out_path.resolve().as_posix()}' (FORMAT PARQUET)
        """
    )
    aux = out_path.parent
    con.execute(f"COPY (SELECT session_code, item, etype FROM eval_gt) TO '{(aux / 'eval_gt.parquet').resolve().as_posix()}' (FORMAT PARQUET)")
    con.execute(f"COPY (SELECT session_code, item FROM eval_hist) TO '{(aux / 'eval_hist.parquet').resolve().as_posix()}' (FORMAT PARQUET)")
    con.execute(
        f"COPY (SELECT session_code, n_events FROM eval_sessions) TO '{(aux / 'eval_sessions.parquet').resolve().as_posix()}' (FORMAT PARQUET)"
    )
    n_candidates = con.execute(f"SELECT count(*) FROM '{out_path.resolve().as_posix()}'").fetchone()[0]
    n_sessions = con.execute("SELECT count(*) FROM eval_sessions").fetchone()[0]
    con.close()
    stats = {
        "n_eval_sessions": int(n_sessions),
        "n_candidates": int(n_candidates),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    print(f"[eval] frame: {stats}", flush=True)
    return stats


def _score_frame(model_path: Path, frame_path: Path, top_k: int) -> pd.DataFrame:
    """Score candidates in row-group batches, keeping top-K rows per session."""
    booster = lgb.Booster(model_file=str(model_path))
    parquet = pq.ParquetFile(str(frame_path))
    pieces: list[pd.DataFrame] = []
    pending: pd.DataFrame | None = None

    def emit(chunk: pd.DataFrame | None) -> None:
        if chunk is None or chunk.empty:
            return
        scored = chunk.assign(score=booster.predict(chunk[FEATURE_COLUMNS].astype(np.float32)))
        scored = scored.sort_values("score", ascending=False, kind="stable")
        pieces.append(scored.groupby("session_code", sort=False).head(top_k))

    for batch in parquet.iter_batches(batch_size=400_000, columns=["session_code", "item", *FEATURE_COLUMNS]):
        chunk = batch.to_pandas()
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
            pending = None
        if chunk.empty:
            continue
        last = int(chunk["session_code"].iloc[-1])
        tail = chunk["session_code"] == last
        pending = chunk[tail]
        emit(chunk[~tail])
    emit(pending)
    if not pieces:
        return pd.DataFrame(columns=["session_code", "item", "score"])
    return pd.concat(pieces, ignore_index=True)


def run_evaluation(
    processed_dir: Path,
    model_path: Path,
    report_path: Path,
    outcomes_path: Path,
    n_eval: int = 50_000,
    latency_sample: int = 2000,
) -> dict:
    started = time.perf_counter()
    frame_path = processed_dir / "eval_frame.parquet"
    frame_stats = build_eval_frame(processed_dir, frame_path, n_eval=n_eval)

    gt_df = pq.read_table(processed_dir / "eval_gt.parquet").to_pandas()
    gt: dict[str, dict[str, set]] = {t: {} for t in TYPES}
    for row in gt_df.itertuples(index=False):
        gt[row.etype].setdefault(str(row.session_code), set()).add(row.item)
    del gt_df

    hist_df = pq.read_table(processed_dir / "eval_hist.parquet").to_pandas()
    hist = {str(int(k)): set(v) for k, v in hist_df.groupby("session_code")["item"].apply(list).items()}
    del hist_df

    eval_meta = pq.read_table(processed_dir / "eval_sessions.parquet").to_pandas()
    session_ids = sorted(str(int(c)) for c in eval_meta["session_code"])
    n_events_map = {str(int(r.session_code)): int(r.n_events) for r in eval_meta.itertuples(index=False)}
    del eval_meta

    popularity = pq.read_table(
        processed_dir / "popularity.parquet",
        columns=["item", "clicks_30d", "carts_30d", "orders_30d"],
    ).to_pandas()
    type_col = {"clicks": "clicks_30d", "carts": "carts_30d", "orders": "orders_30d"}
    pop_lists = {
        t: popularity.sort_values(type_col[t], ascending=False)["item"].head(20).tolist()
        for t in TYPES
    }
    del popularity

    print("[eval] scoring two-stage candidates ...", flush=True)
    t0 = time.perf_counter()
    two_stage_df = _score_frame(model_path, frame_path, top_k=20)
    score_s = time.perf_counter() - t0
    two_stage_preds: dict[str, list[str]] = {
        str(int(sid)): g["item"].tolist()
        for sid, g in two_stage_df.groupby("session_code", sort=False)
    }
    del two_stage_df

    frame = pq.read_table(frame_path, columns=["session_code", "item", "covis_score"]).to_pandas()
    frame = frame.sort_values(["session_code", "covis_score", "item"], ascending=[True, False, True])
    covis_preds: dict[str, list[str]] = {
        str(int(sid)): g["item"].tolist()[:20]
        for sid, g in frame.groupby("session_code", sort=False)
    }
    del frame

    def clean(items: list[str], seen: set | None) -> list[str]:
        seen = seen or set()
        out = [i for i in items if i not in seen]
        return out[:20]

    systems: dict[str, dict[str, dict[str, list[str]]]] = {
        "popularity": {t: {sid: clean(pop_lists[t], hist.get(sid)) for sid in session_ids} for t in TYPES},
        "covisitation": {t: {sid: clean(covis_preds.get(sid, []), hist.get(sid)) for sid in session_ids} for t in TYPES},
        "two_stage": {t: {sid: clean(two_stage_preds.get(sid, []), hist.get(sid)) for sid in session_ids} for t in TYPES},
    }

    print("[eval] computing IR metrics ...", flush=True)
    report: dict = {
        "config": {
            "n_eval_sessions": len(session_ids),
            "ks": list(KS),
            "history_rule": "first 70% history / last 30% ground truth",
            "frame": frame_stats,
            "scoring_wall_s": round(score_s, 2),
        },
        "systems": {},
    }
    for name, per_type in systems.items():
        type_metrics = {}
        for etype in TYPES:
            pairs = [
                (per_type[etype][sid], gt[etype][sid])
                for sid in session_ids
                if gt[etype].get(sid)
            ]
            type_metrics[etype] = evaluate_sessions(pairs, ks=KS) if pairs else dict(EMPTY_METRICS)
        macro = {
            key: round(sum(type_metrics[t][key] for t in TYPES) / len(TYPES), 5)
            for key in EMPTY_METRICS
        }
        weighted = otto_weighted_recall(
            per_type,
            {t: {sid: sorted(v) for sid, v in gt[t].items()} for t in TYPES},
            k=20,
        )
        report["systems"][name] = {
            "metrics": {t: {k: round(v, 5) for k, v in m.items()} for t, m in type_metrics.items()},
            "macro": macro,
            "otto_weighted_recall@20": round(weighted, 5),
        }

    print("[eval] latency sample (single-row scoring path) ...", flush=True)
    rng = np.random.default_rng(42)
    sampled_idx = rng.choice(len(session_ids), size=min(latency_sample, len(session_ids)), replace=False)
    sample_codes = {int(session_ids[int(i)]) for i in sampled_idx}
    lat_frame = pq.read_table(
        frame_path, columns=["session_code", *FEATURE_COLUMNS]
    ).to_pandas()
    lat_frame = lat_frame[lat_frame["session_code"].isin(sample_codes)]
    lat_groups = dict(tuple(lat_frame.groupby("session_code", sort=False)))
    del lat_frame
    booster = lgb.Booster(model_file=str(model_path))
    latencies: list[float] = []
    for code in sample_codes:
        rows = lat_groups.get(code)
        if rows is None or rows.empty:
            continue
        t0 = time.perf_counter()
        scores = booster.predict(rows[FEATURE_COLUMNS].astype(np.float32))
        _ = rows.iloc[np.argsort(-scores)[:20]]
        latencies.append((time.perf_counter() - t0) * 1000)
    report["latency_ms"] = {
        "n_sampled": len(latencies),
        "p50": round(float(np.percentile(latencies, 50)), 3),
        "p95": round(float(np.percentile(latencies, 95)), 3),
        "p99": round(float(np.percentile(latencies, 99)), 3),
        "note": "single-row LightGBM scoring + top-K selection; feature fetch excluded",
    }

    outcomes_rows = []
    for sid in session_ids:
        row: dict = {"session_code": int(sid), "n_events": n_events_map.get(sid, 0)}
        for etype in TYPES:
            relevant = gt[etype].get(sid, set())
            for name in systems:
                pred = systems[name][etype][sid][:20]
                row[f"hit20_{name}_{etype}"] = int(bool(relevant) and any(i in relevant for i in pred))
                row[f"recall20_{name}_{etype}"] = round(len(set(pred) & relevant) / len(relevant), 5) if relevant else 0.0
        outcomes_rows.append(row)
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outcomes_rows).to_parquet(outcomes_path, index=False)

    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[eval] report -> {report_path} ({report['elapsed_s']}s)", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal-holdout offline evaluation")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--model", default="models/ranker.txt")
    parser.add_argument("--report", default="reports/offline_eval.json")
    parser.add_argument("--outcomes", default="reports/session_outcomes.parquet")
    parser.add_argument("--n-eval", type=int, default=50_000)
    args = parser.parse_args()
    result = run_evaluation(
        Path(args.processed),
        Path(args.model),
        Path(args.report),
        Path(args.outcomes),
        n_eval=args.n_eval,
    )
    print(json.dumps({name: s["macro"] for name, s in result["systems"].items()}, indent=2))
    print(json.dumps({name: s["otto_weighted_recall@20"] for name, s in result["systems"].items()}, indent=2))
