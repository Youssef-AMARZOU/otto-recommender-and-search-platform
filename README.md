# OTTO Recommender and Search Platform

[2026-09-30]

Two-stage e-commerce recommender + search platform: hybrid retrieval (co-visitation + BM25 + two-tower ANN) feeding a multi-objective GBDT ranker, with temporal offline IR evaluation, FastAPI serving behind Redis, and a simulated A/B experimentation layer.

> Project context, approved decisions, milestones, and changelog live in the workspace at `My Projects/OTTO Recommender and Search Platform - mimo-v2.6-flash-free/README.md`.

**Status:** M1–M6 complete — everything implemented and run end-to-end on the full OTTO dataset (216.7M events): offline eval two-stage OTTO weighted Recall@20 0.0805 (+18% over covisitation); two-tower valid hit@1 0.614 with FAISS IVF recall@100 0.784 vs exact; serving verified live (uncached server-side p99 24.5 ms vs the 50 ms budget, ANN extras present in 52/60 sessions); drift gate PASS; `docker compose up` (api + redis + mlflow) verified; lint/compile/68 tests green.

## Layout

```text
.
├── configs/                   # Feast / model / eval YAML configs
├── dags/                      # Airflow DAGs (feature ETL, training, eval)
├── src/otto_rec/
│   ├── features/              # Feast feature definitions (M1)
│   ├── retrieval/             # co-visitation, BM25, two-tower, FAISS (M2)
│   ├── ranking/               # LightGBM LambdaMART (M3)
│   ├── evaluation/            # temporal split + IR metrics (implemented)
│   ├── serving/               # FastAPI app + two-stage serving pipeline (M4)
│   ├── experimentation/       # A/B assignment, power calc, CUPED (implemented)
│   └── monitoring/            # Evidently drift reports (M6)
├── scripts/                   # OTTO download, sample generator, eval, A/B replay
├── tests/                     # stdlib unittest
├── docker-compose.yml         # api, redis, mlflow (+ airflow via profile)
├── Dockerfile                 # serving image
├── Makefile
└── .github/workflows/ci.yml   # ruff + compile + unit tests
```

## Pipeline

Data (no competition-rules acceptance needed — use the dataset API, then extract to `data/raw/otto/train.jsonl`):

```bash
kaggle datasets download -d otto/recsys-dataset -f otto-recsys-train.jsonl -p data/raw/otto
```

```bash
make pipeline   # etl -> covis -> popularity -> train -> eval -> ab
```

Each stage also runs standalone (`python -m otto_rec.features.etl`, `python -m otto_rec.retrieval.covisitation`,
`python -m otto_rec.retrieval.popularity`, `python -m otto_rec.ranking.train`, `python scripts/evaluate.py`,
`python scripts/run_ab_replay.py`; prefix with `PYTHONPATH=src`). Reports land in `reports/`
(`offline_eval.json`, `session_outcomes.parquet`, `ab_replay.json`).

On the full dataset (216.7M events, 12.9M sessions), with default budgets (6 GB DuckDB memory, 8 build
buckets, 16 merge partitions):

| Stage | Result |
|---|---|
| ETL | 216,716,096 events → 2.45 GB parquet + temporal split, ~17 min |
| Popularity | 1,855,603 items, ~4.5 min |
| Co-visitation | 129,279,774 neighbour pairs over 1,854,438 items (882 MB) |
| Ranker | LightGBM LambdaMART, valid NDCG@10 0.682 / NDCG@20 0.703 |
| Offline eval (50k holdout) | weighted Recall@20: popularity 0.0024, covisitation 0.0681, **two-stage 0.0805**; latency p99 3.12 ms |
| A/B replay | adequately powered (24,925/arm); hit20_clicks +0.32%, p=0.85 → inconclusive; latency guardrail passed |

Co-visitation is bucketed (session-hash build + item-hash, two-statement merge on fresh connections)
because a single self-join over 216M events exceeds a 6 GB memory budget; `--buckets` and
`--merge-start` allow resuming an interrupted build.

## Two-tower retrieval + ANN (M2-advanced)

```bash
make two-tower   # python -m otto_rec.retrieval.two_tower
make ann         # scripts/build_ann.py (build + quality report)
```

Training uses in-batch softmax with uniform-noise negatives (temperature 0.07, Adagrad,
lr 0.01 — swept: 0.001 is chance, 0.01 reaches ~56% hit@1 on an 80k-session probe). Real
run: 736,066 examples / 1,855,603 items in ~18 min → **valid hit@1 0.614** (chance ≈
0.00024); exports `item_embeddings.npy`, `item_ids.parquet`, `session_tower.npz`.
The session tower also has a numpy-only forward pass (`session_query_vector`) so serving
never needs torch.

FAISS IVFFlat (nlist 4096, nprobe 512, L2-normalised cosine) quality on 2,000 holdout
queries (`reports/ann_quality.json`): **recall@10 0.853 / recall@100 0.784** vs exact
brute force; next-item hit@100 0.2365 vs exact 0.266.

## Drift monitoring (M6)

```bash
make drift   # scripts/monitor_drift.py
```

Splits `reports/session_outcomes.parquet` into temporal halves (identifier `session_code`
excluded), runs an Evidently data-drift report (`reports/drift.html` + JSON summary;
builtin KS/PSI report when Evidently is absent) and applies a **fail-closed gate**: a
column counts as drifted only when p < 0.05 AND PSI ≥ 0.1, a system fails at
drift_share > 0.3. Current run: **PASS, 0/19 columns drifted**.

## Quickstart

Unit tests need no third-party packages:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m compileall -q src dags scripts tests
```

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src dags scripts tests
```

Full stack (once M1+ land: download data, train, then):

```bash
docker compose up --build                 # api + redis + mlflow
docker compose --profile orchestration up # + postgres + airflow
```

Verified [2026-09-30]: image builds on `python:3.12-slim` (+ `libgomp1` for
lightgbm/faiss OpenMP), all three services start, `/health` reports
`{"pipeline": true, "redis": true, "ann": true}`, all four endpoints exercised over
HTTP, `docker compose down` clean.

`make setup` installs the lean runtime `requirements.txt`; training extras
(torch, faiss, duckdb, evidently, mlflow, feast) live in `requirements-train.txt`.

Credentials and endpoints go in `.env` (copy from `.env.example`); never commit real values.

## Serving API (M4)

| Route | Purpose | State |
|---|---|---|
| `GET /health` | liveness + pipeline/model/redis/ann status | working |
| `POST /recommend` | session → two-stage ranked candidates (covis + popular pool + ANN extras tagged `ann`, LambdaMART score), Redis-cached by input digest | working |
| `POST /search` | dense two-tower session retrieval when `history`/`events` are present; deterministic placeholder for query-only (OTTO has no text corpus) | working |
| `POST /explain` | TreeSHAP attribution of a candidate's score | working (first call ~2.5 s: shap import) |

Start it with `make serve` (or `PYTHONPATH=src uvicorn otto_rec.serving.app:app --port 8000`).
At startup the service loads `data/processed/{neighbours,popularity}.parquet` + `models/ranker.txt`;
the first start converts the 129M-pair neighbour graph into a compact CSR cache under
`data/processed/serving_cache/` (~90 s, ~2 GB peak RAM), then loads the FAISS index when
`SERVING_ANN=auto` (default; `off` disables it, `TWO_TOWER_DIR` points at the artifacts,
`SERVING_ANN_TOP_K` caps dense extras per request). Env: `PROCESSED_DIR`, `MODEL_PATH`,
`SERVING_PIPELINE=off` (force placeholder mode), `REDIS_URL`, `CACHE_TTL_SECONDS`,
`MODEL_VERSION`.

Verified on real artifacts (host uvicorn): warm p50 5.2 ms / p95 5.4 ms / p99 ≤ 12.6 ms,
co-visitation pair count matches the build exactly (129,279,774). Inside
`docker compose up` (server-side `latency_ms`, uncached sessions): k=20 p50 13.8 /
p99 24.5 ms (max 37.5) vs the 50 ms budget, dense `/search` p50 14.5 ms, ANN extras
present in top-100 for 52/60 sessions, zero ANN failures. (Client-side timings from
Windows through Docker's port proxy add 15–90 ms of transport — trust `latency_ms`.)

MLflow: `make mlflow-log` (or `PYTHONPATH=src python scripts/log_mlflow.py --register`)
logs params/metrics/model to `sqlite:///mlflow.db` (override with `MLFLOW_TRACKING_URI`).

## Make targets

`setup`, `compile`, `test`, `lint`, `download`, `etl`, `covis`, `popularity`, `two-tower`, `ann`, `train`, `eval`, `ab`, `pipeline`, `drift`, `mlflow-log`, `serve`, `up`, `down`
