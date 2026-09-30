# OTTO Recommender and Search Platform

[2026-09-30]

Two-stage e-commerce recommender + search platform: hybrid retrieval (co-visitation + BM25 + two-tower ANN) feeding a multi-objective GBDT ranker, with temporal offline IR evaluation, FastAPI serving behind Redis, and a simulated A/B experimentation layer.

> Project context, approved decisions, milestones, and changelog live in the workspace at `My Projects/OTTO Recommender and Search Platform - mimo-v2.6-flash-free/README.md`.

**Status:** M1–M3 (ETL, retrieval, ranker) + offline evaluation + A/B replay implemented and **run end-to-end on the full OTTO dataset (216.7M events)**; lint/tests green. Milestones M2-advanced (BM25/two-tower), M4, M6 remain.

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
│   ├── serving/               # FastAPI app (M4 wires Redis/SHAP/MLflow)
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

Credentials and endpoints go in `.env` (copy from `.env.example`); never commit real values.

## Serving API (skeleton)

| Route | Purpose | State |
|---|---|---|
| `GET /health` | liveness + model/redis status | working |
| `POST /recommend` | session → ranked candidates | placeholder until M4 |
| `POST /search` | query → ranked items (BM25 + dense) | placeholder until M2/M4 |
| `POST /explain` | SHAP explanation for a score | placeholder until M4 |

## Make targets

`setup`, `compile`, `test`, `lint`, `download`, `etl`, `covis`, `popularity`, `train`, `eval`, `ab`, `pipeline`, `serve`, `up`, `down`
