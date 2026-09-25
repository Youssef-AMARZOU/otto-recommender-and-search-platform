# OTTO Recommender and Search Platform

[2026-09-25]

Two-stage e-commerce recommender + search platform: hybrid retrieval (co-visitation + BM25 + two-tower ANN) feeding a multi-objective GBDT ranker, with temporal offline IR evaluation, FastAPI serving behind Redis, and a simulated A/B experimentation layer.

> Project context, approved decisions, milestones, and changelog live in the workspace at `My Projects/OTTO Recommender and Search Platform - mimo-v2.6-flash-free/README.md`.

**Status:** M0 scaffold complete. Modules marked `NotImplementedError` are milestone stubs (M1-M6) — the evaluation metrics, temporal split, A/B assignment, power calculation, CUPED, and BM25 modules are implemented and unit-tested.

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
├── scripts/                   # OTTO download
├── tests/                     # stdlib unittest
├── docker-compose.yml         # api, redis, mlflow (+ airflow via profile)
├── Dockerfile                 # serving image
├── Makefile
└── .github/workflows/ci.yml   # ruff + compile + unit tests
```

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

`setup`, `compile`, `test`, `lint`, `download`, `serve`, `up`, `down`
