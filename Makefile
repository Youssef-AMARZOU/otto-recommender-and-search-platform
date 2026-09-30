.PHONY: setup compile test lint download etl covis popularity train eval ab two-tower ann pipeline mlflow-log drift serve up down

setup:
	pip install -r requirements-train.txt

compile:
	python -m compileall -q src dags scripts tests

test:
	python -m unittest discover -s tests -v

lint:
	ruff check src dags scripts tests

download:
	python scripts/download_otto.py

etl:
	PYTHONPATH=src python -m otto_rec.features.etl --jsonl data/raw/otto/train.jsonl --out data/processed

covis:
	PYTHONPATH=src python -m otto_rec.retrieval.covisitation --events data/processed/events.parquet --out data/processed/neighbours.parquet

popularity:
	PYTHONPATH=src python -m otto_rec.retrieval.popularity --events data/processed/events.parquet --out data/processed/popularity.parquet

train:
	PYTHONPATH=src python -m otto_rec.ranking.train --processed data/processed --model-out models/ranker.txt

two-tower:
	PYTHONPATH=src python -m otto_rec.retrieval.two_tower --events data/processed/events.parquet --out models/two_tower

ann: two-tower
	PYTHONPATH=src python scripts/build_ann.py --model-dir models/two_tower --events data/processed/events.parquet --report reports/ann_quality.json

eval:
	PYTHONPATH=src python scripts/evaluate.py --processed data/processed --model models/ranker.txt --report reports/offline_eval.json --outcomes reports/session_outcomes.parquet

ab:
	PYTHONPATH=src python scripts/run_ab_replay.py --outcomes reports/session_outcomes.parquet --eval-report reports/offline_eval.json --report reports/ab_replay.json

pipeline: etl covis popularity train eval ab

mlflow-log:
	PYTHONPATH=src python scripts/log_mlflow.py

drift:
	PYTHONPATH=src python scripts/monitor_drift.py

serve:
	PYTHONPATH=src uvicorn otto_rec.serving.app:app --port 8000

up:
	docker compose up --build

down:
	docker compose down
