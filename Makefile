.PHONY: setup compile test lint download serve up down

setup:
	pip install -r requirements.txt

compile:
	python -m compileall -q src dags scripts tests

test:
	python -m unittest discover -s tests -v

lint:
	ruff check src dags scripts tests

download:
	python scripts/download_otto.py

serve:
	uvicorn otto_rec.serving.app:app --reload --port 8000

up:
	docker compose up --build

down:
	docker compose down
