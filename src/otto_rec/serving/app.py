"""FastAPI serving layer: hybrid retrieval + ranking behind a low-latency API.

Endpoints respond end-to-end with a deterministic placeholder ranker so the
service is runnable and testable before M2/M4 wire in the trained retriever,
LightGBM ranker, Redis cache, MLflow model registry, and SHAP explainer.
Latency budget: p99 under ``serving.latency_budget_ms`` from configs/model.yaml.
"""

from __future__ import annotations

import json
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from otto_rec.experimentation.assignment import hash_bucket

try:
    import redis
except ImportError:
    redis = None

MODEL_VERSION = os.getenv("MODEL_VERSION", "untrained")
REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))

app = FastAPI(title="OTTO Recommender and Search API", version="0.1.0")

_redis_client = None


def _get_redis():
    global _redis_client
    if not REDIS_URL or redis is None:
        return None
    if _redis_client is None:
        try:
            _redis_client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.2, socket_timeout=0.2)
        except (redis.RedisError, ValueError, OSError):
            return None
    return _redis_client


def _cache_get(client, key: str):
    if client is None:
        return None
    try:
        cached = client.get(key)
        return json.loads(cached) if cached is not None else None
    except (redis.RedisError, ValueError, OSError):
        return None


def _cache_set(client, key: str, value: list) -> None:
    if client is None:
        return
    try:
        client.setex(key, CACHE_TTL_SECONDS, json.dumps(value))
    except (redis.RedisError, OSError):
        return


def _placeholder_candidates(seed: str, k: int, exclude: list[str] | None = None) -> list[dict[str, object]]:
    blocked = set(exclude or [])
    candidates: list[dict[str, object]] = []
    bucket = hash_bucket(seed, salt="placeholder", buckets=1_000_000)
    cursor = bucket
    while len(candidates) < k:
        item_id = f"item_{cursor % 500_000:06d}"
        cursor = (cursor * 31 + 17) % 1_000_000
        if item_id in blocked:
            continue
        score = 1.0 - len(candidates) / (k + 1)
        candidates.append({"item_id": item_id, "score": round(score, 6), "sources": ["placeholder"]})
    return candidates


class RecommendRequest(BaseModel):
    session_id: str = Field(min_length=1)
    k: int = Field(default=20, ge=1, le=100)
    history: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    k: int = Field(default=20, ge=1, le=100)


class ExplainRequest(BaseModel):
    session_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)


class Candidate(BaseModel):
    item_id: str
    score: float
    sources: list[str] = []


class RecommendResponse(BaseModel):
    session_id: str
    model_version: str
    candidates: list[Candidate]
    latency_ms: float


class SearchResponse(BaseModel):
    query: str
    model_version: str
    results: list[Candidate]
    latency_ms: float


class ExplainResponse(BaseModel):
    session_id: str
    item_id: str
    model_version: str
    shap_values: dict[str, float] = Field(default_factory=dict)
    note: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    redis: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    redis_up = False
    client = _get_redis()
    if client is not None:
        try:
            redis_up = bool(client.ping())
        except (redis.RedisError, OSError):
            redis_up = False
    return HealthResponse(status="ok", model_version=MODEL_VERSION, redis=redis_up)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    started = time.perf_counter()
    cache_key = f"rec:{request.session_id}:{request.k}"
    client = _get_redis()
    payload = _cache_get(client, cache_key)
    if payload is not None:
        return RecommendResponse(
            session_id=request.session_id,
            model_version=MODEL_VERSION,
            candidates=payload,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    candidates = _placeholder_candidates(request.session_id, request.k, exclude=request.history)
    _cache_set(client, cache_key, candidates)

    return RecommendResponse(
        session_id=request.session_id,
        model_version=MODEL_VERSION,
        candidates=candidates,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    started = time.perf_counter()
    candidates = _placeholder_candidates(request.query, request.k)
    return SearchResponse(
        query=request.query,
        model_version=MODEL_VERSION,
        results=candidates,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    if not request.item_id:
        raise HTTPException(status_code=422, detail="item_id is required")
    return ExplainResponse(
        session_id=request.session_id,
        item_id=request.item_id,
        model_version=MODEL_VERSION,
        shap_values={},
        note="SHAP explainer is wired in milestone M4; returning an empty attribution set.",
    )
