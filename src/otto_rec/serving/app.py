"""FastAPI serving layer: two-stage retrieval + ranking behind a low-latency API.

Startup (lifespan) loads the offline artifacts through
``otto_rec.serving.pipeline.ServingPipeline`` - co-visitation neighbours,
popularity features and the trained LambdaMART ranker - so ``/recommend``
answers the same candidate pool the model was trained and evaluated on.
If artifacts are missing (or ``SERVING_PIPELINE=off``) the service still
boots in deterministic placeholder mode so it stays testable anywhere.

``/explain`` returns TreeSHAP attributions for the scoring model once the
pipeline is loaded. ``/search`` uses dense session retrieval when the M2
two-tower + FAISS artifacts are present (history in the request), and falls
back to a deterministic placeholder otherwise (OTTO has no text corpus).
When the ANN retriever is loaded, ``/recommend`` also unions its top
candidates into the ranker pool (``sources`` include ``ann``).

Latency budget: p99 under ``serving.latency_budget_ms`` (50 ms) from
configs/model.yaml; Redis caches finished recommendations per input digest
(``serving.redis_ttl_seconds``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from otto_rec.experimentation.assignment import hash_bucket
from otto_rec.serving.pipeline import ServingPipeline, build_session_context

try:
    import redis
except ImportError:
    redis = None

MODEL_VERSION = os.getenv("MODEL_VERSION", "untrained")
REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))


def _load_pipeline() -> ServingPipeline | None:
    if os.getenv("SERVING_PIPELINE", "auto") == "off":
        return None
    pipeline = ServingPipeline(
        processed_dir=os.getenv("PROCESSED_DIR", "data/processed"),
        model_path=os.getenv("MODEL_PATH", "models/ranker.txt"),
    )
    pipeline.load()
    return pipeline


def _load_ann():
    """Load the dense retriever unless disabled; missing artifacts are OK."""
    if os.getenv("SERVING_ANN", "auto") == "off":
        return None
    from otto_rec.retrieval.ann import AnnRetriever

    retriever = AnnRetriever(
        model_dir=os.getenv("TWO_TOWER_DIR", "models/two_tower"),
        top_k=int(os.getenv("SERVING_ANN_TOP_K", "100")),
    )
    retriever.load()
    return retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    app.state.pipeline_error = None
    app.state.ann = None
    app.state.ann_error = None
    try:
        app.state.pipeline = _load_pipeline()
    except FileNotFoundError as exc:
        app.state.pipeline_error = str(exc)
        print(f"[serving] placeholder mode: {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001 - service must boot without artifacts
        app.state.pipeline_error = f"{type(exc).__name__}: {exc}"
        print(f"[serving] placeholder mode: {app.state.pipeline_error}", flush=True)
    try:
        app.state.ann = _load_ann()
    except FileNotFoundError as exc:
        app.state.ann_error = str(exc)
        print(f"[serving] ANN disabled: {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001 - service must boot without faiss
        app.state.ann_error = f"{type(exc).__name__}: {exc}"
        print(f"[serving] ANN disabled: {app.state.ann_error}", flush=True)
    yield
    app.state.pipeline = None
    app.state.ann = None


app = FastAPI(title="OTTO Recommender and Search API", version="0.3.0", lifespan=lifespan)

_redis_client = None


def _pipeline() -> ServingPipeline | None:
    return getattr(app.state, "pipeline", None)


def _ann():
    return getattr(app.state, "ann", None)


def _get_redis():
    global _redis_client
    if not REDIS_URL or redis is None:
        return None
    if _redis_client is None:
        try:
            _redis_client = redis.Redis.from_url(
                REDIS_URL, socket_connect_timeout=0.2, socket_timeout=0.2
            )
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


def _digest(payload: dict) -> str:
    """Stable cache key over the actual model inputs (not the session id)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


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


class Event(BaseModel):
    item_id: str = Field(min_length=1)
    etype: Literal["clicks", "carts", "orders"] = "clicks"
    ts: int | None = None


class RecommendRequest(BaseModel):
    session_id: str = Field(min_length=1)
    k: int = Field(default=20, ge=1, le=100)
    history: list[str] = Field(default_factory=list)
    events: list[Event] | None = Field(
        default=None,
        description="Typed session events; preferred over history when present.",
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    k: int = Field(default=20, ge=1, le=100)
    history: list[str] = Field(
        default_factory=list,
        description="Session items for dense retrieval; OTTO has no text corpus.",
    )
    events: list[Event] | None = None


class ExplainRequest(BaseModel):
    session_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    history: list[str] = Field(default_factory=list)
    events: list[Event] | None = None


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
    base_value: float | None = None
    score: float | None = None
    note: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    pipeline: bool
    redis: bool
    ann: bool


def _request_context(items: list[str], events: list[Event] | None):
    if events:
        return build_session_context(
            [e.item_id for e in events],
            [e.etype for e in events],
            [e.ts for e in events],
        )
    return build_session_context(items)


def _history_items(items: list[str], events: list[Event] | None) -> list[str]:
    if events:
        return [e.item_id for e in events]
    return items


def _ann_candidates(items: list[str], events: list[Event] | None, k: int = 50) -> list[str]:
    """Dense candidates for the ranker pool; empty when ANN is unavailable."""
    retriever = _ann()
    if retriever is None or not retriever.ready:
        return []
    try:
        hits = retriever.search(_history_items(items, events), k=k)
    except Exception as exc:  # noqa: BLE001 - dense recall must not break /recommend
        print(f"[serving] ANN candidates failed: {exc}", flush=True)
        return []
    return [hit["item_id"] for hit in hits]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    redis_up = False
    client = _get_redis()
    if client is not None:
        try:
            redis_up = bool(client.ping())
        except (redis.RedisError, OSError):
            redis_up = False
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        pipeline=_pipeline() is not None,
        redis=redis_up,
        ann=(_ann().ready if _ann() is not None else False),
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    started = time.perf_counter()
    digest = _digest(
        {"k": request.k, "h": request.history, "e": [e.model_dump() for e in (request.events or [])]}
    )
    cache_key = f"rec:{digest}"
    client = _get_redis()
    payload = _cache_get(client, cache_key)
    if payload is None:
        pipeline = _pipeline()
        if pipeline is not None:
            context = _request_context(request.history, request.events)
            extra = _ann_candidates(request.history, request.events)
            candidates = pipeline.recommend(context, request.k, extra_items=extra or None)
        else:
            candidates = _placeholder_candidates(
                request.session_id, request.k, exclude=request.history
            )
        _cache_set(client, cache_key, candidates)
    else:
        candidates = payload

    return RecommendResponse(
        session_id=request.session_id,
        model_version=MODEL_VERSION,
        candidates=candidates,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    started = time.perf_counter()
    results: list[dict[str, object]] = []
    history = _history_items(request.history, request.events)
    retriever = _ann()
    if retriever is not None and retriever.ready and history:
        try:
            results = retriever.search(history, k=request.k)
        except Exception as exc:  # noqa: BLE001 - fall back to placeholder
            print(f"[serving] dense search failed: {exc}", flush=True)
    if not results:
        results = _placeholder_candidates(request.query, request.k)
    return SearchResponse(
        query=request.query,
        model_version=MODEL_VERSION,
        results=results,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    pipeline = _pipeline()
    if pipeline is None:
        return ExplainResponse(
            session_id=request.session_id,
            item_id=request.item_id,
            model_version=MODEL_VERSION,
            shap_values={},
            note="Pipeline not loaded; SHAP attributions require the trained ranker.",
        )
    context = _request_context(request.history, request.events)
    try:
        explanation = pipeline.explain(context, request.item_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="shap is not installed") from exc
    return ExplainResponse(
        session_id=request.session_id,
        item_id=request.item_id,
        model_version=MODEL_VERSION,
        shap_values=explanation["shap_values"],
        base_value=explanation["base_value"],
        score=explanation["score"],
        note="TreeSHAP attribution of the LambdaMART score for this candidate.",
    )
