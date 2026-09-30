"""M4: in-process two-stage serving pipeline (retrieval + LightGBM rerank).

Loads the offline artifacts once so ``/recommend`` can answer within the
50 ms p99 latency budget from ``configs/model.yaml``:

- ``neighbours.parquet`` (129 M co-visitation pairs, ~883 MB) is converted
  once into compact CSR-style arrays keyed by integer item codes: the sorted
  string vocabulary comes from ``popularity.parquet``, pair endpoints are
  remapped through Arrow dictionary encoding so no Python string is ever
  materialised per pair, and rows are grouped with an 8-bucket counting sort
  (each item has at most ``top_n`` neighbours, so buckets stay small) to keep
  peak host memory around 2 GB. The conversion is cached under
  ``<processed>/serving_cache/`` and reloaded on later starts.
- ``popularity.parquet`` backs the 10 popularity features and the per-type
  popular pool unioned into the candidate set.
- ``models/ranker.txt`` (LambdaMART) scores the candidate union.

Feature values, candidate construction and tie-breaks mirror
``scripts/evaluate.py::build_eval_frame`` and
``otto_rec.ranking.train.prepare_training_frame`` - covis top-100
(score DESC, item ASC) unioned with the per-type popular top-30, the same
18 ``FEATURE_COLUMNS``, history items dropped only after ranking - so the
model sees at serving time the same distribution it was trained on.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from otto_rec.ranking.train import FEATURE_COLUMNS

POP_FEATURE_COLUMNS = [
    "clicks_all", "carts_all", "orders_all", "total_all",
    "clicks_30d", "carts_30d", "orders_30d", "total_30d",
    "total_7d", "total_90d",
]
SESSION_FEATURE_COLUMNS = ["n_events", "n_clicks", "n_carts", "n_orders", "span_ms"]
COVIS_FEATURE_COLUMNS = ["covis_score", "covis_max"]

COVIS_TOP = 100
POP_POOL_PER_TYPE = 30
CONVERT_BUCKETS = 8
CACHE_VERSION = 1

VALID_ETYPES = ("clicks", "carts", "orders")


@dataclass(frozen=True)
class SessionContext:
    """Session state derived from the request, shaped like the training SQL."""

    items: tuple[str, ...]
    n_events: int
    n_clicks: int
    n_carts: int
    n_orders: int
    span_ms: int


def build_session_context(
    items: list[str],
    etypes: list[str] | None = None,
    timestamps: list[int | None] | None = None,
) -> SessionContext:
    """Aggregate request history into the five session features.

    Untyped history is treated as all-clicks (the dominant OTTO event type);
    ``span_ms`` needs at least two timestamps, otherwise it is 0 - the same
    ``max(ts) - min(ts)`` expression used when the training frame was built.
    """
    etypes = etypes or []
    timestamps = [t for t in (timestamps or []) if t is not None]
    counts = dict.fromkeys(VALID_ETYPES, 0)
    for index in range(len(items)):
        etype = etypes[index] if index < len(etypes) else "clicks"
        counts[etype if etype in counts else "clicks"] += 1
    span = 0
    if len(timestamps) >= 2:
        span = int(max(timestamps) - min(timestamps))
    seen: set[str] = set()
    unique = tuple(item for item in items if not (item in seen or seen.add(item)))
    return SessionContext(
        items=unique,
        n_events=len(items),
        n_clicks=counts["clicks"],
        n_carts=counts["carts"],
        n_orders=counts["orders"],
        span_ms=span,
    )


class ServingPipeline:
    """Two-stage retrieval + ranking over the offline OTTO artifacts."""

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        model_path: str | Path = "models/ranker.txt",
        covis_top: int = COVIS_TOP,
        pop_pool_per_type: int = POP_POOL_PER_TYPE,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.model_path = Path(model_path)
        self.covis_top = covis_top
        self.pop_pool_per_type = pop_pool_per_type
        self.cache_dir = self.processed_dir / "serving_cache"
        self._vocab: np.ndarray | None = None
        self._n_vocab = 0
        self._nb_offsets: np.ndarray | None = None
        self._nb_j: np.ndarray | None = None
        self._nb_w: np.ndarray | None = None
        self._pop: pd.DataFrame | None = None
        self._pool_codes: np.ndarray | None = None
        self._booster: lgb.Booster | None = None
        self._explainer = None
        self.loaded = False

    # ------------------------------------------------------------------ load

    def load(self) -> dict:
        """Load artifacts (converting the neighbour cache on first start)."""
        started = time.perf_counter()
        neighbours_path = self.processed_dir / "neighbours.parquet"
        popularity_path = self.processed_dir / "popularity.parquet"
        missing = [
            str(path)
            for path in (neighbours_path, popularity_path, self.model_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"serving artifacts missing: {', '.join(missing)}")

        items = pq.read_table(popularity_path, columns=["item"]).column("item")
        vocab = np.sort(np.asarray(items.to_pylist(), dtype=str))
        if vocab.size == 0:
            raise ValueError("popularity.parquet has no items")
        self._vocab = vocab
        self._n_vocab = int(vocab.size)

        cache_state = self._load_neighbour_cache(neighbours_path)

        pop = pq.read_table(popularity_path).to_pandas().set_index("item")
        self._pop = pop
        self._pool_codes = self._build_popular_pool(pop)

        self._booster = lgb.Booster(model_file=str(self.model_path))
        self.loaded = True
        stats = {
            "n_items": self._n_vocab,
            "n_pairs": int(self._nb_offsets[-1]),
            "neighbour_cache": cache_state,
            "n_pool_items": int(self._pool_codes.size),
            "model_trees": len(self._booster.dump_model()["tree_info"]),
            "elapsed_s": round(time.perf_counter() - started, 1),
        }
        print(f"[serving] pipeline loaded: {stats}", flush=True)
        return stats

    def _build_popular_pool(self, pop: pd.DataFrame) -> np.ndarray:
        pool: list[str] = []
        for column in ("clicks_30d", "carts_30d", "orders_30d"):
            top = pop.sort_values(column, ascending=False, kind="stable").head(
                self.pop_pool_per_type
            )
            pool.extend(top.index.astype(str).tolist())
        codes = self._codes_of(dict.fromkeys(pool))
        if (codes < 0).any():
            raise ValueError("popular pool contains items outside the vocabulary")
        return codes

    def _load_neighbour_cache(self, neighbours_path: Path) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.cache_dir / "meta.json"
        stat = neighbours_path.stat()
        expected = {
            "version": CACHE_VERSION,
            "neighbours_mtime": stat.st_mtime,
            "neighbours_size": stat.st_size,
            "n_vocab": self._n_vocab,
        }
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, ValueError):
                meta = {}
            if meta == expected:
                try:
                    offsets = np.load(self.cache_dir / "offsets.npy")
                    j = np.load(self.cache_dir / "j.npy")
                    w = np.load(self.cache_dir / "w.npy")
                except MemoryError:
                    offsets = np.load(self.cache_dir / "offsets.npy", mmap_mode="r")
                    j = np.load(self.cache_dir / "j.npy", mmap_mode="r")
                    w = np.load(self.cache_dir / "w.npy", mmap_mode="r")
                except (OSError, ValueError):
                    offsets = j = w = None
                if (
                    offsets is not None
                    and offsets.size == self._n_vocab + 1
                    and offsets[-1] == j.size == w.size
                ):
                    self._nb_offsets, self._nb_j, self._nb_w = offsets, j, w
                    return "loaded"

        offsets, j, w = self._convert_neighbours(neighbours_path)
        try:
            np.save(self.cache_dir / "offsets.npy", offsets)
            np.save(self.cache_dir / "j.npy", j)
            np.save(self.cache_dir / "w.npy", w)
            meta_path.write_text(json.dumps(expected))
            state = "rebuilt"
        except OSError:
            state = "rebuilt (cache write failed)"
        self._nb_offsets, self._nb_j, self._nb_w = offsets, j, w
        return state

    def _convert_neighbours(
        self, neighbours_path: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One-shot conversion: Arrow dictionary remap -> CSR arrays.

        Rows are grouped into ``CONVERT_BUCKETS`` item-code ranges and each
        bucket is sorted independently (counting-sort placement via a cursor),
        which avoids a single 129 M-element global index array and keeps peak
        memory near 2 GB on a constrained host.
        """
        print("[serving] converting neighbours.parquet (first start only) ...", flush=True)
        i_codes = self._encode_column(pq.read_table(neighbours_path, columns=["i"]).column(0))
        j_codes = self._encode_column(pq.read_table(neighbours_path, columns=["j"]).column(0))
        weights = (
            pq.read_table(neighbours_path, columns=["weight"])
            .column(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.float32)
        )
        total = int(i_codes.size)
        counts = np.bincount(i_codes, minlength=self._n_vocab)
        offsets = np.zeros(self._n_vocab + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        starts = offsets.copy()
        j_out = np.empty(total, dtype=np.int32)
        w_out = np.empty(total, dtype=np.float32)

        edges = np.linspace(0, self._n_vocab, CONVERT_BUCKETS + 1, dtype=np.int64)
        for bucket in range(CONVERT_BUCKETS):
            lo, hi = int(edges[bucket]), int(edges[bucket + 1])
            if lo >= hi:
                continue
            selected = np.flatnonzero((i_codes >= lo) & (i_codes < hi))
            if selected.size == 0:
                continue
            bucket_codes = i_codes[selected]
            order = np.argsort(bucket_codes, kind="stable")
            selected = selected[order]
            bucket_codes = bucket_codes[order]
            # Each code lives entirely inside one bucket, so within-bucket
            # rank + the bucket's global start is the row's final position.
            first = np.searchsorted(bucket_codes, bucket_codes, side="left")
            ranks = np.arange(bucket_codes.size, dtype=np.int64) - first
            positions = starts[bucket_codes] + ranks
            j_out[positions] = j_codes[selected]
            w_out[positions] = weights[selected]
        return offsets, j_out, w_out

    def _encode_column(self, column) -> np.ndarray:
        """Map an Arrow string column to int32 vocab codes via dictionary encoding."""
        encoded = pc.dictionary_encode(column.combine_chunks())
        if encoded.indices.null_count:
            raise ValueError("neighbours.parquet contains null items")
        values = encoded.dictionary.to_numpy(zero_copy_only=False)
        indices = encoded.indices.to_numpy(zero_copy_only=False)
        remap = self._codes_of(list(values))
        if (remap < 0).any():
            unknown = [str(v) for v, c in zip(values, remap) if c < 0][:5]
            raise ValueError(f"neighbour items missing from popularity: {unknown}")
        return remap[indices].astype(np.int32)

    # ---------------------------------------------------------------- lookup

    @property
    def ready(self) -> bool:
        return self.loaded

    def _codes_of(self, values) -> np.ndarray:
        """String -> vocab code (-1 for unknown, guards U-dtype truncation)."""
        if self._vocab is None:
            raise RuntimeError("pipeline is not loaded; call load() first")
        values = list(values)
        if not values:
            return np.empty(0, dtype=np.int64)
        width = self._vocab.dtype.itemsize // 4
        strings = [str(v) for v in values]
        # overlong values would be truncated by the U-dtype cast; park them on
        # a sentinel that cannot match so only they get -1
        overlong = np.array([len(v) > width for v in strings], dtype=bool)
        padded = [v if len(v) <= width else "" for v in strings]
        query = np.asarray(padded, dtype=self._vocab.dtype)
        codes = np.searchsorted(self._vocab, query)
        clipped = np.clip(codes, 0, self._n_vocab - 1)
        valid = (codes < self._n_vocab) & (self._vocab[clipped] == query) & ~overlong
        return np.where(valid, codes, -1).astype(np.int64)

    def _items_of(self, codes: np.ndarray) -> list[str]:
        if self._vocab is None:
            raise RuntimeError("pipeline is not loaded; call load() first")
        return [str(item) for item in self._vocab[codes]]

    # ------------------------------------------------------------- retrieval

    def _covis_scores(self, hist_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate neighbour weights over history -> (codes, sum, max)."""
        empty = (np.empty(0, np.int64), np.empty(0, np.float64), np.empty(0, np.float64))
        if hist_codes.size == 0 or self._nb_offsets is None:
            return empty
        starts = self._nb_offsets[hist_codes]
        counts = self._nb_offsets[hist_codes + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return empty
        flat = np.repeat(starts, counts) + (
            np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        )
        j = self._nb_j[flat].astype(np.int64)
        w = self._nb_w[flat]
        uniq, inverse = np.unique(j, return_inverse=True)
        sums = np.bincount(inverse, weights=w, minlength=uniq.size)
        maxima = np.zeros(uniq.size, dtype=np.float64)
        np.maximum.at(maxima, inverse, w)
        return uniq, sums, maxima

    def recommend(
        self, session: SessionContext, k: int, extra_items: list[str] | None = None
    ) -> list[dict]:
        """Score the candidate union and return the top-k items.

        ``extra_items`` appends dense-retrieval (ANN) candidates to the
        covisitation + popular pool; they carry the same ranker features
        (covis scores default to 0) and are tagged ``ann`` in their sources.
        """
        if self._booster is None or self._pool_codes is None or self._pop is None:
            raise RuntimeError("pipeline is not loaded; call load() first")
        hist_codes = self._codes_of(session.items)
        hist_codes = hist_codes[hist_codes >= 0]

        covis_codes, covis_sum, covis_max = self._covis_scores(hist_codes)
        top_codes = np.empty(0, np.int64)
        top_sum = np.empty(0, np.float64)
        top_max = np.empty(0, np.float64)
        if covis_codes.size:
            # Same window and tie-break as the offline QUALIFY clause.
            order = np.lexsort((covis_codes, -covis_sum))[: self.covis_top]
            top_codes = covis_codes[order]
            top_sum = covis_sum[order]
            top_max = covis_max[order]

        in_top = np.zeros(self._n_vocab, dtype=bool)
        if top_codes.size:
            in_top[top_codes] = True
        pool_extra = self._pool_codes[~in_top[self._pool_codes]]
        codes = np.concatenate([top_codes, pool_extra])

        ann_codes = np.empty(0, np.int64)
        if extra_items:
            extras = np.unique(self._codes_of(list(extra_items)))
            extras = extras[extras >= 0]
            seen = np.zeros(self._n_vocab, dtype=bool)
            seen[codes] = True
            ann_codes = extras[~seen[extras]]
            if ann_codes.size:
                codes = np.concatenate([codes, ann_codes])
        if codes.size == 0:
            return []

        n_top = int(top_codes.size)
        covis_score = np.concatenate([top_sum, np.zeros(codes.size - n_top)])
        covis_max = np.concatenate([top_max, np.zeros(codes.size - n_top)])
        in_history = np.isin(codes, hist_codes)

        features = self._features(codes, covis_score, covis_max, in_history, session)
        scores = self._booster.predict(features)
        rank = np.argsort(-scores, kind="stable")
        keep = ~in_history[rank]
        ranked_codes = codes[rank][keep][:k]
        ranked_scores = scores[rank][keep][:k]

        top_set = set(int(c) for c in top_codes)
        pool_set = set(int(c) for c in self._pool_codes)
        ann_set = set(int(c) for c in ann_codes)
        results: list[dict[str, object]] = []
        for code, score in zip(ranked_codes, ranked_scores):
            in_c, in_p = int(code) in top_set, int(code) in pool_set
            sources = [name for name, flag in (("covisitation", in_c), ("popularity", in_p)) if flag]
            if int(code) in ann_set:
                sources.append("ann")
            results.append(
                {
                    "item_id": self._items_of(np.asarray([code]))[0],
                    "score": round(float(score), 6),
                    "sources": sources or ["ranker"],
                }
            )
        return results

    def _features(
        self,
        codes: np.ndarray,
        covis_score: np.ndarray,
        covis_max: np.ndarray,
        in_history: np.ndarray,
        session: SessionContext,
    ) -> pd.DataFrame:
        if self._pop is None:
            raise RuntimeError("pipeline is not loaded; call load() first")
        items = self._items_of(codes)
        pop = self._pop.reindex(items).fillna(0)
        n = codes.size
        columns: dict[str, np.ndarray] = {
            column: pop[column].to_numpy(dtype=np.float32) for column in POP_FEATURE_COLUMNS
        }
        columns["covis_score"] = covis_score.astype(np.float32)
        columns["covis_max"] = covis_max.astype(np.float32)
        for column in SESSION_FEATURE_COLUMNS:
            columns[column] = np.full(n, getattr(session, column), dtype=np.float32)
        columns["in_history"] = in_history.astype(np.float32)
        return pd.DataFrame({column: columns[column] for column in FEATURE_COLUMNS})

    # --------------------------------------------------------------- explain

    def explain(self, session: SessionContext, item_id: str) -> dict:
        """TreeSHAP attribution for one (session, item) candidate row."""
        if self._booster is None:
            raise RuntimeError("pipeline is not loaded; call load() first")
        codes = self._codes_of([item_id])
        if codes[0] < 0:
            raise KeyError(f"unknown item: {item_id}")
        zeros = np.zeros(1, dtype=np.float64)
        flag = np.array([float(item_id in session.items)], dtype=np.float64)
        features = self._features(codes, zeros, zeros, flag, session)
        score = float(self._booster.predict(features)[0])
        if self._explainer is None:
            import shap  # heavy import, deferred to the first /explain call

            self._explainer = shap.TreeExplainer(self._booster)
        raw = self._explainer.shap_values(features)
        values = raw[0] if isinstance(raw, list) else raw
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        base = np.asarray(getattr(self._explainer, "expected_value", 0.0), dtype=np.float64)
        return {
            "shap_values": {
                name: round(float(v), 6) for name, v in zip(FEATURE_COLUMNS, values)
            },
            "base_value": round(float(base.reshape(-1)[0]), 6),
            "score": round(score, 6),
        }
