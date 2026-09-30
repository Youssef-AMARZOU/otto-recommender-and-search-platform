"""FAISS approximate nearest-neighbour index over item embeddings (M2).

faiss is imported lazily inside build/search so the module imports cleanly in
environments without ML dependencies (CI unit tests, API image builds).

Vectors are L2-normalised before indexing, so the returned distance is the
squared Euclidean distance of cosine-equivalent vectors (smaller = closer);
``AnnRetriever`` converts it back to a cosine similarity for API responses.

Default ``nprobe=512``: measured on the full 1.86M-item index it recovers
~79% of exact top-100 (vs 43% at nprobe=64) at p50 ~4.4 ms search latency.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

DEFAULT_NPROBE = 512


class AnnIndex:
    def __init__(self, index_type: str = "ivf_flat", nlist: int = 4096, nprobe: int = DEFAULT_NPROBE):
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self._index = None
        self._ids: list[str] = []

    @property
    def size(self) -> int:
        return len(self._ids)

    def build(self, vectors: Sequence[Sequence[float]], item_ids: Sequence[str]) -> None:
        """Build the index from item embeddings."""
        import faiss

        data = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if data.ndim != 2 or data.shape[0] != len(item_ids):
            raise ValueError("vectors and item_ids must align as a 2D matrix")
        faiss.normalize_L2(data)
        n, dim = data.shape
        if self.index_type == "ivf_flat" and n > max(self.nlist, 39):
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_L2)
            index.train(data)
            index.add(data)
            index.nprobe = min(self.nprobe, self.nlist)
        else:
            # too few vectors (or flat requested): exact search
            index = faiss.IndexFlatL2(dim)
            index.add(data)
        self._index = index
        self._ids = [str(item) for item in item_ids]

    def search(self, query_vector: Sequence[float], k: int = 100) -> list[tuple[str, float]]:
        """Nearest-K (item_id, distance) pairs for one query embedding."""
        import faiss

        if self._index is None:
            raise RuntimeError("index is not built; call build() or load first")
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        distances, indices = self._index.search(query, max(1, min(k, len(self._ids))))
        results: list[tuple[str, float]] = []
        for row, distance in zip(indices[0], distances[0]):
            if row < 0:
                continue
            results.append((self._ids[int(row)], float(distance)))
        return results

    def save(self, path: str | Path) -> str:
        """Persist the faiss index plus its id sidecar; returns the index path."""
        import faiss

        if self._index is None:
            raise RuntimeError("index is not built")
        index_path = Path(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        sidecar = index_path.with_suffix(index_path.suffix + ".ids.json")
        sidecar.write_text(
            json.dumps(
                {
                    "ids": self._ids,
                    "index_type": self.index_type,
                    "nlist": self.nlist,
                    "nprobe": int(getattr(self._index, "nprobe", self.nprobe)),
                }
            )
        )
        return str(index_path)

    @classmethod
    def load(cls, path: str | Path, nprobe: int | None = None) -> AnnIndex:
        """Load a persisted index (search params come from the sidecar)."""
        import faiss

        index_path = Path(path)
        instance = cls()
        instance._index = faiss.read_index(str(index_path))
        sidecar = index_path.with_suffix(index_path.suffix + ".ids.json")
        meta = json.loads(sidecar.read_text())
        instance._ids = meta["ids"]
        instance.index_type = meta.get("index_type", "ivf_flat")
        instance.nlist = meta.get("nlist", 4096)
        probe = int(nprobe or meta.get("nprobe") or DEFAULT_NPROBE)
        instance.nprobe = probe
        if hasattr(instance._index, "nprobe"):
            instance._index.nprobe = probe
        return instance


class AnnRetriever:
    """Serving-side dense retrieval: session tower (numpy) + FAISS lookup."""

    def __init__(
        self,
        model_dir: str | Path = "models/two_tower",
        index_path: str | Path | None = None,
        top_k: int = 100,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.index_path = Path(index_path or self.model_dir / "faiss.index")
        self.top_k = top_k
        self._index: AnnIndex | None = None
        self._embeddings = None
        self._sorted_ids: np.ndarray | None = None
        self.loaded = False

    def load(self) -> dict:
        required = [
            self.index_path,
            self.model_dir / "item_ids.parquet",
            self.model_dir / "item_embeddings.npy",
            self.model_dir / "session_tower.npz",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"ANN artifacts missing: {', '.join(missing)}")
        import pandas as pd

        from otto_rec.retrieval.two_tower import session_query_vector

        self._session_forward = session_query_vector
        self._index = AnnIndex.load(self.index_path)
        ids = pd.read_parquet(self.model_dir / "item_ids.parquet")["item"].astype(str).to_numpy()
        self._sorted_ids = np.sort(ids)
        self._embeddings = np.load(self.model_dir / "item_embeddings.npy", mmap_mode="r")
        self.loaded = True
        stats = {
            "n_indexed": self._index.size,
            "n_items": int(len(self._sorted_ids)),
            "index": str(self.index_path),
        }
        print(f"[serving] ANN retriever loaded: {stats}", flush=True)
        return stats

    @property
    def ready(self) -> bool:
        return self.loaded

    def _rows_for(self, item_ids: Sequence[str]) -> np.ndarray:
        if self._sorted_ids is None:
            raise RuntimeError("ANN retriever is not loaded")
        query = np.asarray(list(item_ids), dtype=self._sorted_ids.dtype)
        positions = np.searchsorted(self._sorted_ids, query)
        clipped = np.clip(positions, 0, len(self._sorted_ids) - 1)
        valid = (positions < len(self._sorted_ids)) & (self._sorted_ids[clipped] == query)
        return clipped[valid]

    def search(self, history: Sequence[str], k: int | None = None) -> list[dict]:
        """Dense top-k for a session history; skips items already in it."""
        if not self.loaded or self._index is None:
            raise RuntimeError("ANN retriever is not loaded")
        rows = self._rows_for(history)
        if rows.size == 0:
            return []
        # memory-mapped rows are gathered lazily; history is at most a few items
        vectors = np.asarray(self._embeddings[rows], dtype=np.float32)
        query = self._session_forward(vectors, str(self.model_dir))
        hits = self._index.search(query, k=(k or self.top_k) + len(history))
        seen = set(history)
        results: list[dict] = []
        for item_id, distance in hits:
            if item_id in seen:
                continue
            results.append(
                {
                    "item_id": item_id,
                    "score": round(1.0 - distance / 2.0, 6),  # squared-L2 -> cosine
                    "sources": ["ann"],
                }
            )
            if len(results) >= (k or self.top_k):
                break
        return results
