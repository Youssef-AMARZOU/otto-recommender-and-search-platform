"""FAISS approximate nearest-neighbour index over item embeddings (M2).

faiss is imported lazily inside build/search so the module imports cleanly in
environments without ML dependencies (CI unit tests, API image builds).
"""

from __future__ import annotations

from collections.abc import Sequence


class AnnIndex:
    def __init__(self, index_type: str = "ivf_flat", nlist: int = 4096, nprobe: int = 64):
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self._index = None

    def build(self, vectors: Sequence[Sequence[float]], item_ids: Sequence[str]) -> None:
        """Build the index from item embeddings."""
        raise NotImplementedError("M2: FAISS index build")

    def search(self, query_vector: Sequence[float], k: int = 100) -> list[tuple[str, float]]:
        """Nearest-K (item_id, distance) pairs for one query embedding."""
        raise NotImplementedError("M2: FAISS ANN search")
