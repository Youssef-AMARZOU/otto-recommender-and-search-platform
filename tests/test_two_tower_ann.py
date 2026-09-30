"""Tests for the M2 two-tower trainer, FAISS index and ANN retriever.

The heavy dependencies (torch, faiss, pandas) are optional: the whole module
self-skips in the bare CI environment and runs inside the full venv.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

_HAS_PANDAS = importlib.util.find_spec("pandas") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_FAISS = importlib.util.find_spec("faiss") is not None
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _write_events(path: Path) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for session in range(60):
        size = int(rng.integers(4, 10))
        items = rng.choice([f"p{i:04d}" for i in range(40)], size=size, replace=False)
        for position, item in enumerate(items):
            rows.append(
                (session, position, str(item), "clicks", 1_660_000_000 + session * 1000 + position)
            )
    pd.DataFrame(
        rows, columns=["session_code", "pos", "item", "etype", "ts"]
    ).to_parquet(path, index=False)
    (path.parent / "split.json").write_text(json.dumps({"threshold_min_ts": 1_660_200_000}))


@unittest.skipUnless(_HAS_PANDAS and _HAS_TORCH, "pandas/torch not installed")
class TestTwoTower(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="tt_test_"))
        self.events = self.tmp / "events.parquet"
        _write_events(self.events)
        self.output = self.tmp / "model"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_train_export_and_forward(self) -> None:
        import numpy as np
        import pandas as pd

        from otto_rec.retrieval.two_tower import (
            TwoTowerConfig,
            export_embeddings,
            session_query_vector,
            train_two_tower,
        )

        config = TwoTowerConfig(
            epochs=1, n_sessions=60, examples_per_session=2, n_noise=8, batch_size=32
        )
        result = train_two_tower(str(self.events), config, str(self.output))
        self.assertEqual(result, str(self.output))
        for name in ("item_embeddings.npy", "item_ids.parquet", "session_tower.npz", "metrics.json"):
            self.assertTrue((self.output / name).exists(), name)

        item_ids = pd.read_parquet(self.output / "item_ids.parquet")["item"].tolist()
        embeddings = np.load(self.output / "item_embeddings.npy")
        self.assertEqual(embeddings.shape, (len(item_ids), config.embedding_dim))

        vectors_path = export_embeddings(result, item_ids[:5])
        self.assertEqual(np.load(vectors_path).shape[0], 5)

        query = session_query_vector(embeddings[:4], result)
        self.assertEqual(query.shape, (config.embedding_dim,))
        self.assertAlmostEqual(float(np.linalg.norm(query)), 1.0, places=4)

    def test_export_skips_unknown_ids(self) -> None:
        import numpy as np

        from otto_rec.retrieval.two_tower import TwoTowerConfig, export_embeddings, train_two_tower

        config = TwoTowerConfig(epochs=1, n_sessions=60, n_noise=8, batch_size=32)
        train_two_tower(str(self.events), config, str(self.output))
        vectors_path = export_embeddings(str(self.output), ["nope", "p0001"])
        self.assertEqual(np.load(vectors_path).shape[0], 1)


@unittest.skipUnless(_HAS_NUMPY and _HAS_FAISS, "faiss not installed")
class TestAnnIndex(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        rng = np.random.default_rng(1)
        self.vectors = rng.normal(size=(300, 16)).astype(np.float32)
        self.ids = [f"i{n:04d}" for n in range(300)]
        self.tmp = Path(tempfile.mkdtemp(prefix="ann_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_nearest_to_query_is_itself(self) -> None:
        from otto_rec.retrieval.ann import AnnIndex

        index = AnnIndex()
        index.build(self.vectors, self.ids)
        hits = index.search(self.vectors[7], k=3)
        self.assertEqual(hits[0][0], self.ids[7])
        self.assertLess(hits[0][1], hits[-1][1])

    def test_save_load_roundtrip(self) -> None:
        from otto_rec.retrieval.ann import AnnIndex

        index = AnnIndex()
        index.build(self.vectors, self.ids)
        path = index.save(self.tmp / "idx.index")
        loaded = AnnIndex.load(path)
        self.assertEqual(loaded.size, len(self.ids))
        self.assertEqual(loaded.search(self.vectors[3], k=1)[0][0], self.ids[3])

    def test_search_before_build_raises(self) -> None:
        from otto_rec.retrieval.ann import AnnIndex

        with self.assertRaises(RuntimeError):
            AnnIndex().search(self.vectors[0])

    def test_build_rejects_mismatched_shapes(self) -> None:
        from otto_rec.retrieval.ann import AnnIndex

        with self.assertRaises(ValueError):
            AnnIndex().build(self.vectors[:5], self.ids)


@unittest.skipUnless(
    _HAS_PANDAS and _HAS_NUMPY and _HAS_FAISS, "pandas/faiss not installed"
)
class TestAnnRetriever(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np
        import pandas as pd

        from otto_rec.retrieval.ann import AnnIndex

        self.tmp = Path(tempfile.mkdtemp(prefix="retr_test_"))
        rng = np.random.default_rng(2)
        ids = [f"i{n:04d}" for n in range(120)]
        vectors = rng.normal(size=(120, 8)).astype(np.float32)
        pd.DataFrame({"item": ids}).to_parquet(self.tmp / "item_ids.parquet", index=False)
        np.save(self.tmp / "item_embeddings.npy", vectors)
        # minimal session tower: identity-through-one-linear (mean -> linear)
        np.savez(
            self.tmp / "session_tower.npz",
            **{"0.weight": np.eye(8, dtype=np.float32), "0.bias": np.zeros(8, dtype=np.float32)},
        )
        index = AnnIndex()
        index.build(vectors, ids)
        index.save(self.tmp / "faiss.index")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_returns_dicts_and_skips_history(self) -> None:
        from otto_rec.retrieval.ann import AnnRetriever

        retriever = AnnRetriever(model_dir=self.tmp)
        stats = retriever.load()
        self.assertTrue(retriever.ready)
        self.assertEqual(stats["n_items"], 120)
        history = ["i0000", "i0001"]
        results = retriever.search(history, k=10)
        self.assertEqual(len(results), 10)
        for hit in results:
            self.assertEqual(hit["sources"], ["ann"])
            self.assertNotIn(hit["item_id"], history)
            self.assertTrue(-1.0 <= hit["score"] <= 1.0)

    def test_missing_artifacts_raise(self) -> None:
        from otto_rec.retrieval.ann import AnnRetriever

        with self.assertRaises(FileNotFoundError):
            AnnRetriever(model_dir=self.tmp / "missing").load()

    def test_not_loaded_search_raises(self) -> None:
        from otto_rec.retrieval.ann import AnnRetriever

        with self.assertRaises(RuntimeError):
            AnnRetriever(model_dir=self.tmp).search(["i0000"])


if __name__ == "__main__":
    unittest.main()
