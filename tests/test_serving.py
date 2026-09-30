"""M4 serving tests: pipeline feature parity, ranking behaviour, API contract.

Every third-party dependency is optional: CI installs nothing beyond the
standard library, so these tests self-skip there and run in full inside the
project venv (fastapi, pandas, pyarrow, lightgbm, shap).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

_HAS_SCIENCE = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "pandas", "pyarrow", "lightgbm")
)
_HAS_FASTAPI = _HAS_SCIENCE and all(
    importlib.util.find_spec(name) is not None for name in ("fastapi", "httpx")
)
_HAS_SHAP = importlib.util.find_spec("shap") is not None

if _HAS_SCIENCE:
    import numpy as np
    import pandas as pd

    from otto_rec.ranking.train import FEATURE_COLUMNS
    from otto_rec.serving.pipeline import (
        POP_FEATURE_COLUMNS,
        SessionContext,
        ServingPipeline,
        build_session_context,
    )

if _HAS_FASTAPI:
    from fastapi.testclient import TestClient

    from otto_rec.serving.app import _digest, app

POP_ITEMS = [f"10{idx:02d}" for idx in range(1, 13)]  # 1001..1012


def _build_fixtures(root: Path) -> tuple[Path, Path]:
    """Tiny popularity/neighbourhood artifacts + a small regression booster."""
    import lightgbm as lgb

    rows = []
    for position, item in enumerate(POP_ITEMS):
        rows.append(
            {
                "item": item,
                "clicks_all": 100 - position,
                "carts_all": 90 - position,
                "orders_all": 80 - position,
                "total_all": 270 - 3 * position,
                "clicks_7d": 0,
                "carts_7d": 0,
                "orders_7d": 0,
                "total_7d": 0,
                "clicks_30d": 0,
                "carts_30d": 0,
                "orders_30d": 0,
                "total_30d": 0,
                "clicks_90d": 0,
                "carts_90d": 0,
                "orders_90d": 0,
                "total_90d": 0,
            }
        )
    by_item = {row["item"]: row for row in rows}
    by_item["1010"]["clicks_30d"] = 90
    by_item["1011"]["clicks_30d"] = 80
    by_item["1011"]["carts_30d"] = 70
    by_item["1012"]["carts_30d"] = 60
    by_item["1012"]["orders_30d"] = 50
    by_item["1010"]["orders_30d"] = 45
    for position, item in enumerate(POP_ITEMS):
        if by_item[item]["clicks_30d"] == 0:
            by_item[item]["clicks_30d"] = 7 - position % 7
        if by_item[item]["carts_30d"] == 0:
            by_item[item]["carts_30d"] = 6 - position % 6
        if by_item[item]["orders_30d"] == 0:
            by_item[item]["orders_30d"] = 5 - position % 5
        by_item[item]["total_30d"] = (
            by_item[item]["clicks_30d"]
            + by_item[item]["carts_30d"]
            + by_item[item]["orders_30d"]
        )
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(processed / "popularity.parquet", index=False)

    neighbours = pd.DataFrame(
        [
            {"i": "1001", "j": "1002", "weight": 2.0},
            {"i": "1001", "j": "1003", "weight": 1.0},
            {"i": "1002", "j": "1001", "weight": 1.5},
            {"i": "1002", "j": "1003", "weight": 0.5},
            {"i": "1002", "j": "1004", "weight": 0.25},
            {"i": "1003", "j": "1004", "weight": 0.75},
        ]
    )
    neighbours.to_parquet(processed / "neighbours.parquet", index=False)

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.random((400, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    booster = lgb.train(
        {"objective": "regression", "num_leaves": 7, "verbosity": -1},
        lgb.Dataset(frame, label=rng.random(400)),
        num_boost_round=20,
    )
    model_path = root / "model.txt"
    booster.save_model(str(model_path))
    return processed, model_path


def _fixture_session() -> SessionContext:
    return build_session_context(
        ["1001", "1002"], etypes=["clicks", "orders"], timestamps=[1000, 4000]
    )


class FeaturePartitionTest(unittest.TestCase):
    """Pure-stdlib sanity checks on the feature layout contract."""

    @unittest.skipUnless(_HAS_SCIENCE, "lightgbm/pandas required to import the pipeline")
    def test_feature_groups_cover_training_columns(self) -> None:
        combined = [
            *POP_FEATURE_COLUMNS,
            *["covis_score", "covis_max"],
            *["n_events", "n_clicks", "n_carts", "n_orders", "span_ms"],
            "in_history",
        ]
        self.assertEqual(sorted(combined), sorted(FEATURE_COLUMNS))
        self.assertEqual(len(FEATURE_COLUMNS), 18)


@unittest.skipUnless(_HAS_SCIENCE, "numpy/pandas/pyarrow/lightgbm required")
class ServingPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp(prefix="otto_serving_"))
        cls.processed, cls.model_path = _build_fixtures(cls.root)
        cls.pipeline = ServingPipeline(
            processed_dir=cls.processed, model_path=cls.model_path, pop_pool_per_type=2
        )
        cls.stats = cls.pipeline.load()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_load_stats(self) -> None:
        self.assertEqual(self.stats["n_items"], 12)
        self.assertEqual(self.stats["n_pairs"], 6)
        self.assertEqual(self.stats["n_pool_items"], 3)
        self.assertTrue(self.pipeline.ready)

    def test_session_context_aggregation(self) -> None:
        session = _fixture_session()
        self.assertEqual(session.n_events, 2)
        self.assertEqual(session.n_clicks, 1)
        self.assertEqual(session.n_orders, 1)
        self.assertEqual(session.span_ms, 3000)
        self.assertEqual(session.items, ("1001", "1002"))
        untyped = build_session_context(["1001", "1001", "1002"])
        self.assertEqual(untyped.n_events, 3)
        self.assertEqual(untyped.n_clicks, 3)
        self.assertEqual(untyped.span_ms, 0)
        self.assertEqual(untyped.items, ("1001", "1002"))

    def test_covis_aggregation_matches_table(self) -> None:
        session = _fixture_session()
        codes = self.pipeline._codes_of(session.items)
        uniq, sums, maxima = self.pipeline._covis_scores(codes)
        by_item = dict(
            zip(self.pipeline._items_of(uniq), zip(sums.tolist(), maxima.tolist()))
        )
        self.assertEqual(by_item["1003"], (1.5, 1.0))
        self.assertEqual(by_item["1002"], (2.0, 2.0))
        self.assertEqual(by_item["1001"], (1.5, 1.5))
        self.assertEqual(by_item["1004"], (0.25, 0.25))

    def test_feature_values_parity(self) -> None:
        session = _fixture_session()
        codes = self.pipeline._codes_of(["1003"])
        features = self.pipeline._features(
            codes,
            covis_score=np.array([1.5]),
            covis_max=np.array([1.0]),
            in_history=np.array([0.0]),
            session=session,
        )
        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        row = features.iloc[0]
        self.assertEqual(float(row["covis_score"]), 1.5)
        self.assertEqual(float(row["covis_max"]), 1.0)
        self.assertEqual(float(row["in_history"]), 0.0)
        self.assertEqual(float(row["n_events"]), 2.0)
        self.assertEqual(float(row["n_orders"]), 1.0)
        self.assertEqual(float(row["span_ms"]), 3000.0)
        self.assertEqual(float(row["clicks_all"]), 98.0)  # 1003 -> 100 - position 2

    def test_recommend_candidate_set_and_sources(self) -> None:
        session = _fixture_session()
        results = self.pipeline.recommend(session, k=12)
        item_ids = [r["item_id"] for r in results]
        # history dropped after ranking; covis neighbours + popular pool remain
        self.assertEqual(set(item_ids), {"1003", "1004", "1010", "1011", "1012"})
        self.assertNotIn("1001", item_ids)
        self.assertNotIn("1002", item_ids)
        by_item = {r["item_id"]: r for r in results}
        self.assertEqual(by_item["1003"]["sources"], ["covisitation"])
        self.assertEqual(by_item["1010"]["sources"], ["popularity"])
        self.assertEqual(by_item["1004"]["sources"], ["covisitation"])
        self.assertTrue(all(r["score"] == r["score"] for r in results))  # no NaN

    def test_recommend_is_deterministic(self) -> None:
        session = _fixture_session()
        first = self.pipeline.recommend(session, k=5)
        second = self.pipeline.recommend(session, k=5)
        self.assertEqual(first, second)

    def test_recommend_extra_items_tagged_ann(self) -> None:
        session = _fixture_session()
        results = self.pipeline.recommend(
            session, k=12, extra_items=["1005", "1010", "1001", "bogus"]
        )
        by_item = {r["item_id"]: r for r in results}
        self.assertIn("ann", by_item["1005"]["sources"])  # fresh dense candidate
        self.assertNotIn("ann", by_item["1010"]["sources"])  # already in the pool: deduped
        self.assertNotIn("1001", by_item)  # history stays excluded
        self.assertNotIn("bogus", by_item)  # unknown ids are dropped

    def test_empty_history_falls_back_to_popular_pool(self) -> None:
        session = build_session_context([])
        results = self.pipeline.recommend(session, k=3)
        self.assertEqual({r["item_id"] for r in results}, {"1010", "1011", "1012"})

    @unittest.skipUnless(_HAS_SHAP, "shap required")
    def test_explain_additivity(self) -> None:
        session = _fixture_session()
        explanation = self.pipeline.explain(session, "1003")
        self.assertEqual(sorted(explanation["shap_values"]), sorted(FEATURE_COLUMNS))
        total = explanation["base_value"] + sum(explanation["shap_values"].values())
        self.assertAlmostEqual(total, explanation["score"], places=3)

    def test_explain_unknown_item(self) -> None:
        with self.assertRaises(KeyError):
            self.pipeline.explain(_fixture_session(), "9999")

    def test_cache_roundtrip(self) -> None:
        self.assertTrue((self.processed / "serving_cache" / "meta.json").exists())
        reloaded = ServingPipeline(
            processed_dir=self.processed, model_path=self.model_path, pop_pool_per_type=2
        )
        stats = reloaded.load()
        self.assertEqual(stats["neighbour_cache"], "loaded")
        session = _fixture_session()
        self.assertEqual(
            reloaded.recommend(session, k=12), self.pipeline.recommend(session, k=12)
        )


@unittest.skipUnless(_HAS_FASTAPI, "fastapi/httpx required")
class ServingAppTest(unittest.TestCase):
    ENV_KEYS = ("SERVING_PIPELINE", "PROCESSED_DIR", "MODEL_PATH")

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp(prefix="otto_serving_app_"))
        cls.processed, cls.model_path = _build_fixtures(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self) -> None:
        self._saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}

    def tearDown(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_placeholder_mode(self) -> None:
        os.environ["SERVING_PIPELINE"] = "off"
        with TestClient(app) as client:
            health = client.get("/health").json()
            self.assertFalse(health["pipeline"])
            response = client.post(
                "/recommend", json={"session_id": "s1", "k": 5, "history": ["1001"]}
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(len(payload["candidates"]), 5)
            self.assertEqual(payload["candidates"][0]["sources"], ["placeholder"])
            explained = client.post(
                "/explain", json={"session_id": "s1", "item_id": "1001"}
            ).json()
            self.assertIn("not loaded", explained["note"])
            searched = client.post("/search", json={"query": "shoes", "k": 3}).json()
            self.assertEqual(len(searched["results"]), 3)

    def test_pipeline_mode_end_to_end(self) -> None:
        os.environ["SERVING_PIPELINE"] = "auto"
        os.environ["PROCESSED_DIR"] = str(self.processed)
        os.environ["MODEL_PATH"] = str(self.model_path)
        with TestClient(app) as client:
            health = client.get("/health").json()
            self.assertTrue(health["pipeline"])
            response = client.post(
                "/recommend",
                json={"session_id": "s1", "k": 10, "history": ["1001", "1002"]},
            )
            payload = response.json()
            items = [c["item_id"] for c in payload["candidates"]]
            # default pop pool (30/type) covers the whole 12-item fixture
            expected = set(POP_ITEMS) - {"1001", "1002"}
            self.assertEqual(set(items), expected)
            self.assertTrue(all(c["sources"] for c in payload["candidates"]))
            self.assertLess(payload["latency_ms"], 50.0)

            typed = client.post(
                "/recommend",
                json={
                    "session_id": "s2",
                    "k": 5,
                    "events": [
                        {"item_id": "1001", "etype": "clicks", "ts": 1000},
                        {"item_id": "1002", "etype": "orders", "ts": 5000},
                    ],
                },
            ).json()
            typed_items = [c["item_id"] for c in typed["candidates"]]
            self.assertEqual(len(typed_items), 5)
            self.assertTrue(set(typed_items) <= expected)

    def test_explain_endpoint(self) -> None:
        if not _HAS_SHAP:
            self.skipTest("shap required")
        os.environ["SERVING_PIPELINE"] = "auto"
        os.environ["PROCESSED_DIR"] = str(self.processed)
        os.environ["MODEL_PATH"] = str(self.model_path)
        with TestClient(app) as client:
            response = client.post(
                "/explain",
                json={"session_id": "s1", "item_id": "1003", "history": ["1001"]},
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(len(payload["shap_values"]), 18)
            self.assertIsNotNone(payload["base_value"])
            missing = client.post(
                "/explain",
                json={"session_id": "s1", "item_id": "9999", "history": ["1001"]},
            )
            self.assertEqual(missing.status_code, 404)

    def test_latency_budget(self) -> None:
        os.environ["SERVING_PIPELINE"] = "auto"
        os.environ["PROCESSED_DIR"] = str(self.processed)
        os.environ["MODEL_PATH"] = str(self.model_path)
        with TestClient(app) as client:
            client.post("/recommend", json={"session_id": "warm", "k": 10, "history": ["1001"]})
            latencies = []
            for round_index in range(150):
                history = [POP_ITEMS[round_index % 6], POP_ITEMS[(round_index + 1) % 6]]
                payload = {"session_id": f"lat{round_index}", "k": 20, "history": history}
                started = time.perf_counter()
                response = client.post("/recommend", json=payload)
                latencies.append((time.perf_counter() - started) * 1000)
                self.assertEqual(response.status_code, 200)
            p99 = float(np.percentile(latencies, 99))
            self.assertLess(p99, 50.0, f"p99={p99:.3f}ms exceeds 50ms budget")

    def test_digest_covers_model_inputs(self) -> None:
        base = {"k": 5, "h": ["1001"], "e": []}
        self.assertEqual(_digest(dict(base)), _digest(dict(base)))
        self.assertNotEqual(_digest(dict(base)), _digest({**base, "k": 6}))
        self.assertNotEqual(_digest(dict(base)), _digest({**base, "h": ["1002"]}))


if __name__ == "__main__":
    unittest.main()
