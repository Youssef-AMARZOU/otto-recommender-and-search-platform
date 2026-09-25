import unittest

from otto_rec.evaluation.metrics import (
    aggregate,
    average_precision_at_k,
    evaluate_session,
    evaluate_sessions,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    otto_weighted_recall,
    precision_at_k,
    recall_at_k,
)

REC = ["b", "a", "d", "e", "c"]
REL = {"a", "c", "f"}


class TestSessionMetrics(unittest.TestCase):
    def test_recall(self):
        self.assertAlmostEqual(recall_at_k(REC, REL, 3), 1 / 3)
        self.assertAlmostEqual(recall_at_k(REC, REL, 5), 2 / 3)
        self.assertEqual(recall_at_k(REC, set(), 5), 0.0)
        self.assertEqual(recall_at_k(REC, REL, 0), 0.0)

    def test_precision(self):
        self.assertAlmostEqual(precision_at_k(REC, REL, 3), 1 / 3)
        self.assertAlmostEqual(precision_at_k(["a"], {"a"}, 5), 0.2)

    def test_hit_rate(self):
        self.assertEqual(hit_rate_at_k(REC, REL, 3), 1.0)
        self.assertEqual(hit_rate_at_k(["x", "y"], REL, 2), 0.0)

    def test_mrr(self):
        self.assertEqual(mrr_at_k(REC, REL, 3), 0.5)
        self.assertEqual(mrr_at_k(["x", "a"], REL, 1), 0.0)

    def test_average_precision(self):
        self.assertAlmostEqual(average_precision_at_k(REC, REL, 3), (0.5) / 3)
        self.assertEqual(average_precision_at_k(["a", "c", "f"], REL, 3), 1.0)

    def test_ndcg(self):
        one_over_log2_3 = 1.0 / 1.584962500721156
        expected = one_over_log2_3 / (1.0 + one_over_log2_3 + 0.5)
        self.assertAlmostEqual(ndcg_at_k(REC, REL, 3), expected, places=6)
        self.assertEqual(ndcg_at_k(["a", "c", "f"], REL, 3), 1.0)
        self.assertEqual(ndcg_at_k(["x", "y"], REL, 2), 0.0)

    def test_perfect_ranking(self):
        perfect = ["a", "c", "f", "z"]
        for metric in (recall_at_k, precision_at_k, hit_rate_at_k, mrr_at_k, average_precision_at_k, ndcg_at_k):
            self.assertAlmostEqual(metric(perfect, REL, 3), 1.0, msg=metric.__name__)


class TestAggregation(unittest.TestCase):
    def test_evaluate_session_keys(self):
        scores = evaluate_session(REC, REL, ks=[3, 10])
        for name in ("recall@3", "precision@3", "hit_rate@3", "mrr@3", "map@3", "ndcg@3", "recall@10"):
            self.assertIn(name, scores)

    def test_evaluate_sessions_means(self):
        pairs = [(["a"], {"a"}), (["z"], {"a"})]
        scores = evaluate_sessions(pairs, ks=[1])
        self.assertAlmostEqual(scores["recall@1"], 0.5)
        self.assertAlmostEqual(scores["hit_rate@1"], 0.5)

    def test_aggregate(self):
        self.assertAlmostEqual(aggregate([1.0, 0.0, 0.5]), 0.5)
        self.assertEqual(aggregate([]), 0.0)


class TestOttoWeightedRecall(unittest.TestCase):
    def test_weighted_score(self):
        ground_truth = {
            "clicks": {"s1": ["a", "b"], "s2": ["c"], "s3": []},
            "carts": {"s1": ["x"]},
            "orders": {"s2": ["y"]},
        }
        predictions = {
            "clicks": {"s1": ["a", "z"], "s2": ["d"]},
            "carts": {"s1": ["x"]},
            "orders": {"s2": ["n"]},
        }
        expected = 0.10 * 0.25 + 0.25 * 1.0 + 0.65 * 0.0
        self.assertAlmostEqual(otto_weighted_recall(predictions, ground_truth), expected)

    def test_perfect_predictions(self):
        ground_truth = {"clicks": {"s1": ["a"]}, "carts": {"s1": ["x"]}, "orders": {"s1": ["y"]}}
        predictions = {"clicks": {"s1": ["a"]}, "carts": {"s1": ["x"]}, "orders": {"s1": ["y"]}}
        self.assertAlmostEqual(otto_weighted_recall(predictions, ground_truth), 1.0)

    def test_missing_event_type_scores_zero(self):
        ground_truth = {"clicks": {"s1": ["a"]}}
        predictions = {}
        self.assertEqual(otto_weighted_recall(predictions, ground_truth), 0.0)


if __name__ == "__main__":
    unittest.main()
