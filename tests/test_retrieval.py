import unittest

from otto_rec.retrieval.lexical import BM25Okapi

CORPUS = [
    ["red", "shoe", "sale"],
    ["blue", "shoe"],
    ["red", "hat"],
]


class TestBM25(unittest.TestCase):
    def test_ranks_best_match_first(self):
        index = BM25Okapi(CORPUS)
        results = index.search(["red", "shoe"], top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0][0], 0)
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_zero_score_documents_dropped(self):
        index = BM25Okapi(CORPUS)
        results = index.search(["purple"], top_k=3)
        self.assertEqual(results, [])

    def test_empty_query_and_corpus(self):
        index = BM25Okapi(CORPUS)
        self.assertEqual(index.search([], top_k=3), [])
        self.assertEqual(BM25Okapi([]).search(["red"], top_k=3), [])

    def test_top_k_respected(self):
        index = BM25Okapi(CORPUS)
        self.assertLessEqual(len(index.search(["red", "shoe"], top_k=2)), 2)

    def test_rejects_invalid_params(self):
        with self.assertRaises(ValueError):
            BM25Okapi(CORPUS, k1=0.0)
        with self.assertRaises(ValueError):
            BM25Okapi(CORPUS, b=1.5)


if __name__ == "__main__":
    unittest.main()
