"""Okapi BM25 over tokenized documents — the lexical arm of hybrid retrieval.

Pure standard library so it runs in CI without ML dependencies. Corpus
documents and queries must already be tokenized (lowercased, stop words
removed) by the caller.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import log


class BM25Okapi:
    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0.0 <= b <= 1.0:
            raise ValueError("require k1 > 0 and b in [0, 1]")
        self.k1 = k1
        self.b = b
        self._n = len(corpus)
        self._doc_freqs = [Counter(doc) for doc in corpus]
        self._doc_lens = [sum(freqs.values()) for freqs in self._doc_freqs]
        self._avgdl = (sum(self._doc_lens) / self._n) if self._n else 0.0
        vocabulary = set()
        for freqs in self._doc_freqs:
            vocabulary.update(freqs)
        self._idf = {}
        for term in vocabulary:
            df = sum(1 for freqs in self._doc_freqs if term in freqs)
            self._idf[term] = log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: Sequence[str], doc_index: int) -> float:
        """BM25 score of one document for one query."""
        if self._n == 0 or self._avgdl == 0.0:
            return 0.0
        freqs = self._doc_freqs[doc_index]
        doc_len = self._doc_lens[doc_index]
        score = 0.0
        for term in set(query):
            if term not in freqs or term not in self._idf:
                continue
            tf = freqs[term]
            denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self._avgdl)
            score += self._idf[term] * tf * (self.k1 + 1.0) / denominator
        return score

    def search(self, query: Sequence[str], top_k: int = 10) -> list[tuple[int, float]]:
        """Top-K (doc_index, score) pairs, best first; zero scores dropped."""
        if self._n == 0 or not query or top_k <= 0:
            return []
        scored = [(index, self.score(query, index)) for index in range(self._n)]
        positive = [pair for pair in scored if pair[1] > 0.0]
        positive.sort(key=lambda pair: (-pair[1], pair[0]))
        return positive[:top_k]
