import unittest

from otto_rec.evaluation.split import temporal_split


def _sessions():
    return [
        {"session_id": "s3", "timestamp": 300},
        {"session_id": "s1", "timestamp": 100},
        {"session_id": "s4", "timestamp": 400},
        {"session_id": "s2", "timestamp": 200},
    ]


class TestTemporalSplit(unittest.TestCase):
    def test_chronological_order(self):
        train, test = temporal_split(_sessions(), train_ratio=0.5)
        self.assertEqual([s["session_id"] for s in train], ["s1", "s2"])
        self.assertEqual([s["session_id"] for s in test], ["s3", "s4"])

    def test_train_precedes_test(self):
        train, test = temporal_split(_sessions(), train_ratio=0.75)
        self.assertLessEqual(train[-1]["timestamp"], test[0]["timestamp"])

    def test_empty_input(self):
        train, test = temporal_split([], train_ratio=0.8)
        self.assertEqual(train, [])
        self.assertEqual(test, [])

    def test_rejects_bad_ratio(self):
        with self.assertRaises(ValueError):
            temporal_split(_sessions(), train_ratio=1.0)
        with self.assertRaises(ValueError):
            temporal_split(_sessions(), train_ratio=0.0)

    def test_rejects_missing_time_key(self):
        with self.assertRaises(KeyError):
            temporal_split([{"session_id": "s1"}], train_ratio=0.5)


if __name__ == "__main__":
    unittest.main()
