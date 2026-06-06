import json
import os
import tempfile
import unittest

from scripts.summarize_mmsearch_val_result import summarize


class MMSearchValSummaryTest(unittest.TestCase):
    def test_summarize_val_result(self):
        rows = [
            {
                "score": 1.0,
                "output_text": ["<reason>x</reason><answer>Paris</answer>"],
                "reward_model": {"ground_truth": "Paris", "candidate_answers": '["France"]'},
            },
            {
                "score": 0.0,
                "output_text": ["<reason>x</reason><text_search>query</text_search>"],
                "reward_model": {"ground_truth": "Rome", "candidate_answers": []},
            },
        ]
        metrics = summarize(rows, "unit.json")
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["score_mean"], 0.5)
        self.assertEqual(metrics["answer_em"], 0.5)
        self.assertEqual(metrics["text_search_rate"], 0.5)

    def test_json_round_trip_shape(self):
        rows = [{"score": 0.25, "output_text": "no answer", "reward_model": {"ground_truth": "vegas"}}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "val_result_1.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f)
            with open(path, "r", encoding="utf-8") as f:
                metrics = summarize(json.load(f), path)
        self.assertEqual(metrics["n"], 1)
        self.assertEqual(metrics["answer_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
