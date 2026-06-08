import unittest

from scripts.analyze_mmsearch_search_need import build_diagnostic


def row(question: str, answer: str, prediction: str, score: float, responses: list[str]):
    return {
        "input_text": question,
        "output_text": responses,
        "score": score,
        "reward_model": {"ground_truth": answer, "candidate_answers": "[]"},
        "image_url": f"{question}.png",
    }


class MMSearchSearchNeedTest(unittest.TestCase):
    def test_search_need_groups(self):
        direct_rows = [
            row("q_helpful", "Paris", "wrong", 0.0, ["<reason>x</reason><answer>wrong</answer>"]),
            row("q_unnecessary", "Rome", "Rome", 1.0, ["<reason>x</reason><answer>Rome</answer>"]),
            row("q_harmful", "Agra", "Agra", 1.0, ["<reason>x</reason><answer>Agra</answer>"]),
            row("q_hard", "Cairo", "wrong", 0.0, ["<reason>x</reason><answer>wrong</answer>"]),
            row("q_failed", "Sydney", "wrong", 0.0, ["<reason>x</reason><answer>wrong</answer>"]),
        ]
        search_rows = [
            row(
                "q_helpful",
                "Paris",
                "Paris",
                1.0,
                ["<reason>x</reason><search><img></search>", "<reason>y</reason><answer>Paris</answer>"],
            ),
            row(
                "q_unnecessary",
                "Rome",
                "Rome",
                1.0,
                ["<reason>x</reason><search><img></search>", "<reason>y</reason><answer>Rome</answer>"],
            ),
            row(
                "q_harmful",
                "Agra",
                "wrong",
                0.0,
                ["<reason>x</reason><search><img></search>", "<reason>y</reason><answer>wrong</answer>"],
            ),
            row(
                "q_hard",
                "Cairo",
                "wrong",
                0.0,
                ["<reason>x</reason><search><img></search>", "<reason>y</reason><answer>wrong</answer>"],
            ),
            row("q_failed", "Sydney", "wrong", 0.0, ["<reason>x</reason><answer>wrong</answer>"]),
        ]
        summary, samples = build_diagnostic(
            direct_rows=direct_rows,
            search_rows=search_rows,
            threshold=0.1001,
            method="unit",
            direct_path="direct.json",
            search_path="search.json",
        )
        groups = {sample["sample_key"].split("|", 1)[0]: sample["group"] for sample in samples}

        self.assertEqual(summary["n_aligned"], 5)
        self.assertEqual(summary["search_helpful_n"], 1)
        self.assertEqual(summary["search_unnecessary_n"], 1)
        self.assertEqual(summary["search_harmful_n"], 1)
        self.assertEqual(summary["hard_n"], 1)
        self.assertEqual(summary["search_protocol_failed_n"], 1)
        self.assertEqual(groups["q_helpful.png"], "search_helpful")
        self.assertEqual(groups["q_failed.png"], "search_protocol_failed")

    def test_semantic_mode_falls_back_to_full_response(self):
        direct_rows = [
            row(
                "q_vegas",
                "vegas",
                "",
                0.0,
                ['The iconic "Welcome to Fabulous Las Vegas" sign is located in Las Vegas, Nevada.'],
            )
        ]
        search_rows = [
            row(
                "q_vegas",
                "vegas",
                "Las Vegas",
                1.0,
                ["<reason>x</reason><search><img></search>", "<reason>y</reason><answer>Las Vegas</answer>"],
            )
        ]
        summary, samples = build_diagnostic(
            direct_rows=direct_rows,
            search_rows=search_rows,
            threshold=0.1001,
            method="unit",
            direct_path="direct.json",
            search_path="search.json",
        )

        self.assertEqual(summary["correctness_mode"], "semantic")
        self.assertEqual(summary["direct_score_correct_rate"], 0.0)
        self.assertEqual(summary["direct_semantic_correct_rate"], 1.0)
        self.assertEqual(summary["search_helpful_n"], 0)
        self.assertEqual(summary["search_unnecessary_n"], 1)
        self.assertEqual(samples[0]["group"], "search_unnecessary")

    def test_score_mode_remains_available_for_format_sensitive_analysis(self):
        direct_rows = [
            row("q_vegas", "vegas", "", 0.0, ['The iconic sign is in Las Vegas.'])
        ]
        search_rows = [
            row(
                "q_vegas",
                "vegas",
                "Las Vegas",
                1.0,
                ["<reason>x</reason><search><img></search>", "<reason>y</reason><answer>Las Vegas</answer>"],
            )
        ]
        summary, samples = build_diagnostic(
            direct_rows=direct_rows,
            search_rows=search_rows,
            threshold=0.1001,
            method="unit",
            direct_path="direct.json",
            search_path="search.json",
            correctness_mode="score",
        )

        self.assertEqual(summary["correctness_mode"], "score")
        self.assertEqual(summary["search_helpful_n"], 1)
        self.assertEqual(samples[0]["group"], "search_helpful")


if __name__ == "__main__":
    unittest.main()
