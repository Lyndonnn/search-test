import unittest
from tempfile import TemporaryDirectory

from data.schema import toy_samples
from eval.run_offline_dependency_relabel import (
    build_dependency_trajectories,
    ensure_support_indexes,
    mmsearch_row_to_sample,
    summarize_edges,
)
from reward.dag_ig import DAGIGLiteReward
from reward.typed_pool import TypedCounterfactualPool
from utils.io import read_jsonl


class OfflineDependencyRelabelTest(unittest.TestCase):
    def test_redacted_teacher_chain_has_three_steps(self):
        trajectories = build_dependency_trajectories(
            toy_samples()[:2],
            text_index_path="data/indexes/text_corpus.jsonl",
            image_index_path="data/indexes/image_corpus.jsonl",
            search_topk=5,
            redact_early_answer=True,
        )
        self.assertEqual(len(trajectories), 2)
        self.assertTrue(all([step.tool_type for step in traj.steps] == ["image_search", "text_search", "stop"] for traj in trajectories))
        for trajectory in trajectories:
            step0_summary = trajectory.steps[0].evidence_summary.lower()
            self.assertFalse(any(answer.lower() in step0_summary for answer in trajectory.gold_answers))

    def test_selected_summary_counts_eligible_edges(self):
        rows = [
            {"future_edge_active": True, "future_credit_eligible": True, "answer_in_step0_observation": False, "d01_future_action_ig": 0.1, "r0_minus_g0": 0.2},
            {"future_edge_active": False, "future_credit_eligible": False, "answer_in_step0_observation": False, "d01_future_action_ig": 0.0, "r0_minus_g0": 0.0},
        ]
        selected = [rows[0]]
        summary = summarize_edges(rows, selected, "test")
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["selected_n"], 1)
        self.assertAlmostEqual(summary["selected_rate"], 0.5)
        self.assertGreater(summary["selected_r0_minus_g0_mean"], 0.0)

    def test_reward_smoke_on_dependency_chain(self):
        trajectories = build_dependency_trajectories(
            toy_samples()[:1],
            text_index_path="data/indexes/text_corpus.jsonl",
            image_index_path="data/indexes/image_corpus.jsonl",
            search_topk=5,
            redact_early_answer=True,
        )
        cf_pool = TypedCounterfactualPool()
        cf_pool.add_many(trajectories[0].steps)
        output = DAGIGLiteReward(cf_pool=cf_pool).compute(trajectories[0])
        self.assertEqual(len(output.step_rewards), 3)
        self.assertIn("0->1", output.diagnostics["future_action_ig"])

    def test_mmsearch_parquet_row_adapter(self):
        row = {
            "prompt": [{"role": "user", "content": "Where is this iconic sign located?"}],
            "reward_model": {"ground_truth": "vegas", "candidate_answers": '["Las Vegas"]'},
            "data_source": "mmsearch_r1/fvqa_train",
            "image_urls": "mmsearch_r1/data/fvqa_debug_images/train_0.png",
            "extra_info": {"question_id": "fvqa_train_0"},
        }
        sample = mmsearch_row_to_sample(row, 0)
        self.assertEqual(sample.sample_id, "fvqa_train_0")
        self.assertEqual(sample.question, "Where is this iconic sign located?")
        self.assertIn("vegas", sample.gold_answers)
        self.assertIn("Las Vegas", sample.gold_answers)
        self.assertEqual(sample.images[0], "mmsearch_r1/data/fvqa_debug_images/train_0.png")

    def test_auto_builds_diagnostic_support_indexes(self):
        samples = toy_samples()[:3]
        with TemporaryDirectory() as tmpdir:
            text_path = f"{tmpdir}/missing_text.jsonl"
            image_path = f"{tmpdir}/missing_image.jsonl"
            support_text, support_image = ensure_support_indexes(samples, text_path, image_path)
            self.assertTrue(support_text.endswith(".dagig_support.jsonl"))
            self.assertTrue(support_image.endswith(".dagig_support.jsonl"))
            self.assertEqual(len(read_jsonl(support_text)), 3)
            self.assertEqual(len(read_jsonl(support_image)), 3)


if __name__ == "__main__":
    unittest.main()
