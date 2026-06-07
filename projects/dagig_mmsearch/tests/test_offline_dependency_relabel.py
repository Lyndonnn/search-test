import unittest

from data.schema import toy_samples
from eval.run_offline_dependency_relabel import build_dependency_trajectories, summarize_edges
from reward.dag_ig import DAGIGLiteReward
from reward.typed_pool import TypedCounterfactualPool


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


if __name__ == "__main__":
    unittest.main()
