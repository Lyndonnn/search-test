import unittest

from agent.rollout import prompted_search_rollout
from data.schema import toy_samples
from eval.run_reference_ablation import AblationVariant, build_delta_rows, materialize_variant_rows
from reward.dag_ig import DAGIGLiteReward


class ReferenceAblationTest(unittest.TestCase):
    def test_local_ig_only_materialization_disables_future(self):
        rows, trajectories = prompted_search_rollout(toy_samples()[:2])
        reward = DAGIGLiteReward(lambda_dep=0.5, alpha=0.4, beta=0.2, gamma=0.05)
        outputs = [reward.compute(trajectory) for trajectory in trajectories]
        variant = AblationVariant("local_ig_only", lambda_dep=0.0, alpha=0.4, beta=0.2, gamma=0.05, use_future=False)

        ablated = materialize_variant_rows(rows, trajectories, outputs, variant, "unit_local_ig_only")

        self.assertEqual(ablated[0]["method"], "unit_local_ig_only")
        for row in ablated:
            for step in row["steps"][:-1]:
                self.assertEqual(step["future_action_ig"], 0.0)
                self.assertEqual(step["propagated_return"], step["local_ig"])

    def test_delta_rows_compare_local_and_dagig(self):
        rows, trajectories = prompted_search_rollout(toy_samples()[:2])
        reward = DAGIGLiteReward(lambda_dep=0.5, alpha=0.4, beta=0.2, gamma=0.05)
        outputs = [reward.compute(trajectory) for trajectory in trajectories]
        local = materialize_variant_rows(
            rows,
            trajectories,
            outputs,
            AblationVariant("local_ig_only", 0.0, 0.4, 0.2, 0.05, use_future=False),
            "unit_local",
        )
        dagig = materialize_variant_rows(
            rows,
            trajectories,
            outputs,
            AblationVariant("dagig_lite", 0.5, 0.4, 0.2, 0.05, use_future=True),
            "unit_dagig",
        )

        delta = build_delta_rows(local, dagig)

        self.assertTrue(delta)
        self.assertTrue(any("total_reward_delta" in row for row in delta))
        self.assertTrue(any(row["tool_type"] != "stop" for row in delta))


if __name__ == "__main__":
    unittest.main()
