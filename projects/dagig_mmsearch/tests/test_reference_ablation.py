import unittest

from agent.rollout import prompted_search_rollout
from data.schema import toy_samples
from eval.run_reference_ablation import AblationVariant, materialize_variant_rows
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


if __name__ == "__main__":
    unittest.main()
