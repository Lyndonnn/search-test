import unittest

from reward.dag_ig import DAGIGLiteReward
from reward.types import ToolStep, Trajectory


class FixedLocal:
    def __init__(self, values):
        self.values = values
        self.last_diagnostics = {}

    def score_step(self, trajectory, step):
        return self.values[step.step_id]


class FixedFuture:
    def __init__(self, value):
        self.value = value
        self.last_diagnostics = {}

    def score_edge(self, trajectory, source_step, target_step):
        return self.value


class ZeroGate:
    def score(self, trajectory, step):
        return 0.0


class ZeroCost:
    def score(self, step):
        return 0.0


def step(step_id, span):
    return ToolStep(
        step_id=step_id,
        tool_type="text_search" if step_id == 0 else "stop",
        action_text=f"action {step_id}",
        action_tokens=[step_id + 1] * (span[1] - span[0]),
        action_span=span,
        raw_observation="obs",
        evidence_summary="obs",
        context_before_action="",
        context_after_observation="",
        metadata={},
    )


class DAGIGTest(unittest.TestCase):
    def test_return_formula_and_token_injection(self):
        steps = [step(0, (2, 4)), step(1, (5, 6))]
        traj = Trajectory("toy", "q", [], ["a"], steps, "a", True, "", "", {"response_token_count": 8})
        reward = DAGIGLiteReward(
            local_ig_scorer=FixedLocal({0: 0.1, 1: 0.4}),
            future_action_ig_scorer=FixedFuture(0.5),
            gate_scorer=ZeroGate(),
            cost_model=ZeroCost(),
            lambda_dep=0.5,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
        )
        output = reward.compute(traj)
        self.assertAlmostEqual(output.step_rewards[0].propagated_return, 0.1 + 0.5 * 0.5 * 0.4)
        self.assertEqual(output.token_rewards[0], 0.0)
        self.assertGreater(output.token_rewards[2], 0.0)
        self.assertGreater(output.token_rewards[3], 0.0)
        self.assertEqual(output.token_rewards[4], 0.0)

    def test_length_normalization(self):
        steps = [step(0, (0, 2)), step(1, (2, 3))]
        traj = Trajectory("toy", "q", [], ["a"], steps, "a", True, "", "", {"response_token_count": 4})
        reward = DAGIGLiteReward(
            local_ig_scorer=FixedLocal({0: 1.0, 1: 0.0}),
            future_action_ig_scorer=FixedFuture(0.0),
            gate_scorer=ZeroGate(),
            cost_model=ZeroCost(),
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            action_length_norm=True,
        )
        output = reward.compute(traj)
        self.assertAlmostEqual(output.token_rewards[0], 0.5)
        self.assertAlmostEqual(output.token_rewards[1], 0.5)


if __name__ == "__main__":
    unittest.main()

