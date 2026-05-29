import unittest

from reward.gate import GateRewardScorer
from reward.types import ToolStep, Trajectory


def make_step(tool_type, action):
    return ToolStep(
        step_id=0,
        tool_type=tool_type,
        action_text=action,
        action_tokens=[1],
        action_span=(0, 1),
        raw_observation={},
        evidence_summary="",
        context_before_action="",
        context_after_observation="",
        metadata={},
    )


def make_traj(tool_free_answers):
    return Trajectory(
        sample_id="toy",
        question="Which city contains the Colosseum?",
        images=[],
        gold_answers=["Rome"],
        steps=[],
        final_answer="San Francisco",
        final_correct=False,
        full_prompt="",
        full_response="",
        metadata={"tool_free_answers": tool_free_answers},
    )


class GateRewardTest(unittest.TestCase):
    def test_consistent_stop_must_match_consensus(self):
        scorer = GateRewardScorer(consistency_threshold=0.8)
        traj = make_traj(["Rome", "Rome", "Rome"])
        self.assertLess(scorer.score(traj, make_step("stop", "San Francisco")), 0.0)
        self.assertGreater(scorer.score(traj, make_step("stop", "Rome")), 0.0)

    def test_inconsistent_search_gets_positive_gate(self):
        scorer = GateRewardScorer(consistency_threshold=0.8)
        traj = make_traj(["unknown", "Rome", "uncertain"])
        self.assertGreater(scorer.score(traj, make_step("text_search", "Colosseum city")), 0.0)


if __name__ == "__main__":
    unittest.main()
