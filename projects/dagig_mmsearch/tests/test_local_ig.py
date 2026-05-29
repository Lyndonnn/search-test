import unittest

from reward.local_ig import LocalIGScorer
from reward.typed_pool import TypedCounterfactualPool
from reward.types import ToolStep, Trajectory


def make_step(step_id, summary, action="search"):
    return ToolStep(
        step_id=step_id,
        tool_type="text_search",
        action_text=action,
        action_tokens=[1, 2],
        action_span=(0, 2),
        raw_observation=[{"title": summary, "snippet": summary}],
        evidence_summary=summary,
        context_before_action="",
        context_after_observation=summary,
        metadata={"sample_id": f"s{step_id}"},
    )


def trajectory(step):
    return Trajectory(
        sample_id="toy",
        question="Where is the Eiffel Tower?",
        images=[],
        gold_answers=["Paris"],
        steps=[step],
        final_answer="Paris",
        final_correct=True,
        full_prompt="",
        full_response="",
        metadata={"response_token_count": 2},
    )


class LocalIGTest(unittest.TestCase):
    def test_toy_trajectory_local_ig(self):
        pool = TypedCounterfactualPool()
        current = make_step(0, "The answer is Paris", "real")
        pool.add(make_step(1, "The answer is London", "cf"))
        score = LocalIGScorer(cf_pool=pool, dead_zone=0.0).score_step(trajectory(current), current)
        self.assertGreater(score, 0.0)

    def test_dead_zone(self):
        pool = TypedCounterfactualPool()
        current = make_step(0, "The answer is Paris", "real")
        pool.add(make_step(1, "The answer is London", "cf"))
        score = LocalIGScorer(cf_pool=pool, dead_zone=10.0).score_step(trajectory(current), current)
        self.assertEqual(score, 0.0)

    def test_negative_scaling(self):
        pool = TypedCounterfactualPool()
        current = make_step(0, "No useful answer", "real")
        pool.add(make_step(1, "The answer is Paris", "cf"))
        score = LocalIGScorer(cf_pool=pool, dead_zone=0.0, negative_scale=0.25).score_step(trajectory(current), current)
        self.assertLess(score, 0.0)
        self.assertGreater(score, -1.0)


if __name__ == "__main__":
    unittest.main()

