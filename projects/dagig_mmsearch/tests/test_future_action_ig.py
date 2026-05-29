import unittest

from reward.future_action_ig import FutureActionIGScorer
from reward.typed_pool import TypedCounterfactualPool
from reward.types import ToolStep, Trajectory


def step(step_id, tool_type, action, summary):
    return ToolStep(
        step_id=step_id,
        tool_type=tool_type,
        action_text=action,
        action_tokens=[step_id + 1],
        action_span=(step_id, step_id + 1),
        raw_observation=[{"title": summary, "snippet": summary}],
        evidence_summary=summary,
        context_before_action="",
        context_after_observation=summary,
        metadata={"sample_id": f"s{step_id}"},
    )


class FutureActionIGTest(unittest.TestCase):
    def test_next_step_dependency(self):
        source = step(0, "text_search", "find museum", "The relevant next action is Louvre Museum query")
        target = step(1, "text_search", "Louvre Museum query", "The answer is Louvre Museum")
        cf = step(2, "text_search", "other", "The relevant next action is bridge query")
        pool = TypedCounterfactualPool()
        pool.add(source)
        pool.add(cf)
        traj = Trajectory("toy", "Where is Mona Lisa?", [], ["Louvre Museum"], [source, target], "Louvre Museum", True, "", "")
        score = FutureActionIGScorer(cf_pool=pool, dead_zone=0.0).score_edge(traj, source, target)
        self.assertGreaterEqual(score, 0.0)

    def test_positive_only(self):
        source = step(0, "text_search", "find", "No target action")
        target = step(1, "text_search", "Paris", "answer")
        cf = step(2, "text_search", "other", "Paris")
        pool = TypedCounterfactualPool()
        pool.add(cf)
        traj = Trajectory("toy", "Where?", [], ["Paris"], [source, target], "Paris", True, "", "")
        score = FutureActionIGScorer(cf_pool=pool, dead_zone=0.0, positive_only=True).score_edge(traj, source, target)
        self.assertGreaterEqual(score, 0.0)

    def test_action_span_logprob_uses_action_text(self):
        source = step(0, "text_search", "find", "Paris")
        target = step(1, "stop", "Paris", "final")
        cf = step(2, "text_search", "other", "London")
        pool = TypedCounterfactualPool()
        pool.add(cf)
        traj = Trajectory("toy", "Where?", [], ["Paris"], [source, target], "Paris", True, "", "")
        scorer = FutureActionIGScorer(cf_pool=pool, dead_zone=0.0)
        scorer.score_edge(traj, source, target)
        self.assertEqual(scorer.last_diagnostics["target_action"], "Paris")


if __name__ == "__main__":
    unittest.main()

