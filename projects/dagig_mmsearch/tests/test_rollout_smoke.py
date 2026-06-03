import unittest

from agent.policy_wrapper import PolicyWrapper
from agent.rollout import (
    agentic_search_rollout,
    dagig_reward_debug_rollout,
    direct_vqa_rollout,
    model_agent_two_turn_rollout,
    prompted_search_rollout,
)
from data.schema import toy_samples


class RolloutSmokeTest(unittest.TestCase):
    def test_direct_vqa_8_samples(self):
        rows = direct_vqa_rollout(toy_samples())
        self.assertEqual(len(rows), 8)
        self.assertTrue(all("final_answer" in row for row in rows))

    def test_prompted_search_8_samples(self):
        rows, trajectories = prompted_search_rollout(toy_samples())
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(trajectories), 8)
        self.assertTrue(all(len(row["steps"]) >= 2 for row in rows))

    def test_agentic_search_has_stop_and_search_branches(self):
        rows, trajectories = agentic_search_rollout(toy_samples())
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(trajectories), 8)
        by_id = {row["sample_id"]: row for row in rows}
        self.assertEqual(len(by_id["toy_eiffel"]["steps"]), 1)
        self.assertEqual(by_id["toy_eiffel"]["steps"][0]["tool_type"], "stop")
        self.assertGreaterEqual(len(by_id["toy_mona_lisa"]["steps"]), 2)
        self.assertNotEqual(by_id["toy_mona_lisa"]["steps"][0]["tool_type"], "stop")

    def test_model_agent_mode_does_not_force_search(self):
        class StopModel:
            def generate_text(self, prompt, **kwargs):
                return '{"tool":"stop","action":"unknown"}'

        rows, _ = agentic_search_rollout(
            toy_samples()[1:2],
            policy=PolicyWrapper(model=StopModel()),
            scripted_direct_stop=False,
            force_search_when_needed=False,
            fallback_on_invalid=False,
        )

        self.assertEqual(rows[0]["steps"][0]["tool_type"], "stop")
        self.assertEqual(len(rows[0]["steps"]), 1)

    def test_placeholder_search_action_repairs_to_question(self):
        class PlaceholderSearchModel:
            def generate_text(self, prompt, **kwargs):
                return '{"tool":"text_search","action":"query"}'

        sample = toy_samples()[1]
        rows, _ = agentic_search_rollout(
            [sample],
            policy=PolicyWrapper(model=PlaceholderSearchModel()),
            scripted_direct_stop=False,
            force_search_when_needed=False,
            fallback_on_invalid=False,
        )

        self.assertEqual(rows[0]["steps"][0]["tool_type"], "text_search")
        self.assertEqual(rows[0]["steps"][0]["action_text"], sample.question)

    def test_two_turn_does_not_extract_tool_answer(self):
        class SearchThenUnknownModel:
            def __init__(self):
                self.calls = 0

            def generate_text(self, prompt, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return '{"action":"text_search","query":"Mona Lisa museum"}'
                return '{"action":"stop","answer":"unknown"}'

        rows, trajectories = model_agent_two_turn_rollout(
            toy_samples()[1:2],
            policy=PolicyWrapper(model=SearchThenUnknownModel()),
        )

        self.assertEqual(rows[0]["steps"][0]["tool_type"], "text_search")
        self.assertEqual(rows[0]["steps"][1]["tool_type"], "stop")
        self.assertEqual(rows[0]["final_answer"], "unknown")
        self.assertFalse(rows[0]["final_correct"])
        self.assertEqual(trajectories[0].final_answer, "unknown")

    def test_two_turn_strips_answer_fields_from_search_observation(self):
        class SearchThenAnswerModel:
            def __init__(self):
                self.calls = 0

            def generate_text(self, prompt, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return '{"action":"text_search","query":"Mona Lisa museum"}'
                return '{"action":"stop","answer":"Louvre Museum"}'

        rows, _ = model_agent_two_turn_rollout(
            toy_samples()[1:2],
            policy=PolicyWrapper(model=SearchThenAnswerModel()),
        )

        search_raw = rows[0]["steps"][0]["raw_observation"]
        self.assertTrue(search_raw)
        self.assertFalse(any("answer" in item for item in search_raw))

    def test_dagig_reward_debug_8_samples(self):
        rows = dagig_reward_debug_rollout(toy_samples())
        self.assertEqual(len(rows), 8)
        self.assertTrue(all("token_rewards" in row for row in rows))
        self.assertTrue(all("local_ig" in row["steps"][0] for row in rows))


if __name__ == "__main__":
    unittest.main()
