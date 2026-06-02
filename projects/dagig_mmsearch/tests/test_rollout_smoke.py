import unittest

from agent.policy_wrapper import PolicyWrapper
from agent.rollout import agentic_search_rollout, dagig_reward_debug_rollout, direct_vqa_rollout, prompted_search_rollout
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

    def test_dagig_reward_debug_8_samples(self):
        rows = dagig_reward_debug_rollout(toy_samples())
        self.assertEqual(len(rows), 8)
        self.assertTrue(all("token_rewards" in row for row in rows))
        self.assertTrue(all("local_ig" in row["steps"][0] for row in rows))


if __name__ == "__main__":
    unittest.main()
