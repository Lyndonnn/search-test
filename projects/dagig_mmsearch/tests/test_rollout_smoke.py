import unittest

from agent.rollout import dagig_reward_debug_rollout, direct_vqa_rollout, prompted_search_rollout
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

    def test_dagig_reward_debug_8_samples(self):
        rows = dagig_reward_debug_rollout(toy_samples())
        self.assertEqual(len(rows), 8)
        self.assertTrue(all("token_rewards" in row for row in rows))
        self.assertTrue(all("local_ig" in row["steps"][0] for row in rows))


if __name__ == "__main__":
    unittest.main()

