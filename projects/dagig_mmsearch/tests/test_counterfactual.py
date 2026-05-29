import unittest

from reward.typed_pool import TypedCounterfactualPool
from reward.types import ToolStep


def step(step_id, tool_type, summary, answer=""):
    return ToolStep(
        step_id=step_id,
        tool_type=tool_type,
        action_text=summary,
        action_tokens=[step_id + 1],
        action_span=(step_id, step_id + 1),
        raw_observation=[{"title": summary, "snippet": summary, "answer": answer}],
        evidence_summary=summary,
        context_before_action="",
        context_after_observation=summary,
        metadata={"answer": answer, "sample_id": f"s{step_id}"},
    )


class CounterfactualTest(unittest.TestCase):
    def test_typed_pool_does_not_cross_modality(self):
        pool = TypedCounterfactualPool()
        text = step(0, "text_search", "Paris result", "Paris")
        image = step(1, "image_search", "Paris image", "Paris")
        other = step(2, "text_search", "Louvre result", "Louvre")
        pool.add(text)
        pool.add(image)
        pool.add(other)
        cfs = pool.sample("text_search", text, 2)
        self.assertEqual(len(cfs), 2)
        self.assertTrue(all(cf.metadata["cf_tool_type"] == "text_search" for cf in cfs))

    def test_cf_samples_count_and_fallback(self):
        pool = TypedCounterfactualPool()
        current = step(0, "ocr", "ABC")
        cfs = pool.sample("ocr", current, 3)
        self.assertEqual(len(cfs), 3)
        self.assertTrue(all(cf.metadata["cf_tool_type"] == "ocr" for cf in cfs))

    def test_hard_negative_metadata(self):
        pool = TypedCounterfactualPool()
        current = step(0, "text_search", "Paris entity", "Paris")
        negative = step(1, "text_search", "Paris entity wrong", "London")
        pool.add(current)
        pool.add(negative)
        cf = pool.sample("text_search", current, 1)[0]
        self.assertTrue(cf.metadata["whether_hard_negative"])


if __name__ == "__main__":
    unittest.main()

