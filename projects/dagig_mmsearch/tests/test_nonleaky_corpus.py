import unittest

from data.prepare_nonleaky_corpus import build_nonleaky_indexes
from data.schema import VQASample


class NonLeakyCorpusTest(unittest.TestCase):
    def test_nonleaky_indexes_redact_answer_fields_and_text(self):
        samples = [
            VQASample(
                sample_id="s1",
                question="Where is the landmark?",
                images=["image://s1"],
                gold_answers=["Paris"],
                metadata={"evidence": "Question: Where is the landmark? Answer: Paris. It is a visual landmark."},
            )
        ]

        text_rows, image_rows = build_nonleaky_indexes(samples, include_question_docs=True)

        self.assertNotIn("answer", text_rows[0])
        self.assertNotIn("answer", image_rows[0])
        self.assertNotIn("Paris", text_rows[0]["snippet"])
        self.assertNotIn("Paris", image_rows[0]["caption"])
        self.assertFalse(text_rows[0]["contains_gold_answer"])
        self.assertFalse(image_rows[0]["contains_gold_answer"])


if __name__ == "__main__":
    unittest.main()
