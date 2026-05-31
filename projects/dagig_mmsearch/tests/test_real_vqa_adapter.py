import unittest

from data.dataset_mixer import build_indexes_from_samples
from data.real_vqa_adapter import row_to_sample
from utils.io import read_jsonl


class RealVQAAdapterTest(unittest.TestCase):
    def test_common_row_to_sample(self):
        row = {
            "question": "What landmark is shown?",
            "answers": ["Eiffel Tower"],
            "image_url": "https://example.test/eiffel.jpg",
            "caption": "A tower in Paris.",
            "id": "ex1",
        }
        sample = row_to_sample(row, 0, "unit")
        self.assertEqual(sample.sample_id, "ex1")
        self.assertEqual(sample.gold_answers[0], "Eiffel Tower")
        self.assertEqual(sample.images[0], "https://example.test/eiffel.jpg")
        self.assertTrue(sample.metadata["needs_search"])

    def test_build_indexes_from_samples(self):
        row = {
            "question": "Which city?",
            "answer": "Paris",
            "image": {"id": "img1"},
            "id": "ex2",
        }
        sample = row_to_sample(row, 0, "unit")
        build_indexes_from_samples(
            [sample],
            text_path="data/cache/test_real_text_index.jsonl",
            image_path="data/cache/test_real_image_index.jsonl",
        )
        text_rows = read_jsonl("data/cache/test_real_text_index.jsonl")
        image_rows = read_jsonl("data/cache/test_real_image_index.jsonl")
        self.assertEqual(text_rows[0]["answer"], "Paris")
        self.assertEqual(image_rows[0]["answer"], "Paris")


if __name__ == "__main__":
    unittest.main()
