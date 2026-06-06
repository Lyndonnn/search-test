import unittest
from dataclasses import replace
from io import BytesIO

from PIL import Image

from mmsearch_r1.utils.tools.offline_search import OfflineDoc, format_image_results


def png_bytes(color):
    image = Image.new("RGB", (16, 16), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class MMSearchOfflineSearchTest(unittest.TestCase):
    def test_image_results_include_one_placeholder_per_image(self):
        doc = OfflineDoc(
            question_id="q0",
            question="Which city?",
            answer="Paris",
            candidate_answers=[],
            category="unit",
            image_thumb_bytes=png_bytes((255, 0, 0)),
            image_dhash=[],
            image_hist=[],
            tokens=[],
            tf={},
            doc_len=0,
        )
        results = [(doc, 1.0), (replace(doc, question_id="q1"), 0.5)]
        text, images, titles = format_image_results(results)
        self.assertEqual(len(images), 2)
        self.assertEqual(len(titles), 2)
        self.assertEqual(text.count("<|image_pad|>"), 2)


if __name__ == "__main__":
    unittest.main()
