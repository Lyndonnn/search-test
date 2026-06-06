import unittest

from mmsearch_r1.utils.tools.image_search import call_image_search


class MMSearchImageSearchTest(unittest.TestCase):
    def test_empty_image_source_returns_failed_observation(self):
        text, images, stat = call_image_search("")
        self.assertIn("skipped", text)
        self.assertEqual(images, [])
        self.assertFalse(stat["success"])
        self.assertEqual(stat["backend"], "missing_image_source")


if __name__ == "__main__":
    unittest.main()
