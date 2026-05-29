import unittest

from tools.crop import CropTool
from tools.image_search import ImageSearchTool
from tools.ocr import OCRTool
from tools.text_search import TextSearchTool


class ToolsTest(unittest.TestCase):
    def test_text_search_returns_topk(self):
        result = TextSearchTool(topk=2).run("Eiffel Tower location", topk=2)
        self.assertTrue(result.success)
        self.assertEqual(len(result.raw_observation), 2)
        self.assertIn("Eiffel", result.evidence_summary)

    def test_image_search_returns_topk(self):
        result = ImageSearchTool(topk=2).run("Golden Gate Bridge", topk=2)
        self.assertTrue(result.success)
        self.assertEqual(len(result.raw_observation), 2)

    def test_crop_does_not_crash(self):
        result = CropTool().run("[0, 0, 10, 10]")
        self.assertIn("bbox", result.raw_observation)

    def test_ocr_does_not_crash(self):
        result = OCRTool().run("missing.png", fallback_text="ABC 123")
        self.assertIn("OCR text", result.evidence_summary)


if __name__ == "__main__":
    unittest.main()

