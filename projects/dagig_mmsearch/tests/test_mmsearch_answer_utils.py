import unittest
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ANSWER_UTILS_PATH = ROOT / "mmsearch_r1" / "workers" / "multimodal" / "reward" / "answer_utils.py"
spec = importlib.util.spec_from_file_location("mmsearch_answer_utils", ANSWER_UTILS_PATH)
answer_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(answer_utils)
normalize_answer_list = answer_utils.normalize_answer_list


class MMSearchAnswerUtilsTest(unittest.TestCase):
    def test_json_string_answers(self):
        self.assertEqual(normalize_answer_list('["Paris", "France"]'), ["Paris", "France"])

    def test_plain_string_answer(self):
        self.assertEqual(normalize_answer_list("Paris"), ["Paris"])

    def test_numpy_like_array_answers(self):
        class NumpyLike:
            def tolist(self):
                return ["1990", "1989"]

        self.assertEqual(normalize_answer_list(NumpyLike()), ["1990", "1989"])

    def test_nested_answers(self):
        self.assertEqual(normalize_answer_list(["Paris", ["France"]]), ["Paris", "France"])


if __name__ == "__main__":
    unittest.main()
