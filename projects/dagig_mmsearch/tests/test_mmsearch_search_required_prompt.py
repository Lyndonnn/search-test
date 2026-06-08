import importlib.util
import pathlib
import unittest


def load_prompt_module():
    root = pathlib.Path(__file__).resolve().parents[3]
    path = root / "scripts" / "create_mmsearch_search_required_prompts.py"
    spec = importlib.util.spec_from_file_location("create_mmsearch_search_required_prompts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MMSearchSearchRequiredPromptTest(unittest.TestCase):
    def test_image_required_prompt_uses_strict_two_turn_format(self):
        module = load_prompt_module()
        prompt = module.IMAGE_SEARCH_REQUIRED_PROMPT

        self.assertIn("<image>", prompt)
        self.assertIn("<reason>brief reason for why image search is needed</reason><search><img></search>", prompt)
        self.assertIn("<reason>briefly use the returned visual evidence</reason><answer>final answer</answer>", prompt)
        self.assertIn("Do not output a bare <search><img></search> or bare <answer>.", prompt)

    def test_search_and_text_prompts_forbid_bare_answer_examples(self):
        module = load_prompt_module()
        for prompt in [module.SEARCH_REQUIRED_PROMPT, module.TEXT_SEARCH_REQUIRED_PROMPT]:
            self.assertIn("<image>", prompt)
            self.assertIn("<reason>", prompt)
            self.assertIn("<answer>final answer</answer>", prompt)
            self.assertNotIn("For example: <answer>Titanic</answer>", prompt)


if __name__ == "__main__":
    unittest.main()
