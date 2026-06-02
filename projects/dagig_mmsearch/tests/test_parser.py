import unittest

from agent.parser import parse_action


class ParserTest(unittest.TestCase):
    def test_action_json_parse(self):
        parsed = parse_action('{"tool": "text_search", "action": "Eiffel Tower location"}')
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.tool_type, "text_search")
        self.assertEqual(parsed.action_text, "Eiffel Tower location")

    def test_invalid_json_fallback(self):
        parsed = parse_action("{bad json")
        self.assertFalse(parsed.valid)
        self.assertEqual(parsed.tool_type, "stop")
        self.assertIn("bad json", parsed.action_text)

    def test_multiple_json_objects_prefers_valid_search(self):
        parsed = parse_action(
            '{"tool":"text_search","action":"query"} '
            '{"tool":"stop","action":"final answer"} '
            "I should search first."
        )
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.tool_type, "text_search")

    def test_query_field_overrides_placeholder_action(self):
        parsed = parse_action('{"tool":"text_search","action":"query","query":"name of the organization"}')
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.tool_type, "text_search")
        self.assertEqual(parsed.action_text, "name of the organization")


if __name__ == "__main__":
    unittest.main()
