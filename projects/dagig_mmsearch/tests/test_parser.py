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

    def test_action_as_tool_schema(self):
        parsed = parse_action('{"action":"text_search","query":"What country does this building belong to?"} {"action":"stop"}')
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.tool_type, "text_search")
        self.assertEqual(parsed.action_text, "What country does this building belong to?")

    def test_action_as_tool_prefers_specific_image_search(self):
        parsed = parse_action(
            '{"action":"image_search","query":"system shown in the image"} '
            '{"action":"stop"} '
            '{"action":"text_search","query":"What is the name of the system shown in the image?"}'
        )
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.tool_type, "image_search")
        self.assertEqual(parsed.action_text, "system shown in the image")


if __name__ == "__main__":
    unittest.main()
