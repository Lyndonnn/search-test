import unittest

from agent.parser import parse_action, parse_final_answer


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

    def test_final_answer_parser_accepts_answer_action(self):
        parsed = parse_final_answer('{"action":"answer","answer":"hoarders"}')
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.tool_type, "stop")
        self.assertEqual(parsed.action_text, "hoarders")

    def test_final_answer_parser_rejects_search_observation_json(self):
        parsed = parse_final_answer('{"action":"text_search","observation":[{"answer":"Sochi"}]}')
        self.assertFalse(parsed.valid)
        self.assertEqual(parsed.tool_type, "stop")


if __name__ == "__main__":
    unittest.main()
