import json
import unittest
from tempfile import NamedTemporaryFile

from mmsearch_r1.utils.dagig_offline import dagig_edge_for_extra_info, edge_tool_type, edge_weight, load_dagig_offline_edges


class MMSearchDAGIGOfflineTest(unittest.TestCase):
    def test_loads_selected_edges_by_question_id(self):
        row = {
            "sample_id": "fvqa_train_0",
            "selected_for_dependency_training": True,
            "dependency_edge": {
                "sample_id": "fvqa_train_0",
                "step0_tool": "image_search",
                "r0_minus_g0": 0.42,
            },
        }
        with NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            load_dagig_offline_edges.cache_clear()
            edge = dagig_edge_for_extra_info({"question_id": "fvqa_train_0"}, f.name)
            self.assertIsNotNone(edge)
            self.assertEqual(edge_tool_type(edge), "image_search")
            self.assertAlmostEqual(edge_weight(edge, "r0_minus_g0"), 0.42)

    def test_ignores_unselected_rows(self):
        row = {
            "sample_id": "fvqa_train_1",
            "selected_for_dependency_training": False,
            "dependency_edge": {"sample_id": "fvqa_train_1", "step0_tool": "image_search"},
        }
        with NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            load_dagig_offline_edges.cache_clear()
            self.assertIsNone(dagig_edge_for_extra_info({"question_id": "fvqa_train_1"}, f.name))


if __name__ == "__main__":
    unittest.main()
