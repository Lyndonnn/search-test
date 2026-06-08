import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from utils.hf_reference import resolve_local_model_path


class HFReferenceTest(unittest.TestCase):
    def test_resolves_huggingface_snapshot_from_hub_cache(self):
        with TemporaryDirectory() as tmpdir:
            snapshot = os.path.join(
                tmpdir,
                "hub",
                "models--Qwen--Qwen2.5-VL-3B-Instruct",
                "snapshots",
                "abc123",
            )
            os.makedirs(snapshot)
            with open(os.path.join(snapshot, "config.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            with open(os.path.join(snapshot, "model-00001-of-00001.safetensors"), "w", encoding="utf-8") as f:
                f.write("")
            with patch.dict(os.environ, {"HF_HOME": tmpdir}, clear=False):
                self.assertEqual(
                    resolve_local_model_path("Qwen/Qwen2.5-VL-3B-Instruct"),
                    snapshot,
                )


if __name__ == "__main__":
    unittest.main()
