import unittest

from scripts.patch_mmsearch_r1_verl_flash_attn import parse_args, patch_text


class MMSearchVerlPatchTest(unittest.TestCase):
    def test_flash_attn_import_becomes_optional(self):
        source = """from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

class Actor:
    def forward(self):
        if False:
            pass
            if self.use_remove_padding:
                input_ids_rmpad = unpad_input(input_ids.unsqueeze(-1), attention_mask)
"""
        patched, changed = patch_text(source)
        self.assertTrue(changed)
        self.assertIn("try:", patched)
        self.assertIn("_FLASH_ATTN_AVAILABLE = False", patched)
        self.assertIn("flash_attn is required when use_remove_padding=True", patched)

    def test_patch_is_idempotent(self):
        source = """try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    index_first_axis = None
    pad_input = None
    rearrange = None
    unpad_input = None
    _FLASH_ATTN_AVAILABLE = False

class Actor:
    def forward(self):
        if False:
            pass
            if self.use_remove_padding:
                if not _FLASH_ATTN_AVAILABLE:
                    raise ImportError(
                        "flash_attn is required when use_remove_padding=True. "
                        "Install flash-attn or set actor_rollout_ref.model.use_remove_padding=False."
                    )
                input_ids_rmpad = unpad_input(input_ids.unsqueeze(-1), attention_mask)
"""
        patched, changed = patch_text(source)
        self.assertFalse(changed)
        self.assertEqual(patched, source)

    def test_reset_only_arg(self):
        args = parse_args(["--reset-only", "--verl-root", "/tmp/verl"])
        self.assertTrue(args.reset_only)
        self.assertEqual(args.verl_root, "/tmp/verl")


if __name__ == "__main__":
    unittest.main()
