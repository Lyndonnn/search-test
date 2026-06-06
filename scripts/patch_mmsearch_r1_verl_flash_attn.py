#!/usr/bin/env python3
"""Make flash-attn optional for the pinned MMSearch-R1 veRL debug stack.

The original veRL actor imports flash_attn at module import time. Our A100
debug scripts explicitly set use_remove_padding=False and eager attention, so
flash-attn is not needed for correctness. This patch keeps flash-attn required
only when remove-padding is actually enabled.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


IMPORT_LINE = "from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input"
PATCHED_IMPORT = """try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    index_first_axis = None
    pad_input = None
    rearrange = None
    unpad_input = None
    _FLASH_ATTN_AVAILABLE = False"""

REMOVE_PADDING_LINE = "            if self.use_remove_padding:\n"
REMOVE_PADDING_GUARD = """            if self.use_remove_padding:
                if not _FLASH_ATTN_AVAILABLE:
                    raise ImportError(
                        "flash_attn is required when use_remove_padding=True. "
                        "Install flash-attn or set actor_rollout_ref.model.use_remove_padding=False."
                    )
"""


def patch_text(text: str) -> tuple[str, bool]:
    changed = False
    if "_FLASH_ATTN_AVAILABLE" not in text and IMPORT_LINE in text:
        text = text.replace(IMPORT_LINE, PATCHED_IMPORT, 1)
        changed = True

    if "_FLASH_ATTN_AVAILABLE" not in text:
        raise RuntimeError("Could not locate the flash_attn import block to patch.")

    if "flash_attn is required when use_remove_padding=True" not in text:
        if REMOVE_PADDING_LINE not in text:
            raise RuntimeError("Could not locate use_remove_padding branch to guard.")
        text = text.replace(REMOVE_PADDING_LINE, REMOVE_PADDING_GUARD, 1)
        changed = True

    return text, changed


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    patched, changed = patch_text(text)
    if changed:
        path.write_text(patched, encoding="utf-8")
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verl-root",
        default=os.environ.get("MMSEARCH_R1_VERL_ROOT", "third_party/mmsearch_r1_verl"),
        help="Path to the pinned MMSearch-R1 veRL checkout.",
    )
    parser.add_argument(
        "--reset-first",
        action="store_true",
        help="Restore the target file from the pinned veRL checkout before patching.",
    )
    return parser.parse_args()


def reset_target(verl_root: Path) -> None:
    if not (verl_root / ".git").is_dir():
        return
    subprocess.run(
        ["git", "-C", str(verl_root), "checkout", "--", "verl/workers/actor/dp_actor.py"],
        check=True,
    )


def main() -> None:
    args = parse_args()
    verl_root = Path(args.verl_root)
    if args.reset_first:
        reset_target(verl_root)
    target = verl_root / "verl" / "workers" / "actor" / "dp_actor.py"
    if not target.is_file():
        raise FileNotFoundError(f"Missing veRL actor file: {target}")
    changed = patch_file(target)
    status = "patched" if changed else "already patched"
    print(f"MMSearch-R1 veRL flash-attn fallback: {status} {target}")


if __name__ == "__main__":
    main()
