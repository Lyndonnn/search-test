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
BROKEN_PATCHED_IMPORT = """try:
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    index_first_axis = None
    pad_input = None
    rearrange = None
    unpad_input = None
    _FLASH_ATTN_AVAILABLE = False"""
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
FLASH_FALLBACK_NAMES = {"index_first_axis", "pad_input", "rearrange", "unpad_input"}


def _is_flash_fallback_tail_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped in {"_FLASH_ATTN_AVAILABLE = True", "except ImportError:", "_FLASH_ATTN_AVAILABLE = False"}:
        return True
    return any(stripped == f"{name} = None" for name in FLASH_FALLBACK_NAMES)


def normalize_flash_import_block(text: str) -> tuple[str, bool]:
    """Replace original or previously broken flash-attn fallback blocks.

    Older versions of this script could leave `try:` followed by an unindented
    `from flash_attn...` line. Do not rely on exact string matching here; repair
    any block whose current or next line contains the flash-attn import.
    """

    lines = text.splitlines()
    out: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        if stripped == "try:" and IMPORT_LINE in next_line:
            out.extend(PATCHED_IMPORT.splitlines())
            i += 2
            while i < len(lines) and _is_flash_fallback_tail_line(lines[i]):
                i += 1
            changed = True
            continue

        if IMPORT_LINE in line:
            out.extend(PATCHED_IMPORT.splitlines())
            i += 1
            while i < len(lines) and _is_flash_fallback_tail_line(lines[i]):
                i += 1
            changed = True
            continue

        out.append(line)
        i += 1

    normalized = "\n".join(out)
    if text.endswith("\n"):
        normalized += "\n"
    return normalized, changed


def patch_text(text: str) -> tuple[str, bool]:
    changed = False
    text, import_changed = normalize_flash_import_block(text)
    changed = changed or import_changed

    # Kept for backward compatibility with exact bad files from early debug
    # sessions. The line-wise normalizer above should normally catch this.
    if BROKEN_PATCHED_IMPORT in text:
        text = text.replace(BROKEN_PATCHED_IMPORT, PATCHED_IMPORT, 1)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Restore the target file from the pinned veRL checkout and do not patch.",
    )
    return parser.parse_args(argv)


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
    if args.reset_first or args.reset_only:
        reset_target(verl_root)
    target = verl_root / "verl" / "workers" / "actor" / "dp_actor.py"
    if not target.is_file():
        raise FileNotFoundError(f"Missing veRL actor file: {target}")
    if args.reset_only:
        print(f"MMSearch-R1 veRL flash-attn fallback: reset original {target}")
        return
    changed = patch_file(target)
    status = "patched" if changed else "already patched"
    print(f"MMSearch-R1 veRL flash-attn fallback: {status} {target}")


if __name__ == "__main__":
    main()
