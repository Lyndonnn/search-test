#!/usr/bin/env python3
import numpy as np

from mmsearch_r1.workers.multimodal.rollout.vllm_rollout_spmd import _extend_prompt_token_ids


def main() -> None:
    prompt = np.array([1, 2, 3, 4], dtype=np.int64)
    new_tokens = np.array([5, 6, 7, 8, 9], dtype=np.int64)
    max_model_len = 6

    updated, appended = _extend_prompt_token_ids(prompt, new_tokens, max_model_len)

    assert isinstance(updated, list), f"expected list, got {type(updated)}"
    assert len(updated) == max_model_len, f"expected len {max_model_len}, got {len(updated)}"
    assert appended == 2, f"expected appended=2, got {appended}"
    print("check_rollout_tokens: ok")


if __name__ == "__main__":
    main()
