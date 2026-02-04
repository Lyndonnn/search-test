**M0 Sanity**
Goal: run a 5-step training sanity check and fix two known bugs (token concat / prompt None).

**How To Verify**
1. Confirm data and prompt files exist:
`ls -lh mmsearch_r1/data/mini_data.pq`
`ls -lh mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl`
2. Run M0:
`HYDRA_FULL_ERROR=1 bash scripts/run_m0_sanity.sh`
3. Pass criteria:
Logs include `Total training steps: 5` and `Training Progress`
`Global Step` reaches 5
No `broadcast`, `NoneType`, `OOM`, `NaN`

**How To Roll Back**
To revert the two bug fixes:
`git apply -R patches/0001_fix_token_concat.diff`
`git apply -R patches/0002_fix_prompt_fallback.diff`
