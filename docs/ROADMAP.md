**M0 Sanity**
- Acceptance: single-GPU run completes `total_steps=5` with no crash, NaN, or OOM; at least one rollout finishes; logs and checkpoint are produced.
- Success log keywords: `step=5`, `train finished`, `rollout finished`, `checkpoint saved`
- Failure log keywords: `NaN`, `OOM`, `broadcast`, `NoneType`, `FileNotFoundError`, `CUDA error`

**M1 Zoom Grid**
- Acceptance: `zoom` tool can be triggered in training/inference; trajectory logs include `action_id` and `box`; zoom can be enabled/disabled via config; trace_summary reports zoom rate.
- Success log keywords: `tool=zoom`, `zoom_action`, `crop`, `trace_summary`
- Failure log keywords: `invalid action_id`, `zoom spam`, `crop failed`, `shape mismatch`

**M2 Zoom Continuous**
- Acceptance: continuous box is clipped to [0,1]; max 2 zooms per episode; area/boundary penalties apply; trace_summary writes distribution CSV.
- Success log keywords: `continuous zoom`, `box clipped`, `area penalty`, `csv written`
- Failure log keywords: `box out of range`, `too many zoom`, `unstable loss`

**M3 Reward Judge**
- Acceptance: reward_mode=judge is switchable; evidence-grounded rule applies; offline_eval outputs accuracy/evidence_rate/avg_tools.
- Success log keywords: `reward_mode=judge`, `evidence_rate`, `offline_eval finished`
- Failure log keywords: `judge failed`, `missing evidence`, `eval crash`

**M4 Scale 2k**
- Acceptance: 2k parquet training runs; profile script outputs VRAM peak and throughput; single/multi-GPU configs documented.
- Success log keywords: `parquet loaded`, `tokens/s`, `gpu utilization`
- Failure log keywords: `parquet read error`, `OOM`, `throughput drop`

**M5 Eval Release**
- Acceptance: ablation_runner writes CSV comparison; repro guide is runnable from clone to results; optional CI passes lint + check_rollout_tokens.
- Success log keywords: `ablation done`, `csv saved`, `ci passed`
- Failure log keywords: `ablation failed`, `csv write error`, `ci failed`
