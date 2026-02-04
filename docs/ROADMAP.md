**M0 Sanity**
- 验收标准: 使用单卡跑 `total_steps=5` 完成训练，无崩溃、无 NaN、无 OOM；完成至少一次 rollout；生成 checkpoint 与日志。
- 成功日志关键词: `step=5`, `train finished`, `rollout finished`, `checkpoint saved`
- 失败日志关键词: `NaN`, `OOM`, `broadcast`, `NoneType`, `FileNotFoundError`, `CUDA error`

**M1 Zoom Grid**
- 验收标准: 训练/推理中可触发 `zoom` tool；轨迹里记录 `action_id` 与 `box`；可配置启用/禁用 zoom；trace_summary 能统计 zoom 触发率。
- 成功日志关键词: `tool=zoom`, `zoom_action`, `crop`, `trace_summary`
- 失败日志关键词: `invalid action_id`, `zoom spam`, `crop failed`, `shape mismatch`

**M2 Zoom Continuous**
- 验收标准: 连续 box 输出被裁剪到 [0,1]；最多 2 次 zoom/episode；area/boundary penalty 生效；trace_summary 输出分布统计 CSV。
- 成功日志关键词: `continuous zoom`, `box clipped`, `area penalty`, `csv written`
- 失败日志关键词: `box out of range`, `too many zoom`, `unstable loss`

**M3 Reward Judge**
- 验收标准: reward_mode=judge 可切换；evidence grounded 规则生效；offline_eval 输出 accuracy/evidence_rate/avg_tools。
- 成功日志关键词: `reward_mode=judge`, `evidence_rate`, `offline_eval finished`
- 失败日志关键词: `judge failed`, `missing evidence`, `eval crash`

**M4 Scale 2k**
- 验收标准: 2k parquet 训练可跑通；profile 脚本输出显存峰值与吞吐；单卡/多卡推荐配置在文档中可用。
- 成功日志关键词: `parquet loaded`, `tokens/s`, `gpu utilization`
- 失败日志关键词: `parquet read error`, `OOM`, `throughput drop`

**M5 Eval Release**
- 验收标准: ablation_runner 输出 CSV 对比表；复现指南从 clone 到结果可执行；可选 CI 通过 lint + check_rollout_tokens。
- 成功日志关键词: `ablation done`, `csv saved`, `ci passed`
- 失败日志关键词: `ablation failed`, `csv write error`, `ci failed`
