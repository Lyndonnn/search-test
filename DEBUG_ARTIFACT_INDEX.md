# DAG-IG Debug Artifact Index

This branch contains code plus a lightweight debug bundle for reviewing where the
current DAG-IG experiments fail. Large artifacts are intentionally not committed:
raw data packages, extracted image data, checkpoints, third-party model repos,
model weights, full train rollouts, and zip files remain local only.

## Main Code

- `scripts/dagig_train/`: experiment, scoring, evaluation, verifier, and runner scripts.

## High-Level Reports

- `DAGIG_EXPERIMENT_REPORT.md`
- `RN03_10_EXPERIMENT_REPORT.md`
- `LOCATE_ONLY_EXPERIMENT_REPORT.md`
- `results/dagig_rn03_10_grounded/GROUNDED_EXPERIMENT_REPORT.md`
- `results/dagig_rn03_10_grounded_rl/GROUNDED_DAGIG_RL_REPORT.md`
- `results/dagig_rn03_10_counterfactual_dagig/COUNTERFACTUAL_DAGIG_REPORT.md`
- `results/dagig_rn03_10_counterfactual_dagig_v2/DAGIG_V2_REPORT.md`
- `results/dagig_rn03_10_counterfactual_dagig_v31/DAGIG_V31_SCORER_REPORT.md`
- `results/dagig_rn03_10_answer_equivalence_v3/answer_equivalence_summary.md`

## Key Failure Chain

1. Grounded action SFT plus GroundingDINO works substantially better than direct
   bbox generation on localization.
2. Heuristic and v1 counterfactual DAG-IG were diagnostic only; they did not
   stably beat generic process rewards.
3. DAG-IG v2 correctly stopped because search counterfactuals were too weak.
4. Search counterfactual v3 fixed the search edge: search credit became strongly
   predictive of retrieval metrics.
5. DAG-IG v3.1 still failed on final supported-answer prediction. The current
   bottleneck is answer equivalence / evidence-to-answer scoring, not the search
   counterfactual.
6. Strict Answer Equivalence Verifier v3 uses a frozen judge for every rollout,
   but the positive rate remains very low and some false positives remain due to
   answer-type classification issues.

## Trajectory-Level Debug Files

These are committed for dev/test only so reviewers can inspect concrete rollouts:

- `results/dagig_rn03_10_counterfactual_dagig_v31/scored_rollouts_dev.jsonl`
- `results/dagig_rn03_10_counterfactual_dagig_v31/scored_rollouts_test.jsonl`
- `results/dagig_rn03_10_answer_equivalence_v3/answer_equivalence_dev.jsonl`
- `results/dagig_rn03_10_answer_equivalence_v3/answer_equivalence_test.jsonl`

## Not Committed

- `data/`
- `checkpoints/`
- `third_party/`
- `*.zip`
- full train rollout / verifier JSONL outputs
- generated images, crops, contact sheets, and model weights
