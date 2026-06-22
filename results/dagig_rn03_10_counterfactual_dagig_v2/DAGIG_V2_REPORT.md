# DAG-IG v2 Report

## Status

DAG-IG v2 stopped before scoring/training because the rebuilt search counterfactuals failed the required quality gate.

The prompt required stopping if fewer than 60% of rows have the real query retrieval score higher than the counterfactual query score. The v2 search counterfactuals did not meet this threshold:

| split | n | real beats best CF | real beats mean CF | pass |
| --- | ---: | ---: | ---: | --- |
| train | 458 | 0.3450 | 0.6092 | false |
| dev | 98 | 0.3673 | 0.5204 | false |
| test | 64 | 0.3125 | 0.6719 | false |

Because of this, I did not run v2 preference building, DPO/rejection SFT, model evaluation, or ablations.

## What Was Completed

Created and ran:

- `scripts/dagig_train/36_diagnose_dagig_v1_failure.py`
- `scripts/dagig_train/37_audit_counterfactual_quality.py`
- `scripts/dagig_train/38_rebuild_search_counterfactuals_v2.py`
- `scripts/dagig_train/39_evidence_support_verifier.py`

Created but not executed due to the search-counterfactual stop condition:

- `scripts/dagig_train/40_score_counterfactual_dagig_v2.py`
- `scripts/dagig_train/41_validate_dagig_v2_predictiveness.py`

The later training/evaluation scripts were not created or run because the prompt makes them conditional on v2 predictiveness passing.

## V1 Failure Diagnosis

V1 had usable grounding and evidence signals, but weak search credit:

- `R_ground -> IoU` was strong: dev Spearman 0.4904, test Spearman 0.5219.
- `R_evidence -> evidence_support` was moderate: dev Spearman 0.3038, test Spearman 0.4072.
- `R_search -> R@5` was weak: dev Spearman 0.0311, test Spearman 0.1815.
- `DAGIG_total -> answer F1` was weak or negative.

Generic-process matched or beat v1 because it directly rewards retrieval/evidence process metrics, while v1 search credit used weak lexical proxies and noisy counterfactuals.

## Counterfactual Quality Audit

Counterfactual audit results:

| split | n | pass | fail | fail rate |
| --- | ---: | ---: | ---: | ---: |
| train | 458 | 327 | 131 | 0.2860 |
| dev | 98 | 74 | 24 | 0.2449 |
| test | 64 | 47 | 17 | 0.2656 |

The most common hard failure was duplicated search counterfactuals. This means many rows do not provide a meaningful contrast between real and counterfactual search behavior.

## Evidence Verifier

Hybrid heuristic verifier on teacher evidence:

| split | n | supports answer rate | mean support score |
| --- | ---: | ---: | ---: |
| train | 458 | 0.7227 | 0.6554 |
| dev | 98 | 0.8163 | 0.7195 |
| test | 64 | 0.7344 | 0.6553 |

This verifier is usable as a lightweight diagnostic, but it is not yet strong enough for a paper-level entailment claim.

## Why Search Counterfactuals Failed

Many query counterfactuals are not actually worse than the real query under the current BM25 retrieval setup.

Common patterns:

- Entity-removed query still contains enough task-specific terms to retrieve the correct document.
- Generic query still retains location/task words such as "Los Angeles home loan specialist branches count".
- Hard-negative entity substitution often does not change BM25 ranking enough when the remaining query terms dominate.
- Some real queries are themselves weak, so counterfactuals can tie or slightly outperform them.

Example failure pattern:

`Bank of America Los Angeles home loan specialist branches count`

The entity-removed and generic versions still retrieve the same target at rank 1 because the non-entity terms are sufficient.

This makes the edge delta unfair: a good real query receives little or no credit because the counterfactual query is not truly counterfactual.

## Reviewer Questions

1. What is the contribution beyond generic process reward?

Not established yet for v2. The search counterfactual gate failed before training.

2. Are counterfactuals typed and fair?

They are typed, but not fair enough. Search counterfactuals are too often equivalent to the real query under retrieval.

3. Does DAG-IG score predict downstream success?

V2 predictiveness was not run because search counterfactual quality failed first. V1 predicts grounding well but search poorly.

4. Does DAG-IG training improve process metrics over generic-process?

Not tested for v2. Training was correctly blocked.

5. Which edge contributes most?

From v1, the grounding edge is strongest. Evidence is moderate. Search is the bottleneck.

6. Do ablations prove dependency gates and counterfactual deltas matter?

No. Ablations are not justified until the v2 scorer passes predictiveness thresholds and beats generic-process.

7. What remains weak?

Search counterfactual construction, retrieval-sensitive entity substitution, and evidence verification strength.

## Required Next Fix

Do not train yet. The next step is to rebuild search counterfactuals so that counterfactual queries are truly retrieval-worse while still plausible.

Concrete fixes:

- Build hard negatives from the retrieval index, not random other anchors.
- Choose replacement entities whose documents are retrieved by the hard-negative query.
- Remove or mask not just brand/entity tokens, but also target-specific disambiguators when constructing generic queries.
- Use retrieval-top overlap constraints: a valid counterfactual should reduce target rank or remove target from top 5.
- Reject counterfactuals that tie the real query score.
- Store per-query rank deltas and use only rows where real query beats all counterfactuals.

Only after search counterfactuals pass the 60% gate should DAG-IG v2 scoring, predictiveness validation, preference pairs, DPO/rejection SFT, evaluation, and ablations proceed.

