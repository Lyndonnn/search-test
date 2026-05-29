# DAG-IG Paper Artifacts

This directory stores tables, figures, and case studies for the DAG-IG research prototype.

## Claims

Claim 1: IG-Search's answer-centered local IG is useful for text multi-hop retrieval, but it under-credits early enabling actions in multimodal tool chains.

Claim 2: Future-action IG treats later action spans as hindsight subgoals, assigning credit to crop, OCR, image search, text search, and select actions without human subgoal labels.

Claim 3: Typed counterfactuals are necessary for multimodal IG. Untyped replacement can reward modality and format differences instead of real information gain.

Claim 4: DAG-IG-Lite is the stable first implementation. Next-step dependency propagation is enough to test delayed credit assignment before adding full DAG propagation.

## Expected Outputs

- `tables/main_table.csv`
- `tables/efficiency_table.csv`
- `tables/ablation_table.csv`
- `tables/attribution_diagnostic.csv`
- `figures/accuracy_vs_toolcalls.png`
- `figures/local_ig_hist.png`
- `figures/future_action_ig_hist.png`
- `figures/dependency_heatmap.png`
- `figures/reward_by_tool_type.png`
- `figures/case_study_dagig.png`
- `case_studies/*.md`

