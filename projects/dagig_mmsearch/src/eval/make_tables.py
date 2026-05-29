from __future__ import annotations

from pathlib import Path

from eval.metrics import aggregate_rollouts
from eval.statistics import reward_diagnostics
from utils.io import read_jsonl, write_csv


METHOD_FILES = {
    "direct_vqa": "results/direct_vqa/direct_vqa_smoke.jsonl",
    "prompted_search": "results/prompted_search/prompted_search_smoke.jsonl",
    "outcome_only_rl": "results/outcome_rl/outcome_rl_smoke.jsonl",
    "local_ig_only": "results/local_ig/local_ig_smoke.jsonl",
    "dagig_lite_with_gate_cost": "results/dagig_lite/dagig_lite_smoke.jsonl",
}


def main() -> None:
    main_rows = []
    efficiency_rows = []
    ablation_rows = []
    attribution_rows = []
    for method, path in METHOD_FILES.items():
        rows = read_jsonl(path)
        if not rows:
            continue
        aggregate = aggregate_rollouts(rows, method)
        aggregate.update(reward_diagnostics(rows))
        main_rows.append(aggregate)
        efficiency_rows.append(
            {
                "method": method,
                "avg_tool_calls": aggregate["avg_tool_calls"],
                "avg_latency": aggregate["avg_latency"],
                "answer_per_tool_call": aggregate["answer_per_tool_call"],
                "tool_call_success_rate": aggregate["tool_call_success_rate"],
            }
        )
    ablations = [
        "No-search direct VQA",
        "Prompted search",
        "Outcome-only RL",
        "Local-IG only",
        "Local-IG + typed counterfactual",
        "Local-IG + self-evidence summary",
        "DAG-IG-Lite without gate",
        "DAG-IG-Lite with gate",
        "DAG-IG-Lite with cost",
        "Full DAG-IG",
        "Full DAG-IG + crop/OCR/select",
    ]
    for idx, name in enumerate(ablations):
        ablation_rows.append(
            {
                "ablation": name,
                "implemented_stage": idx <= 8,
                "lambda_dep": "0.0,0.25,0.5,1.0" if "DAG-IG" in name else "",
                "cf_samples": "1,2,4,8" if "IG" in name else "",
                "status": "smoke" if idx <= 8 else "planned_a100",
            }
        )
    attribution_rows.append(
        {
            "method": "local_ig_vs_dagig_lite",
            "leave_one_step_out_rank_corr": 0.5,
            "note": "Toy diagnostic placeholder; replace with reference-policy logprob recomputation for paper runs.",
        }
    )
    Path("paper_artifacts/tables").mkdir(parents=True, exist_ok=True)
    write_csv("paper_artifacts/tables/main_table.csv", main_rows)
    write_csv("paper_artifacts/tables/efficiency_table.csv", efficiency_rows)
    write_csv("paper_artifacts/tables/ablation_table.csv", ablation_rows)
    write_csv("paper_artifacts/tables/attribution_diagnostic.csv", attribution_rows)


if __name__ == "__main__":
    main()

