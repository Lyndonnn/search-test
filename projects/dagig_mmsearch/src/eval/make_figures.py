from __future__ import annotations

from collections import defaultdict

from utils.io import read_jsonl
from utils.visualization import plot_bar, plot_heatmap, plot_hist, plot_scatter, write_case_study_markdown


def main() -> None:
    rows = read_jsonl("results/dagig_lite/dagig_lite_smoke.jsonl") or read_jsonl(
        "results/dagig_lite/dagig_reward_debug.jsonl"
    )
    local = []
    future = []
    total = []
    tool_rewards: dict[str, list[float]] = defaultdict(list)
    matrix = [[0.0, 0.0], [0.0, 0.0]]
    xs = []
    ys = []
    for row in rows:
        xs.append(len(row.get("steps", [])))
        ys.append(1.0 if row.get("final_correct") else 0.0)
        for step in row.get("steps", []):
            local.append(float(step.get("local_ig", 0.0)))
            future.append(float(step.get("future_action_ig", 0.0)))
            total_reward = float(step.get("total_step_reward", 0.0))
            total.append(total_reward)
            tool_rewards[step.get("tool_type", "unknown")].append(total_reward)
            sid = int(step.get("step_id", 0))
            if sid < 2:
                matrix[sid][min(1, sid + 1)] = max(matrix[sid][min(1, sid + 1)], float(step.get("future_action_ig", 0.0)))
    plot_hist(total, "paper_artifacts/figures/reward_hist.png", "Reward histogram")
    plot_hist(local, "paper_artifacts/figures/local_ig_hist.png", "Local IG histogram")
    plot_hist(future, "paper_artifacts/figures/future_action_ig_hist.png", "Future-action IG histogram")
    plot_heatmap(matrix, "paper_artifacts/figures/dependency_heatmap.png", "Dependency edge heatmap")
    plot_scatter(xs, ys, "paper_artifacts/figures/accuracy_vs_toolcalls.png", "Accuracy vs tool calls", "tool calls", "accuracy")
    labels = list(tool_rewards)
    values = [sum(tool_rewards[label]) / max(1, len(tool_rewards[label])) for label in labels]
    plot_bar(labels, values, "paper_artifacts/figures/reward_by_tool_type.png", "Reward by tool type")
    if rows:
        write_case_study_markdown("paper_artifacts/case_studies/toy_dagig_case.md", rows[0])
        plot_heatmap(matrix, "paper_artifacts/figures/case_study_dagig.png", "DAG-IG case study")


if __name__ == "__main__":
    main()

