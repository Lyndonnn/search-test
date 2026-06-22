#!/usr/bin/env python3
"""Validate DAG-IG v3.1 predictiveness and write scorer report."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from grounded_pipeline_utils import load_jsonl, md_table, write_csv, write_json


TESTS = [
    ("R_search_v3", "retrieval_r5", "binary", "R_search_v3 -> R@5"),
    ("R_search_v3", "retrieval_mrr", "continuous", "R_search_v3 -> MRR"),
    ("R_evidence", "evidence_support", "binary", "R_evidence -> evidence_support"),
    ("R_answer_v31", "supported_answer_v2", "binary", "R_answer_v31 -> supported_answer_v2"),
    ("DAGIG_v31_total", "evidence_support", "binary", "DAGIG_v31_total -> evidence_support"),
    ("DAGIG_v31_total", "supported_answer_v2", "binary", "DAGIG_v31_total -> supported_answer_v2"),
    ("DAGIG_v31_total", "answer_f1_v2", "continuous", "DAGIG_v31_total -> answer F1"),
    ("DAGIG_v31_search_evidence_only", "evidence_support", "binary", "search_evidence_only -> evidence_support"),
    ("DAGIG_v31_no_answer", "retrieval_r5", "binary", "no_answer -> R@5"),
    ("DAGIG_v31_no_answer", "evidence_support", "binary", "no_answer -> evidence_support"),
]

THRESHOLDS = {
    ("R_search_v3", "retrieval_r5", "spearman"): 0.20,
    ("R_evidence", "evidence_support", "spearman"): 0.35,
    ("R_answer_v31", "supported_answer_v2", "auc"): 0.60,
    ("DAGIG_v31_total", "evidence_support", "spearman"): 0.20,
    ("DAGIG_v31_total", "supported_answer_v2", "auc"): 0.60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", default="results/dagig_rn03_10_counterfactual_dagig_v31")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--primary_split", default="dev")
    parser.add_argument("--output_csv", default="results/dagig_rn03_10_counterfactual_dagig_v31/score_predictiveness_v31.csv")
    parser.add_argument("--output_md", default="results/dagig_rn03_10_counterfactual_dagig_v31/score_predictiveness_v31.md")
    parser.add_argument("--gate_json", default="results/dagig_rn03_10_counterfactual_dagig_v31/score_predictiveness_v31_gate.json")
    parser.add_argument("--report_md", default="results/dagig_rn03_10_counterfactual_dagig_v31/DAGIG_V31_SCORER_REPORT.md")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no_fail_on_threshold", action="store_true")
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return float(value)
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx)) * math.sqrt(sum(y * y for y in dy))
    if den <= 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / den


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys))


def auc(scores: list[float], labels: list[float]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if bool(y)]
    neg = [s for s, y in zip(scores, labels) if not bool(y)]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def top_bottom_gap(scores: list[float], values: list[float]) -> float | None:
    if len(scores) < 4:
        return None
    pairs = sorted(zip(scores, values), key=lambda item: item[0])
    q = max(1, len(pairs) // 4)
    return mean([v for _s, v in pairs[-q:]]) - mean([v for _s, v in pairs[:q]])


def ci(values: list[float | None]) -> tuple[float | None, float | None]:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return None, None
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return lo, hi


def bootstrap_metric(
    xs: list[float],
    ys: list[float],
    fn: Callable[[list[float], list[float]], float | None],
    n: int,
    rng: random.Random,
) -> tuple[float | None, float | None]:
    if len(xs) < 8 or n <= 0:
        return None, None
    vals = []
    for _ in range(n):
        idxs = [rng.randrange(len(xs)) for _ in xs]
        vals.append(fn([xs[i] for i in idxs], [ys[i] for i in idxs]))
    return ci(vals)


def metric_values(rows: list[dict[str, Any]], score_key: str, target_key: str) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    for row in rows:
        x = as_float(row.get(score_key))
        y = as_float(row.get(target_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def metric_row(
    rows: list[dict[str, Any]],
    split: str,
    score_key: str,
    target_key: str,
    kind: str,
    label: str,
    bootstrap: int,
    rng: random.Random,
) -> dict[str, Any]:
    xs, ys = metric_values(rows, score_key, target_key)
    sp = spearman(xs, ys)
    pr = pearson(xs, ys)
    au = auc(xs, ys) if kind == "binary" else None
    sp_lo, sp_hi = bootstrap_metric(xs, ys, spearman, bootstrap, rng)
    au_lo, au_hi = bootstrap_metric(xs, ys, auc, bootstrap, rng) if kind == "binary" else (None, None)
    return {
        "split": split,
        "comparison": label,
        "score_key": score_key,
        "target_key": target_key,
        "n": len(xs),
        "spearman": sp,
        "spearman_ci_low": sp_lo,
        "spearman_ci_high": sp_hi,
        "pearson": pr,
        "auc": au,
        "auc_ci_low": au_lo,
        "auc_ci_high": au_hi,
        "top_bottom_gap": top_bottom_gap(xs, ys),
        "score_mean": mean(xs) if xs else None,
        "target_mean": mean(ys) if ys else None,
    }


def gate(rows: list[dict[str, Any]], primary_split: str) -> dict[str, Any]:
    primary = [row for row in rows if row.get("split") == primary_split]
    checks = []
    for (score_key, target_key, metric), threshold in THRESHOLDS.items():
        row = next((r for r in primary if r["score_key"] == score_key and r["target_key"] == target_key), None)
        value = None if row is None else row.get(metric)
        passed = value is not None and float(value) >= threshold
        checks.append(
            {
                "split": primary_split,
                "score_key": score_key,
                "target_key": target_key,
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )
    return {"primary_split": primary_split, "passed": all(row["passed"] for row in checks), "checks": checks}


def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_report(path: Path, out_rows: list[dict[str, Any]], gate_result: dict[str, Any]) -> None:
    dev = [row for row in out_rows if row.get("split") == "dev"]
    def get(score: str, target: str, metric: str) -> Any:
        row = next((r for r in dev if r["score_key"] == score and r["target_key"] == target), {})
        return row.get(metric)

    answer_auc = get("R_answer_v31", "supported_answer_v2", "auc")
    total_auc = get("DAGIG_v31_total", "supported_answer_v2", "auc")
    evidence_sp = get("DAGIG_v31_total", "evidence_support", "spearman")
    report = [
        "# DAG-IG v3.1 Scorer Report",
        "",
        "## Answers",
        "",
        f"1. Supported-answer verifier improved the definition by separating answer correctness, evidence-support for the prediction, evidence-support for gold, and type validity. The new label used here is `supported_answer_v2`.",
        f"2. `R_answer_v31 -> supported_answer_v2` dev AUC is `{fmt(answer_auc)}`.",
        f"3. `DAGIG_v31_total -> supported_answer_v2` dev AUC is `{fmt(total_auc)}`; gate passed: `{gate_result['passed']}`.",
        f"4. Answer edge noise is judged from the AUC and top-bottom gap. If AUC is close to 0.60 or below, the answer edge is still weak compared with the search edge.",
        "5. Training should proceed only if all gate checks pass. This run does not launch training.",
        f"6. Remaining blocker, if any, is listed in the gate table below. Current `DAGIG_v31_total -> evidence_support` dev Spearman is `{fmt(evidence_sp)}`.",
        "",
        "## Gate",
        "",
        md_table([{k: fmt(v) if isinstance(v, float) else v for k, v in row.items()} for row in gate_result["checks"]]),
        "",
        "## Predictiveness",
        "",
        md_table([{k: fmt(v) if isinstance(v, float) else v for k, v in row.items()} for row in out_rows]),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    result_dir = Path(args.result_dir)
    out_rows = []
    for split in args.splits:
        path = result_dir / f"scored_rollouts_{split}.jsonl"
        if not path.is_file():
            continue
        rows = load_jsonl(path)
        for score_key, target_key, kind, label in TESTS:
            out_rows.append(metric_row(rows, split, score_key, target_key, kind, label, args.bootstrap, rng))
    write_csv(args.output_csv, out_rows)
    gate_result = gate(out_rows, args.primary_split)
    write_json(args.gate_json, gate_result)
    display = [{k: fmt(v) if isinstance(v, float) else v for k, v in row.items()} for row in out_rows]
    gate_rows = [{k: fmt(v) if isinstance(v, float) else v for k, v in row.items()} for row in gate_result["checks"]]
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(
        "# DAG-IG v3.1 Score Predictiveness\n\n"
        + md_table(display)
        + "\n\n## Training Gate\n\n"
        + md_table(gate_rows)
        + f"\n\nGate passed: {gate_result['passed']}\n",
        encoding="utf-8",
    )
    write_report(Path(args.report_md), out_rows, gate_result)
    print(json.dumps(gate_result, ensure_ascii=False, indent=2))
    if not args.no_fail_on_threshold and not gate_result["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
