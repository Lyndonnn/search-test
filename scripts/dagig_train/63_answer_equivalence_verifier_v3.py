#!/usr/bin/env python3
"""Answer-equivalence verifier v3 with type-aware rules and optional frozen judge."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import extract_segments, load_jsonl, md_table, token_f1, tokenize, write_jsonl


SPLITS = ["train", "dev", "test"]
TAGS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]
STRUCTURED_TYPES = {"phone", "email", "company_id", "count", "date", "price"}
JUDGE_TYPES = {"address", "entity_name", "title", "description", "other"}
DEFAULT_JUDGE_PATHS = [
    Path("/storage/zhengxiang/models/Qwen2.5-7B-Instruct"),
    Path("/data/zhengxiang/code/dagig/models/Qwen2.5-7B-Instruct"),
]
MONTHS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
GENERIC_ANSWERS = {
    "company",
    "organization",
    "website",
    "store",
    "address",
    "phone",
    "number",
    "email",
    "flag",
    "building",
    "unknown",
    "none",
    "not available",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_dir", default="results/dagig_rn03_10_counterfactual_dagig_v31")
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_answer_equivalence_v3")
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--judge_model_path", default="auto")
    parser.add_argument("--judge_device", default="cuda")
    parser.add_argument("--judge_max_new_tokens", type=int, default=192)
    parser.add_argument("--judge_max_input_tokens", type=int, default=3072)
    parser.add_argument("--judge_batch_size", type=int, default=8)
    parser.add_argument("--disable_judge", action="store_true")
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


def clean_text(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def normalize_text(text: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", clean_text(text).lower()).strip()


def compact_alnum(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(text).lower())


def digit_text(text: Any) -> str:
    return re.sub(r"\D+", "", clean_text(text))


def extract_phone(text: Any) -> list[str]:
    raw = clean_text(text)
    out = []
    for match in re.finditer(r"(?:\+?\d[\d\s()./-]{5,}\d)", raw):
        digits = digit_text(match.group(0))
        if len(digits) >= 7:
            out.append(digits)
    return sorted(set(out), key=len, reverse=True)


def phone_match(a: Any, b: Any) -> bool:
    aa = extract_phone(a)
    bb = extract_phone(b)
    for x in aa:
        for y in bb:
            if x == y:
                return True
            suffix = 8 if min(len(x), len(y)) >= 8 else 7
            if len(x) >= suffix and len(y) >= suffix and x[-suffix:] == y[-suffix:]:
                return True
    return False


def extract_emails(text: Any) -> list[str]:
    return sorted(set(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", clean_text(text).lower())))


def extract_numbers(text: Any) -> list[str]:
    vals = [x.replace(",", "") for x in re.findall(r"\d+(?:[,.]\d+)?", clean_text(text))]
    words = tokenize(str(text or ""))
    for tok in words:
        if tok in NUMBER_WORDS:
            vals.append(str(NUMBER_WORDS[tok]))
    return sorted(set(vals))


def extract_dates(text: Any) -> list[str]:
    raw = clean_text(text).lower()
    dates = set()
    for y, m, d in re.findall(r"\b(20\d{2}|19\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", raw):
        dates.add(f"{int(y):04d}-{int(m):02d}-{int(d):02d}")
    for d, m, y in re.findall(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2}|19\d{2})\b", raw):
        dates.add(f"{int(y):04d}-{int(m):02d}-{int(d):02d}")
    month_re = "|".join(sorted(MONTHS, key=len, reverse=True))
    for mon, d, y in re.findall(rf"\b({month_re})\s+(\d{{1,2}}),?\s+(20\d{{2}}|19\d{{2}})\b", raw):
        dates.add(f"{int(y):04d}-{MONTHS[mon]}-{int(d):02d}")
    for d, mon, y in re.findall(rf"\b(\d{{1,2}})\s+({month_re})\s+(20\d{{2}}|19\d{{2}})\b", raw):
        dates.add(f"{int(y):04d}-{MONTHS[mon]}-{int(d):02d}")
    return sorted(dates)


def extract_prices(text: Any) -> list[str]:
    raw = clean_text(text)
    out = []
    for match in re.finditer(
        r"(?:[$€£¥]\s*)?\d+(?:[,.]\d{3})*(?:\.\d+)?(?:\s*(?:usd|eur|gbp|dollars?|euros?|pounds?))?",
        raw,
        flags=re.I,
    ):
        token = match.group(0)
        if any(sym in token.lower() for sym in ["$", "€", "£", "¥", "usd", "eur", "gbp", "dollar", "euro", "pound"]):
            out.append(re.sub(r"[, ]+", "", token.lower()))
    return sorted(set(out))


def strip_org_suffix(text: str) -> str:
    toks = [
        tok
        for tok in tokenize(text)
        if tok
        not in {
            "inc",
            "incorporated",
            "llc",
            "ltd",
            "limited",
            "corp",
            "corporation",
            "company",
            "co",
            "plc",
            "group",
            "the",
        }
    ]
    return " ".join(toks)


def is_generic_answer(text: Any) -> bool:
    norm = normalize_text(text)
    toks = tokenize(str(text or ""))
    return bool(not norm or norm in GENERIC_ANSWERS or (len(toks) <= 2 and all(tok in GENERIC_ANSWERS for tok in toks)))


def classify_answer_type(question: Any, question_type: Any, gold: Any, pred: Any) -> str:
    q = normalize_text(f"{question} {question_type}")
    gold_text = clean_text(gold)
    pred_text = clean_text(pred)
    combo = f"{gold_text} {pred_text}"
    if extract_phone(combo) or any(k in q for k in ["phone", "telephone", "contact number", "call", "电话号码"]):
        return "phone"
    if extract_emails(combo) or "email" in q or "e mail" in q:
        return "email"
    if extract_prices(combo) or any(k in q for k in ["price", "cost", "fee", "usd", "eur", "多少钱"]):
        return "price"
    if extract_dates(combo) or any(k in q for k in ["date", "when", "release", "opened", "founded"]):
        return "date"
    if any(k in q for k in ["how many", "count", "number of", "branches", "stores", "reports"]) and re.search(r"\d", combo):
        return "count"
    if any(k in q for k in ["registration", "license", "company id", "vat", "ein", "code", "identifier"]) or (
        re.search(r"[a-zA-Z]", combo) and re.search(r"\d", combo) and len(compact_alnum(gold_text)) <= 24
    ):
        return "company_id"
    if any(k in q for k in ["address", "where", "location", "located", "store address", "branch address"]):
        return "address"
    if any(k in q for k in ["title", "book", "movie", "film", "song", "report title"]):
        return "title"
    if any(k in q for k in ["who", "organization", "company", "brand", "name of", "which company", "which organization"]):
        return "entity_name"
    if len(tokenize(gold_text)) >= 8 or any(k in q for k in ["describe", "what does", "explain", "why"]):
        return "description"
    return "other"


def normalized_value(answer_type: str, text: Any) -> str:
    if answer_type == "phone":
        return "|".join(extract_phone(text))
    if answer_type == "email":
        return "|".join(extract_emails(text))
    if answer_type == "date":
        dates = extract_dates(text)
        return "|".join(dates) if dates else compact_alnum(text)
    if answer_type == "price":
        prices = extract_prices(text)
        return "|".join(prices) if prices else "|".join(extract_numbers(text))
    if answer_type == "count":
        return "|".join(extract_numbers(text))
    if answer_type == "company_id":
        return compact_alnum(text)
    if answer_type in {"address", "entity_name", "title"}:
        return strip_org_suffix(normalize_text(text))
    return normalize_text(text)


def structured_match(answer_type: str, gold: Any, pred: Any) -> bool:
    if answer_type == "phone":
        return phone_match(gold, pred)
    gold_norm = normalized_value(answer_type, gold)
    pred_norm = normalized_value(answer_type, pred)
    if not gold_norm or not pred_norm:
        return False
    if answer_type in {"email", "company_id", "count", "date", "price"}:
        return bool(set(gold_norm.split("|")) & set(pred_norm.split("|")) or gold_norm == pred_norm)
    return gold_norm == pred_norm


def fuzzy_alias_score(gold: Any, pred: Any) -> float:
    if is_generic_answer(pred):
        return 0.0
    g = normalize_text(gold)
    p = normalize_text(pred)
    cg = compact_alnum(gold)
    cp = compact_alnum(pred)
    if not g or not p:
        return 0.0
    scores = [token_f1(str(pred), str(gold)), SequenceMatcher(None, cg, cp).ratio()]
    gs = strip_org_suffix(g)
    ps = strip_org_suffix(p)
    if gs and ps:
        scores.append(SequenceMatcher(None, gs.replace(" ", ""), ps.replace(" ", "")).ratio())
    if len(cg) >= 4 and len(cp) >= 4 and (cg in cp or cp in cg):
        scores.append(0.90)
    if gs and ps and (gs in ps or ps in gs) and min(len(gs), len(ps)) >= 4:
        scores.append(0.88)
    return max(scores)


def appears_in_evidence(answer_type: str, answer: Any, evidence: Any) -> bool:
    if not clean_text(answer) or not clean_text(evidence):
        return False
    if answer_type == "phone":
        return phone_match(answer, evidence)
    if answer_type == "email":
        return bool(set(extract_emails(answer)) & set(extract_emails(evidence)))
    if answer_type in {"company_id", "count", "date", "price"}:
        ans_norm = normalized_value(answer_type, answer)
        ev_norm = normalized_value(answer_type, evidence)
        return bool(ans_norm and ev_norm and (set(ans_norm.split("|")) & set(ev_norm.split("|")) or ans_norm in ev_norm))
    if is_generic_answer(answer):
        return False
    ans = normalize_text(answer)
    body = normalize_text(evidence)
    cans = compact_alnum(answer)
    cbody = compact_alnum(evidence)
    return bool((ans and ans in body) or (len(cans) >= 4 and cans in cbody) or token_f1(str(answer), str(evidence)) >= 0.55)


def semantic_anchor_match(anchor: Any, evidence: Any) -> bool:
    toks = [tok for tok in tokenize(str(anchor or "")) if len(tok) >= 3]
    if not toks:
        return True
    body = set(tokenize(str(evidence or "")))
    return bool(set(toks) & body)


class FrozenJudge:
    def __init__(self, model_path: str, device: str = "cuda", max_new_tokens: int = 128, max_input_tokens: int = 3072):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.enabled = False
        self.error = ""
        self.tokenizer = None
        self.model = None
        if not model_path or model_path == "none":
            self.error = "judge_disabled"
            return
        if "checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora" in model_path:
            self.error = "policy_checkpoint_rejected_as_judge"
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                local_files_only=True,
            )
            if device != "cuda":
                self.model.to(device)
            self.model.eval()
            self.enabled = True
        except Exception as exc:  # pragma: no cover - diagnostic path
            self.error = f"judge_load_failed: {type(exc).__name__}: {exc}"

    @staticmethod
    def prompt(question: str, gold: str, pred: str, evidence: str) -> list[dict[str, str]]:
        task = (
            "Question:\n"
            f"{question}\n\n"
            "Gold answer:\n"
            f"{gold}\n\n"
            "Predicted answer:\n"
            f"{pred}\n\n"
            "Evidence:\n"
            f"{evidence}\n\n"
            "Task:\n"
            "Decide whether the predicted answer is semantically equivalent to the gold answer for this question and supported by the evidence.\n"
            "Return exactly one valid JSON object and no other text. Use booleans true/false and score from 0 to 1.\n"
            "Required keys: semantic_equivalent, evidence_supports_prediction, missing_key_information, added_wrong_information, too_generic, score, reason.\n"
            "JSON:"
        )
        return [
            {"role": "system", "content": "You are a strict frozen answer-equivalence judge. Return JSON only."},
            {"role": "user", "content": task},
        ]

    def unavailable_result(self) -> dict[str, Any]:
        return {
            "semantic_equivalent": False,
            "evidence_supports_prediction": False,
            "missing_key_information": True,
            "added_wrong_information": False,
            "too_generic": False,
            "score": 0.0,
            "reason": self.error or "judge_disabled",
            "malformed": True,
        }

    def judge_many(self, contexts: list[dict[str, str]], batch_size: int = 8) -> list[dict[str, Any]]:
        if not self.enabled:
            return [self.unavailable_result() for _ in contexts]
        results: list[dict[str, Any]] = []
        try:
            import torch
        except Exception as exc:  # pragma: no cover - diagnostic path
            self.error = f"torch_import_failed: {type(exc).__name__}: {exc}"
            return [self.unavailable_result() for _ in contexts]
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            try:
                texts = [
                    self.tokenizer.apply_chat_template(
                        self.prompt(ctx["question"], ctx["gold"], ctx["pred"], ctx["evidence"]),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    + "{"
                    for ctx in batch
                ]
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_tokens,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                prompt_len = inputs["input_ids"].shape[1]
                for seq in output:
                    decoded = "{" + self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    results.append(parse_judge_json(decoded))
            except Exception as exc:  # pragma: no cover - diagnostic path
                reason = f"judge_runtime_failed: {type(exc).__name__}: {exc}"
                results.extend(
                    {
                        "semantic_equivalent": False,
                        "evidence_supports_prediction": False,
                        "missing_key_information": True,
                        "added_wrong_information": False,
                        "too_generic": False,
                        "score": 0.0,
                        "reason": reason,
                        "malformed": True,
                    }
                    for _ in batch
                )
        return results


def parse_judge_json(text: str) -> dict[str, Any]:
    raw = clean_text(text)
    candidates = [raw]
    if raw.startswith("{{"):
        candidates.append(raw[1:])
    decoder = json.JSONDecoder()
    for start in [idx for idx, char in enumerate(raw) if char == "{"]:
        try:
            payload, _end = decoder.raw_decode(raw[start:])
            if isinstance(payload, dict):
                return normalize_judge_payload(payload)
        except json.JSONDecodeError:
            continue
    match = re.search(r"\{.*?\}", raw, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for json_text in candidates:
        try:
            payload = json.loads(json_text)
            if isinstance(payload, dict):
                return normalize_judge_payload(payload)
        except json.JSONDecodeError:
            continue
    if not raw:
        return {
            "semantic_equivalent": False,
            "evidence_supports_prediction": False,
            "missing_key_information": True,
            "added_wrong_information": False,
            "too_generic": False,
            "score": 0.0,
            "reason": f"malformed_judge_output: {raw[:120]}",
            "malformed": True,
        }
    return {
        "semantic_equivalent": False,
        "evidence_supports_prediction": False,
        "missing_key_information": True,
        "added_wrong_information": False,
        "too_generic": False,
        "score": 0.0,
        "reason": f"invalid_judge_json: {raw[:120]}",
        "malformed": True,
    }


def normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_equivalent": bool(payload.get("semantic_equivalent")),
        "evidence_supports_prediction": bool(payload.get("evidence_supports_prediction")),
        "missing_key_information": bool(payload.get("missing_key_information")),
        "added_wrong_information": bool(payload.get("added_wrong_information")),
        "too_generic": bool(payload.get("too_generic")),
        "score": max(0.0, min(1.0, float(payload.get("score") or 0.0))),
        "reason": str(payload.get("reason") or ""),
        "malformed": False,
    }


def resolve_judge_path(value: str) -> str:
    if value == "auto":
        for path in DEFAULT_JUDGE_PATHS:
            if path.is_dir():
                return str(path)
        return "none"
    return value


def row_context(row: dict[str, Any]) -> dict[str, str]:
    segments = row.get("prediction_segments")
    if not isinstance(segments, dict):
        segments = extract_segments(str(row.get("prediction") or ""), TAGS)
    question = str(row.get("question") or row.get("supported_answer_verifier_v2", {}).get("question") or "")
    question_type = str(row.get("question_type") or "")
    gold = str(row.get("gold_answer") or row.get("supported_answer_verifier_v2", {}).get("gold_answer") or "")
    pred = str(segments.get("answer") or row.get("predicted_answer") or "")
    selected_evidence = str(segments.get("evidence") or "")
    retrieved_evidence = str(row.get("supported_answer_verifier_v2", {}).get("retrieved_evidence") or "")
    semantic_anchor = str(row.get("semantic_anchor") or row.get("supported_answer_verifier_v2", {}).get("semantic_anchor") or "")
    evidence_all = "\n".join(x for x in [selected_evidence, retrieved_evidence] if x)
    return {
        "question": question,
        "question_type": question_type,
        "gold": gold,
        "pred": pred,
        "selected_evidence": selected_evidence,
        "retrieved_evidence": retrieved_evidence,
        "semantic_anchor": semantic_anchor,
        "evidence": evidence_all,
    }


def verify_row(row: dict[str, Any], judge_result: dict[str, Any], judge_model_name: str) -> dict[str, Any]:
    ctx = row_context(row)
    question = ctx["question"]
    question_type = ctx["question_type"]
    gold = ctx["gold"]
    pred = ctx["pred"]
    selected_evidence = ctx["selected_evidence"]
    retrieved_evidence = ctx["retrieved_evidence"]
    semantic_anchor = ctx["semantic_anchor"]
    evidence_all = ctx["evidence"]
    answer_type = classify_answer_type(question, question_type, gold, pred)
    normalized_gold = normalized_value(answer_type, gold)
    normalized_prediction = normalized_value(answer_type, pred)
    token_score = token_f1(pred, gold)
    normalized_exact = structured_match(answer_type, gold, pred) if answer_type in STRUCTURED_TYPES else normalized_gold == normalized_prediction and bool(normalized_gold)
    fuzzy_score = fuzzy_alias_score(gold, pred) if answer_type in JUDGE_TYPES else 0.0
    soft_match = bool(normalized_exact or (answer_type in JUDGE_TYPES and fuzzy_score >= 0.78))
    evidence_supports_prediction_rule = appears_in_evidence(answer_type, pred, evidence_all)
    evidence_supports_gold_rule = appears_in_evidence(answer_type, gold, evidence_all)
    if answer_type in STRUCTURED_TYPES:
        semantic_equivalent = bool(normalized_exact)
        semantic_score = 1.0 if semantic_equivalent else 0.0
        evidence_supports_prediction = evidence_supports_prediction_rule
        evidence_supports_gold = evidence_supports_gold_rule
    else:
        judge_uncertain = bool(0.35 <= float(judge_result.get("score") or 0.0) < 0.60)
        judge_ok = bool(
            not judge_result.get("malformed")
            and not judge_uncertain
            and not judge_result.get("too_generic")
            and not judge_result.get("added_wrong_information")
            and not judge_result.get("missing_key_information")
        )
        semantic_score = max(fuzzy_score, token_score, float(judge_result.get("score") or 0.0) if judge_ok else 0.0)
        semantic_equivalent = bool(soft_match or (judge_ok and judge_result.get("semantic_equivalent") and semantic_score >= 0.60))
        evidence_supports_prediction = bool(
            evidence_supports_prediction_rule or (judge_ok and judge_result.get("evidence_supports_prediction"))
        )
        evidence_supports_gold = bool(evidence_supports_gold_rule or (semantic_equivalent and evidence_supports_prediction))
    anchor_ok = semantic_anchor_match(semantic_anchor, evidence_all)
    if is_generic_answer(pred):
        semantic_equivalent = False
        semantic_score = min(semantic_score, 0.20)
    if answer_type in STRUCTURED_TYPES and not normalized_exact:
        failure = "wrong_value" if normalized_prediction else "format_mismatch"
    elif is_generic_answer(pred):
        failure = "too_generic"
    elif not semantic_equivalent:
        failure = "semantic_mismatch"
    elif not evidence_supports_prediction:
        failure = "unsupported"
    elif judge_result.get("malformed") or (answer_type not in STRUCTURED_TYPES and 0.35 <= float(judge_result.get("score") or 0.0) < 0.60):
        failure = "judge_uncertain"
    elif semantic_anchor and not anchor_ok and answer_type in {"entity_name", "address", "title", "description"}:
        failure = "wrong_entity"
    else:
        failure = "none"
    answer_correct_score = max(0.0, min(1.0, semantic_score))
    support_factor = 1.0 if evidence_supports_prediction else (0.45 if evidence_supports_gold and semantic_equivalent else 0.0)
    supported_answer_soft = answer_correct_score * support_factor
    supported_answer_hard = bool(semantic_equivalent and evidence_supports_prediction and answer_correct_score >= 0.60 and failure == "none")
    return {
        "sample_id": row.get("sample_id"),
        "split": row.get("split"),
        "rollout_index": row.get("rollout_index", 0),
        "answer_type": answer_type,
        "gold_answer": gold,
        "predicted_answer": pred,
        "normalized_gold": normalized_gold,
        "normalized_prediction": normalized_prediction,
        "normalized_exact": bool(normalized_exact),
        "token_f1": float(token_score),
        "soft_match": bool(soft_match),
        "semantic_equivalent": bool(semantic_equivalent),
        "semantic_equivalence_score": float(semantic_score),
        "evidence_supports_prediction": bool(evidence_supports_prediction),
        "evidence_supports_gold": bool(evidence_supports_gold),
        "answer_correct_score": float(answer_correct_score),
        "supported_answer_soft": float(supported_answer_soft),
        "supported_answer_hard": bool(supported_answer_hard),
        "failure_type": failure,
        "judge_model": judge_model_name,
        "judge_reason": str(judge_result.get("reason") or ""),
        "judge_semantic_equivalent": bool(judge_result.get("semantic_equivalent")),
        "judge_evidence_supports_prediction": bool(judge_result.get("evidence_supports_prediction")),
        "judge_score": float(judge_result.get("score") or 0.0),
        "judge_malformed": bool(judge_result.get("malformed")),
        "question": question,
        "question_type": question_type,
        "selected_evidence": selected_evidence,
        "retrieved_evidence": retrieved_evidence,
        "semantic_anchor": semantic_anchor,
        "supported_answer_v2": bool(row.get("supported_answer_v2")),
    }


def summarize(rows_by_split: dict[str, list[dict[str, Any]]]) -> str:
    summary = []
    by_type: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for split, rows in rows_by_split.items():
        n = len(rows)
        summary.append(
            {
                "split": split,
                "n": n,
                "supported_answer_v2_positive_rate": mean([float(r.get("supported_answer_v2")) for r in rows]) if n else 0.0,
                "supported_answer_hard_v3_positive_rate": mean([float(r["supported_answer_hard"]) for r in rows]) if n else 0.0,
                "supported_answer_soft_mean": mean([float(r["supported_answer_soft"]) for r in rows]) if n else 0.0,
                "semantic_equivalent_rate": mean([float(r["semantic_equivalent"]) for r in rows]) if n else 0.0,
                "answer_correct_score_mean": mean([float(r["answer_correct_score"]) for r in rows]) if n else 0.0,
                "judge_uncertain_or_malformed": mean(
                    [float(r["failure_type"] == "judge_uncertain" or r["judge_malformed"]) for r in rows]
                )
                if n
                else 0.0,
            }
        )
        for row in rows:
            by_type.setdefault((split, row["answer_type"]), []).append(row)
    type_rows = []
    for (split, answer_type), rows in sorted(by_type.items()):
        n = len(rows)
        type_rows.append(
            {
                "split": split,
                "answer_type": answer_type,
                "n": n,
                "hard_rate": mean([float(r["supported_answer_hard"]) for r in rows]) if n else 0.0,
                "soft_mean": mean([float(r["supported_answer_soft"]) for r in rows]) if n else 0.0,
                "semantic_equivalent_rate": mean([float(r["semantic_equivalent"]) for r in rows]) if n else 0.0,
                "answer_correct_score_mean": mean([float(r["answer_correct_score"]) for r in rows]) if n else 0.0,
            }
        )
    false_pos = []
    false_neg = []
    uncertain = []
    for split, rows in rows_by_split.items():
        false_pos.extend(
            {
                "split": split,
                "sample_id": r["sample_id"],
                "answer_type": r["answer_type"],
                "gold": r["gold_answer"],
                "pred": r["predicted_answer"],
                "soft": f"{r['supported_answer_soft']:.3f}",
                "reason": r["judge_reason"][:120],
            }
            for r in rows
            if r["supported_answer_hard"] and not r["evidence_supports_gold"]
        )
        false_neg.extend(
            {
                "split": split,
                "sample_id": r["sample_id"],
                "answer_type": r["answer_type"],
                "gold": r["gold_answer"],
                "pred": r["predicted_answer"],
                "soft": f"{r['supported_answer_soft']:.3f}",
                "failure": r["failure_type"],
            }
            for r in rows
            if r.get("supported_answer_v2") and not r["supported_answer_hard"]
        )
        uncertain.extend(
            {
                "split": split,
                "sample_id": r["sample_id"],
                "answer_type": r["answer_type"],
                "gold": r["gold_answer"],
                "pred": r["predicted_answer"],
                "judge_score": f"{r['judge_score']:.3f}",
                "reason": r["judge_reason"][:120],
            }
            for r in rows
            if r["failure_type"] == "judge_uncertain" or r["judge_malformed"]
        )
    md = [
        "# Answer Equivalence Verifier v3 Summary",
        "",
        "## Overall",
        "",
        md_table(summary),
        "",
        "## Breakdown By Answer Type",
        "",
        md_table(type_rows),
        "",
        "## Potential False Positive Examples",
        "",
        md_table(false_pos[:30]),
        "",
        "## Potential False Negative Examples",
        "",
        md_table(false_neg[:30]),
        "",
        "## Judge Uncertain Examples",
        "",
        md_table(uncertain[:30]),
        "",
    ]
    return "\n".join(md)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_path = "none" if args.disable_judge else resolve_judge_path(args.judge_model_path)
    judge = None if args.disable_judge else FrozenJudge(judge_path, args.judge_device, args.judge_max_new_tokens)
    judge_name = judge_path if judge and judge.enabled else f"disabled_or_unavailable:{getattr(judge, 'error', 'disabled')}"
    rows_by_split = {}
    for split in args.splits:
        rows = load_jsonl(Path(args.scored_dir) / f"scored_rollouts_{split}.jsonl", limit=args.max_rows)
        contexts = [row_context(row) for row in rows]
        if judge and judge.enabled:
            judge_results = judge.judge_many(contexts, batch_size=args.judge_batch_size)
        else:
            disabled = FrozenJudge("none").unavailable_result()
            judge_results = [disabled for _ in rows]
        out = [verify_row(row, judge_result, judge_name) for row, judge_result in zip(rows, judge_results)]
        write_jsonl(out_dir / f"answer_equivalence_{split}.jsonl", out)
        rows_by_split[split] = out
    (out_dir / "answer_equivalence_summary.md").write_text(summarize(rows_by_split), encoding="utf-8")
    print(json.dumps({"judge_model": judge_name, **{split: len(rows) for split, rows in rows_by_split.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
