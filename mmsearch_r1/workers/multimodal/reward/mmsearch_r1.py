import re

import torch
from verl import DataProto

from mmsearch_r1.workers.multimodal.reward.answer_utils import normalize_answer_list
from mmsearch_r1.utils.reward_score_mm import _default_compute_score
from mmsearch_r1.utils.reward_score_mm.mmsearch_r1_score import normalize_answer


TEXT_SEARCH_RE = re.compile(r"<text_search>.*?</text_search>", re.DOTALL)
IMAGE_SEARCH_TEXT = "<search><img></search>"


def find_subsequence(haystack: list[int], needle: list[int], start_at: int = 0) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    first = needle[0]
    max_start = len(haystack) - len(needle)
    for start in range(max(start_at, 0), max_start + 1):
        if haystack[start] != first:
            continue
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


class MMSearchR1_RewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def extract_responses_list(
        self,
        tokenizer,
        input_ids: torch.Tensor,  # User Prompt + All Responses
        multi_turn_response_mask: torch.Tensor,  # 0,0,0,...,1,1,1,...,0,0,0,...,1,1,1
    ) -> list:
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat(
            [multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0
        )
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0] - 1
        segments = []
        for s, e in zip(starts, ends):
            segments.append(input_ids[s : e + 1].tolist())

        # Decode each segment
        # decoded_responses = [tokenizer.decode(seg, skip_special_tokens=True) for seg in segments]
        decoded_responses = tokenizer.batch_decode(segments, skip_special_tokens=True)
        return decoded_responses

    def search_action_spans(self, valid_response_ids: list[int], decoded_responses: list[str]) -> list[tuple[int, int, str]]:
        spans = []
        used_starts = set()
        for response in decoded_responses:
            action_texts = []
            action_texts.extend(match.group(0) for match in TEXT_SEARCH_RE.finditer(response))
            if IMAGE_SEARCH_TEXT in response:
                action_texts.append(IMAGE_SEARCH_TEXT)

            for action_text in action_texts:
                action_ids = self.tokenizer.encode(action_text, add_special_tokens=False)
                start_at = 0
                start = find_subsequence(valid_response_ids, action_ids, start_at=start_at)
                while start in used_starts:
                    start_at = start + 1
                    start = find_subsequence(valid_response_ids, action_ids, start_at=start_at)
                if start >= 0:
                    used_starts.add(start)
                    tool_type = "image_search" if action_text == IMAGE_SEARCH_TEXT else "text_search"
                    spans.append((start, start + len(action_ids), tool_type))
        return spans

    def apply_search_success_shaping(
        self,
        reward_tensor: torch.Tensor,
        row_idx: int,
        valid_response_ids: list[int],
        decoded_responses: list[str],
        score: float,
        extra_info,
        ground_truth: list[str] | None = None,
        observation_text: str = "",
    ) -> None:
        if not extra_info:
            return
        mode = extra_info.get("reward_shaping_mode", "outcome_only")
        if mode not in {"search_success_shaping", "dagig_lite_proxy"}:
            return

        bonus = float(extra_info.get("search_action_bonus", 0.0) or 0.0)
        if bonus <= 0:
            return

        correct_threshold = float(extra_info.get("format_penalty", 0.1) or 0.1) + 1e-4
        if mode == "search_success_shaping":
            if as_bool(extra_info.get("search_action_bonus_correct_only", True)) and score <= correct_threshold:
                return
        elif mode == "dagig_lite_proxy":
            if as_bool(extra_info.get("dagig_proxy_require_correct", True)) and score <= correct_threshold:
                return
            if not self.has_dagig_proxy_support(observation_text, ground_truth or []):
                return

        spans = self.search_action_spans(valid_response_ids, decoded_responses)
        if not spans:
            return

        for start, end, _tool_type in spans:
            length = max(end - start, 1)
            reward_tensor[row_idx, start:end] += bonus / length

    @staticmethod
    def has_dagig_proxy_support(observation_text: str, ground_truth: list[str]) -> bool:
        """Cheap DAG-IG proxy for debug runs.

        A search action is eligible only when real non-assistant search
        observations contain a gold answer substring. Query-only matches are
        intentionally ignored to avoid reward hacking.
        """

        if not ground_truth:
            return False
        if "[Text Search Results]" not in observation_text and "[Image Search Results]" not in observation_text:
            return False

        normalized_context = normalize_answer(observation_text)
        for gold in ground_truth:
            normalized_gold = normalize_answer(gold)
            if len(normalized_gold) < 3:
                continue
            if normalized_gold in normalized_context:
                return True
        return False

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # shape: (B*R, response_length_total)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # Get valid prompt_ids w/o padding tokens
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # Get valid response_ids w/o padding tokens
            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            observation_text = ""
            if 'multi_turn_response_mask' in data_item.batch:
                response_mask = data_item.batch['multi_turn_response_mask'][-valid_response_length:]
                observation_ids = valid_response_ids[response_mask < 0.1]
                observation_text = self.tokenizer.decode(observation_ids, skip_special_tokens=True)

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str = [response_str]
            # For multi turn, we maybe need `response_str` in a list format
            if 'multi_turn_response_mask' in data_item.batch:
                # `response_str` is a list now
                response_str = self.extract_responses_list(
                    self.tokenizer, data_item.batch['input_ids'], data_item.batch['multi_turn_response_mask']
                )

            # We need `ground_truth` to be a list to support multiple candidate answers
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            ground_truth = normalize_answer_list(ground_truth)
            if 'candidate_answers' in data_item.non_tensor_batch['reward_model']:
                candidate_answers = data_item.non_tensor_batch['reward_model']['candidate_answers']
                ground_truth += normalize_answer_list(candidate_answers)
            ground_truth = [g for g in ground_truth if isinstance(g, str)]
            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            reward_tensor[i, valid_response_length - 1] = score
            self.apply_search_success_shaping(
                reward_tensor,
                i,
                valid_response_ids.tolist(),
                response_str,
                score,
                extra_info,
                ground_truth,
                observation_text,
            )

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor
