from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class HFReferencePolicyConfig:
    name_or_path: str
    cache_dir: str | None = None
    dtype: str = "bf16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    max_context_tokens: int = 4096


class HFReferencePolicy:
    """Frozen HF policy used only for logprob scoring.

    The scorer is text-context first. For Qwen2.5-VL this is enough for the
    DAG-IG reward prototype because observations are summarized as text.
    """

    def __init__(self, cfg: HFReferencePolicyConfig) -> None:
        self.cfg = cfg
        self.model: Any = None
        self.tokenizer: Any = None
        self.device = "cpu"
        self._load()

    def score_text(self, context: str, target: str) -> float:
        import torch

        if not target:
            return 0.0
        with torch.no_grad():
            context_ids = self._encode(context, add_special_tokens=True)
            target_ids = self._encode(target, add_special_tokens=False)
            if target_ids.numel() == 0:
                return 0.0
            input_ids = torch.cat([context_ids, target_ids], dim=1)
            attention_mask = torch.ones_like(input_ids)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            start = context_ids.shape[1] - 1
            end = start + target_ids.shape[1]
            token_log_probs = log_probs[:, start:end, :].gather(2, labels[:, start:end].unsqueeze(-1)).squeeze(-1)
            return float(token_log_probs.mean().item())

    def _load(self) -> None:
        import torch
        import transformers

        dtype = self._torch_dtype(torch)
        tokenizer_kwargs = {
            "cache_dir": self.cfg.cache_dir or os.environ.get("DAGIG_MODEL_CACHE") or os.environ.get("HF_HOME"),
            "trust_remote_code": self.cfg.trust_remote_code,
        }
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.cfg.name_or_path, **tokenizer_kwargs)
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "cache_dir": self.cfg.cache_dir or os.environ.get("DAGIG_MODEL_CACHE") or os.environ.get("HF_HOME"),
            "torch_dtype": dtype,
            "device_map": self.cfg.device_map,
            "trust_remote_code": self.cfg.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.cfg.attn_implementation:
            model_kwargs["attn_implementation"] = self.cfg.attn_implementation

        errors: list[str] = []
        for class_name in (
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "Qwen2_5_VLForConditionalGeneration",
        ):
            model_cls = getattr(transformers, class_name, None)
            if model_cls is None:
                continue
            try:
                self.model = model_cls.from_pretrained(self.cfg.name_or_path, **model_kwargs)
                break
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
        if self.model is None:
            raise RuntimeError("Failed to load HF reference policy:\n" + "\n".join(errors[-4:]))
        self.model.eval()
        try:
            self.device = str(next(self.model.parameters()).device)
        except Exception:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _encode(self, text: str, add_special_tokens: bool) -> Any:
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_context_tokens,
        ).input_ids
        return encoded.to(self.device)

    def _torch_dtype(self, torch_module: Any) -> Any:
        if self.cfg.dtype in {"bf16", "bfloat16"}:
            return torch_module.bfloat16
        if self.cfg.dtype in {"fp16", "float16"}:
            return torch_module.float16
        if self.cfg.dtype in {"fp32", "float32"}:
            return torch_module.float32
        return "auto"


def load_reference_policy_from_config(config: dict[str, Any]) -> HFReferencePolicy:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    return HFReferencePolicy(
        HFReferencePolicyConfig(
            name_or_path=model_cfg.get("name_or_path", "Qwen/Qwen2.5-VL-3B-Instruct"),
            dtype=model_cfg.get("precision", "bf16"),
            cache_dir=model_cfg.get("cache_dir"),
            attn_implementation=model_cfg.get("attn_implementation"),
            max_context_tokens=int(data_cfg.get("max_prompt_length", 3072)),
        )
    )
