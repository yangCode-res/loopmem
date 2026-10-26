"""Target-only LoRA SFT for structured two-action boundary decisions.

Input JSONL rows contain ``messages`` (system, user, assistant). Only assistant
target tokens contribute to loss. Optional dependencies are imported in main().
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def fit_prompt(prompt_ids: Sequence[int], target_ids: Sequence[int], *, max_length: int,
               prefix_tokens: int, max_input_length: int | None = None,
               max_target_length: int | None = None) -> tuple[list[int], list[int]]:
    """Preserve system prefix and newest prompt tail; never truncate targets."""
    target = list(target_ids)
    if max_length <= 0 or (max_target_length is not None and max_target_length <= 0):
        raise ValueError("length limits must be positive")
    if max_target_length is not None and len(target) > max_target_length:
        raise ValueError("structured target exceeds max_target_length")
    if len(target) >= max_length:
        raise ValueError("target leaves no prompt budget")
    budget = max_length - len(target)
    if max_input_length is not None:
        if max_input_length <= 0:
            raise ValueError("max_input_length must be positive")
        budget = min(budget, max_input_length)
    prompt = list(prompt_ids)
    if not 0 <= prefix_tokens <= len(prompt):
        raise ValueError("invalid system-prefix length")
    if len(prompt) > budget:
        prefix = min(prefix_tokens, budget)
        tail = budget - prefix
        prompt = prompt[:prefix] + (prompt[-tail:] if tail else [])
    return prompt, target


def target_only_labels(prompt_length: int, target_ids: Sequence[int]) -> list[int]:
    return [-100] * prompt_length + list(target_ids)


def read_messages(path: Path) -> list[list[dict[str, str]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = [row["messages"] for row in rows]
    if not result or any(len(m) != 3 or [x.get("role") for x in m] != ["system", "user", "assistant"] for m in result):
        raise ValueError("each SFT row must have system/user/assistant messages")
    return result


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    class Rows(Dataset):
        def __init__(self, path: Path, tok: object) -> None:
            self.rows = read_messages(path)
            self.tok = tok
        def __len__(self) -> int: return len(self.rows)
        def __getitem__(self, i: int) -> dict[str, list[int]]:
            messages = self.rows[i]
            prompt = self.tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
            target = self.tok.encode(messages[-1]["content"], add_special_tokens=False)
            if self.tok.eos_token_id is not None: target.append(self.tok.eos_token_id)
            prefix = self.tok.apply_chat_template([messages[0]], tokenize=True, add_generation_prompt=False)
            prompt, target = fit_prompt(prompt, target, max_length=args.max_length,
                prefix_tokens=len(prefix), max_input_length=args.max_input_length)
            ids = prompt + target
            return {"input_ids": ids, "attention_mask": [1] * len(ids),
                    "labels": target_only_labels(len(prompt), target)}

    class Pad:
        def __init__(self, pad_id: int): self.pad_id = pad_id
        def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, object]:
            n = max(len(x["input_ids"]) for x in features)
            return {key: torch.tensor([x[key] + ([fill] * (n-len(x[key]))) for x in features])
                    for key, fill in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100))}

    class TargetTrainer(Trainer):
        def compute_loss(self, model: object, inputs: dict[str, object], return_outputs: bool = False, **kwargs: object):
            out = model(**inputs)
            logits = out.logits[..., :-1, :].contiguous()
            labels = inputs["labels"][..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            return (loss, out) if return_outputs else loss

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only,
                                                  torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    training_args = TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, weight_decay=0.0, max_grad_norm=1.0, bf16=True,
        gradient_checkpointing=True, remove_unused_columns=False, report_to="none", seed=args.seed,
        save_strategy="epoch", eval_strategy="epoch")
    trainer = TargetTrainer(model=model, args=training_args, train_dataset=Rows(args.train, tok),
        eval_dataset=Rows(args.validation, tok), data_collator=Pad(tok.pad_token_id))
    trainer.train(); trainer.save_model(str(args.output / "final_adapter")); tok.save_pretrained(args.output / "final_adapter")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True); p.add_argument("--train", required=True, type=Path)
    p.add_argument("--validation", required=True, type=Path); p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=float, default=2); p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--gradient-accumulation", type=int, default=2)
    p.add_argument("--max-length", type=int, default=6144); p.add_argument("--max-input-length", type=int)
    p.add_argument("--seed", type=int, default=20260803); p.add_argument("--local-files-only", action="store_true")
    train(p.parse_args())


if __name__ == "__main__": main()
