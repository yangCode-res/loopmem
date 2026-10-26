#!/usr/bin/env python3
"""Train BGE with exact event-level listwise policy gradients.

Each retrieval event is one finite action space: the candidate loops or records
that were available at that switch.  The initial BGE distribution is frozen in
memory before optimization.  Training maximizes graded hindsight reward with a
clipped policy ratio, while a listwise target and exact categorical KL keep the
update stable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


SCHEMA_VERSION = "cl-gism-bge-listwise-policy-gradient-v1"
GAINS = {0: -1.0, 1: 0.25, 2: 1.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-7)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--clip-range", type=float, default=0.20)
    parser.add_argument("--kl-coef", type=float, default=0.03)
    parser.add_argument("--anchor-coef", type=float, default=0.20)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=0.35)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def read_events(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError("reranker event file is empty")
    identities: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("schema_version") != "cl-gism-reranker-listwise-event-v1":
            raise ValueError(f"row {index}: unexpected event schema")
        identity = str(row.get("event_id") or "")
        if not identity or identity in identities:
            raise ValueError(f"row {index}: missing or duplicate event_id")
        identities.add(identity)
        if any(key not in row for key in ("qid", "trajectory_id", "level", "query")):
            raise ValueError(f"row {index}: missing event metadata")
        if row["level"] not in {"loop", "record"} or not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError(f"row {index}: invalid retrieval level or query")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError(f"row {index}: at least two candidates are required")
        if any(not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str)
               or not candidate["text"].strip() or candidate.get("label") not in GAINS
               for candidate in candidates):
            raise ValueError(f"row {index}: invalid candidate text or relevance label")
        labels = {int(candidate["label"]) for candidate in row.get("candidates") or []}
        if not labels.issubset(GAINS) or len(labels) < 2:
            raise ValueError(f"row {index}: event is not rankable")
    return rows


def split_qids(
    rows: list[dict[str, Any]], fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["qid"])].append(row)
    qids = sorted(groups)
    if len(qids) < 2:
        raise ValueError("listwise RL requires at least two train qids")
    random.Random(seed).shuffle(qids)
    count = max(1, min(len(qids) - 1, round(len(qids) * fraction)))
    validation = set(qids[:count])
    return (
        [row for qid in qids if qid not in validation for row in groups[qid]],
        [row for qid in qids if qid in validation for row in groups[qid]],
        {
            "total_qids": len(qids),
            "train_qids": len(qids) - count,
            "validation_qids": count,
            "qid_overlap": 0,
        },
    )


class EventDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        counts = Counter(str(row["trajectory_id"]) for row in rows)
        self.rows = rows
        self.weights = [
            float(row.get("reward_multiplier") or 1.0) / counts[str(row["trajectory_id"])]
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {**self.rows[index], "index": index, "event_weight": self.weights[index]}


def event_scores(
    model: Any, tokenizer: Any, event: dict[str, Any], device: torch.device, max_length: int
) -> torch.Tensor:
    candidates = event["candidates"]
    encoded = tokenizer(
        [event["query"]] * len(candidates),
        [candidate["text"] for candidate in candidates],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return model(**encoded).logits.reshape(-1).float()


@torch.no_grad()
def collect_old_policy(
    model: Any, tokenizer: Any, dataset: EventDataset, device: torch.device,
    max_length: int, temperature: float,
) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    model.eval()
    for number, event in enumerate(dataset.rows, 1):
        scores = event_scores(model, tokenizer, event, device, max_length)
        result.append(torch.log_softmax(scores / temperature, dim=-1).cpu())
        if number % 100 == 0:
            print(json.dumps({"phase": "collect_old_policy", "events": number}))
    model.train()
    return result


def listwise_ppo_loss(
    scores: torch.Tensor,
    old_log_policy: torch.Tensor,
    labels: torch.Tensor,
    *,
    event_weight: float,
    policy_temperature: float,
    target_temperature: float,
    clip_range: float,
    kl_coef: float,
    anchor_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    new_log_policy = torch.log_softmax(scores / policy_temperature, dim=-1)
    old_log_policy = old_log_policy.to(scores.device)
    old_prob = old_log_policy.exp()
    gains = torch.tensor([GAINS[int(label)] for label in labels], device=scores.device)
    old_value = (old_prob * gains).sum()
    advantages = gains - old_value
    log_ratio = (new_log_policy - old_log_policy).clamp(-20, 20)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    policy_loss = -(old_prob * torch.minimum(ratio * advantages, clipped * advantages)).sum()
    new_prob = new_log_policy.exp()
    kl = (new_prob * (new_log_policy - old_log_policy)).sum()
    target = torch.softmax(gains / target_temperature, dim=-1)
    anchor = -(target * new_log_policy).sum()
    loss = event_weight * (policy_loss + kl_coef * kl + anchor_coef * anchor)
    return loss, {
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "anchor": float(anchor.detach().cpu()),
        "expected_gain": float((new_prob * gains).sum().detach().cpu()),
        "clip_fraction": float(((ratio - 1).abs() > clip_range).float().mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate(
    model: Any, tokenizer: Any, dataset: EventDataset, device: torch.device,
    max_length: int,
) -> dict[str, Any]:
    pair_correct = pair_total = recall3 = 0
    reciprocal_ranks: list[float] = []
    ndcg3: list[float] = []
    by_level: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    model.eval()
    for event in dataset.rows:
        scores = event_scores(model, tokenizer, event, device, max_length).cpu()
        labels = torch.tensor([int(candidate["label"]) for candidate in event["candidates"]])
        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                if labels[left] == labels[right]:
                    continue
                expected = labels[left] > labels[right]
                actual = scores[left] > scores[right]
                correct = bool(expected == actual)
                pair_correct += int(correct)
                pair_total += 1
                level = str(event["level"])
                by_level[level][0] += int(correct)
                by_level[level][1] += 1
        order = torch.argsort(scores, descending=True)
        maximum = int(labels.max())
        ranks = [rank + 1 for rank, index in enumerate(order.tolist()) if int(labels[index]) == maximum]
        reciprocal_ranks.append(1.0 / min(ranks))
        recall3 += int(any(rank <= 3 for rank in ranks))
        gains = torch.tensor([(2 ** int(label)) - 1 for label in labels], dtype=torch.float32)
        discounts = 1.0 / torch.log2(torch.arange(2, min(3, len(labels)) + 2).float())
        dcg = (gains[order[:3]] * discounts).sum()
        ideal = (torch.sort(gains, descending=True).values[:3] * discounts).sum().clamp_min(1e-8)
        ndcg3.append(float(dcg / ideal))
    model.train()
    return {
        "events": len(dataset),
        "pair_accuracy": pair_correct / max(1, pair_total),
        "pair_comparisons": pair_total,
        "recall_at_3": recall3 / max(1, len(dataset)),
        "mrr": sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)),
        "ndcg_at_3": sum(ndcg3) / max(1, len(ndcg3)),
        "pair_accuracy_by_level": {
            level: correct / max(1, total) for level, (correct, total) in by_level.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.gradient_accumulation < 1 or args.learning_rate <= 0:
        raise SystemExit("epochs, gradient accumulation, and learning rate must be positive")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    if not 0 < args.validation_fraction < 0.5:
        raise SystemExit("validation fraction must be in (0, 0.5)")
    if args.policy_temperature <= 0 or args.target_temperature <= 0:
        raise SystemExit("temperatures must be positive")
    data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    if not data_manifest.get("safety", {}).get("openresearcher_train_only"):
        raise SystemExit("data manifest is not train-only")
    if data_manifest.get("safety", {}).get("test_qid_overlap") != 0:
        raise SystemExit("data manifest reports test contamination")
    if sha256_file(args.events) != data_manifest["reranker"]["file"]["sha256"]:
        raise SystemExit("reranker event hash differs from train-only manifest")
    if not args.model.joinpath("config.json").is_file():
        raise SystemExit("initial BGE model is incomplete")
    set_seed(args.seed)
    rows = read_events(args.events)
    train_rows, validation_rows, split = split_qids(rows, args.validation_fraction, args.seed)
    train = EventDataset(train_rows)
    validation = EventDataset(validation_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, low_cpu_mem_usage=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    old_train = collect_old_policy(
        model, tokenizer, train, device, args.max_length, args.policy_temperature
    )
    initial_validation = evaluate(model, tokenizer, validation, device, args.max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    last_metrics: dict[str, float] = {}
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        for position, index in enumerate(order, 1):
            event = train[index]
            scores = event_scores(model, tokenizer, event, device, args.max_length)
            labels = torch.tensor(
                [int(candidate["label"]) for candidate in event["candidates"]],
                dtype=torch.long,
                device=device,
            )
            loss, last_metrics = listwise_ppo_loss(
                scores,
                old_train[index],
                labels,
                event_weight=float(event["event_weight"]),
                policy_temperature=args.policy_temperature,
                target_temperature=args.target_temperature,
                clip_range=args.clip_range,
                kl_coef=args.kl_coef,
                anchor_coef=args.anchor_coef,
            )
            window_start = ((position - 1) // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(order) - window_start)
            (loss / window_size).backward()
            if position % args.gradient_accumulation == 0 or position == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if position % 50 == 0:
                print(json.dumps({"epoch": epoch + 1, "event": position, "loss": float(loss.detach().cpu()), **last_metrics}))

    final_validation = evaluate(model, tokenizer, validation, device, args.max_length)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "exact_enumerated_listwise_clipped_policy_gradient",
        "initial_model": str(args.model.resolve()),
        "events": {"path": str(args.events.resolve()), "sha256": sha256_file(args.events)},
        "data_manifest": {"path": str(args.data_manifest.resolve()), "sha256": sha256_file(args.data_manifest)},
        "split": split,
        "train_events": len(train),
        "validation_events": len(validation),
        "train_levels": dict(Counter(row["level"] for row in train_rows)),
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "optimization": {
            "epochs": args.epochs,
            "optimizer_steps": optimizer_steps,
            "learning_rate": args.learning_rate,
            "clip_range": args.clip_range,
            "kl_coef": args.kl_coef,
            "anchor_coef": args.anchor_coef,
            "policy_temperature": args.policy_temperature,
            "target_temperature": args.target_temperature,
            "trajectory_normalized": True,
            "seed": args.seed,
        },
        "last_train_metrics": last_metrics,
        "safety": {
            "openresearcher_train_only": True,
            "test_used_for_training_or_tuning": False,
            "old_policy_collected_before_first_update": True,
            "loop_and_record_policies_present": set(row["level"] for row in rows) == {"loop", "record"},
        },
    }
    (args.output_dir / "rl_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
