# Training core code

Exact input contracts and synthetic schema examples are in
[data formats](data_formats.md).

This directory contains the core algorithms described in the paper: target-only controller SFT, reviewed offline boundary-policy refinement, graded listwise reranker training, and the two-pass retrospective workstream annotation prompts. Trained weights and training trajectories are obtained separately.

## Controller SFT

`training/controller_sft.py` accepts train and validation JSONL files. Each row has a `messages` array with exactly `system`, `user`, and `assistant` roles. The assistant message is the structured target; prompt tokens are masked with `-100`. Valid target-token losses are summed per example and then averaged across examples. LoRA uses rank 32, alpha 64, dropout 0.05. Inputs and outputs are explicit CLI arguments:

```sh
python -m training.controller_sft --base-model MODEL --train TRAIN.jsonl \
  --validation VALIDATION.jsonl --output OUTPUT_DIR
```

The script truncates old prompt evidence while preserving the system prefix and newest tail. It fails if a structured target cannot fit. Optional dependencies: PyTorch, Transformers, PEFT, and a compatible CUDA/bfloat16 runtime.

## Offline controller RL

`training/build_controller_rl.py` accepts a reviewed annotation bundle and materializes qid-disjoint train/validation data. It validates source hashes and causal `system`/`user` messages. `training/train_controller_rl.py` enumerates both actions on each context, computes exact old-policy expected-reward advantages, applies clipped updates, refreshes the behavior policy at each epoch start, and keeps the KL reference fixed at RL initialization. Reward order is `[CONTINUE, SWITCH_LOOP]`: `[1,-1.5]` for a reviewed continue and `[-2,2]` for a reviewed switch. A correct recorded final outcome adds `0.25` only to a switch matching the reviewed boundary label.

```sh
python -m training.build_controller_rl --review REVIEW_BUNDLE --output DATA_DIR
python -m training.train_controller_rl --base-model MODEL --adapter INITIAL_ADAPTER \
  --data DATA_DIR --output OUTPUT_DIR
```

This entry point requires PyTorch, Transformers, PEFT, and a compatible CUDA/bfloat16 runtime.

## Reranker

`training/train_reranker.py` trains one cross-encoder over loop-level and record-level candidate events. Each event has `qid`, `trajectory_id`, `level`, `query`, and at least two `candidates`, each with a `text` and integer label in `{0,1,2}`. The gains are `{-1,0.25,1}`. The loss combines a clipped listwise policy term, categorical KL to the initial distribution, and soft-label cross-entropy. The initial distribution is collected once before optimization and remains the fixed reference. Use a train-only event file and a manifest with candidate/test separation checks; this package contains no research trajectories or qid lists.

```sh
python -m training.train_reranker --model MODEL --events EVENTS.jsonl \
  --data-manifest MANIFEST.json --output-dir OUTPUT_DIR
```

This entry point requires PyTorch and Transformers. Hyperparameters are command-line options.

## Retrospective annotation

`training/retrospective_annotation.py` retains two core prompts and the original
multi-view agreement/adjudication strategy. Each stage defaults to three views
and three candidates (`views` and `candidate_count` are keyword options).
Inventory agreement is accepted directly; disagreements are adjudicated using
the inventory prompt. Assignment uses the fixed inventory, accepting agreement
or a strict per-action majority, and adjudicating unresolved disagreements.
Unused workstreams are removed and IDs are renumbered by first action use.

The runner accepts `complete_json(system_prompt, user_payload)` as a callback.
Production trajectory validation, outcome labeling, and reconstruction of causal
controller inputs remain data-pipeline responsibilities. Only the decision-time
prefix belongs in each controller input, even when retrospective labels use the
whole recorded action sequence.
