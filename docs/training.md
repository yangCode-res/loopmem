# Training core code

Exact input contracts and synthetic schema examples are in
[data formats](data_formats.md).

This directory contains the core algorithms described in the paper: target-only controller SFT, reviewed offline boundary-policy refinement, graded listwise reranker training, and the two-pass retrospective workstream annotation prompts. It is not a complete reproduction of every final checkpoint. The release provides training entry points; trained weights and training trajectories are obtained separately.

## Controller SFT

`training/controller_sft.py` accepts train and validation JSONL files. Each row has a `messages` array with exactly `system`, `user`, and `assistant` roles. The assistant message is the structured target; prompt tokens are masked with `-100`. LoRA uses rank 32, alpha 64, dropout 0.05. Inputs and outputs are explicit CLI arguments:

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

The reward is a retrospective training signal, not a causal counterfactual estimate. Optional dependencies: PyTorch, Transformers, PEFT, and a compatible CUDA/bfloat16 runtime.

## Reranker

`training/train_reranker.py` trains one cross-encoder over loop-level and record-level candidate events. Each event has `qid`, `trajectory_id`, `level`, `query`, and at least two `candidates`, each with a `text` and integer label in `{0,1,2}`. The gains are `{-1,0.25,1}`. The loss combines a clipped listwise policy term, categorical KL to the initial distribution, and soft-label cross-entropy. The initial distribution is collected once before optimization and remains the fixed reference. Use a train-only event file and a manifest with candidate/test separation checks; this package contains no research trajectories or qid lists.

```sh
python -m training.train_reranker --model MODEL --events EVENTS.jsonl \
  --data-manifest MANIFEST.json --output-dir OUTPUT_DIR
```

Optional dependencies: PyTorch and Transformers. Hyperparameters are CLI options; defaults are those of the extracted listwise training implementation. Set the command-line arguments for the training configuration you intend to run.

## Retrospective annotation

`training/retrospective_annotation.py` includes the paper's two core prompts and a small two-pass runner. Pass one identifies atomic workstreams from recorded action intents. Pass two maps each action to the immutable inventory. The runner accepts a callable `complete_json(system_prompt, user_payload)` so no API provider or credential is embedded. The research pipeline also uses multiple candidate annotations, reconciliation/adjudication, trajectory validation, and causal-prefix reconstruction; those data-pipeline operations are not reproduced by this minimal prompt module. Retrospective annotation may inspect later progress, but the controller training inputs must be reconstructed from the context available at each decision.

## Provenance and limits

The release includes code and synthetic examples only. It excludes model weights, private manifests, qid lists, raw trajectories, outcomes, API keys, local paths, and cluster job files. The extracted reranker trainer's reference distribution is fixed for the run; it does not refresh per epoch. No training job was run as part of packaging. A successful unit test of pure data/loss helpers does not establish that a model checkpoint was reproduced.
