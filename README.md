# LoopMem

Core runtime and training code for LoopMem, a memory management framework for
long-horizon Web research. LoopMem organizes interactions into research loops
and connects them through progress handoff and hierarchical evidence recall.

## Code structure

- `src/loopmem/`: loop formation, progress handoff, memory storage, and evidence recall.
- `training/`: controller SFT, offline boundary-policy refinement, reranker training,
  and retrospective annotation.

## Setup

Use Python 3.10 or later. From the repository root:

```sh
python -m pip install 'torch>=2.5,<3' 'transformers>=4.56,<5' 'peft>=0.13,<1' 'accelerate>=1,<2'
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

The core memory runtime uses the Python standard library. The dependencies above
are needed for model loading and training; controller training requires a CUDA
GPU with bfloat16 support.

## Runtime

Use `OnlineMemorySession` with `ContextOnlyLoopBoundaryJudge`, `TransitionPlanner`,
`BGEReranker`, and `RetrievalBudgetGate`. The controller and planner accept clients
with a `complete_json(system_prompt, user_prompt)` method; an OpenAI-compatible
client is provided in `src/loopmem/client.py`.

In the agent's execution loop:

1. Call `session.build_prompt(system_message, user_message)` before each researcher step.
2. Execute the researcher's tool calls, then pass the complete batch to
   `session.submit_batch(action_message, result_messages)`. Each result must carry
   the matching `tool_call_id`.
3. Call `session.finalize()` when the task ends.

The host agent supplies the researcher, search/browser tools, and model endpoints.

## Training

```sh
# Controller supervised fine-tuning
python -m training.controller_sft --base-model MODEL --train TRAIN.jsonl --validation VALIDATION.jsonl --output SFT_DIR

# Prepare reviewed boundary samples and refine the controller
python -m training.build_controller_rl --review REVIEWED.jsonl --output DATA_DIR
python -m training.train_controller_rl --base-model MODEL --adapter SFT_DIR/final_adapter --data DATA_DIR --output RL_DIR

# Train the shared loop- and record-level reranker
python -m training.train_reranker --model BGE_MODEL_DIR --events EVENTS.jsonl --data-manifest MANIFEST.json --output-dir RERANKER_DIR
```

Replace the uppercase arguments with your model and data paths. Each training
entry point provides additional options through `--help`. Retrospective annotation
is implemented in `training/retrospective_annotation.py`.

This repository contains core code; pretrained weights, training trajectories,
and a complete benchmark runner are not included.
