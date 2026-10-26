# LoopMem

Core code for **LoopMem: Preserving Research Continuity through Loop-Structured Memory**.

LoopMem organizes an agent's interaction history into research loops. A controller
detects loop boundaries after complete tool interactions. At a boundary, the
outgoing investigation is archived and a planner uses prior progress and the
latest observation to establish the next objective and evidence needs. Historical
evidence is recalled by selecting loops first and records second.

## Included code

| Paper component | Implementation |
| --- | --- |
| Loop memories and interaction records, Section 3.1 | `src/loopmem/schema.py`, `session.py` |
| Autonomous boundaries and progress handoff, Section 3.3 | `boundary.py`, `controller_context.py`, `transition.py`, `prompts.py` |
| Hierarchical evidence recall, Section 3.4 | `retrieval.py`, `reranker.py`, `session.py` |
| Adaptive retrieval budgets, Appendix D.1 | `retrieval_budget.py`, `configs/runtime.json` |
| Controller supervised fine-tuning, Section 3.5 | `training/controller_sft.py` |
| Offline boundary-policy refinement, Section 3.5 | `training/build_controller_rl.py`, `train_controller_rl.py` |
| Loop- and record-level reranker training, Section 3.5 | `training/train_reranker.py` |
| Retrospective annotation, Appendices B.2 and D.3 | `training/retrospective_annotation.py` |

The runtime was extracted from the experimental implementation, retaining its
controller and planner prompts, memory retrieval logic, and budget thresholds.
The entry points are simplified to remove deployment-specific dependencies.
This release contains the core implementation, not a complete benchmark runner.
Model weights, training trajectories, and experiment logs are not included.

## Quick start

Python 3.10 or later is required. The memory runtime and offline example use only
the standard library; loading the learned reranker requires the model extras.

```sh
python -m pip install -e .
python examples/offline_mock.py
python -m unittest discover -s tests -v
```

The example uses synthetic tool results and scripted model responses to exercise
the protocol. It is not an evaluation of a trained controller or planner.

For model inference and training:

```sh
python -m pip install -e '.[models]'
```

See [runtime usage](docs/runtime.md) for model clients and agent integration, and
[training](docs/training.md) for input formats and commands. The training helpers
can be inspected without running model training. GPU training and separately
obtained base models are required to produce learned adapters.

## Data and models

The controller uses a Llama-3.1-8B-Instruct base model with LoRA. Evidence scoring
uses a BGE cross-encoder. The researcher and planner connect through
OpenAI-compatible model endpoints. Model names, endpoints, and local checkpoint
paths are supplied by the user; credentials are read from the environment.

Training trajectories can be obtained from the upstream
[OpenResearcher dataset](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset).
The code accepts user-prepared JSONL files; it does not download or redistribute
benchmark questions, reference answers, or pretrained weights. Follow the
upstream dataset and model terms when obtaining those resources.
