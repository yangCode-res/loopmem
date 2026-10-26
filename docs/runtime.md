# Core runtime

This directory contains the core memory runtime extracted from the experimental implementation. It accepts complete assistant tool-call batches and their paired results, lets a context-only boundary controller decide whether the batch begins a new work unit, archives the outgoing unit on a switch, plans the next unit from the archived handoff and observed batch, and recalls memory before each packed Researcher call.

The package keeps the runtime's JSON schema, context projection, lexical loop index, local cross-encoder adapter, deterministic retrieval budget, context-only boundary contract, OpenAI-compatible JSON client, and transition-planning prompt. `OnlineMemorySession` is a narrow extraction of the original online context-only branch. It excludes the unified/global-state and legacy branches. The transport integration is intentionally outside this package.

Run the offline protocol example with:

```sh
PYTHONPATH=src python examples/offline_mock.py
```

Run the offline tests with:

```sh
python -m unittest discover -s tests -v
```

The offline example uses scripted JSON responses only to demonstrate the runtime interface; it does not approximate the trained controller or planner. The API client can connect to an OpenAI-compatible endpoint through `OPENAI_API_KEY`, `LOOPMEM_OPENAI_BASE_URL`, and `LOOPMEM_OPENAI_MODEL`. The controller adapter and transition planner accept a client object implementing `complete_json(system_prompt, user_prompt)`.

The local BGE reranker requires `torch`, `transformers`, and a separately provisioned model directory. No model weights are included. Those weights are required for the learned controller and cross-encoder behavior used by experiments. This extracted core implementation does not claim to reproduce every experiment or provide a complete benchmark reproduction environment.

## Extraction notes

- Boundary classification is post-result: a multi-tool action is classified only after every unique `tool_call_id` has a matching result.
- On `SWITCH_LOOP`, the prior active events are archived; the just-observed action and all its results seed the incoming work unit.
- Retrieval runs each time `build_prompt` packs context. Selected evidence is returned with its parent loop packet and active loop contract.
- Budget defaults are 2/3 loops and 5/8 records, with score-gap, coverage, stagnation, repetition, and cooldown thresholds retained in `RetrievalBudgetConfig`.

## Connecting model clients

The controller and planner accept a client exposing
`complete_json(system_prompt, user_prompt)`. Instantiate separate clients, using
your controller serving name and researcher serving name respectively:

```python
import os
from loopmem import (
    BGEReranker, ContextOnlyLoopBoundaryJudge, OnlineMemorySession,
    OpenAIChatJSONClient, RetrievalBudgetGate, TransitionPlanner,
)

controller_client = OpenAIChatJSONClient(
    api_key=os.environ["CONTROLLER_API_KEY"],
    model=os.environ["CONTROLLER_MODEL"],
    base_url=os.environ["CONTROLLER_BASE_URL"],
)
planner_client = OpenAIChatJSONClient(
    api_key=os.environ["RESEARCHER_API_KEY"],
    model=os.environ["RESEARCHER_MODEL"],
    base_url=os.environ["RESEARCHER_BASE_URL"],
)
session = OnlineMemorySession(
    question="Which source establishes the requested date?",
    controller=ContextOnlyLoopBoundaryJudge(controller_client),
    planner=TransitionPlanner(planner_client),
    reranker=BGEReranker(os.environ["RERANKER_PATH"], device="cpu"),
    budget_gate=RetrievalBudgetGate(),
)
```

The agent host calls `session.build_prompt(system_message, user_message)` before
each researcher step, and `session.submit_batch(action_message, result_messages)`
after executing all tools in a batch. Call `session.finalize()` when the task
ends. Tool results need matching unique `tool_call_id` fields; incomplete batches
are rejected. The researcher remains responsible for tool selection and answers.

For servers without authentication, supply the server's documented dummy API
key. Use the actual trained controller and cross-encoder checkpoints for learned
inference. Without a reranker, the runtime uses a lexical fallback for the
offline example. Browser and search tools are supplied by the agent host.
