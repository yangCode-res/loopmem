# Runtime usage

The controller classifies a complete tool-action/result batch after all results
arrive. A switch archives the outgoing loop, seeds a new loop with the incoming
batch, and invokes the planner. Continuation retains the objective and evidence
needs. Historical evidence is recalled on every `build_prompt` call.

## Connecting model clients

The controller and planner accept a client exposing
`complete_json(system_prompt, user_prompt)`. Instantiate separate clients, using
your controller serving name and researcher serving name respectively:

```python
import os
import json
from pathlib import Path
from loopmem import (
    BGEReranker, ContextOnlyLoopBoundaryJudge, OnlineMemorySession,
    OpenAIChatJSONClient, RetrievalBudgetConfig, RetrievalBudgetGate, TransitionPlanner,
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
config = json.loads(Path("configs/runtime.json").read_text())
budget = RetrievalBudgetGate(RetrievalBudgetConfig(**config.pop("retrieval_budget")))
session = OnlineMemorySession(
    question="Which source establishes the requested date?",
    controller=ContextOnlyLoopBoundaryJudge(controller_client),
    planner=TransitionPlanner(planner_client),
    reranker=BGEReranker(os.environ["RERANKER_PATH"], device="cpu"),
    budget_gate=budget,
    **config,
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
