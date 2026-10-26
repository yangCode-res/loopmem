"""Offline walkthrough using scripted protocol responses, with no model weights."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loopmem import ContextOnlyLoopBoundaryJudge, OnlineMemorySession, TransitionPlanner


class ScriptedJSONClient:
    """A tiny protocol fixture; it does not implement boundary/planning logic."""
    model = "offline-fixture"
    def __init__(self): self.boundary_calls = 0
    def complete_json(self, system_prompt, user_prompt):
        if "context-boundary controller" in system_prompt.lower() or "context-only" in system_prompt.lower():
            self.boundary_calls += 1
            return {"decision": "SWITCH_LOOP", "closed_loop_key": "L001"}
        payload=json.loads(user_prompt)
        action=payload.get("observed_action",{})
        label="the date check" if "date" in str(action).lower() else "the source check"
        return {"next_loop":{"loop_description":label,"target_relation":label,"completion_test":"observed evidence establishes or rules out this relation"},"retrieval_request":{"subject_scope":"example subject","target_relation":label,"required_evidence":"a source-backed result","key_entities":[]}}


def tool_call(call_id, query):
    return {"id":call_id,"type":"function","function":{"name":"search","arguments":json.dumps({"query":query})}}

client=ScriptedJSONClient()
session=OnlineMemorySession(
    question="Find a source and its publication date.",
    controller=ContextOnlyLoopBoundaryJudge(client),
    planner=TransitionPlanner(client),
)
system={"role":"system","content":"Researcher system prompt."}
user={"role":"user","content":session.question}
session.submit_batch(
    {"role":"assistant","tool_calls":[tool_call("a1","source") ]},
    [{"role":"tool","tool_call_id":"a1","name":"search","content":"A source page was found."}],
)
session.build_prompt(system,user)
session.submit_batch(
    {"role":"assistant","tool_calls":[tool_call("a2","publication date") ]},
    [{"role":"tool","tool_call_id":"a2","name":"search","content":"The page states the date as 2024-01-01."}],
)
packed=session.build_prompt(system,user)
print(json.dumps({"switch":session.traces[-1].loop_switched,"retrieved_loop_ids":session.traces[-1].selected_loop_ids,"memory_context_present":"<cross_loop_memory>" in packed[0]["content"]},indent=2))
