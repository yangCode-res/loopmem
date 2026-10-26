"""Causal semantic handoff planner from the frozen context-only runtime."""
from __future__ import annotations
import json
import os
from typing import Any
from .client import OpenAIChatJSONClient
from .prompts import BATCH_ALIGNMENT_PROMPT, PLANNER_PROMPT

ATOMIC_CONTRACT_PROMPT = PLANNER_PROMPT

TRANSITION_SYSTEM_PROMPT = """You are a semantic Loop transition planner for a deep-research agent.
The Researcher independently chooses concrete tools and queries. Given a causal handoff and the just-observed
action/result, define the smallest independently decidable next evidence workstream and a retrieval intent for
prior Loop Memory. Do not answer the user's question, invent evidence, choose tools, URLs, or exact queries.
Return exactly one JSON object with next_loop and retrieval_request. next_loop must contain loop_description,
target_relation, and completion_test. retrieval_request must contain subject_scope, target_relation,
required_evidence, and key_entities. A proposal is a hypothesis, not evidence."""

class TransitionPlanner:
    def __init__(self, client: OpenAIChatJSONClient | None, *, contract_mode: str | None = None) -> None:
        self.client = client
        self.contract_mode = contract_mode or os.getenv("LOOPMEM_LOOP_CONTRACT_MODE", "atomic")
        if self.contract_mode not in {"legacy", "atomic"}:
            raise ValueError("LOOPMEM_LOOP_CONTRACT_MODE must be legacy or atomic")

    @staticmethod
    def _text(value: Any, limit: int = 600) -> str:
        text = str(value or "").strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    def propose(
        self,
        *,
        task_summary: str,
        closed_loop_handoff: dict[str, Any] | None,
        current_loop: dict[str, Any] | None,
        observed_action: dict[str, Any],
        observed_result: str | list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.client:
            return {
                "next_loop": {
                    "loop_description": "Observed evidence workstream",
                    "target_relation": "Observed evidence relation",
                    "completion_test": "The required evidence relation is established or ruled out.",
                },
                "retrieval_request": {
                    "subject_scope": "",
                    "target_relation": "Observed evidence relation",
                    "required_evidence": "Relevant prior evidence",
                    "key_entities": [],
                },
                "planner_error": "no transition planner client",
            }
        payload = {
            "task_summary": self._text(task_summary, 800),
            "closed_loop_handoff": closed_loop_handoff or {},
            "current_loop": current_loop or {},
            "observed_action": observed_action,
            "observed_result": (
                [{"tool_call_id": str(item.get("tool_call_id") or ""),
                  "text": self._text(item.get("text"), 900)} for item in observed_result]
                if isinstance(observed_result, list) else self._text(observed_result, 1200)
            ),
        }
        system_prompt = TRANSITION_SYSTEM_PROMPT + BATCH_ALIGNMENT_PROMPT
        if self.contract_mode == "atomic":
            system_prompt += ATOMIC_CONTRACT_PROMPT
        raw = self.client.complete_json(system_prompt, json.dumps(payload, ensure_ascii=False))
        # Models may include harmless metadata (for example, a rationale or
        # confidence) alongside the two contracts.  Ignore such extras while
        # enforcing the required contract structure below; rejecting the whole
        # response would unnecessarily disable semantic handoff/retrieval.
        if not isinstance(raw, dict):
            raise ValueError("transition planner response must be a JSON object")
        loop = raw.get("next_loop")
        request = raw.get("retrieval_request")
        if not isinstance(loop, dict) or not isinstance(request, dict):
            raise ValueError("transition planner contracts must be objects")
        required_loop = {"loop_description", "target_relation", "completion_test"}
        required_request = {"subject_scope", "target_relation", "required_evidence", "key_entities"}
        if not required_loop <= set(loop) or not required_request <= set(request):
            raise ValueError("transition planner returned an incomplete contract")
        clean_loop = {
            key: self._text(loop.get(key), 500)
            for key in required_loop
        }
        entities = request.get("key_entities")
        if not isinstance(entities, list):
            entities = []
        clean_request = {
            "subject_scope": self._text(request.get("subject_scope"), 500),
            "target_relation": self._text(request.get("target_relation"), 500),
            "required_evidence": self._text(request.get("required_evidence"), 700),
            "key_entities": [self._text(item, 160) for item in entities[:8] if self._text(item, 160)],
        }
        if not all(clean_loop.values()) or not clean_request["target_relation"]:
            raise ValueError("transition planner returned empty semantic fields")
        return {"next_loop": clean_loop, "retrieval_request": clean_request,
                "contract_mode": self.contract_mode}
