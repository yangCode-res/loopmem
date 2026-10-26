"""Extracted context-only boundary decision contract."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Protocol

from .controller_context import (
    compact_context_only_input, compact_context_only_post_result_input,
)
from .schema import LoopMemory, TaskAnchor
from .prompts import BOUNDARY_PROMPT
from .client import OpenAIChatJSONClient


class TrajectoryEvent(Protocol):
    sequence: int
    role: str
    text: str
    message: dict[str, Any]


def _truncate(text: str, limit: int = 400) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit - 1].rstrip() + "…"

@dataclass
class LoopBoundaryDecision:
    split: bool
    reason: str
    confidence: float = 0.5
    decision: str = "CONTINUE"
    resume_loop_id: str = ""
    current_loop_signature: dict[str, str] = field(default_factory=dict)
    candidate_loop_signature: dict[str, str] = field(default_factory=dict)
    current_loop_subgoal: str = ""
    next_loop_subgoal: str = ""
    loop_outcome: str = "IN_PROGRESS"
    current_loop_summary: str = ""
    requested_decision: str = "CONTINUE"
    normalization_reason: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)
    controller_system_prompt: str = ""
    controller_input: dict[str, Any] | None = None
    controller_model_response: dict[str, Any] | None = None
    controller_provider: str = ""
    controller_request_count: int = 0
    controller_prompt_tokens: int = 0
    controller_completion_tokens: int = 0
    controller_completion_text: str = ""
    controller_sequence_logprob: float | None = None
    controller_token_logprobs: list[dict[str, Any]] = field(default_factory=list)

CONTEXT_ONLY_DECISION_SYSTEM_PROMPT = BOUNDARY_PROMPT

CONTEXT_ONLY_POST_RESULT_DECISION_SYSTEM_PROMPT = BOUNDARY_PROMPT

class ContextOnlyLoopBoundaryJudge:
    """Deployment adapter for the context-only Controller SFT contract.

    The model observes an action that the Researcher already selected.  It may
    segment context, but it never selects a research action or decides task
    completion.  One instance belongs to one online research session.
    """

    def __init__(self, client: OpenAIChatJSONClient | None) -> None:
        self.client = client
        self.active_contract: dict[str, str] | None = None
        self.completed_packets: list[dict[str, Any]] = []
        self._next_loop_number = 1

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        parts: list[str] = []
        content = message.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            parts.append(reasoning)
        calls = message.get("tool_calls")
        if calls:
            parts.append(json.dumps(calls, ensure_ascii=False))
        return "\n".join(parts).strip()

    @classmethod
    def _compact_event(cls, event: TrajectoryEvent, *, text_limit: int) -> dict[str, Any]:
        text = cls._message_text(event.message)
        return {
            "message_id": f"msg_{event.sequence:04d}",
            "index": event.sequence,
            "role": event.role,
            "name": event.message.get("name"),
            "recipient": event.message.get("recipient"),
            "channel": event.message.get("channel"),
            "text": text[:text_limit],
            "truncated": len(text) > text_limit,
        }

    @staticmethod
    def _tool_action_label(event: TrajectoryEvent) -> str:
        calls = event.message.get("tool_calls") or []
        if not calls or not isinstance(calls[0], dict):
            return "Observed evidence workstream"
        function = calls[0].get("function") or {}
        if not isinstance(function, dict):
            return "Observed evidence workstream"
        name = str(function.get("name") or "research action").split(".")[-1]
        arguments = str(function.get("arguments") or "").strip()
        arguments = re.sub(r"\s+", " ", arguments)[:360]
        if arguments:
            return _truncate(f"Evidence relation pursued in the observed {name} action: {arguments}", 480)
        return _truncate(f"Evidence relation pursued in the observed {name} action", 480)

    @staticmethod
    def _signature(contract: dict[str, str] | None) -> dict[str, str]:
        if not contract:
            return {}
        description = str(contract.get("loop_description") or "").strip()
        completion = str(contract.get("completion_test") or "").strip()
        return {
            "subject_scope": _truncate(description or "Observed evidence scope", 300),
            "target_relation": _truncate(description or "Observed evidence relation", 300),
            "success_condition": _truncate(
                completion or "The required observable evidence is established", 300
            ),
        }

    def observe_initial_event(self, event: TrajectoryEvent) -> dict[str, str]:
        """Initialize L001 deterministically without spending a model call."""

        if self.active_contract is None:
            description = self._tool_action_label(event)
            self.active_contract = {
                "loop_key": "L001",
                "loop_description": description,
                "completion_test": "The observable evidence relation pursued by this workstream is established or ruled out.",
            }
            self._next_loop_number = 2
        return dict(self.active_contract)

    @staticmethod
    def _evidence_items(events: list[TrajectoryEvent], loop_key: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for event in events:
            if event.role != "tool":
                continue
            compact = ContextOnlyLoopBoundaryJudge._compact_event(event, text_limit=1600)
            items.append(
                {
                    "memory_id": f"{loop_key}:{compact['message_id']}",
                    "message_id": compact["message_id"],
                    "name": compact.get("name"),
                    "text": compact.get("text") or "",
                    "truncated": bool(compact.get("truncated")),
                }
            )
        return items

    @staticmethod
    def _tokens(value: Any) -> set[str]:
        return set(re.findall(r"[a-z0-9_\-]{3,}", json.dumps(value, ensure_ascii=False).casefold()))

    def _retrieved_context(
        self,
        *,
        question: str,
        candidate_messages: list[dict[str, Any]],
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        if not self.completed_packets:
            return []
        query_tokens = self._tokens(
            {
                "question": question,
                "active_loop": self.active_contract,
                "candidate_research_action": candidate_messages,
            }
        )
        ranked: list[tuple[float, int, int, dict[str, Any]]] = []
        for recency, packet in enumerate(self.completed_packets):
            for item_index, item in enumerate(packet.get("evidence_items") or []):
                item_tokens = self._tokens(item.get("text") or "")
                overlap = len(query_tokens & item_tokens)
                if overlap:
                    score = overlap / max(
                        1.0, (len(query_tokens) * max(1, len(item_tokens))) ** 0.5
                    )
                    ranked.append(
                        (score, recency, item_index, {"loop_key": packet["loop_key"], **item})
                    )
        ranked.sort(key=lambda value: (value[0], value[1], value[2]), reverse=True)
        selected = [item for _, _, _, item in ranked[:top_k]]
        if selected:
            return selected
        latest = self.completed_packets[-1]
        return [
            {"loop_key": latest["loop_key"], **item}
            for item in (latest.get("evidence_items") or [])[-min(2, top_k) :]
        ]

    @staticmethod
    def _validate_contract(raw: Any, expected_key: str) -> dict[str, str]:
        if not isinstance(raw, dict) or set(raw) != {
            "loop_key",
            "loop_description",
            "completion_test",
        }:
            raise ValueError("new_loop must contain exactly loop_key, loop_description, completion_test")
        contract = {key: str(raw.get(key) or "").strip() for key in raw}
        if contract["loop_key"] != expected_key:
            raise ValueError(f"new_loop.loop_key must be {expected_key}")
        if not contract["loop_description"] or not contract["completion_test"]:
            raise ValueError("new_loop description and completion_test must be non-empty")
        return contract

    def judge(
        self,
        anchor: TaskAnchor,
        current_loop_events: list[TrajectoryEvent],
        candidate_next_event: TrajectoryEvent,
        prior_loops: list[LoopMemory] | None = None,
        committed_current_signature: dict[str, str] | None = None,
        candidate_result_event: TrajectoryEvent | None = None,
    ) -> LoopBoundaryDecision:
        del prior_loops, committed_current_signature
        if candidate_next_event.role != "assistant" or not (
            candidate_next_event.message.get("tool_calls") or []
        ):
            return LoopBoundaryDecision(
                split=False,
                reason="candidate is not an observed research tool action",
                decision="CONTINUE",
                requested_decision="CONTINUE",
                current_loop_signature=self._signature(self.active_contract),
                candidate_loop_signature=self._signature(self.active_contract),
                normalization_reason="candidate_is_not_a_research_tool_action",
                controller_system_prompt=CONTEXT_ONLY_DECISION_SYSTEM_PROMPT,
            )
        if self.active_contract is None:
            self.observe_initial_event(current_loop_events[0] if current_loop_events else candidate_next_event)
        if not self.client:
            return LoopBoundaryDecision(
                split=False,
                reason="no client available",
                confidence=0.0,
                current_loop_signature=self._signature(self.active_contract),
                candidate_loop_signature=self._signature(self.active_contract),
            )

        # Keep complete prior packets locally for audit/retrieval, but send the
        # boundary classifier only a small current-workstream projection.
        if candidate_result_event is not None:
            payload = compact_context_only_post_result_input(
                question=anchor.original_goal,
                active_loop=self.active_contract,
                current_events=current_loop_events,
                candidate_action=candidate_next_event,
                candidate_result=candidate_result_event.message.get("batch_results", candidate_result_event.text),
            )
            system_prompt = CONTEXT_ONLY_POST_RESULT_DECISION_SYSTEM_PROMPT
        else:
            payload = compact_context_only_input(
                question=anchor.original_goal,
                active_loop=self.active_contract,
                current_events=current_loop_events,
                candidate_action=candidate_next_event,
            )
            system_prompt = CONTEXT_ONLY_DECISION_SYSTEM_PROMPT
        before = (
            int(getattr(self.client, "request_count", 0)),
            int(getattr(self.client, "total_prompt_tokens", 0)),
            int(getattr(self.client, "total_completion_tokens", 0)),
        )
        raw = self.client.complete_json(system_prompt, json.dumps(payload, ensure_ascii=False))
        after = (
            int(getattr(self.client, "request_count", 0)),
            int(getattr(self.client, "total_prompt_tokens", 0)),
            int(getattr(self.client, "total_completion_tokens", 0)),
        )
        if not isinstance(raw, dict) or set(raw) != {"decision", "closed_loop_key"}:
            raise ValueError("Controller response must contain exactly decision and closed_loop_key")
        requested = str(raw.get("decision") or "").strip().upper()
        closed = str(raw.get("closed_loop_key") or "").strip()
        current_contract = dict(self.active_contract or {})
        expected_new_key = f"L{self._next_loop_number:03d}"
        if requested == "CONTINUE":
            if closed:
                raise ValueError("CONTINUE must not close a Loop")
            split = False
            decision = "CONTINUE"
            candidate_contract = current_contract
            outcome = "IN_PROGRESS"
        elif requested == "SWITCH_LOOP":
            if closed != current_contract.get("loop_key"):
                raise ValueError("SWITCH_LOOP must close the active loop_key")
            # Semantic next-Loop planning runs as a separate Researcher/
            # DeepSeek transition step. Keep a provisional contract here so
            # the boundary state machine can archive and advance safely.
            candidate_contract = {
                "loop_key": expected_new_key,
                "loop_description": self._tool_action_label(candidate_next_event),
                "completion_test": (
                    "The independently decidable evidence relation pursued by this workstream "
                    "is established or ruled out."
                ),
            }
            split = True
            decision = "SWITCH_NEW"
            outcome = "SUPERSEDED"
            self.completed_packets.append(
                {
                    **current_contract,
                    "outcome": outcome,
                    "evidence_items": self._evidence_items(
                        current_loop_events, current_contract["loop_key"]
                    ),
                }
            )
            self.active_contract = candidate_contract
            self._next_loop_number += 1
        else:
            raise ValueError("decision must be CONTINUE or SWITCH_LOOP")
        return LoopBoundaryDecision(
            split=split,
            reason=(
                "context-only Controller opened a new independently decidable evidence workstream"
                if split
                else "candidate remains in the active evidence workstream"
            ),
            confidence=1.0,
            decision=decision,
            current_loop_signature=self._signature(current_contract),
            candidate_loop_signature=self._signature(candidate_contract),
            current_loop_subgoal=current_contract.get("loop_description", ""),
            next_loop_subgoal=candidate_contract.get("loop_description", "") if split else "",
            loop_outcome=outcome,
            current_loop_summary=(
                _fallback_summary(current_loop_events) if split else ""
            ),
            requested_decision=requested,
            raw_response=dict(raw),
            controller_system_prompt=system_prompt,
            controller_input=payload,
            controller_model_response=dict(raw),
            controller_provider=str(getattr(self.client, "model", "")),
            controller_request_count=after[0] - before[0],
            controller_prompt_tokens=after[1] - before[1],
            controller_completion_tokens=after[2] - before[2],
            controller_completion_text=str(
                getattr(self.client, "last_completion_text", "") or ""
            ),
            controller_sequence_logprob=getattr(self.client, "last_completion_logprob", None),
            controller_token_logprobs=list(
                getattr(self.client, "last_completion_token_logprobs", []) or []
            ),
        )


def _fallback_summary(events: list[Any]) -> str:
    observations = [_truncate(event.text, 500) for event in events if event.role == "tool" and event.text]
    if observations:
        return "Observed evidence in this segment: " + " | ".join(observations[-3:])
    return "The observed segment recorded research actions but no usable tool observation."
