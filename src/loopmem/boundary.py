"""Extracted context-only boundary decision contract."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Protocol

from .controller_context import (
    compact_context_only_post_result_input,
)
from .schema import TaskAnchor
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
    current_loop_subgoal: str = ""
    current_loop_summary: str = ""
    controller_input: dict[str, Any] | None = None


class ContextOnlyLoopBoundaryJudge:
    """Deployment adapter for the context-only Controller SFT contract.

    The model observes an action that the Researcher already selected.  It may
    segment context, but it never selects a research action or decides task
    completion.  One instance belongs to one online research session.
    """

    def __init__(self, client: OpenAIChatJSONClient | None) -> None:
        self.client = client
        self.active_contract: dict[str, str] | None = None
        self._next_loop_number = 1


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


    def judge(
        self,
        anchor: TaskAnchor,
        current_loop_events: list[TrajectoryEvent],
        candidate_next_event: TrajectoryEvent,
        candidate_result_event: TrajectoryEvent | None = None,
    ) -> LoopBoundaryDecision:
        if candidate_next_event.role != "assistant" or not (
            candidate_next_event.message.get("tool_calls") or []
        ):
            return LoopBoundaryDecision(
                split=False,
                reason="candidate is not an observed research tool action",
                decision="CONTINUE",
            )
        if self.active_contract is None:
            self.observe_initial_event(current_loop_events[0] if current_loop_events else candidate_next_event)
        if not self.client:
            raise ValueError("a boundary controller client is required")

        # Keep complete prior packets locally for audit/retrieval, but send the
        # boundary classifier only a small current-workstream projection.
        if candidate_result_event is None:
            raise ValueError("loop boundaries are classified only after tool results")
        payload = compact_context_only_post_result_input(
            question=anchor.original_goal,
            active_loop=self.active_contract,
            current_events=current_loop_events,
            candidate_action=candidate_next_event,
            candidate_result=candidate_result_event.message.get("batch_results", candidate_result_event.text),
        )
        system_prompt = BOUNDARY_PROMPT
        raw = self.client.complete_json(system_prompt, json.dumps(payload, ensure_ascii=False))
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
        elif requested == "SWITCH_LOOP":
            if closed != current_contract.get("loop_key"):
                raise ValueError("SWITCH_LOOP must close the active loop_key")
            # Semantic next-loop planning runs as a separate model call. Keep a provisional contract here so
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
            current_loop_subgoal=current_contract.get("loop_description", ""),
            current_loop_summary=(
                _fallback_summary(current_loop_events) if split else ""
            ),
            controller_input=payload,
        )


def _fallback_summary(events: list[Any]) -> str:
    observations = [_truncate(event.text, 500) for event in events if event.role == "tool" and event.text]
    if observations:
        return "Observed evidence in this segment: " + " | ".join(observations[-3:])
    return "The observed segment recorded research actions but no usable tool observation."
