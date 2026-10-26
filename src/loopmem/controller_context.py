"""Context-window helpers shared by Controller training and evaluation.

The context-only Controller has a deliberately small input contract.  The
full trajectory remains available to the runtime trace/audit layer, but the
boundary classifier only needs the active Loop contract, a compact view of
the current workstream, and the candidate action.  Keeping this projection in
one module lets online inference and every SFT data producer use the exact
same schema.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any


CONTEXT_ONLY_INPUT_SCHEMA = "context_only_boundary_input_v2"
CONTEXT_ONLY_POST_RESULT_INPUT_SCHEMA = "context_only_boundary_input_v3_post_result"


def _compact_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    # Preserve both the beginning (usually the action/subject) and the end
    # (often the useful result) without allowing a single web page to consume
    # the Controller budget.
    head = max(1, int(limit * 0.58))
    tail = max(1, limit - head - 29)
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"


def _compact_event(value: Any, *, limit: int = 600) -> dict[str, str]:
    """Project a recorded event to role plus bounded action/result text."""

    if not isinstance(value, dict):
        return {"role": "unknown", "text": _compact_text(value, limit)}
    role = str(value.get("role") or "unknown")
    text = str(value.get("text") or value.get("content") or "").strip()
    calls = value.get("tool_calls")
    if calls:
        text = f"[tool_calls] {_compact_text(json.dumps(calls, ensure_ascii=False), 420)}"
    elif role == "assistant" and not value.get("recipient") and "[tool_calls]" not in text:
        # Free-form planning/reasoning is not a boundary signal.  The action
        # itself is supplied separately as candidate_action.
        text = ""
    # Old materialized samples already flatten calls into ``text``.  Keep only
    # a bounded action/result snippet; IDs and transport metadata are audit
    # fields and do not belong in the boundary decision input.
    return {"role": role, "text": _compact_text(text, limit)}


def _compact_candidate(value: Any) -> dict[str, str]:
    """Extract only the candidate tool action from old/new sample shapes."""

    if hasattr(value, "message"):
        value = getattr(value, "message")
    items = value if isinstance(value, list) else [value]
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "unknown")
        text = str(item.get("text") or item.get("content") or "").strip()
        calls = item.get("tool_calls")
        if isinstance(calls, list) and len(calls) > 1:
            actions = []
            for call in calls:
                function = call.get("function") or {}
                actions.append({
                    "tool_call_id": str(call.get("id") or ""),
                    "tool": str(function.get("name") or "research_action"),
                    "arguments": _compact_text(function.get("arguments"), 420),
                })
            return {"tool": "tool_batch", "arguments": json.dumps(actions, ensure_ascii=False)}
        if calls and isinstance(calls, list) and isinstance(calls[0], dict):
            function = calls[0].get("function") or {}
            if isinstance(function, dict):
                return {
                    "tool": str(function.get("name") or "research_action"),
                    "arguments": _compact_text(function.get("arguments"), 420),
                }
        # compact_message serializes tool calls into text.  Do not attempt to
        # parse arbitrary prose; retaining one bounded action line is safer.
        if "[tool_calls]" in text:
            return {
                "tool": str(item.get("recipient") or item.get("name") or "research_action"),
                "arguments": _compact_text(text.split("[tool_calls]", 1)[-1], 420),
            }
        # The runtime's compact event representation stores the serialized
        # function call directly in ``text``.  Recover only its name and
        # arguments when it is valid JSON; arbitrary tool output is ignored.
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
        candidates = parsed if isinstance(parsed, list) else []
        for parsed_call in candidates:
            if not isinstance(parsed_call, dict):
                continue
            function = parsed_call.get("function") or {}
            if isinstance(function, dict) and function.get("name"):
                return {
                    "tool": str(function.get("name")),
                    "arguments": _compact_text(function.get("arguments"), 420),
                }
        tool_hint = item.get("recipient") or item.get("name")
        if tool_hint and role == "assistant":
            return {
                "tool": str(tool_hint),
                "arguments": _compact_text(text, 420),
            }
    return {"tool": "research_action", "arguments": ""}


def compact_context_only_input(
    *,
    question: Any,
    active_loop: Any,
    current_events: Sequence[Any] | None = None,
    candidate_action: Any = None,
    task_summary_limit: int = 800,
    recent_event_limit: int = 4,
) -> dict[str, Any]:
    """Build the v2 compact input used by online and SFT Controllers.

    ``current_events`` may contain runtime ``TrajectoryEvent`` objects or the
    compact dictionaries emitted by the retrospective labelers.  The helper
    intentionally ignores completed-loop catalogs and raw retrieved evidence:
    retrieval is a separate memory concern and the context-only Controller
    only classifies the current boundary.
    """

    projected: list[dict[str, str]] = []
    for event in current_events or []:
        raw = getattr(event, "message", event)
        item = _compact_event(raw)
        # Assistant prose/reasoning is not needed for boundary classification;
        # keep tool actions and tool observations only.
        if item["role"] in {"assistant", "tool"} and item["text"]:
            projected.append(item)
    projected = projected[-max(1, recent_event_limit) :]
    summary_parts = []
    for item in projected:
        prefix = "action" if item["role"] == "assistant" else "result"
        summary_parts.append(f"{prefix}: {item['text']}")
    return {
        "schema_version": CONTEXT_ONLY_INPUT_SCHEMA,
        "task_summary": _compact_text(question, task_summary_limit),
        "active_loop": dict(active_loop or {}),
        "current_loop_summary": _compact_text(" | ".join(summary_parts), 1400),
        "recent_evidence": projected,
        "candidate_action": _compact_candidate(candidate_action),
    }


def compact_context_only_post_result_input(
    *,
    question: Any,
    active_loop: Any,
    current_events: Sequence[Any] | None = None,
    candidate_action: Any = None,
    candidate_result: Any = None,
    task_summary_limit: int = 800,
    recent_event_limit: int = 4,
) -> dict[str, Any]:
    """Build the causal post-result boundary input.

    The Controller sees the Researcher's action and the paired tool result,
    but never any later action.  This resolves retrospective boundaries that
    are not identifiable from a bare ``open(cursor)`` or ``find(pattern)``
    request while keeping the payload bounded.
    """

    value = compact_context_only_input(
        question=question,
        active_loop=active_loop,
        current_events=current_events,
        candidate_action=candidate_action,
        task_summary_limit=task_summary_limit,
        recent_event_limit=recent_event_limit,
    )
    value["schema_version"] = CONTEXT_ONLY_POST_RESULT_INPUT_SCHEMA
    if isinstance(candidate_result, list):
        value["candidate_results"] = [
            {"tool_call_id": str(item.get("tool_call_id") or ""),
             "text": _compact_text(item.get("text"), 900)}
            for item in candidate_result
        ]
        value["candidate_result"] = "See candidate_results: one result for each observed batch action."
    else:
        value["candidate_result"] = _compact_text(candidate_result, 900)
    return value


__all__ = [
    "CONTEXT_ONLY_INPUT_SCHEMA",
    "CONTEXT_ONLY_POST_RESULT_INPUT_SCHEMA",
    "compact_context_only_input",
    "compact_context_only_post_result_input",
]
