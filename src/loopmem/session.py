"""Minimal online context-only memory session extracted from the frozen runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import hashlib
import re
from typing import Any, Protocol

from .boundary import ContextOnlyLoopBoundaryJudge, LoopBoundaryDecision, _fallback_summary
from .retrieval import LexicalMemoryIndex, MemoryHit, tokenize
from .retrieval_budget import RetrievalBudgetGate
from .reranker import BGEReranker
from .schema import LoopMemory, MemoryStatus, RawMemory, SourceType, TaskAnchor, utc_now
from .transition import TransitionPlanner
from .controller_context import _compact_candidate


class JSONClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


@dataclass
class Event:
    sequence: int
    role: str
    text: str
    raw_memory: RawMemory
    message: dict[str, Any]
    name: str | None = None
    tool_call_id: str | None = None


@dataclass
class RuntimeTrace:
    round_number: int
    query: str
    selected_loop_ids: list[str]
    selected_record_ids: list[str]
    loop_switched: bool
    retrieval_budget: dict[str, Any] = field(default_factory=dict)


def _message_text(message: dict[str, Any]) -> str:
    parts = []
    content = message.get("content")
    if isinstance(content, str) and content:
        parts.append(content)
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append(f"[reasoning]\n{reasoning}")
    calls = message.get("tool_calls")
    if calls:
        parts.append(f"[tool_calls]\n{json.dumps(calls, ensure_ascii=False)}")
    return "\n".join(parts).strip()


def _relevant_excerpt(text: str, query: str, limit: int) -> str:
    """Keep the passage that caused lexical retrieval, not only the prefix.

    Search-result memories are often several thousand characters long. The
    useful result can occur near the end, so prefix truncation can turn a
    successful BM25 recall into an empty signal for the research model.
    """

    if len(text) <= limit:
        return text
    query_terms = {
        term for term in tokenize(query)
        if len(term) >= 3 and term not in {"the", "and", "for", "with", "from", "that", "this"}
    }
    if not query_terms:
        return text[:limit]

    window_size = max(400, limit - 360)
    stride = max(200, window_size // 3)
    best_start = 0
    best_score = -1.0
    lowered = text.casefold()
    document_tokens = tokenize(text)
    document_frequency = {term: document_tokens.count(term) for term in query_terms}
    final_start = max(0, len(text) - window_size)
    starts = list(range(0, final_start + 1, stride))
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    for start in starts:
        window = lowered[start : start + window_size]
        score = 0.0
        for term in query_terms:
            count = window.count(term)
            if count:
                # Rare terms such as names dominate generic clue words.
                score += count / max(1, document_frequency.get(term, 1))
        if score > best_score:
            best_score = score
            best_start = start
    best_start = max(0, min(best_start, len(text) - window_size))
    passage = text[best_start : best_start + window_size]
    if best_start == 0:
        return passage[:limit]
    header = text[:120].rstrip()
    return (header + "\n...[retrieval-matched passage]...\n" + passage)[:limit]

class OnlineMemorySession:
    """Accept complete action/result batches, archive loops, and repack context."""

    def __init__(
        self,
        *,
        question: str,
        controller: ContextOnlyLoopBoundaryJudge,
        planner: TransitionPlanner,
        top_k: int = 3,
        record_top_k: int = 5,
        memory_text_limit: int = 1800,
        reranker: BGEReranker | None = None,
        budget_gate: RetrievalBudgetGate | None = None,
    ) -> None:
        self.question = question
        self.anchor = TaskAnchor(
            task_id="task_loopmem",
            original_goal=question,
            success_criteria=["produce an evidence-backed answer"],
            immutable_constraints=["preserve source evidence"],
            domain="deep_research",
        )
        self.controller = controller
        self.planner = planner
        self.top_k = max(1, int(top_k))
        self.record_top_k = max(1, int(record_top_k))
        self.memory_text_limit = max(1, int(memory_text_limit))
        self.reranker = reranker
        self.budget_gate = budget_gate
        self.index = LexicalMemoryIndex()
        self.current_events: list[Event] = []
        self.completed_loops: list[LoopMemory] = []
        self.retrieval_request: dict[str, Any] = {}
        self.current_loop_signature: dict[str, str] = {}
        self.latest_decision = LoopBoundaryDecision(False, "initial loop")
        self.traces: list[RuntimeTrace] = []
        self._sequence = 0
        self._latest_observation = ""
        self._last_budget_decision = None
        self._workunit_seen_text: set[str] = set()
        self._loop_rounds = 0
        self._stagnant_rounds = 0
        self._forced_memory_ids: list[str] = []

    def _observe_batch(self, results: list[dict[str, Any]]) -> None:
        """Count a complete native batch; text novelty remains a proxy for progress."""
        novel = False
        for message in results:
            text = str(message.get("content") or "").strip()
            failed = re.match(r"^(?:Error\b|Invalid\b|No (?:search )?results\b)", text, re.I)
            soft404 = "404 - Page not found" in text[:500]
            if not text or failed or soft404:
                continue
            lines=[]
            for line in text.splitlines():
                if re.match(r"^\[\d+\] |^\*\*viewing lines|^L\d+: URL:",line):
                    continue
                lines.append(re.sub(r"^L\d+:\s*", "", line))
            body="\n".join(lines).strip()
            if body:
                fingerprint=hashlib.sha256(body.encode()).hexdigest()
                novel |= fingerprint not in self._workunit_seen_text
                self._workunit_seen_text.add(fingerprint)
        self._loop_rounds += 1
        self._stagnant_rounds = 0 if novel else self._stagnant_rounds + 1

    def _event(self, message: dict[str, Any]) -> Event:
        self._sequence += 1
        role = str(message.get("role") or "unknown")
        text = _message_text(message)
        raw = RawMemory(
            task_id=self.anchor.task_id,
            source_type={"user": SourceType.USER, "assistant": SourceType.AGENT, "tool": SourceType.TOOL}.get(role, SourceType.SYSTEM),
            content={"sequence": self._sequence, "role": role, "text": text},
            content_type="application/json",
            metadata={"sequence": self._sequence, "role": role},
        )
        return Event(self._sequence, role, text, raw, dict(message), message.get("name"), message.get("tool_call_id"))

    @staticmethod
    def _validate_batch(action: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        calls = action.get("tool_calls") or []
        expected = [str(call.get("id") or "") for call in calls if isinstance(call, dict)]
        observed = [str(result.get("tool_call_id") or "") for result in results]
        if not expected or any(not item for item in expected) or len(set(expected)) != len(expected):
            raise ValueError("action must contain tool calls with unique nonempty IDs")
        if len(set(observed)) != len(observed) or set(observed) != set(expected):
            raise ValueError("tool results must match every action tool_call_id exactly once")
        by_id = {str(item["tool_call_id"]): item for item in results}
        return [{"tool_call_id": key, "text": _message_text(by_id[key])} for key in expected]

    @staticmethod
    def _action_payload(action: dict[str, Any]) -> dict[str, Any]:
        return _compact_candidate(action)

    def _plan(self, action: dict[str, Any], paired_results: list[dict[str, Any]]) -> None:
        handoff = self.completed_loops[-1].to_dict() if self.completed_loops else None
        proposal = self.planner.propose(
            task_summary=self.question,
            closed_loop_handoff=handoff,
            current_loop=dict(self.current_loop_signature),
            observed_action=self._action_payload(action),
            observed_result=(paired_results if len(paired_results) > 1 else paired_results[0]["text"]),
        )
        loop = proposal.get("next_loop") or {}
        self.current_loop_signature = {
            "subject_scope": str(loop.get("loop_description") or "")[:300],
            "target_relation": str(loop.get("target_relation") or "")[:300],
            "success_condition": str(loop.get("completion_test") or "")[:300],
        }
        self.retrieval_request = dict(proposal.get("retrieval_request") or {})
        self.controller.active_contract = {
            "loop_key": self.controller.active_contract.get("loop_key", "L001") if self.controller.active_contract else "L001",
            "loop_description": str(loop.get("loop_description") or "Observed evidence workstream"),
            "completion_test": str(loop.get("completion_test") or "The evidence relation is established or ruled out."),
        }

    def submit_batch(self, action: dict[str, Any], results: list[dict[str, Any]]) -> LoopBoundaryDecision:
        """Commit a full native tool batch and decide its boundary post-result."""
        if action.get("role") != "assistant" or not action.get("tool_calls"):
            raise ValueError("action must be an assistant tool-call message")
        paired = self._validate_batch(action, results)
        self._forced_memory_ids = []
        self._observe_batch(results)
        action_event = self._event(action)
        result_events = [self._event(item) for item in results]
        self._latest_observation = result_events[-1].text
        if not self.current_events:
            self.current_events = [action_event, *result_events]
            initial = self.controller.observe_initial_event(action_event)
            self.current_loop_signature = {
                "subject_scope": str(initial.get("loop_description") or "")[:300],
                "target_relation": str(initial.get("loop_description") or "")[:300],
                "success_condition": str(initial.get("completion_test") or "")[:300],
            }
            self._plan(action, paired)
            self.latest_decision = LoopBoundaryDecision(False, "first observed batch initializes the loop")
            return self.latest_decision

        projection = Event(
            sequence=result_events[-1].sequence,
            role="tool",
            text="paired tool batch",
            raw_memory=result_events[-1].raw_memory,
            message={"role": "tool", "batch_results": paired},
        ) if len(paired) > 1 else result_events[0]
        outgoing_contract = dict(self.controller.active_contract or {})
        decision = self.controller.judge(
            self.anchor,
            self.current_events,
            action_event,
            candidate_result_event=projection,
        )
        self.latest_decision = decision
        if decision.split:
            archived = self._archive(decision, outgoing_contract=outgoing_contract)
            if archived is not None:
                self._forced_memory_ids = [archived.loop_id]
            self._loop_rounds = 1
            self._stagnant_rounds = 0
            self.current_loop_signature = {}
            self.current_events = [action_event, *result_events]
            self._plan(action, paired)
        else:
            self.current_events.extend([action_event, *result_events])
        return decision

    def _archive(
        self,
        decision: LoopBoundaryDecision | None = None,
        *,
        outgoing_contract: dict[str, Any] | None = None,
    ) -> LoopMemory | None:
        if not self.current_events:
            return None
        tools = [event for event in self.current_events if event.role == "tool"]
        assistants = [event for event in self.current_events if event.role == "assistant"]
        contract = dict(outgoing_contract or self.controller.active_contract or {})
        if decision and decision.current_loop_subgoal:
            contract["loop_description"] = decision.current_loop_subgoal
        relation = str((self.current_loop_signature or {}).get("target_relation") or "").strip()
        subject = str((self.current_loop_signature or {}).get("subject_scope") or "").strip()
        subgoal = relation if not subject or subject.casefold() in relation.casefold() else f"{relation} — {subject}"
        subgoal = (subgoal or str(contract.get("loop_description") or "Observed evidence workstream"))[:240]
        summary = str(getattr(decision, "current_loop_summary", "") or "").strip() if decision else ""
        conclusion = summary[:2400] or (assistants[-1].text[:2400] if assistants else tools[-1].text[:2400] if tools else None)
        loop = LoopMemory(
            task_id=self.anchor.task_id,
            subgoal=subgoal,
            context=decision.reason if decision else "finalized at task completion",
            actions=[{"sequence": e.sequence, "text": e.text} for e in assistants],
            observations=[{"sequence": e.sequence, "text": e.text} for e in tools],
            conclusion=conclusion,
            evidence_ids=[e.raw_memory.raw_id for e in self.current_events],
            started_at=self.current_events[0].raw_memory.occurred_at,
            ended_at=utc_now(),
            status=MemoryStatus.SUPERSEDED if decision and decision.split else MemoryStatus.RESOLVED,
            semantic_signature=dict(self.current_loop_signature),
        )
        for event in self.current_events:
            self.index.add_raw(event.raw_memory)
        self.index.add_loop(loop)
        self.completed_loops.append(loop)
        return loop

    def _query(self) -> str:
        request = self.retrieval_request
        pieces = [request.get("subject_scope"), request.get("target_relation"), request.get("required_evidence"), *(request.get("key_entities") or []), self._latest_observation]
        return "\n".join(dict.fromkeys(str(item).strip() for item in pieces if str(item or "").strip()))

    def _retrieve(self, query: str) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
        broad_limit = max(max(self.top_k, 12) * 3, max(self.top_k, 12))
        pool = [h for h in self.index.search(query, top_k=broad_limit, task_id=self.anchor.task_id) if h.memory_type == "loop"][:max(self.top_k, 12)]
        if self.reranker and pool:
            try:
                pool = self.reranker.rerank(query, pool, top_k=max(self.top_k, 2))
            except Exception:
                pool = pool[:max(self.top_k, 2)]
        # Keep the outgoing loop available for the first incoming context,
        # as in the runtime's progress-handoff path. The shared budget still
        # limits the combined handoff and ranked candidates.
        combined = [*self.index.lookup(self._forced_memory_ids), *pool]
        hits = []
        seen_ids: set[str] = set()
        for hit in combined:
            if hit.memory_id in seen_ids:
                continue
            seen_ids.add(hit.memory_id)
            hits.append(hit)
            if len(hits) >= self.top_k:
                break
        budget = None
        if self.budget_gate:
            budget = self.budget_gate.begin(query=query, loop_scores=[float(h.metadata.get("rerank_score",h.score)) for h in hits], candidate_loop_count=len(hits), stagnant_rounds=self._stagnant_rounds)
            hits = hits[:budget.loop_k]
        records=[]
        packets=[]
        for hit in hits:
            packet=dict(hit.metadata.get("loop_packet") or {})
            candidates=packet.get("records") or []
            packet["available_record_count"]=len(candidates)
            packet["records"]=[]
            for record in candidates:
                excerpt=_relevant_excerpt(str(record.get("content") or ""),query,min(1800,self.memory_text_limit))
                records.append({**record,"content":excerpt,"ranking_text":excerpt,"loop_memory_id":hit.memory_id})
            packets.append((hit,packet))
        if self.reranker and records:
            try:
                ranking_limit = max(self.record_top_k, self.budget_gate.config.expanded_record_k) if self.budget_gate else self.record_top_k
                ranked=self.reranker.rerank_records(query,records,top_k=ranking_limit)
            except Exception:
                ranked=records[:self.record_top_k]
        else:
            ranked=records[:self.record_top_k]
        selected=ranked[:self.record_top_k]
        if budget:
            compact_ids=[str(x.get("record_id") or "") for x in ranked[:self.budget_gate.config.default_record_k]]
            aspect_coverage = None
            missing_aspects: list[str] = []
            if self.reranker and ranked:
                try:
                    aspect_coverage, missing_aspects = self.reranker.aspect_coverage(self._retrieval_aspects(), ranked, compact_k=self.budget_gate.config.default_record_k, normalized_tolerance=self.budget_gate.config.aspect_score_tolerance)
                except Exception:
                    aspect_coverage, missing_aspects = None, []
            budget.missing_aspects = missing_aspects
            budget=self.budget_gate.finish(budget,query=query,record_scores=[float(x["record_rerank_score"]) for x in ranked if x.get("record_rerank_score") is not None],compact_record_ids=compact_ids,candidate_record_count=len(ranked),aspect_coverage=aspect_coverage,stagnant_rounds=self._stagnant_rounds)
            selected=ranked[:budget.record_k]
        chosen_by_loop: dict[str,list[dict[str,Any]]]={}
        selected_ids=[]
        for record in selected:
            loop_id=str(record.get("loop_memory_id") or "")
            clean={k:v for k,v in record.items() if k not in {"loop_memory_id","ranking_text"}}
            chosen_by_loop.setdefault(loop_id,[]).append(clean)
            selected_ids.append(str(record.get("record_id") or ""))
        out=[]
        for hit,packet in packets:
            packet["records"]=chosen_by_loop.get(hit.memory_id,[])
            packet["selected_record_count"]=len(packet["records"])
            packet["loop_memory_id"]=hit.memory_id
            packet["routing_score"]=hit.score
            out.append(packet)
        return out,[h.memory_id for h in hits],(budget.to_dict() if budget else {})

    def _retrieval_aspects(self) -> list[str]:
        request = self.retrieval_request
        parts = [request.get("required_evidence"), request.get("target_relation"),
                 request.get("subject_scope"), *(request.get("key_entities") or [])]
        contract = " | ".join(dict.fromkeys(
            " ".join(str(item or "").split()) for item in parts
            if " ".join(str(item or "").split())
        ))
        fallback = str((self.controller.active_contract or {}).get("loop_description") or "")
        text = " ".join((contract or fallback).split())[:300]
        return [text] if len(text) >= 3 else []

    def build_prompt(self, system_message: dict[str, Any], user_message: dict[str, Any]) -> list[dict[str, Any]]:
        """Re-run hierarchical recall on every call and restore parent loop context."""
        query=self._query()
        packets,loop_ids,budget=self._retrieve(query)
        selected_records=[str(r.get("record_id") or "") for p in packets for r in p.get("records") or []]
        memory={
            "memory_role":"retrieved prior evidence context only",
            "active_context_loop":dict(self.controller.active_contract or {}),
            "retrieval_query":query,
            "retrieved_completed_loop_packets":packets,
        }
        system=dict(system_message)
        if packets or self.completed_loops:
            instruction=("The following material is retrieved prior evidence/context only. It does not choose a "
                         "research direction, tool, query, answer, or stopping point. Independently make the same "
                         "research judgments you would make without a memory controller, and use only relevant "
                         "evidence from these completed Loops.\n")
            system["content"]=str(system.get("content") or "")+"\n\n<cross_loop_memory>\n"+instruction+json.dumps(memory,ensure_ascii=False,indent=2)+"\n</cross_loop_memory>"
        messages=[system,dict(user_message),*(dict(event.message) for event in self.current_events)]
        self.traces.append(RuntimeTrace(len(self.traces)+1,query,loop_ids,selected_records,self.latest_decision.split,budget))
        self.latest_decision=LoopBoundaryDecision(False,"same loop")
        return messages

    def finalize(self) -> LoopMemory | None:
        loop = self._archive(None)
        self.current_events = []
        return loop
