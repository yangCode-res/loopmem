"""Lexical candidate retrieval over archived loop descriptions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any

from .schema import LoopMemory, RawMemory


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


@dataclass
class MemoryDocument:
    memory_id: str
    memory_type: str
    text: str
    metadata: dict[str, Any]
    index_text: str | None = None


@dataclass
class MemoryHit:
    memory_id: str
    memory_type: str
    score: float
    text: str
    metadata: dict[str, Any]


class LexicalMemoryIndex:
    """BM25 loop index with a separate immutable source-record store."""

    def __init__(self) -> None:
        self.documents: list[MemoryDocument] = []
        self._tokens: dict[str, list[str]] = {}
        self.raw_records: dict[str, RawMemory] = {}

    def add_raw(self, memory: RawMemory) -> None:
        memory.validate()
        self.raw_records[memory.raw_id] = memory

    def add_loop(self, memory: LoopMemory) -> None:
        # Put durable conclusions and tool observations before assistant actions.
        # Prompt packing truncates long memories, so evidence must appear first.
        text = "\n".join(
            [
                f"SUBGOAL: {memory.subgoal}",
                f"CONCLUSION: {memory.conclusion or ''}",
                f"TOOL_EVIDENCE: {json.dumps(memory.observations, ensure_ascii=False)}",
                f"CONTEXT: {memory.context or ''}",
                f"ACTIONS: {json.dumps(memory.actions, ensure_ascii=False)}",
            ]
        )
        loop_summary = "\n".join(
            part for part in (
                f"SUBGOAL: {memory.subgoal}",
                f"CONCLUSION: {memory.conclusion or ''}",
            )
            if part.strip()
        )
        records: list[dict[str, Any]] = []
        if memory.conclusion:
            records.append(
                {
                    "record_id": f"{memory.loop_id}_record_001",
                    "record_type": "DURABLE_FINDING",
                    "content": str(memory.conclusion),
                    "evidence_ids": list(memory.evidence_ids),
                }
            )
        for observation in memory.observations:
            records.append(
                {
                    "record_id": f"{memory.loop_id}_record_{len(records) + 1:03d}",
                    "record_type": "TOOL_EVIDENCE",
                    "content": json.dumps(observation, ensure_ascii=False),
                    "evidence_ids": ([memory.provenance[observation["sequence"]]]
                                     if observation.get("sequence") in memory.provenance else []),
                }
            )
        loop_packet = {
            "loop_memory_id": memory.loop_id,
            "semantic_signature": dict(memory.semantic_signature),
            "subgoal": memory.subgoal,
            "completion_test": str(memory.semantic_signature.get("success_condition") or ""),
            "outcome": memory.status.value,
            "summary": loop_summary,
            "records": records,
            "evidence_ids": list(memory.evidence_ids),
            "provenance": {record["record_id"]: list(record["evidence_ids"]) for record in records},
        }
        self._add(
            MemoryDocument(
                memory_id=memory.loop_id,
                memory_type="loop",
                text=text,
                metadata={
                    "task_id": memory.task_id,
                    "status": memory.status.value,
                    "subgoal": memory.subgoal,
                    "semantic_signature": dict(memory.semantic_signature),
                    "loop_summary": loop_summary,
                    "record_count": len(records),
                    "evidence_ids": list(memory.evidence_ids),
                    "loop_packet": loop_packet,
                },
                index_text=loop_summary,
            )
        )

    def _add(self, document: MemoryDocument) -> None:
        self.documents.append(document)
        self._tokens[document.memory_id] = tokenize(document.index_text or document.text)


    def search(self, query: str, top_k: int = 5, task_id: str | None = None) -> list[MemoryHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        n = len(self.documents)
        lengths = [len(self._tokens[d.memory_id]) for d in self.documents]
        avgdl = sum(lengths) / max(n, 1)
        df = {term: sum(term in self._tokens[d.memory_id] for d in self.documents) for term in set(query_tokens)}
        k1, b = 1.2, 0.75
        hits: list[MemoryHit] = []
        for document, dl in zip(self.documents, lengths):
            if task_id and document.metadata.get("task_id") != task_id:
                continue
            terms = self._tokens[document.memory_id]
            score = 0.0
            for term in query_tokens:
                tf = terms.count(term)
                if not tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1))))
            if score > 0:
                hits.append(MemoryHit(document.memory_id, document.memory_type, score, document.text, document.metadata))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


__all__ = [
    "LexicalMemoryIndex",
    "MemoryDocument",
    "MemoryHit",
    "tokenize",
]
