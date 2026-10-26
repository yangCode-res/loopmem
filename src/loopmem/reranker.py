"""Local bge-reranker-v2-m3 scorer for Loop Memory packets."""

from __future__ import annotations

from typing import Any, Sequence


class BGEReranker:
    """Cross-encoder reranker with lazy local model loading.

    The model scores compact query/Loop-summary pairs only; raw evidence is
    never sent to an external service.  A CPU fallback keeps the pipeline
    runnable when the Controller/search GPUs are occupied.
    """

    def __init__(self, model_path: str, *, device: str = "cpu", max_length: int = 2048) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.max_length = max(128, int(max_length))
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, use_fast=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, local_files_only=True, low_cpu_mem_usage=True
        ).eval()
        self._model.to(self.device)
        self._torch = torch

    @staticmethod
    def _loop_text(hit: Any) -> str:
        metadata = getattr(hit, "metadata", {}) or {}
        packet = metadata.get("loop_packet") if isinstance(metadata, dict) else None
        if isinstance(packet, dict):
            signature = packet.get("semantic_signature") or {}
            return "\n".join(
                [
                    str(packet.get("subgoal") or ""),
                    str(packet.get("summary") or ""),
                    str(signature.get("subject_scope") or ""),
                    str(signature.get("target_relation") or ""),
                    str(signature.get("success_condition") or ""),
                ]
            )
        return str(metadata.get("loop_summary") or getattr(hit, "text", ""))

    def _score_documents(self, query: str, documents: Sequence[str]) -> list[float]:
        """Score query/document pairs with the local cross encoder."""

        if not documents:
            return []
        self._load()
        pairs = [[str(query), str(document)] for document in documents]
        encoded = self._tokenizer(
            pairs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            return self._model(**encoded).logits.reshape(-1).detach().cpu().tolist()

    def rerank(self, query: str, hits: Sequence[Any], *, top_k: int) -> list[Any]:
        if not hits:
            return []
        logits = self._score_documents(query, [self._loop_text(hit) for hit in hits])
        ranked = sorted(zip(logits, hits), key=lambda item: item[0], reverse=True)
        for score, hit in ranked:
            if hasattr(hit, "metadata") and isinstance(hit.metadata, dict):
                hit.metadata["rerank_score"] = float(score)
        return [hit for _, hit in ranked[: max(1, int(top_k))]]

    def rerank_records(
        self,
        query: str,
        records: Sequence[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rank evidence records across all routed Loops."""

        if not records:
            return []
        documents = [
            str(record.get("ranking_text") or record.get("content") or "")
            for record in records
        ]
        logits = self._score_documents(query, documents)
        ranked = sorted(zip(logits, records), key=lambda item: item[0], reverse=True)
        selected: list[dict[str, Any]] = []
        for rank, (score, record) in enumerate(ranked[: max(1, int(top_k))], start=1):
            item = dict(record)
            item["record_rerank_score"] = float(score)
            item["record_rank"] = rank
            selected.append(item)
        return selected

    def aspect_coverage(
        self,
        aspects: Sequence[str],
        ranked_records: Sequence[dict[str, Any]],
        *,
        compact_k: int,
        normalized_tolerance: float = 0.25,
    ) -> tuple[float | None, list[str]]:
        """Measure whether compact Top-K retains each aspect's best record.

        Coverage is relative within the already bounded record candidate list,
        so it does not treat uncalibrated BGE logits as probabilities.
        """

        clean_aspects = list(
            dict.fromkeys(str(aspect).strip() for aspect in aspects if str(aspect).strip())
        )[:8]
        if not clean_aspects or not ranked_records:
            return None, []
        documents = [
            str(record.get("ranking_text") or record.get("content") or "")
            for record in ranked_records
        ]
        compact_limit = min(max(1, int(compact_k)), len(documents))
        covered = 0
        missing: list[str] = []
        for aspect in clean_aspects:
            scores = self._score_documents(aspect, documents)
            spread = max(self._score_std(scores), 1e-6)
            best_score = max(scores)
            best_compact_score = max(scores[:compact_limit])
            if (best_score - best_compact_score) / spread <= max(
                0.0, float(normalized_tolerance)
            ):
                covered += 1
            else:
                missing.append(aspect)
        return covered / len(clean_aspects), missing

    @staticmethod
    def _score_std(values: Sequence[float]) -> float:
        """Dependency-free population standard deviation for score diagnostics."""

        if not values:
            return 0.0
        mean = sum(float(value) for value in values) / len(values)
        variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
        return variance ** 0.5


__all__ = ["BGEReranker"]
