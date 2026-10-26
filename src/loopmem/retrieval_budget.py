"""Deterministic adaptive budgets for hierarchical Loop/record retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Sequence

from .retrieval import tokenize


@dataclass(frozen=True)
class RetrievalBudgetConfig:
    """Compact defaults and one-round expansion thresholds."""

    default_loop_k: int = 2
    expanded_loop_k: int = 3
    default_record_k: int = 5
    expanded_record_k: int = 8
    loop_margin_threshold: float = 0.25
    record_margin_threshold: float = 0.15
    aspect_coverage_min: float = 0.60
    hard_aspect_coverage_min: float = 0.35
    aspect_score_tolerance: float = 0.25
    stagnation_threshold: int = 2
    hard_stagnation_threshold: int = 3
    repeated_query_threshold: float = 0.80
    repeated_record_overlap_threshold: float = 0.80
    cooldown_rounds: int = 2

    def __post_init__(self) -> None:
        if self.default_loop_k < 1 or self.expanded_loop_k < self.default_loop_k:
            raise ValueError("Loop retrieval budgets must be positive and ordered")
        if self.default_record_k < 1 or self.expanded_record_k < self.default_record_k:
            raise ValueError("Record retrieval budgets must be positive and ordered")
        for name in (
            "loop_margin_threshold",
            "record_margin_threshold",
            "aspect_coverage_min",
            "hard_aspect_coverage_min",
            "aspect_score_tolerance",
            "repeated_query_threshold",
            "repeated_record_overlap_threshold",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "aspect_coverage_min",
            "hard_aspect_coverage_min",
            "repeated_query_threshold",
            "repeated_record_overlap_threshold",
        ):
            if float(getattr(self, name)) > 1:
                raise ValueError(f"{name} must not exceed 1")


@dataclass
class RetrievalBudgetDecision:
    """One auditable budget decision for a Researcher prompt."""

    loop_k: int
    record_k: int
    candidate_loop_count: int
    default_loop_k: int = 2
    default_record_k: int = 5
    candidate_record_count: int = 0
    reasons: list[str] = field(default_factory=list)
    loop_score_margin: float | None = None
    record_score_margin: float | None = None
    aspect_coverage: float | None = None
    missing_aspects: list[str] = field(default_factory=list)
    repeated_query_score: float = 0.0
    repeated_record_overlap: float = 0.0
    cooldown_before: int = 0
    cooldown_after: int = 0

    @property
    def mode(self) -> str:
        loop_expanded = self.loop_k > self.default_loop_k
        record_expanded = self.record_k > self.default_record_k
        if loop_expanded and record_expanded:
            return "expanded"
        if loop_expanded:
            return "loop_expanded"
        if record_expanded:
            return "record_expanded"
        return "compact"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "loop_k": self.loop_k,
            "record_k": self.record_k,
            "candidate_loop_count": self.candidate_loop_count,
            "candidate_record_count": self.candidate_record_count,
            "expansion_reasons": list(dict.fromkeys(self.reasons)),
            "loop_score_margin": self.loop_score_margin,
            "record_score_margin": self.record_score_margin,
            "aspect_coverage": self.aspect_coverage,
            "missing_aspects": list(self.missing_aspects),
            "repeated_query_score": self.repeated_query_score,
            "repeated_record_overlap": self.repeated_record_overlap,
            "cooldown_before": self.cooldown_before,
            "cooldown_after": self.cooldown_after,
        }


def normalized_boundary_margin(scores: Sequence[float], boundary: int) -> float | None:
    """Return the adjacent rank gap divided by this query's score spread."""

    values = sorted((float(score) for score in scores if math.isfinite(float(score))), reverse=True)
    if boundary < 1 or len(values) <= boundary:
        return None
    spread = max(statistics.pstdev(values), 1e-6)
    return (values[boundary - 1] - values[boundary]) / spread


def _token_set(text: str) -> set[str]:
    return {
        term
        for term in tokenize(str(text))
        if len(term) >= 3 and term not in {"the", "and", "for", "with", "from", "that", "this"}
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class RetrievalBudgetGate:
    """Choose compact or expanded K values without another model call.

    The gate is stateful only for repeat/cooldown detection. A decision never
    remains expanded automatically: every Researcher prompt is reconsidered.
    """

    def __init__(self, config: RetrievalBudgetConfig | None = None) -> None:
        self.config = config or RetrievalBudgetConfig()
        self._previous_query_tokens: set[str] = set()
        self._previous_compact_record_ids: set[str] = set()
        self._cooldown_remaining = 0

    @staticmethod
    def _bounded_k(requested: int, candidate_count: int) -> int:
        if candidate_count <= 0:
            return 0
        return min(max(1, int(requested)), int(candidate_count))

    def begin(
        self,
        *,
        query: str,
        loop_scores: Sequence[float],
        candidate_loop_count: int,
        stagnant_rounds: int = 0,
    ) -> RetrievalBudgetDecision:
        config = self.config
        query_tokens = _token_set(query)
        repeated_query_score = _jaccard(query_tokens, self._previous_query_tokens)
        margin = normalized_boundary_margin(loop_scores, config.default_loop_k)
        hard_stagnation = int(stagnant_rounds) >= config.hard_stagnation_threshold
        expansion_allowed = self._cooldown_remaining == 0 or hard_stagnation
        loop_k = self._bounded_k(config.default_loop_k, candidate_loop_count)
        reasons: list[str] = []
        if expansion_allowed and candidate_loop_count > config.default_loop_k:
            if margin is not None and margin < config.loop_margin_threshold:
                loop_k = self._bounded_k(config.expanded_loop_k, candidate_loop_count)
                reasons.append("loop_score_ambiguous")
            if int(stagnant_rounds) >= config.stagnation_threshold:
                loop_k = self._bounded_k(config.expanded_loop_k, candidate_loop_count)
                reasons.append("stagnation")
        return RetrievalBudgetDecision(
            loop_k=loop_k,
            record_k=0,
            candidate_loop_count=int(candidate_loop_count),
            default_loop_k=config.default_loop_k,
            default_record_k=config.default_record_k,
            reasons=reasons,
            loop_score_margin=margin,
            repeated_query_score=repeated_query_score,
            cooldown_before=self._cooldown_remaining,
        )

    def finish(
        self,
        decision: RetrievalBudgetDecision,
        *,
        query: str,
        record_scores: Sequence[float],
        compact_record_ids: Sequence[str],
        candidate_record_count: int,
        aspect_coverage: float | None = None,
        stagnant_rounds: int = 0,
    ) -> RetrievalBudgetDecision:
        config = self.config
        decision.candidate_record_count = int(candidate_record_count)
        decision.record_score_margin = normalized_boundary_margin(
            record_scores, config.default_record_k
        )
        decision.aspect_coverage = (
            None if aspect_coverage is None else max(0.0, min(1.0, float(aspect_coverage)))
        )
        current_compact_ids = {str(item) for item in compact_record_ids if str(item)}
        decision.repeated_record_overlap = _jaccard(
            current_compact_ids, self._previous_compact_record_ids
        )
        decision.record_k = self._bounded_k(config.default_record_k, candidate_record_count)

        hard_expand = (
            int(stagnant_rounds) >= config.hard_stagnation_threshold
            or (
                decision.aspect_coverage is not None
                and decision.aspect_coverage < config.hard_aspect_coverage_min
            )
        )
        expansion_allowed = self._cooldown_remaining == 0 or hard_expand
        if expansion_allowed and candidate_record_count > config.default_record_k:
            if (
                decision.record_score_margin is not None
                and decision.record_score_margin < config.record_margin_threshold
            ):
                decision.record_k = self._bounded_k(
                    config.expanded_record_k, candidate_record_count
                )
                decision.reasons.append("record_score_ambiguous")
            if (
                decision.aspect_coverage is not None
                and decision.aspect_coverage < config.aspect_coverage_min
            ):
                decision.record_k = self._bounded_k(
                    config.expanded_record_k, candidate_record_count
                )
                decision.reasons.append("aspect_coverage_low")
            if int(stagnant_rounds) >= config.stagnation_threshold:
                decision.record_k = self._bounded_k(
                    config.expanded_record_k, candidate_record_count
                )
                decision.reasons.append("stagnation")
            if (
                decision.repeated_query_score >= config.repeated_query_threshold
                and decision.repeated_record_overlap
                >= config.repeated_record_overlap_threshold
            ):
                decision.record_k = self._bounded_k(
                    config.expanded_record_k, candidate_record_count
                )
                decision.reasons.append("repeated_query_same_records")

        expanded = (
            decision.loop_k
            > self._bounded_k(config.default_loop_k, decision.candidate_loop_count)
            or decision.record_k
            > self._bounded_k(config.default_record_k, decision.candidate_record_count)
        )
        if expanded:
            self._cooldown_remaining = max(0, int(config.cooldown_rounds))
        else:
            self._cooldown_remaining = max(0, self._cooldown_remaining - 1)
        decision.cooldown_after = self._cooldown_remaining
        self._previous_query_tokens = _token_set(query)
        self._previous_compact_record_ids = current_compact_ids
        return decision


__all__ = [
    "RetrievalBudgetConfig",
    "RetrievalBudgetDecision",
    "RetrievalBudgetGate",
    "normalized_boundary_margin",
]
