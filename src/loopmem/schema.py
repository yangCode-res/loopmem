"""Task, loop, and original-evidence records used by the runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, ClassVar, Mapping
from uuid import uuid4


class SchemaValidationError(ValueError):
    """Raised when a schema object violates an invariant."""


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SourceType(_StrEnum):
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    PAPER = "paper"
    WEB = "web"
    EXPERIMENT = "experiment"
    SYSTEM = "system"


class MemoryStatus(_StrEnum):
    ACTIVE = "active"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    RESOLVED = "resolved"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    """Create a stable, human-readable identifier for a schema object."""

    return f"{prefix}_{uuid4().hex}"


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {k: _enum_value(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {k: _enum_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_value(v) for v in value]
    return value


class SchemaModel:
    """Small common API shared by all schema records."""

    _required_prefix: ClassVar[str | None] = None

    def validate(self) -> None:
        """Validate invariants; subclasses extend this method."""

        if self._required_prefix:
            object_id = getattr(self, self._id_field, "")
            if not isinstance(object_id, str) or not object_id.startswith(self._required_prefix + "_"):
                raise SchemaValidationError(
                    f"{self.__class__.__name__}.{self._id_field} must start with "
                    f"'{self._required_prefix}_'"
                )

    @property
    def _id_field(self) -> str:
        return "id"

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _enum_value(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class TaskAnchor(SchemaModel):
    """Stable task intent and success criteria.

    Anchor updates are represented as new versions by the caller; ordinary
    Turn processing must not mutate an existing anchor in place.
    """

    task_id: str
    original_goal: str
    success_criteria: list[str] = field(default_factory=list)
    immutable_constraints: list[str] = field(default_factory=list)
    domain: str | None = None
    version: int = 1
    created_at: str = field(default_factory=utc_now)
    evidence_ids: list[str] = field(default_factory=list)
    supersedes_anchor_id: str | None = None
    anchor_id: str = field(default_factory=lambda: new_id("anchor"))

    _required_prefix: ClassVar[str] = "anchor"

    @property
    def _id_field(self) -> str:
        return "anchor_id"

    def validate(self) -> None:
        super().validate()
        if not self.task_id:
            raise SchemaValidationError("TaskAnchor.task_id cannot be empty")
        if not self.original_goal.strip():
            raise SchemaValidationError("TaskAnchor.original_goal cannot be empty")
        if self.version < 1:
            raise SchemaValidationError("TaskAnchor.version must be >= 1")
        if any(not item.startswith("raw_") for item in self.evidence_ids):
            raise SchemaValidationError("TaskAnchor.evidence_ids must contain raw_* IDs")
        if self.supersedes_anchor_id and not self.supersedes_anchor_id.startswith("anchor_"):
            raise SchemaValidationError("TaskAnchor.supersedes_anchor_id must reference an anchor_* object")


@dataclass
class LoopMemory(SchemaModel):
    """A completed local task unit spanning one or more Turns/actions."""

    task_id: str
    subgoal: str
    context: str | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    conclusion: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    semantic_signature: dict[str, str] = field(default_factory=dict)
    provenance: dict[int, str] = field(default_factory=dict)
    loop_id: str = field(default_factory=lambda: new_id("loop"))

    _required_prefix: ClassVar[str] = "loop"

    @property
    def _id_field(self) -> str:
        return "loop_id"

    def validate(self) -> None:
        super().validate()
        if not self.task_id:
            raise SchemaValidationError("LoopMemory.task_id cannot be empty")
        if not self.subgoal.strip():
            raise SchemaValidationError("LoopMemory.subgoal cannot be empty")
        if self.status in (MemoryStatus.RESOLVED, MemoryStatus.SUPERSEDED) and not self.ended_at:
            raise SchemaValidationError("Completed LoopMemory must have ended_at")
        if any(not item.startswith("raw_") for item in self.evidence_ids):
            raise SchemaValidationError("LoopMemory.evidence_ids must contain raw_* IDs")
        if any(not isinstance(sequence, int) or not raw_id.startswith("raw_")
               for sequence, raw_id in self.provenance.items()):
            raise SchemaValidationError("LoopMemory.provenance maps interaction-message positions to raw_* IDs")
        allowed_signature_fields = {"subject_scope", "target_relation", "success_condition"}
        if any(key not in allowed_signature_fields for key in self.semantic_signature):
            raise SchemaValidationError(
                "LoopMemory.semantic_signature contains an unsupported field"
            )


@dataclass(frozen=True)
class RawMemory(SchemaModel):
    """Source event or document retained as ground-truth evidence.

    Attribute replacement is blocked after creation.  Callers should also
    treat nested ``content`` and ``metadata`` values as immutable payloads.
    """

    task_id: str
    source_type: SourceType
    content: Any
    content_type: str = "text/plain"
    occurred_at: str = field(default_factory=utc_now)
    recorded_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_loop_id: str | None = None
    raw_id: str = field(default_factory=lambda: new_id("raw"))

    _required_prefix: ClassVar[str] = "raw"

    @property
    def _id_field(self) -> str:
        return "raw_id"

    def validate(self) -> None:
        super().validate()
        if not self.task_id:
            raise SchemaValidationError("RawMemory.task_id cannot be empty")
        if not self.content_type.strip():
            raise SchemaValidationError("RawMemory.content_type cannot be empty")
        if self.parent_loop_id and not self.parent_loop_id.startswith("loop_"):
            raise SchemaValidationError("RawMemory.parent_loop_id must reference a loop_* object")


__all__ = ["LoopMemory", "MemoryStatus", "RawMemory", "SchemaValidationError", "SourceType", "TaskAnchor", "new_id", "utc_now"]
