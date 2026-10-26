from __future__ import annotations

import json
from typing import Callable, Any

INVENTORY_SYSTEM_PROMPT = 'You define a fixed inventory of semantic workstreams.\nReturn one valid JSON object only. Do not reveal chain-of-thought and do not answer the question.\n\nInput contains the user question and the intent of every recorded research action, but not its later tool result.\nEnumerate the smallest set of independently testable relations actually pursued by those action intents. A\nworkstream has one stable subject scope, target relation, and observable completion condition.\n\nInvalid aggregation example:\n  W001 "Identify film X satisfying title/year, billing property, genre-launching reputation, and studio scandal"\n  completion "All four clues are satisfied"\nThis is invalid because it contains several predicates that can independently succeed or fail. The valid\ninventory uses separate IDs for candidate title/year, billing property, genre-launching reputation, studio\nscandal, and any separately researched original-title attribute. Action intents about each predicate receive\nthat predicate\'s ID. "They support the same final answer" never permits predicate aggregation. If a description\nor completion test needs a list of independently verifiable clauses joined by commas or "and", split it.\n\nDo not collapse independently testable user clues into candidate identification. Candidate localization is one\nworkstream; after a candidate is proposed, actions testing different clue relations use different workstreams\nwhenever each clue could independently be resolved or refuted. For example, a film\'s title/year, whether it\nlaunched a genre, whether its billing has a specified property, and whether its studio had a specified scandal\nare separate relations when the recorded intents research them. They may support one final answer but are not\none context workstream. A dependency chain such as person birthplace, church at that birthplace, and mosaic in\nthat church is three relations when all three are researched.\n\nDifferent wording, spelling, source, URL, tool, failed search, mirror/proxy/API attempt, or repeated verification\nof the same relation stays in one workstream. Formatting, synthesis, citation assembly, and final-answer writing\nare not workstreams. Descriptions and completion tests must name information outcomes, never a query, website,\nURL, tool, or next action.\n\nThis stage does not assign actions. Return exactly workstreams. Workstream IDs are consecutive W001, W002, ...;\nlist no outcomes. Prefer relations pursued by the supplied intents; the program will deterministically discard a\nproposed relation if no assignment candidate ultimately uses it. If input includes\ncandidate_inventories, adjudicate them from scratch: retain atomic independently testable predicates, merge\nsource/query/tool variants, correct every candidate when needed, and return one complete canonical inventory.\nThe later assignment stage is forbidden from changing this inventory.'

ASSIGNMENT_SYSTEM_PROMPT = 'You assign recorded research-action intents to a fixed semantic workstream inventory.\nReturn one valid JSON object only. Do not reveal chain-of-thought and do not answer the question.\n\nThe input gives fixed_workstreams that another stage already defined. Never merge, split, rename, add, remove, or\nreword them. Assign exactly one listed WNNN ID to every action_index in order. Use the relation the candidate\nresearch action is trying to establish, not its website, query wording, tool, access method, or expected result.\nRepeated verification and failed attempts on the same relation keep the same ID; returning to an older relation\nreuses its old ID. It is valid for this concrete trajectory to leave a proposed fixed workstream unused; the\nprogram removes unused proposals after assignment consensus. Return exactly\naction_workstream_ids. If assignment_candidates are supplied, independently resolve their disagreements against\nthe fixed inventory and action intents; do not change the inventory.'


INVENTORY_VIEWS = (
    "QUESTION_OBLIGATIONS_FIRST: enumerate candidate localization, every independently verifiable user clue, "
    "and any requested original name/title as separate relations before checking which action intents pursue them.",
    "DEPENDENCY_GRAPH_FIRST: identify each subject-relation-object dependency actually pursued; merge only "
    "alternate attempts to establish the same edge, never different edges that support one answer.",
    "ANTI_AGGREGATION_REVIEW: actively search for a proposed relation that would require two facts to pass; "
    "split such conjunctions while aggressively merging website/query/tool/access variants.",
)

ASSIGNMENT_VIEWS = (
    "ATOMIC_RELATION_FIRST: classify the explicit relation pursued by each action. Candidate localization is "
    "only for proposing or identifying the entity; an action testing a clue attribute uses that clue's ID.",
    "LOCAL_INTENT_FIRST: classify each candidate action from its own stated intent before considering neighbors. "
    "A shared final answer never makes distinct attribute-verification actions candidate-identification work.",
    "SEQUENCE_AUDIT: review the whole ID sequence for semantic returns and genuine relation transitions while "
    "keeping query/source/tool/access changes inside the same relation.",
)


def validate_inventory(raw: dict[str, Any]) -> list[dict[str, str]]:
    if set(raw) != {"workstreams"} or not isinstance(raw["workstreams"], list) or not raw["workstreams"]:
        raise ValueError("inventory must contain only a nonempty workstreams list")
    result = []
    for i, item in enumerate(raw["workstreams"], 1):
        if set(item) != {"workstream_id", "loop_description", "completion_test"}:
            raise ValueError("invalid workstream fields")
        if item["workstream_id"] != f"W{i:03d}" or not item["loop_description"].strip() or not item["completion_test"].strip():
            raise ValueError("workstreams need consecutive IDs and nonempty definitions")
        result.append(item)
    return result


def validate_assignment(raw: dict[str, Any], action_count: int, workstreams: list[dict[str, str]]) -> list[str]:
    if set(raw) != {"action_workstream_ids"}:
        raise ValueError("assignment must contain only action_workstream_ids")
    ids = raw["action_workstream_ids"]
    allowed = {w["workstream_id"] for w in workstreams}
    if not isinstance(ids, list) or len(ids) != action_count or any(x not in allowed for x in ids):
        raise ValueError("each action must map to one listed workstream")
    return ids


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _majority_assignment(candidates: list[list[str]]) -> tuple[list[str] | None, list[int]]:
    """Return per-action strict majorities, or indices that need adjudication."""
    result: list[str] = []
    disputed: list[int] = []
    for index, votes in enumerate(zip(*candidates)):
        counts = {value: votes.count(value) for value in set(votes)}
        best = max(counts.values())
        winners = [value for value, count in counts.items() if count == best]
        if best <= len(candidates) // 2 or len(winners) != 1:
            disputed.append(index)
            result.append("")
        else:
            result.append(winners[0])
    return (result if not disputed else None), disputed


def _effective_assignment(
    workstreams: list[dict[str, str]], action_ids: list[str]
) -> tuple[list[dict[str, str]], list[str]]:
    """Drop unused proposals and number workstreams by first action use."""
    by_id = {item["workstream_id"]: item for item in workstreams}
    first_use = list(dict.fromkeys(action_ids))
    remap = {old: f"W{index:03d}" for index, old in enumerate(first_use, 1)}
    effective = [{**by_id[old], "workstream_id": remap[old]} for old in first_use]
    return effective, [remap[old] for old in action_ids]


def annotate(
    question: str,
    intents: list[str],
    complete_json: Callable[[str, str], dict[str, Any]],
    *,
    views: int = 3,
    candidate_count: int | None = None,
) -> dict[str, Any]:
    """Run multi-view intent annotation with agreement and adjudication.

    This extracts the core intent-only annotation procedure. It does not include
    production trajectory validation, outcome labeling, or causal-prefix
    reconstruction.
    """
    if not question.strip() or not intents:
        raise ValueError("annotation requires a question and recorded action intents")
    if not 2 <= views <= min(len(INVENTORY_VIEWS), len(ASSIGNMENT_VIEWS)):
        raise ValueError("views must select between two and three distinct perspectives")
    if candidate_count is not None and candidate_count < 2:
        raise ValueError("multi-view annotation requires at least two candidates")
    count = candidate_count if candidate_count is not None else views
    inventory_base = {"question": question, "action_intents": intents}

    inventory_candidates = []
    for index in range(count):
        payload = {
            **inventory_base,
            "candidate_perspective": INVENTORY_VIEWS[index % views],
        }
        candidate = complete_json(INVENTORY_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
        inventory_candidates.append(validate_inventory(candidate))
    if all(_canonical_json(c) == _canonical_json(inventory_candidates[0])
           for c in inventory_candidates[1:]):
        inventory = inventory_candidates[0]
    else:
        payload = {
            **inventory_base,
            "candidate_inventories": [{"workstreams": candidate} for candidate in inventory_candidates],
            "adjudication_instruction": (
                "Return the best corrected canonical inventory. Independently re-check every user clue and "
                "dependency against the recorded action intents. Reject broad multi-predicate aggregation and "
                "source/query/tool-level oversegmentation; do not merely vote for the most common candidate."
            ),
        }
        inventory = validate_inventory(complete_json(
            INVENTORY_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False)
        ))

    assignment_base = {
        "question": question,
        "fixed_workstreams": inventory,
        "action_intents": intents,
    }
    assignment_candidates = []
    for index in range(count):
        payload = {
            **assignment_base,
            "assignment_perspective": ASSIGNMENT_VIEWS[index % views],
        }
        candidate = complete_json(ASSIGNMENT_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
        assignment_candidates.append(validate_assignment(candidate, len(intents), inventory))
    if all(c == assignment_candidates[0] for c in assignment_candidates[1:]):
        assignment = assignment_candidates[0]
    else:
        assignment, disputed = _majority_assignment(assignment_candidates)
        if assignment is None:
            payload = {
                **assignment_base,
                "assignment_candidates": [
                    {"action_workstream_ids": c} for c in assignment_candidates
                ],
                "disputed_action_indices": disputed,
            }
            assignment = validate_assignment(
                complete_json(ASSIGNMENT_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False)),
                len(intents), inventory,
            )

    effective_inventory, effective_assignment = _effective_assignment(inventory, assignment)
    return {"workstreams": effective_inventory, "action_workstream_ids": effective_assignment}
