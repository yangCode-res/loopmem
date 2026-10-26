from __future__ import annotations

import json
from typing import Callable, Any

INVENTORY_SYSTEM_PROMPT = 'You define a fixed inventory of semantic workstreams.\nReturn one valid JSON object only. Do not reveal chain-of-thought and do not answer the question.\n\nInput contains the user question and the intent of every recorded research action, but not its later tool result.\nEnumerate the smallest set of independently testable relations actually pursued by those action intents. A\nworkstream has one stable subject scope, target relation, and observable completion condition.\n\nInvalid aggregation example:\n  W001 "Identify film X satisfying title/year, billing property, genre-launching reputation, and studio scandal"\n  completion "All four clues are satisfied"\nThis is invalid because it contains several predicates that can independently succeed or fail. The valid\ninventory uses separate IDs for candidate title/year, billing property, genre-launching reputation, studio\nscandal, and any separately researched original-title attribute. Action intents about each predicate receive\nthat predicate\'s ID. "They support the same final answer" never permits predicate aggregation. If a description\nor completion test needs a list of independently verifiable clauses joined by commas or "and", split it.\n\nDo not collapse independently testable user clues into candidate identification. Candidate localization is one\nworkstream; after a candidate is proposed, actions testing different clue relations use different workstreams\nwhenever each clue could independently be resolved or refuted. For example, a film\'s title/year, whether it\nlaunched a genre, whether its billing has a specified property, and whether its studio had a specified scandal\nare separate relations when the recorded intents research them. They may support one final answer but are not\none context workstream. A dependency chain such as person birthplace, church at that birthplace, and mosaic in\nthat church is three relations when all three are researched.\n\nDifferent wording, spelling, source, URL, tool, failed search, mirror/proxy/API attempt, or repeated verification\nof the same relation stays in one workstream. Formatting, synthesis, citation assembly, and final-answer writing\nare not workstreams. Descriptions and completion tests must name information outcomes, never a query, website,\nURL, tool, or next action.\n\nThis stage does not assign actions. Return exactly workstreams. Workstream IDs are consecutive W001, W002, ...;\nlist no outcomes. Prefer relations pursued by the supplied intents; the program will deterministically discard a\nproposed relation if no assignment candidate ultimately uses it. If input includes\ncandidate_inventories, adjudicate them from scratch: retain atomic independently testable predicates, merge\nsource/query/tool variants, correct every candidate when needed, and return one complete canonical inventory.\nThe later assignment stage is forbidden from changing this inventory.'

ASSIGNMENT_SYSTEM_PROMPT = 'You assign recorded research-action intents to a fixed semantic workstream inventory.\nReturn one valid JSON object only. Do not reveal chain-of-thought and do not answer the question.\n\nThe input gives fixed_workstreams that another stage already defined. Never merge, split, rename, add, remove, or\nreword them. Assign exactly one listed WNNN ID to every action_index in order. Use the relation the candidate\nresearch action is trying to establish, not its website, query wording, tool, access method, or expected result.\nRepeated verification and failed attempts on the same relation keep the same ID; returning to an older relation\nreuses its old ID. It is valid for this concrete trajectory to leave a proposed fixed workstream unused; the\nprogram removes unused proposals after assignment consensus. Return exactly\naction_workstream_ids. If assignment_candidates are supplied, independently resolve their disagreements against\nthe fixed inventory and action intents; do not change the inventory.'


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


def annotate(question: str, intents: list[str], complete_json: Callable[[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Run separate inventory and assignment passes; inventory is immutable in pass two."""
    if not question.strip() or not intents:
        raise ValueError("annotation requires a question and recorded action intents")
    inventory = validate_inventory(complete_json(INVENTORY_SYSTEM_PROMPT, json.dumps({"question": question, "action_intents": intents}, ensure_ascii=False)))
    assignment = validate_assignment(complete_json(ASSIGNMENT_SYSTEM_PROMPT, json.dumps({"question": question, "fixed_workstreams": inventory, "action_intents": intents}, ensure_ascii=False)), len(intents), inventory)
    used = set(assignment)
    inventory = [w for w in inventory if w["workstream_id"] in used]
    remap = {w["workstream_id"]: f"W{i:03d}" for i, w in enumerate(inventory, 1)}
    return {"workstreams": [{**w, "workstream_id": remap[w["workstream_id"]]} for w in inventory],
            "action_workstream_ids": [remap[x] for x in assignment]}
