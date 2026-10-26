# Training input formats

All examples in this document are synthetic schema examples. They are not
benchmark questions, measured results, or the training corpus used in the paper.

## Controller SFT

One JSON object per line, with `system`, `user`, and `assistant` messages:

```json
{"messages":[{"role":"system","content":"Return a structured loop-boundary decision."},{"role":"user","content":"{\"active_loop\":{\"loop_key\":\"L001\"},\"candidate_action\":{\"tool\":\"search\",\"arguments\":\"verify publication date\"}}"},{"role":"assistant","content":"{\"decision\":\"SWITCH_LOOP\",\"closed_loop_key\":\"L001\"}"}]}
```

Use the actual controller system prompt in `src/loopmem/prompts.py` and causal
inputs produced by `controller_context.py` when constructing training rows.
Split source questions before creating train and validation files. The final
assistant message is the target; its tokens alone receive labels in SFT.

## Reviewed controller RL bundle

`training/build_controller_rl.py` consumes a directory containing:

| File | Required content |
| --- | --- |
| `events.jsonl` | Causal events with `event_id`, `qid`, `round`, `messages` (`system`, `user`), `runtime_action.decision`, and `trace_sha256`. |
| `review-selection.jsonl` | Exact event rows selected for boundary review. |
| `manual-boundaries.json` | `selection_sha256` and `groups`; each group has `ids` (event IDs), `label` (`CONTINUE`, `SWITCH_LOOP`, or `EXCLUDE`), and `reason`. |
| `outcomes-private.jsonl` | One row per question with `qid` and the matching `trace_sha256`. This is an input filename, not a supplied dataset. |
| `manual-outcomes.json` | `groups`, each with `qids`, `label` (`CORRECT`, `INCORRECT`, or `UNKNOWN`), and `reason`. |
| `source-audit.json` | `training_safe`, `source_verified`, `events_sha256`, and `outcomes_sha256`. |

The builder checks hashes, exact event identity, outcome-review coverage, and
question-disjoint splits. It writes `train.jsonl`, `validation.jsonl`, and a
`manifest.json` with file hashes. Each partition must contain both actions.
Do not mark a source as training-safe without independently checking its origin
and separation from held-out questions.

The materialized rows contain `sample_id`, `qid`, `messages`, `target_action`,
`reward` (in `[CONTINUE, SWITCH_LOOP]` order), and `weight`. The trainer also
records question IDs and hashes for local verification; do not publish those
generated files as part of this anonymous code release.

## Reranker events

Each JSONL event has the following structure:

```json
{
  "schema_version": "cl-gism-reranker-listwise-event-v1",
  "event_id": "synthetic-loop-001",
  "qid": "0",
  "trajectory_id": "synthetic-trajectory-001",
  "level": "loop",
  "query": "Alpha report publication date",
  "event_weight": 1.0,
  "candidates": [
    {"text": "Verify Alpha report publication metadata.", "label": 2},
    {"text": "Investigate an unrelated transport route.", "label": 0}
  ]
}
```

`level` is `loop` or `record`. Labels `0`, `1`, and `2` map to gains `-1`, `0.25`,
and `1`. Each event must have at least two relevance levels. Event IDs are unique;
at least two source questions are required for a question-disjoint split.

The dataset manifest must include:

```json
{
  "safety": {"openresearcher_train_only": true, "test_qid_overlap": 0},
  "reranker": {"file": {"sha256": "SHA256_OF_THE_VERIFIED_EVENT_FILE"}}
}
```

The hash string above describes a field to compute for your actual file. The
boolean assertions describe checks to perform on the real training data; they
must not be copied as evidence that synthetic or unverified data passed an audit.

## Retrospective annotation

Call `annotate(question, action_intents, complete_json)` in
`training/retrospective_annotation.py`. The callback returns parsed JSON. The
inventory stage and assignment stage each use multiple candidate perspectives,
with agreement checks and adjudication of unresolved disagreements. Assignment
receives an immutable inventory. Unused definitions are removed, then IDs are
renumbered by first action use. Defaults are three views/candidates; set the
`views` and `candidate_count` keyword options to change those counts.

Only decision-time prefixes should later be exposed to the controller. Full
recorded trajectories inform offline labels, not online boundary decisions.
