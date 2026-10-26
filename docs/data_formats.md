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

## Reviewed controller RL samples

`training/build_controller_rl.py` accepts one reviewed decision per JSONL row:

```json
{
  "sample_id": "synthetic-boundary-001",
  "qid": "0",
  "messages": [
    {"role": "system", "content": "The controller system prompt."},
    {"role": "user", "content": "The serialized decision-time controller input."}
  ],
  "target_action": "SWITCH_LOOP",
  "terminal_label": "CORRECT"
}
```

Use the actual prompt and causal input format from `src/loopmem/prompts.py` and
`controller_context.py`. `sample_id` must be unique and `qid` identifies the source
question. `target_action` is `CONTINUE` or `SWITCH_LOOP`; the optional
`terminal_label` is `CORRECT`, `INCORRECT`, or `UNKNOWN` (default). It describes
the recorded trajectory's final answer, not a controller prediction.

The builder computes both action rewards, splits by question, and writes
`train.jsonl`, `validation.jsonl`, and a manifest of file hashes. Training weights
balance switch labels and question contributions, then normalize to a global
mean of one. Both partitions must contain both actions. The trainer checks file
hashes and question separation. Input provenance and exclusion of benchmark
evaluation questions must be verified before preparing these samples.

## Reranker events

Each JSONL event has the following structure:

```json
{
  "schema_version": "loopmem-reranker-listwise-event-v1",
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
