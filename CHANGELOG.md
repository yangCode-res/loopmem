# Changes to the public implementation

## Paper-alignment update

This update changes the public reference implementation relative to its initial
release. The original development code, manuscript, trained checkpoints, and
reported experiment results have not been changed or regenerated.

- **Working context (Section 3.2):** include the active objective after the first
  tool interaction, even when the archive and recalled evidence are empty.
- **Evidence provenance (Sections 3.1 and 3.4):** retain each original interaction
  ID and message position, and attach source-record IDs to individual tool
  evidence records. Recalled records retain those links and their parent context.
- **Loop selection (Section 3.4):** apply cross-encoder ranking without overriding
  it with a forced outgoing-loop insertion. Progress handoff still supplies the
  outgoing investigation directly to the planner at a boundary. Return parent
  loop context only when at least one of its records is selected.
- **Adaptive budgets (Appendix D.1):** compute score gaps, coverage, and candidate
  counts before Top-K truncation. Raw source records are stored separately from
  the lexical loop index, so only archived loop descriptions enter routing.
- **Controller SFT (Section 3.5):** sum valid target-token losses within each
  example, then average across examples, matching the displayed objective.
- **Retrospective annotation (Appendix B.2):** include the source pipeline's
  multi-view candidate agreement and adjudication core, while keeping assignment
  inventories fixed.
- Remove unused global-state/State Delta schemas, old retrieval and input
  migration helpers, pre-result classification, legacy planner modes, unused
  diagnostics, and repeated documentation. Model-output validation and tests
  remain in place.

Tests exercise protocol and mathematical behavior with synthetic data. They do
not reproduce the benchmark accuracies or establish performance for checkpoints
trained with this revised implementation.
