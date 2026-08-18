# Phase 0 core matrix v2: significant-use state allocation

Seed 1337, six tasks, 2,000 steps/task, batch 32, and 1,536,000 training tokens
per group. A/B/C/E results reproduce v1 exactly; only FIP state allocation was
changed. This remains a single-seed diagnostic, not a final Phase 0 result.

## Direct v1 -> v2 result

| group | v1 retention forgetting | v2 retention forgetting |
|---|---:|---:|
| A dense FT | 0.4676 | 0.4676 |
| B online EWC | 0.4733 | 0.4733 |
| C 5% replay | 0.2527 | 0.2527 |
| E sparse FT | 0.4555 | 0.4555 |
| F FIP | 0.5086 | **0.4915** |

The fix improves FIP by 0.0171 absolute, but FIP remains worse than sparse-only
E by 0.0360 and worse than dense A by 0.0239. It therefore still fails the
frozen Phase 0 forgetting gate and does not justify a five-seed run.

## State-allocation result

The intended mechanism correction worked:

- final `shared` fraction is roughly 20--23% per layer, down from 77--86%;
- final `stable` fraction is roughly 48--52% per layer, up from 1--4 slots;
- per-task significant-feature fraction starts at 24.0% and falls to 6--18%
  as protected capacity accumulates;
- FIP still learns the final task (`unrelated = 1.0`).

Thus the remaining failure cannot be attributed only to the old
"activated once" rule.

## Next falsifiable diagnosis

FIP v0 protects only FFN feature slices. Embeddings, attention projections,
normalization weights, and the tied output head remain fully trainable. The
synthetic mapping tasks can therefore be overwritten through those unprotected
paths even when FFN stable slots are exactly frozen.

The next experiment should be a diagnostic ablation, not five seeds:

1. measure task-to-task parameter drift separately for embedding/head,
   attention, norms, stable FFN, shared FFN, and plastic/free FFN;
2. add a diagnostic FIP variant that freezes non-FFN parameters after task 1;
3. run seed 1337 under the same token/step budget;
4. if retention improves materially, the FFN-only isolation boundary is the
   limiting design assumption; if it does not, the feature allocation itself
   remains ineffective.

This ablation must be reported as a diagnostic group and must not silently
replace the frozen F group.
