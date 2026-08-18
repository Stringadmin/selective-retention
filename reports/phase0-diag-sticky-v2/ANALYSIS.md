# Group I sticky protection — corrected single-seed analysis

Date: 2026-07-31  
Device: RTX 5070 / CUDA, PyTorch 2.11.0+cu128  
Seed: 1337  
Budget: 6 tasks × 2,000 steps × batch 32 (1,536,000 current-task tokens per group)

## Implementation correction

The first group-I run was invalid as a sticky comparison. Its capacity logic
reselected the current EMA top-k rows whenever the cumulative mask exceeded the
cap, so with `protection_quantile=0.5` and `max_protected_fraction=0.5` it
collapsed to the non-sticky H mask. The corrected implementation treats the
capacity as an admission cap: existing protected rows are never evicted, and
only new candidates that fit in the remaining capacity can be admitted.

## Results

| Group | Forgetting | Final average accuracy | Immediate new-task accuracy | Time | Peak memory |
|---|---:|---:|---:|---:|---:|
| H (non-sticky) | 0.3116 | 0.5963 | 1.0000 | 231.0 s | 190.3 MB |
| I (sticky, corrected) | 0.4401 | 0.4666 | 1.0000 | 230.9 s | 226.5 MB |

For context, the comparable F result is forgetting `0.4915`; sparse baseline E
is `0.4555`; and the earlier FFN-boundary diagnostic G is `0.3009`.

The corrected I reduced forgetting relative to F by only **10.5%** and was
**41.2% worse than H** (`0.4401` vs `0.3116`). Relative to E, the reduction
was only **3.4%**, far below the frozen Phase-0 `>=50%` target.

## Mechanism checks

- Protected fraction was `0.5` for embedding/head, attention, and norms after
  every task.
- Turnover was `0.0` after every task, confirming the sticky invariant.
- Immediate accuracy was `1.0` for all six tasks, so the failure is not an
  inability to fit the current task.
- The fixed first-task protected set preserved some modular/shared knowledge,
  but facts and string still ended at `0.0`; protecting the first task is not a
  reliable proxy for protecting future reusable knowledge.

## Decision

Group I does **not** pass the Phase-0 continuation gate. The result is now a
valid negative result for this particular policy: a global, first-task-filling
sticky mask preserves the wrong rows and harms later-task retention. Do not run
the five-seed matrix or claim FIP success from H/I diagnostics. The next
architecture experiment should provide a separately budgeted low-rank plastic
bypass (or an equivalent task-adaptive capacity allocator), with parameter and
token parity explicitly documented before implementation.

