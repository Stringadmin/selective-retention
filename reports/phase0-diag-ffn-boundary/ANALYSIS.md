# FFN-boundary diagnostic ablation (seed 1337)

Groups F and G use the same FIP model, task order, seed, 2,000 steps/task,
batch size 32, and token budget. G freezes embedding/head, attention, and norms
after task 1. G is diagnostic and non-comparable to the frozen Phase 0 groups.

## Validity checks

- F exactly reproduces the v2 retention forgetting result: `0.4915`.
- F and G have identical task-1 accuracy and module drift.
- In G, embedding/head, attention, and norm drift are exactly zero for tasks
  2--6. The freeze is therefore real, not merely a reporting label.
- 36 tests pass after correcting module drift to a true category-level L2 norm
  (`sqrt(sum(delta^2))`).

## Main result

| metric | F | G |
|---|---:|---:|
| retention forgetting | 0.4915 | **0.3009** |
| final average accuracy | **0.3939** | 0.3686 |
| final conflict accuracy | 0.0586 | **1.0000** |
| final modular accuracy | **0.6123** | 0.1128 |
| final shared accuracy | **0.6318** | 0.0986 |
| final string accuracy | 0.0000 | 0.0000 |
| final unrelated accuracy | 1.0000 | 1.0000 |

G reduces measured forgetting by 38.8% relative to F. This establishes that
unprotected non-FFN updates are a major overwrite path. In F, attention has the
largest total L2 drift on most tasks, with embedding/head and norms also moving
substantially; all three are exactly zero in G after task 1.

## Stability--plasticity qualification

The lower G forgetting is not a free improvement. Immediate new-task accuracy
falls sharply:

| newly trained task | F | G |
|---|---:|---:|
| facts | 1.0000 | 1.0000 |
| conflict | 1.0000 | 1.0000 |
| modular | **0.5415** | 0.1963 |
| string | 1.0000 | 1.0000 |
| shared | **0.9072** | 0.5195 |
| unrelated | 1.0000 | 1.0000 |

G preserves conflict facts but loses much of the ability to acquire modular
and shared-rule tasks. It also still forgets string completely (1.0 -> 0.0),
even though all non-FFN parameters are frozen. Therefore:

1. the FFN-only isolation boundary is too narrow; and
2. the current FFN feature-state mechanism is also insufficient on at least
   some tasks.

## Research decision

Do not promote G to the main architecture and do not run five seeds yet.
Freezing all non-FFN parameters merely trades plasticity for stability.

The next architecture iteration should use *graded protection* outside the FFN
rather than a binary freeze: for example optimizer-level update scales for
embedding/head, attention, and norms derived from task importance, with a
reserved plastic subspace or low-rank update path. The next diagnostic must
retain at least 90% of F's immediate new-task accuracy while improving
retention; otherwise any forgetting reduction is not a valid FIP gain.
