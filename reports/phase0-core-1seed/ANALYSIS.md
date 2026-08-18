# Phase 0 core matrix: one-seed analysis

Run: `seed=1337`; six synthetic tasks; 2,000 steps/task; batch 32; equal
1,536,000 training tokens per A/B/C/E/F group.  This is an **interim,
single-seed diagnostic**, not a final Phase 0 result and not an A--F matrix.

## Result

| group | average retention forgetting | final average accuracy |
|---|---:|---:|
| A dense full FT | 0.4676 | 0.3177 |
| B online EWC | 0.4733 | 0.3927 |
| C 5% replay | **0.2527** | **0.7192** |
| E Top-K sparse FT | 0.4555 | 0.4633 |
| F FIP | **0.5086** | 0.3582 |

FIP does not pass the frozen Phase 0 forgetting gate in this seed. It is worse
than the sparse-only attribution baseline E (0.5086 vs 0.4555) and worse than
the dense A baseline (0.5086 vs 0.4676), while it does learn the final task
(`unrelated` accuracy 1.0). Therefore this is not a failure to learn new data;
it is a retention failure.

## Diagnostic signal

At the final task, every FIP layer has only 1--4 `stable` features and roughly
77--86% `shared` features. The current implementation increments a feature's
usage counter if it was activated at least once anywhere in a task. Over 2,000
Top-K steps, this criterion is too permissive: nearly every slot is eventually
touched, promoted to `shared`, and remains 0.1-updatable rather than protected.

This is a concrete v0 state-assignment failure mode, not evidence that the FIP
hypothesis is true. Do not run five seeds from this revision. The next revision
must define and test a task-level *significant-use* rule (for example, a
per-task contribution threshold/quantile) before repeating this single-seed
core matrix under a new implementation version.

## Validity checks

- Parameter count: 6,360,320 for A/B/C/E/F.
- Per-group tokens and optimizer steps: identical.
- C uses 5% old-task replay and records replay storage/tokens.
- EWC Fisher uses current-task gradients only; no extra data pass.
- CUDA memory was verified stable after fixing the FIP hook graph-retention
  bug; FIP peak allocation was 195.9 MB.
- D remains excluded because temporary LoRA adapter parameters do not yet meet
  the frozen maximum-parameter parity condition.
