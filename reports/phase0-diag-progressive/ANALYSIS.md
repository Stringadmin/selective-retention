# Progressive non-FFN protection diagnostic

Seed 1337, six tasks, 2,000 steps/task, batch 32. H adds per-row progressive
protection to non-FFN parameters. All runs use the corrected true L2 module
drift metric. 41 tests pass.

## Default H result

| metric | F | G (hard freeze) | H q=0.5, scale=0.1 |
|---|---:|---:|---:|
| retention forgetting | 0.4915 | **0.3009** | 0.3116 |
| immediate new-task average | 0.9081 | 0.7860 | **1.0000** |
| final average accuracy | 0.3939 | 0.3686 | **0.5963** |

H reduces forgetting by 36.6% relative to F while improving immediate task
learning: all six tasks reach 1.0 immediately after training. It is strongly
Pareto-preferable to the hard-freeze G diagnostic in plasticity and final
accuracy, but its forgetting is 0.0107 worse than G.

H does not pass the frozen Phase 0 gate: relative to sparse E (`0.4555`), the
forgetting reduction is about 31.6%, below the required 50%.

## Targeted parameter checks

| H setting | protected rows | forgetting | immediate avg | final avg |
|---|---:|---:|---:|---:|
| q=0.5, scale=0.1 | 50% | **0.3116** | 1.0000 | **0.5963** |
| q=0.4, scale=0.1 | 60% | 0.3541 | 1.0000 | 0.5496 |
| q=0.5, scale=0.05 | 50% | 0.3388 | 1.0000 | 0.5597 |

Neither broader protection nor a smaller update scale improves the aggregate
result. The response is non-monotonic: increasing protection redirects learning
pressure into the remaining plastic rows and also changes which rows become
important in later tasks.

## Interpretation

The architecture direction is supported: graded non-FFN protection recovers
plasticity that hard freezing destroyed and materially improves retention over
F. But a single global quantile and scale are insufficient.

The next diagnostic should measure protected-set turnover between tasks. The
current lifetime EMA recomputes the top rows at every boundary, so a row that
carried old knowledge can later become unprotected. If turnover is high, the
next design should use sticky/cumulative protection with an explicit capacity
budget and a separate plastic low-rank path. Module-specific policies are also
needed: embedding/head, attention, and norms have different drift and capacity
profiles and should not necessarily share one quantile and scale.

Do not run five seeds or promote H to the frozen F group yet.
