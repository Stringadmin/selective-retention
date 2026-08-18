# FIP Phase 0 execution plan v0.1

Date: 2026-07-29

This is an execution note for the frozen research protocol. It does not modify
the hypothesis, the six tasks, or any pass/fail threshold.

## Completed preconditions

- Optimizer-level FIP isolation is tested under AdamW weight decay and existing
  momentum; gradients alone are not accepted as proof of isolation.
- Registry tensors and cached masks follow `model.to(device)`.
- Rollback can restore model, registry, and optimizer state.
- CUDA PyTorch 2.11.0+cu128 is verified on the RTX 5070 (`sm_120`).
- Deterministic six-task synthetic suite, A/B/C/E/F runner, report hashes, and
  small end-to-end preflights are present.
- A 5% replay quota is accumulated across batches and only samples data from
  previous tasks, never data added in the current task.

## Execution order

1. Run the comparable core matrix, one seed first:

   ```bash
   python -m experiments.phase0 --device cuda --seeds 1337 \
     --output-dir reports/phase0-core-1seed
   ```

2. Inspect metrics and failure modes. This is an interim A/B/C/E/F result, not
   a complete Phase 0 conclusion, because D remains gated below.

3. If no implementation/data-quality failure is found, run the same comparable
   matrix with the five frozen seeds. Keep the generated report and all source
   hashes unchanged.

4. Resolve D's maximum-parameter parity, freeze the resolution, then run D for
   the same five seeds. Do not call the A--F matrix complete before this step.

## D (cumulative merged LoRA) fairness gate

The LoRA implementation is functionally tested: adapters merge into the dense
attention/FFN base without changing output and do not use task IDs. The tied
embedding/output head is deliberately not adapted because merging a head delta
would silently alter input embeddings.

Its temporary low-rank matrices nevertheless add train-time parameters. That
violates the protocol's strict maximum-parameter equality if D is run against
the unmodified dense width. The runner therefore rejects D unless an explicit
diagnostic override is supplied. A valid final matrix needs one of these
decisions documented before D runs:

1. a width-matched dense D architecture with an exact parameter accounting;
2. a protocol amendment that treats temporary adapter parameters separately;
   or
3. D reclassified as a diagnostic baseline, excluded from direct equality
   claims.

## Preliminary throughput only

The A-only full-width preflight (6 layers, 256 hidden, 1024 FFN, batch 32)
processed 38,400 synthetic tokens in 5.64 seconds: **6,813 tokens/s**, with a
138 MB reported PyTorch peak allocation. This is a sizing signal, not a final
wall-clock estimate: B/C/E/F and repeated evaluation have different overhead.

## Commands

```bash
# Inspect the exact frozen budget without training
python -m experiments.phase0 --device cuda --dry-run

# Re-run the small pipeline check; its report is explicitly non-scientific
python -m experiments.phase0_smoke --device cuda
```
