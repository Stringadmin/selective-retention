# FIP implementation gates v0.1

Date: 2026-07-29

This note records an implementation correction discovered after the frozen
research protocol and architecture v0 were written.  It does not alter the
experiment hypothesis, datasets, group matrix, or success thresholds.

## Why a correction was needed

The v0 gradient hooks made the gradient of a `stable` feature zero and scaled
the gradient of a `shared` feature.  That is insufficient for AdamW:

1. decoupled weight decay changes a zero-gradient parameter;
2. a stored Adam momentum vector changes a feature frozen after an earlier
   plastic update; and
3. Adam's normalization means a 0.1 gradient is not necessarily a 0.1
   parameter update.

The old tests therefore established gradient isolation, not feature-update
isolation.  No Phase 0 training result may be interpreted as evidence for FIP
until the latter is tested.

## v0.1 implementation invariant

`FeatureMaskedAdamW` runs ordinary AdamW, then projects the actual update for
each FIP feature slice:

```text
parameter_after = parameter_before + state_scale * (adamw_after - parameter_before)
```

where `stable = 0`, `shared = 0.1`, and `plastic/free = 1`.

It clears Adam state on `stable` slices after the projection.  Non-FIP
parameters retain ordinary AdamW behavior.  Gradient hooks remain in place for
importance measurement and early gradient suppression, but are not the proof
of isolation.  The optimizer also increments a feature's registry version once
per step if any of its three weight slices has a non-zero committed delta.

## Additional correctness requirements

- `FIPTransformer.to(device)` must move all registry tensors and rebuild each
  cached feature mask on that device.
- A rollback intended to continue training must snapshot and restore the
  optimizer state as well as model and registry state.
- Every FIP training script must use `FeatureMaskedAdamW`, not plain AdamW.

## Required CPU gates

1. With AdamW weight decay, all three slices of a `stable` feature have maximum
   actual update below `1e-7`.
2. A feature frozen after an earlier Adam update remains unchanged and its
   stored moment is zeroed.
3. A `shared` feature retains `0.1 ± 0.01` of the corresponding plastic
   parameter delta.
4. Non-FIP parameters match plain AdamW under identical inputs and settings.
5. Rollback plus replay of a step reproduces the original post-step model when
   optimizer state is supplied.

Current verification: 21 CPU tests passing; the same suite contains a
CUDA-conditional FIP forward/backward/optimizer smoke test.
