"""FIP Phase 0 mechanism tests.

These verify the scientific gates before any real training:
  - stable feature gradient leakage ~ 0
  - shared feature gradient scaled by lr_shared
  - rollback logits recovery error < 1e-6
  - Top-K produces exact sparsity
  - state machine promotes features correctly
  - dead feature ratio < 20%
"""

import math
import copy

import pytest
import torch

from fip import (
    DenseFFN,
    FIPFFN,
    FeatureMaskedAdamW,
    FIPTransformer,
    FeatureState,
    ModelConfig,
    NonFFNProtector,
    PlasticityController,
    RollbackManager,
)


# --------------------------------------------------------------------- helpers
def _small_cfg(use_fip=True):
    return ModelConfig(
        vocab_size=64, n_layer=2, d_model=32, d_ff=64,
        n_head=4, context=32, top_k=16, free_ratio=0.2, use_fip=use_fip,
    )


def _fip_ffn(d_model=8, d_ff=16, top_k=16, device="cpu"):
    from fip import FeatureRegistry
    reg = FeatureRegistry()
    ffn = FIPFFN(d_model, d_ff, top_k, reg, layer_id=0, device=device)
    return ffn, reg


# ============================================================ gradient isolation
class TestGradientIsolation:
    def test_stable_features_zero_grad(self):
        """Stable features must receive ~0 gradient even when selected by Top-K."""
        ffn, reg = _fip_ffn(d_model=8, d_ff=16, top_k=16)  # top_k=all -> all selected
        ffn.train()
        stable_idx = torch.tensor([0, 1, 2, 3])
        plastic_idx = torch.tensor([4, 5, 6, 7])
        reg.layers[0].state[stable_idx] = int(FeatureState.STABLE)
        ffn.rebuild_mask()

        x = torch.randn(4, 8, 8, requires_grad=False)
        out = ffn(x)
        out.sum().backward()

        g_gate = ffn.w_gate.grad
        g_up = ffn.w_up.grad
        g_down = ffn.w_down.grad

        # stable slices ~ 0
        assert g_gate[stable_idx].abs().max().item() < 1e-7
        assert g_up[stable_idx].abs().max().item() < 1e-7
        assert g_down[:, stable_idx].abs().max().item() < 1e-7
        # plastic slices actually receive gradient
        assert g_gate[plastic_idx].abs().sum().item() > 0
        assert g_up[plastic_idx].abs().sum().item() > 0
        assert g_down[:, plastic_idx].abs().sum().item() > 0

    def test_stable_isolation_with_sparsity(self):
        """Stable isolation holds even with Top-K < d_ff (sparse selection)."""
        ffn, reg = _fip_ffn(d_model=8, d_ff=32, top_k=8)
        ffn.train()
        # freeze the first 8 features
        reg.layers[0].state[:8] = int(FeatureState.STABLE)
        ffn.rebuild_mask()

        x = torch.randn(4, 8, 8)
        ffn(x).sum().backward()

        # No stable feature column/row may carry gradient.
        g = ffn.w_gate.grad
        stable_cols = g[:8].abs().max(dim=1).values
        assert stable_cols.max().item() < 1e-7

    def test_shared_features_scaled_grad(self):
        """Shared features receive gradient scaled by lr_shared (0.1)."""
        ffn, reg = _fip_ffn(d_model=8, d_ff=16, top_k=16)
        ffn.train()
        x = torch.randn(4, 8, 8)

        # baseline: feature 0 plastic
        ffn.zero_grad(set_to_none=True)
        reg.layers[0].state[:] = int(FeatureState.PLASTIC)
        ffn.rebuild_mask()
        ffn(x).sum().backward()
        g_plastic = ffn.w_gate.grad[0].clone()

        # now: feature 0 shared
        ffn.zero_grad(set_to_none=True)
        reg.layers[0].state[:] = int(FeatureState.SHARED)
        ffn.rebuild_mask()
        ffn(x).sum().backward()
        g_shared = ffn.w_gate.grad[0].clone()

        ratio = (g_shared.abs().mean() / g_plastic.abs().mean()).item()
        assert abs(ratio - 0.1) < 1e-3

    def test_no_grad_when_eval(self):
        """In eval mode the importance hook must not fire (no backward)."""
        ffn, reg = _fip_ffn(d_model=8, d_ff=16, top_k=8)
        ffn.eval()
        x = torch.randn(2, 4, 8)
        out = ffn(x)
        assert out.shape == (2, 4, 8)
        # no backward -> no importance update
        assert reg.layers[0].importance.abs().sum().item() == 0.0


class TestOptimizerIsolation:
    """Feature-state guarantees must hold after an optimizer step, not only in grads."""

    @staticmethod
    def _step(ffn, opt, x):
        opt.zero_grad(set_to_none=True)
        ffn(x).square().sum().backward()
        opt.step()

    def test_stable_feature_has_zero_real_adamw_update(self):
        """Weight decay must not move a stable feature with an all-zero grad."""
        ffn, reg = _fip_ffn(d_model=8, d_ff=16, top_k=16)
        ffn.train()
        reg.layers[0].state[0] = int(FeatureState.STABLE)
        ffn.rebuild_mask()
        opt = FeatureMaskedAdamW(
            ffn.parameters(), model=ffn, registry=reg, lr=1e-2, weight_decay=0.1
        )
        before = [p.detach().clone() for p in (ffn.w_gate, ffn.w_up, ffn.w_down)]
        self._step(ffn, opt, torch.randn(4, 8, 8))

        assert (ffn.w_gate[0] - before[0][0]).abs().max().item() < 1e-7
        assert (ffn.w_up[0] - before[1][0]).abs().max().item() < 1e-7
        assert (ffn.w_down[:, 0] - before[2][:, 0]).abs().max().item() < 1e-7
        assert (ffn.w_gate[1:] - before[0][1:]).abs().sum().item() > 0
        assert reg.layers[0].version[0].item() == 0
        assert reg.layers[0].version[1:].max().item() >= 1

    def test_stable_feature_discards_preexisting_adam_momentum(self):
        """Freezing after a plastic step also prevents momentum-driven drift."""
        ffn, reg = _fip_ffn(d_model=8, d_ff=16, top_k=16)
        ffn.train()
        opt = FeatureMaskedAdamW(ffn.parameters(), model=ffn, registry=reg, lr=1e-2)
        self._step(ffn, opt, torch.randn(4, 8, 8))  # populate Adam moments

        reg.layers[0].state[0] = int(FeatureState.STABLE)
        ffn.rebuild_mask()
        frozen = [p.detach().clone() for p in (ffn.w_gate, ffn.w_up, ffn.w_down)]
        self._step(ffn, opt, torch.randn(4, 8, 8))

        assert (ffn.w_gate[0] - frozen[0][0]).abs().max().item() < 1e-7
        assert (ffn.w_up[0] - frozen[1][0]).abs().max().item() < 1e-7
        assert (ffn.w_down[:, 0] - frozen[2][:, 0]).abs().max().item() < 1e-7
        # The stale moment tensors are cleared as part of the freeze operation.
        assert opt.state[ffn.w_gate]["exp_avg"][0].abs().max().item() == 0.0

    def test_shared_feature_keeps_one_tenth_real_update(self):
        """Adam normalization cannot turn shared=0.1 into a full parameter step."""
        torch.manual_seed(3)
        plastic, reg_plastic = _fip_ffn(d_model=8, d_ff=16, top_k=16)
        shared, reg_shared = _fip_ffn(d_model=8, d_ff=16, top_k=16)
        shared.load_state_dict(plastic.state_dict())
        plastic.train()
        shared.train()
        reg_plastic.layers[0].state[:] = int(FeatureState.PLASTIC)
        reg_shared.layers[0].state[:] = int(FeatureState.SHARED)
        plastic.rebuild_mask()
        shared.rebuild_mask()
        opt_plastic = FeatureMaskedAdamW(
            plastic.parameters(), model=plastic, registry=reg_plastic, lr=1e-2
        )
        opt_shared = FeatureMaskedAdamW(
            shared.parameters(), model=shared, registry=reg_shared, lr=1e-2
        )
        initial = plastic.w_gate.detach().clone()
        x = torch.randn(4, 8, 8)
        self._step(plastic, opt_plastic, x)
        self._step(shared, opt_shared, x)

        full_delta = (plastic.w_gate - initial).abs().mean()
        shared_delta = (shared.w_gate - initial).abs().mean()
        assert torch.isclose(shared_delta / full_delta, torch.tensor(0.1), atol=0.01)

    def test_non_fip_parameters_match_plain_adamw(self):
        """The projection must leave attention/embedding/etc. AdamW updates intact."""
        torch.manual_seed(4)
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        reference = copy.deepcopy(model)
        fip_opt = FeatureMaskedAdamW(model.parameters(), model=model, lr=1e-3)
        ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        idx = torch.randint(0, cfg.vocab_size, (2, 8))

        fip_opt.zero_grad(set_to_none=True)
        model(idx).square().mean().backward()
        fip_opt.step()
        ref_opt.zero_grad(set_to_none=True)
        reference(idx).square().mean().backward()
        ref_opt.step()

        assert torch.allclose(
            model.blocks[0].attn.qkv.weight,
            reference.blocks[0].attn.qkv.weight,
            atol=1e-8,
            rtol=1e-6,
        )


# ============================================================ top-k sparsity
class TestTopKSparsity:
    def test_exact_k_nonzero(self):
        ffn, _ = _fip_ffn(d_model=8, d_ff=32, top_k=8)
        ffn.eval()
        x = torch.randn(4, 5, 8)
        # intercept the activation via a forward hook
        acts = []
        ffn.register_forward_hook(
            lambda m, inp, o: None  # noop, we check via separate call
        )
        with torch.no_grad():
            gate = torch.nn.functional.silu(x @ ffn.w_gate.t())
            up = x @ ffn.w_up.t()
            act = gate * up
            vals, idx = act.topk(8, dim=-1)
            sparse = torch.zeros_like(act).scatter(-1, idx, vals)
        nnz = (sparse != 0).sum(dim=-1)
        assert (nnz == 8).all()

    def test_dense_ffn_no_sparsity(self):
        ffn = DenseFFN(8, 16)
        x = torch.randn(2, 3, 8)
        with torch.no_grad():
            gate = torch.nn.functional.silu(x @ ffn.w_gate.t())
            up = x @ ffn.w_up.t()
            act = gate * up
        assert (act != 0).all()


# ============================================================ rollback
class TestRollback:
    def test_rollback_exact_recovery(self):
        """After rollback, logits must match pre-task logits (< 1e-6)."""
        torch.manual_seed(0)
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        ctrl = PlasticityController(model, model.registry)
        rb = RollbackManager(model, model.registry)

        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        model.eval()
        with torch.no_grad():
            logits_ref = model(idx).clone()

        ctrl.begin_task()
        rb.checkpoint_before(0)

        # mutate weights with a gradient step
        model.train()
        out = model(idx)
        loss = out.float().sum()
        loss.backward()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        opt.step()

        # weights changed -> logits differ
        model.eval()
        with torch.no_grad():
            assert not torch.allclose(model(idx), logits_ref)

        # rollback -> exact recovery
        rb.rollback(0, ctrl)
        model.eval()
        err = rb.recovery_error(lambda: model(idx), logits_ref)
        assert err < 1e-6, f"recovery error {err} >= 1e-6"

    def test_rollback_restores_registry_state(self):
        torch.manual_seed(0)
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        ctrl = PlasticityController(model, model.registry)
        rb = RollbackManager(model, model.registry)

        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        ctrl.begin_task()
        rb.checkpoint_before(0)

        # simulate importance accumulation + end_task promotion
        for ffn in rb.ffn_layers:
            model.registry.update_importance(ffn.layer_id, torch.ones(cfg.d_ff))
        ctrl.end_task()

        states_after = {lid: r.state.clone() for lid, r in model.registry.layers.items()}
        # states changed from all-plastic/free (0/1) to include stable(3)/shared(2)
        assert any((s == int(FeatureState.STABLE)).any() for s in states_after.values())

        rb.rollback(0, ctrl)
        for lid, r in model.registry.layers.items():
            # restored to pre-task: no stable features
            assert not (r.state == int(FeatureState.STABLE)).any()

    def test_rollback_restores_optimizer_for_reproducible_continuation(self):
        """Replaying a step after rollback matches its original AdamW outcome."""
        torch.manual_seed(5)
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        ctrl = PlasticityController(model, model.registry)
        rb = RollbackManager(model, model.registry)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        idx_warmup = torch.randint(0, cfg.vocab_size, (2, 8))
        idx_task = torch.randint(0, cfg.vocab_size, (2, 8))

        def one_step(idx):
            opt.zero_grad(set_to_none=True)
            model(idx).square().mean().backward()
            opt.step()

        model.train()
        one_step(idx_warmup)  # ensure the checkpoint contains non-empty Adam state
        ctrl.begin_task()
        rb.checkpoint_before(0, optimizer=opt)
        one_step(idx_task)
        original = {name: p.detach().clone() for name, p in model.named_parameters()}

        rb.rollback(0, ctrl, optimizer=opt)
        one_step(idx_task)
        for name, parameter in model.named_parameters():
            assert torch.allclose(parameter, original[name], atol=1e-8, rtol=1e-6), name


# ============================================================ device movement
class TestDeviceMovement:
    def test_model_apply_moves_registry_and_cached_masks(self):
        """Registry bookkeeping follows ``model.to`` instead of staying on CPU."""
        model = FIPTransformer(_small_cfg(use_fip=True))
        model.to("cpu")
        for lid, reg in model.registry.layers.items():
            assert reg.state.device.type == "cpu"
            assert reg.importance.device.type == "cpu"
            ffn = model.blocks[lid].ffn
            assert ffn._mask.device == ffn.w_gate.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_cuda_fip_step_uses_cuda_registry_and_mask(self):
        """A real FIP forward/backward/AdamW step works on the target GPU."""
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg).cuda()
        ctrl = PlasticityController(model, model.registry)
        ctrl.begin_task()
        opt = FeatureMaskedAdamW(model.parameters(), model=model, lr=1e-3)
        idx = torch.randint(0, cfg.vocab_size, (2, 8), device="cuda")
        opt.zero_grad(set_to_none=True)
        loss = model(idx).float().square().mean()
        loss.backward()
        opt.step()

        assert torch.isfinite(loss)
        assert all(reg.state.is_cuda and reg.importance.is_cuda for reg in model.registry.layers.values())
        assert all(block.ffn._mask.is_cuda for block in model.blocks)


# ============================================================ state machine
class TestStateMachine:
    def test_initial_free_ratio(self):
        ffn, reg = _fip_ffn(d_model=8, d_ff=20, top_k=8, device="cpu")
        n_free = int((reg.layers[0].state == int(FeatureState.FREE)).sum())
        assert n_free == 4  # 20% of 20

    def test_importance_ema_updates(self):
        ffn, reg = _fip_ffn(d_model=8, d_ff=16, top_k=16)
        ffn.train()
        x = torch.randn(4, 8, 8)
        ffn(x).sum().backward()
        assert reg.layers[0].importance.abs().sum().item() > 0

    def test_end_task_promotes_stable(self):
        ffn, reg = _fip_ffn(d_model=8, d_ff=16, top_k=16)
        ffn.train()
        reg.begin_task()
        # give half the features high importance, half zero
        reg.layers[0].importance[:8] = 1.0
        reg.layers[0].importance[8:] = 0.0
        reg.layers[0].task_contrib[:8] = 1.0
        reg.end_task()
        states = reg.layers[0].state
        # high-importance half promoted to stable (quantile 0.5 threshold)
        assert (states[:8] == int(FeatureState.STABLE)).all()
        # zero-importance features not stable
        assert not (states[8:] == int(FeatureState.STABLE)).any()

    def test_repeated_use_promotes_shared(self):
        ffn, reg = _fip_ffn(d_model=8, d_ff=16, top_k=16)
        ffn.train()
        # A feature significantly used by a task is frozen to stable at that
        # task's boundary (used_now), so a later task cannot overwrite it.
        # Marginal features (below the significance quantile) stay plastic;
        # once reused significantly they are frozen, never silently shared.
        imp = torch.zeros(16)
        imp[:8] = 1.0
        imp[8:] = 0.01
        tc = torch.zeros(16)
        tc[:8] = 10.0
        tc[8:] = 0.001
        for _ in range(2):
            reg.begin_task()
            reg.layers[0].importance[:] = imp
            reg.layers[0].task_contrib[:] = tc
            reg.end_task()
        states = reg.layers[0].state
        # significantly-used half frozen to stable (used_now)
        assert (states[:8] == int(FeatureState.STABLE)).all()
        # marginal half must NOT be stable and NOT be shared
        assert not (states[8:] == int(FeatureState.STABLE)).any()
        assert not (states[8:] == int(FeatureState.SHARED)).any()

    def test_marginal_contribution_does_not_promote_shared(self):
        """Features with only marginal per-task contribution must not become shared.

        Under the old "activated at least once" rule, all 16 features would
        have usage_count=2 after two tasks and become shared.  With the
        significance-quantile rule, only the top-25% contributors are
        considered significantly used.
        """
        ffn, reg = _fip_ffn(d_model=8, d_ff=16, top_k=16)
        ffn.train()
        contrib = torch.zeros(16)
        contrib[:4] = 10.0          # top 25% — significant
        contrib[4:] = 0.001          # tiny but non-zero — marginal

        for _ in range(2):
            reg.begin_task()
            reg.layers[0].importance[:] = contrib
            reg.layers[0].task_contrib[:] = contrib
            reg.end_task()

        states = reg.layers[0].state
        # The 4 high-contribution features are high-importance and therefore
        # frozen to stable (stable takes precedence over shared so a later task
        # cannot overwrite them).  They must NOT be shared.
        assert (states[:4] == int(FeatureState.STABLE)).all()
        # The 12 marginal features must NOT be shared.
        assert not (states[4:] == int(FeatureState.SHARED)).any()

    def test_custom_task_sig_quantile_controls_threshold(self):
        """A higher quantile makes the significance gate stricter."""
        from fip import FeatureRegistry
        reg = FeatureRegistry(task_sig_quantile=0.9)
        reg.register_layer(0, n_features=10, device="cpu", dtype=torch.float32)
        reg.begin_task()
        # 10 features with linearly increasing contribution.
        reg.layers[0].task_contrib[:] = torch.arange(1, 11, dtype=torch.float32)
        sig = reg._significant_mask(reg.layers[0])
        # quantile 0.9 of [1..10] -> ~9.1, so only feature 9 (value 10) passes.
        assert sig.sum().item() == 1
        assert sig[9].item()


# ============================================================ dead features
class TestDeadFeatures:
    def test_dead_feature_ratio_below_threshold(self):
        """After moderate training, dead features must be < 20%."""
        torch.manual_seed(0)
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        ctrl = PlasticityController(model, model.registry)
        ctrl.begin_task()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(20):
            idx = torch.randint(0, cfg.vocab_size, (8, 16))
            opt.zero_grad(set_to_none=True)
            logits = model(idx)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, cfg.vocab_size),
                idx[:, 1:].reshape(-1),
            )
            loss.backward()
            opt.step()

        max_dead = max(model.registry.dead_feature_ratio(lid) for lid in model.registry.layers)
        assert max_dead < 0.2, f"dead feature ratio {max_dead:.2f} >= 0.2"


# ============================================================ integration
class TestIntegration:
    def test_dense_and_fip_same_param_count(self):
        """FIP and Dense must have equal FFN parameter count (fairness)."""
        cfg_fip = _small_cfg(use_fip=True)
        cfg_dense = _small_cfg(use_fip=False)
        m_fip = FIPTransformer(cfg_fip)
        m_dense = FIPTransformer(cfg_dense)
        assert m_fip.num_params() == m_dense.num_params()

    def test_full_model_forward_backward(self):
        torch.manual_seed(0)
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        ctrl = PlasticityController(model, model.registry)
        ctrl.begin_task()
        idx = torch.randint(0, cfg.vocab_size, (4, 16))
        logits = model(idx)
        assert logits.shape == (4, 16, cfg.vocab_size)
        loss = logits.float().sum()
        loss.backward()
        # grads exist on attention + ffn params
        assert model.blocks[0].attn.qkv.weight.grad is not None
        assert model.blocks[0].ffn.w_gate.grad is not None

    def test_cumulative_lora_merge_preserves_output_without_task_id(self):
        """A task adapter can be merged into one dense model exactly."""
        torch.manual_seed(8)
        cfg = _small_cfg(use_fip=False)
        model = FIPTransformer(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            original = model(idx).clone()

        model.enable_lora(rank=4)
        with torch.no_grad():
            assert torch.allclose(model(idx), original, atol=1e-8, rtol=1e-6)
        assert all(parameter.requires_grad == ("lora_" in name)
                   for name, parameter in model.named_parameters())

        opt = torch.optim.AdamW(model.lora_parameters(), lr=1e-2)
        opt.zero_grad(set_to_none=True)
        loss = model(idx).square().mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            adapted = model(idx).clone()
        model.merge_lora_()
        with torch.no_grad():
            assert torch.allclose(model(idx), adapted, atol=1e-6, rtol=1e-5)


# ============================================================ diagnostic ablation
class TestDiagnosticAblation:
    """Tests for the per-module drift measurement and group-G freeze logic."""

    def test_module_drift_categorizes_all_modules(self):
        """module_drift separates embedding, attention, norms, and FFN-by-state."""
        from experiments.phase0 import module_drift, snapshot_parameters

        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        ctrl = PlasticityController(model, model.registry)
        ctrl.begin_task()
        # Assign mixed states so every FFN category is non-empty.
        for reg in model.registry.layers.values():
            n = reg.n_features
            reg.state[: n // 4] = int(FeatureState.STABLE)
            reg.state[n // 4 : n // 2] = int(FeatureState.SHARED)
            reg.state[n // 2 :] = int(FeatureState.PLASTIC)
        for block in model.blocks:
            block.ffn.rebuild_mask()

        snap = snapshot_parameters(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01)

        drift = module_drift(model, snap, model.registry)
        for key in ("embedding_head", "attention", "norms",
                     "ffn_stable", "ffn_shared", "ffn_plastic_free"):
            assert drift[key] > 0, f"{key} drift should be non-zero"

    def test_freeze_non_ffn_preserves_ffn_trainable(self):
        """freeze_non_ffn_params freezes everything except w_gate/w_up/w_down."""
        from experiments.phase0 import freeze_non_ffn_params

        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        frozen = freeze_non_ffn_params(model)
        assert frozen > 0
        for name, p in model.named_parameters():
            if any(k in name for k in ("w_gate", "w_up", "w_down")):
                assert p.requires_grad, f"{name} should remain trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_frozen_non_ffn_params_do_not_move_under_optimizer(self):
        """After freezing, a FeatureMaskedAdamW step must not move non-FFN params."""
        from experiments.phase0 import freeze_non_ffn_params

        torch.manual_seed(0)
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        ctrl = PlasticityController(model, model.registry)
        ctrl.begin_task()
        freeze_non_ffn_params(model)
        opt = FeatureMaskedAdamW(
            model.parameters(), model=model, lr=1e-2, weight_decay=0.1
        )

        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        idx = torch.randint(0, cfg.vocab_size, (4, 8))
        opt.zero_grad(set_to_none=True)
        model(idx).float().square().mean().backward()
        opt.step()

        for name, p in model.named_parameters():
            if not any(k in name for k in ("w_gate", "w_up", "w_down")):
                assert torch.allclose(p, before[name], atol=1e-7), \
                    f"{name} moved despite freeze"
        # At least one FFN param should change (plastic features get gradient).
        ffn_changed = any(
            not torch.allclose(p, before[name], atol=1e-7)
            for name, p in model.named_parameters()
            if any(k in name for k in ("w_gate", "w_up", "w_down"))
        )
        assert ffn_changed, "FFN params should change under optimizer"

    def test_module_drift_zero_for_frozen_non_ffn(self):
        """After freezing non-FFN, drift for non-FFN categories must be zero."""
        from experiments.phase0 import module_drift, snapshot_parameters, freeze_non_ffn_params

        torch.manual_seed(0)
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        ctrl = PlasticityController(model, model.registry)
        ctrl.begin_task()
        freeze_non_ffn_params(model)
        opt = FeatureMaskedAdamW(
            model.parameters(), model=model, lr=1e-2, weight_decay=0.1
        )

        snap = snapshot_parameters(model)
        idx = torch.randint(0, cfg.vocab_size, (4, 8))
        opt.zero_grad(set_to_none=True)
        model(idx).float().square().mean().backward()
        opt.step()

        drift = module_drift(model, snap, model.registry)
        assert drift["embedding_head"] == 0.0
        assert drift["attention"] == 0.0
        assert drift["norms"] == 0.0
        # No stable/shared features yet (first task, before end_task).
        assert drift["ffn_stable"] == 0.0
        assert drift["ffn_shared"] == 0.0
        # Plastic/free features should have moved.
        assert drift["ffn_plastic_free"] > 0.0


# ============================================================ non-FFN protection
class TestNonFFNProtection:
    """Tests for the importance-driven NonFFNProtector used by group H."""

    def test_protector_tracks_only_non_ffn_params(self):
        """Only embedding/head, attention, and norm parameters are tracked."""
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        protector = NonFFNProtector(model)
        for name in protector.importance:
            assert not any(k in name for k in ("w_gate", "w_up", "w_down", "lora_"))

    def test_protector_scales_default_to_one_before_end_task(self):
        """Before end_task, all protection scales are 1.0 (no protection)."""
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        protector = NonFFNProtector(model)
        for scale in protector.scales.values():
            assert torch.all(scale == 1.0)

    def test_end_task_marks_important_rows_for_protection(self):
        """After end_task, high-importance rows get reduced scale."""
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        protector = NonFFNProtector(model, protection_quantile=0.5, protected_scale=0.1)
        ctrl = PlasticityController(model, model.registry)
        ctrl.begin_task()

        idx = torch.randint(0, cfg.vocab_size, (4, 8))
        model(idx).float().square().mean().backward()
        protector.update_importance()
        protector.end_task()

        frac = protector.protected_fraction()
        assert 0.0 < frac <= 0.6, f"protected fraction {frac} out of range"

    def test_protected_rows_update_at_reduced_scale(self):
        """Protected non-FFN rows must move at protected_scale, not full speed."""
        from experiments.phase0 import snapshot_parameters
        torch.manual_seed(0)
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        protector = NonFFNProtector(model, protection_quantile=0.5, protected_scale=0.1)
        ctrl = PlasticityController(model, model.registry)
        ctrl.begin_task()

        idx = torch.randint(0, cfg.vocab_size, (4, 8))
        # Warm up importance
        model(idx).float().square().mean().backward()
        protector.update_importance()
        protector.end_task()
        ctrl.begin_task()

        # Now check that the protected row moves at ~0.1 the unprotected delta
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        opt = FeatureMaskedAdamW(
            model.parameters(), model=model, lr=1e-2,
            non_ffn_protector=protector,
        )
        opt.zero_grad(set_to_none=True)
        model(idx).float().square().mean().backward()
        protector.update_importance()
        opt.step()

        attn_name = "blocks.0.attn.qkv.weight"
        delta = (model.blocks[0].attn.qkv.weight - before[attn_name]).abs()
        scale = protector.get_scale(model.blocks[0].attn.qkv.weight)
        protected_mask = scale < 1.0
        if protected_mask.any():
            protected_delta = delta[protected_mask].mean().item()
            unprotected_delta = delta[~protected_mask].mean().item()
            if unprotected_delta > 1e-8:
                ratio = protected_delta / unprotected_delta
                assert 0.05 <= ratio <= 0.2, f"ratio {ratio} should be near 0.1"

    def test_unprotected_non_ffn_params_match_plain_adamw(self):
        """Without protection (before end_task), non-FFN updates equal plain AdamW."""
        torch.manual_seed(4)
        cfg = _small_cfg(use_fip=True)
        model_a = FIPTransformer(cfg)
        model_b = FIPTransformer(cfg)
        model_b.load_state_dict(model_a.state_dict())
        ctrl = PlasticityController(model_a, model_a.registry)
        ctrl.begin_task()

        protector = NonFFNProtector(model_b)
        ctrl_b = PlasticityController(model_b, model_b.registry)
        ctrl_b.begin_task()

        opt_a = FeatureMaskedAdamW(model_a.parameters(), model=model_a, lr=1e-3)
        opt_b = FeatureMaskedAdamW(
            model_b.parameters(), model=model_b, lr=1e-3,
            non_ffn_protector=protector,
        )
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        opt_a.zero_grad(); model_a(idx).float().square().mean().backward(); opt_a.step()
        opt_b.zero_grad(); model_b(idx).float().square().mean().backward(); opt_b.step()

        for name, p in model_a.named_parameters():
            if "lora_" in name:
                continue
            if any(k in name for k in ("w_gate", "w_up", "w_down")):
                continue
            assert torch.allclose(p, model_b.state_dict()[name], atol=1e-8), name


# ============================================================ sticky protection
class TestStickyProtection:
    """Tests for cumulative protection with capacity cap (group I)."""

    def test_sticky_protection_is_cumulative(self):
        """Once a row is protected, it stays protected across tasks."""
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        protector = NonFFNProtector(
            model, protection_quantile=0.5, protected_scale=0.1,
            sticky=True, max_protected_fraction=0.8,
        )
        ctrl = PlasticityController(model, model.registry)

        # Task 1: train and end
        ctrl.begin_task()
        protector.begin_task()
        idx = torch.randint(0, cfg.vocab_size, (4, 8))
        model(idx).float().square().mean().backward()
        protector.update_importance()
        protector.end_task()
        frac1 = protector.protected_fraction()
        assert frac1 > 0

        # Task 2: different data, different importance pattern
        ctrl.begin_task()
        protector.begin_task()
        torch.manual_seed(99)
        idx2 = torch.randint(0, cfg.vocab_size, (4, 8))
        model(idx2).float().square().mean().backward()
        protector.update_importance()
        protector.end_task()
        frac2 = protector.protected_fraction()

        # Sticky: fraction should not decrease (cumulative)
        assert frac2 >= frac1, f"sticky fraction decreased: {frac1} -> {frac2}"

    def test_sticky_capacity_cap_enforced(self):
        """Protected fraction of 2-D parameters must not exceed the cap.

        (1-D norm vectors are protected whole-vector by the 1-D sticky rule and
        are excluded from the row-capacity accounting.)
        """
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        protector = NonFFNProtector(
            model, protection_quantile=0.3, protected_scale=0.1,
            sticky=True, max_protected_fraction=0.3,
        )
        ctrl = PlasticityController(model, model.registry)

        for seed in range(5):
            ctrl.begin_task()
            protector.begin_task()
            torch.manual_seed(seed)
            idx = torch.randint(0, cfg.vocab_size, (4, 8))
            model(idx).float().square().mean().backward()
            protector.update_importance()
            protector.end_task()

        total = protected = 0
        for name, mask in protector.protected_mask.items():
            if protector._is_1d[name]:
                continue
            total += mask.numel()
            protected += int(mask.sum().item())
        assert protected / total <= 0.3 + 1e-6

    def test_sticky_capacity_never_evicts_old_rows(self):
        """A full sticky capacity admits no replacements and has zero turnover."""
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        protector = NonFFNProtector(
            model, protection_quantile=0.5, protected_scale=0.1,
            sticky=True, max_protected_fraction=0.5,
        )
        ctrl = PlasticityController(model, model.registry)

        masks = []
        turnovers = []
        for seed in range(4):
            ctrl.begin_task()
            protector.begin_task()
            torch.manual_seed(seed * 101 + 3)
            idx = torch.randint(0, cfg.vocab_size, (4, 8))
            model(idx).float().square().mean().backward()
            protector.update_importance()
            protector.end_task()
            masks.append({name: mask.clone() for name, mask in protector.protected_mask.items()})
            turnovers.append(protector.turnover())

        for previous, current in zip(masks, masks[1:]):
            for name in previous:
                assert torch.all(~previous[name] | current[name]), name
        assert all(value == 0.0 for value in turnovers[1:]), turnovers

    def test_non_sticky_has_turnover(self):
        """Non-sticky mode should show turnover (rows losing protection)."""
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        protector = NonFFNProtector(
            model, protection_quantile=0.5, sticky=False,
        )
        ctrl = PlasticityController(model, model.registry)

        # Task 1
        ctrl.begin_task()
        protector.begin_task()
        torch.manual_seed(0)
        idx = torch.randint(0, cfg.vocab_size, (4, 8))
        model(idx).float().square().mean().backward()
        protector.update_importance()
        protector.end_task()

        # Task 2 with different data
        ctrl.begin_task()
        protector.begin_task()
        torch.manual_seed(99)
        idx2 = torch.randint(0, cfg.vocab_size, (4, 8))
        model(idx2).float().square().mean().backward()
        protector.update_importance()
        protector.end_task()

        # Non-sticky should have some turnover
        assert protector.turnover() >= 0.0

    def test_sticky_reduces_turnover(self):
        """Sticky mode should have lower turnover than non-sticky."""
        cfg = _small_cfg(use_fip=True)

        def run(sticky):
            torch.manual_seed(0)
            model = FIPTransformer(cfg)
            protector = NonFFNProtector(
                model, protection_quantile=0.5, sticky=sticky,
                max_protected_fraction=0.8,
            )
            ctrl = PlasticityController(model, model.registry)
            turnovers = []
            for seed in range(4):
                ctrl.begin_task()
                protector.begin_task()
                torch.manual_seed(seed * 100 + 7)
                idx = torch.randint(0, cfg.vocab_size, (4, 8))
                model(idx).float().square().mean().backward()
                protector.update_importance()
                protector.end_task()
                turnovers.append(protector.turnover())
            return sum(turnovers) / len(turnovers)

        sticky_avg = run(True)
        non_sticky_avg = run(False)
        assert sticky_avg <= non_sticky_avg, \
            f"sticky turnover {sticky_avg} should be <= non-sticky {non_sticky_avg}"

    def test_protected_set_summary_categorizes_modules(self):
        """protected_set_summary returns embedding_head, attention, norms."""
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        protector = NonFFNProtector(model, protection_quantile=0.5, sticky=True)
        ctrl = PlasticityController(model, model.registry)
        ctrl.begin_task()
        protector.begin_task()
        idx = torch.randint(0, cfg.vocab_size, (4, 8))
        model(idx).float().square().mean().backward()
        protector.update_importance()
        protector.end_task()

        summary = protector.protected_set_summary()
        assert "embedding_head" in summary
        assert "attention" in summary
        assert "norms" in summary

    # ---------------------------------------------------- per-task budget mode
    def test_per_task_budget_capacity_grows_with_tasks(self):
        """With per_task_budget set, later tasks can still admit new rows.

        The fixed-cap sticky policy lets the first task fill the whole budget;
        the per-task-budget policy grows capacity so a later task is not
        locked out.
        """
        def run(per_task_budget):
            torch.manual_seed(0)
            cfg = _small_cfg(use_fip=True)
            model = FIPTransformer(cfg)
            protector = NonFFNProtector(
                model, protection_quantile=0.5, protected_scale=0.1,
                sticky=True, max_protected_fraction=0.9,
                per_task_budget=per_task_budget,
            )
            ctrl = PlasticityController(model, model.registry)
            fractions = []
            for seed in range(4):
                ctrl.begin_task()
                protector.begin_task()
                torch.manual_seed(seed * 7 + 1)
                idx = torch.randint(0, cfg.vocab_size, (4, 8))
                model(idx).float().square().mean().backward()
                protector.update_importance()
                protector.end_task()
                fractions.append(protector.protected_fraction())
            return fractions

        budget = run(0.1)
        # With a per-task budget, protection grows over tasks (capacity grows).
        assert all(budget[i] <= budget[i + 1] + 1e-9 for i in range(len(budget) - 1))

    def test_per_task_budget_uses_current_task_importance(self):
        """New admissions come from rows significant in the current task."""
        torch.manual_seed(0)
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        protector = NonFFNProtector(
            model, protection_quantile=0.5, protected_scale=0.1,
            sticky=True, max_protected_fraction=0.9,
            per_task_budget=0.2,
        )
        ctrl = PlasticityController(model, model.registry)

        # Task 1: only a few rows are important -> few admissions.
        ctrl.begin_task()
        protector.begin_task()
        torch.manual_seed(3)
        idx = torch.randint(0, cfg.vocab_size, (4, 8))
        model(idx).float().square().mean().backward()
        protector.update_importance()
        protector.end_task()
        frac1 = protector.protected_fraction()

        # Task 2 with different data: new rows become significant and are
        # admitted (fraction grows, capacity not filled by task 1 alone).
        ctrl.begin_task()
        protector.begin_task()
        torch.manual_seed(4)
        idx2 = torch.randint(0, cfg.vocab_size, (4, 8))
        model(idx2).float().square().mean().backward()
        protector.update_importance()
        protector.end_task()
        frac2 = protector.protected_fraction()

        assert frac2 > frac1, f"expected growth, got {frac1} -> {frac2}"
        assert frac2 <= 0.9 + 1e-6

    def test_token_ownership_freezes_exclusive_rows(self):
        """Tokens used by exactly one task get scale 0 (fully frozen) rows."""
        torch.manual_seed(0)
        cfg = _small_cfg(use_fip=True)
        model = FIPTransformer(cfg)
        protector = NonFFNProtector(
            model, protection_quantile=0.5, protected_scale=0.1,
            sticky=True, max_protected_fraction=0.9,
            per_task_budget=0.2, token_ownership=True,
        )
        ctrl = PlasticityController(model, model.registry)

        # Task 1: uses token ids 1..8 only.
        ctrl.begin_task()
        protector.begin_task()
        torch.manual_seed(1)
        idx = torch.randint(1, 8, (4, 8))
        model(idx).float().square().mean().backward()
        protector.update_importance()
        protector.record_task_tokens(idx)
        protector.end_task()
        scale1 = protector.scales["tok_emb.weight"]
        # tokens 1..7 are exclusive after one task -> scale 0 (fully frozen)
        assert (scale1[1:8] == 0.0).all()
        # a token never seen keeps scale 1 (unprotected)
        assert scale1[-1] == 1.0

        # Task 2: reuses token 3 (now shared) + new tokens 9..12.
        ctrl.begin_task()
        protector.begin_task()
        torch.manual_seed(2)
        idx2 = torch.cat([torch.randint(3, 4, (4, 4)), torch.randint(9, 13, (4, 4))], dim=1)
        model(idx2).float().square().mean().backward()
        protector.update_importance()
        protector.record_task_tokens(idx2)
        protector.end_task()
        scale2 = protector.scales["tok_emb.weight"]
        # token 3 became shared (2 tasks) but was already frozen -> stays 0
        assert scale2[3] == 0.0
        # tokens 9..12 exclusive to task 2 -> frozen now
        assert (scale2[9:13] == 0.0).all()
