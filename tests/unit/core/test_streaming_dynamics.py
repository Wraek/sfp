"""Tests for streaming processor dynamics: LR warmup, adaptive surprise, loss
sanity checks, tier2/axiom guidance losses, and salience gradient scaling."""

import math

import torch
import pytest

from sfp.config import FieldConfig, StreamingConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.core.streaming import StreamingProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 32
N_LAYERS = 2


def _make_field() -> SemanticFieldProcessor:
    return SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))


def _make_processor(
    **streaming_overrides,
) -> StreamingProcessor:
    """Build a small, fast StreamingProcessor with caller overrides."""
    defaults = dict(
        surprise_threshold=0.0,
        adaptive_surprise=False,
        warmup_steps=0,
    )
    defaults.update(streaming_overrides)
    field = _make_field()
    return StreamingProcessor(
        field,
        streaming_config=StreamingConfig(**defaults),
    )


# ---------------------------------------------------------------------------
# 1. LR warmup ramps from start_factor to 1.0
# ---------------------------------------------------------------------------


def test_warmup_ramps_lr():
    """With warmup_steps=10, the scheduler should ramp the effective LR from
    start_factor * base_lr up toward base_lr over the warmup window."""
    warmup_steps = 10
    start_factor = 0.1
    base_lr = 1e-4

    proc = _make_processor(
        lr=base_lr,
        warmup_steps=warmup_steps,
        warmup_start_factor=start_factor,
    )

    # Scheduler should have been created
    assert proc._scheduler is not None

    # Before any step the effective LR should be near start_factor * base_lr
    initial_lr = proc._optimizer.param_groups[0]["lr"]
    assert initial_lr == pytest.approx(base_lr * start_factor, rel=1e-4)

    # Run enough steps that updates happen (threshold=0 guarantees updates)
    lrs_seen: list[float] = []
    for _ in range(warmup_steps + 5):
        x = torch.randn(DIM)
        proc.process(x)
        lrs_seen.append(proc._optimizer.param_groups[0]["lr"])

    # LR should have increased monotonically during warmup
    for i in range(1, warmup_steps):
        assert lrs_seen[i] >= lrs_seen[i - 1]

    # After warmup completes the LR should be approximately base_lr
    assert lrs_seen[warmup_steps - 1] == pytest.approx(base_lr, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. Warmup disabled: warmup_steps=0 means no scheduler
# ---------------------------------------------------------------------------


def test_warmup_disabled_no_scheduler():
    """warmup_steps=0 + lr_decay disabled should result in no scheduler object."""
    proc = _make_processor(warmup_steps=0, lr_decay_enabled=False)
    assert proc._scheduler is None

    # LR should be exactly the configured value with no scheduling
    base_lr = proc.config.lr
    assert proc._optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

    # Processing should still work normally
    x = torch.randn(DIM)
    metric = proc.process(x)
    assert metric.grad_norm >= 0

    # LR unchanged after processing
    assert proc._optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)


# ---------------------------------------------------------------------------
# 3. Adaptive surprise threshold uses percentile of recent grad norms
# ---------------------------------------------------------------------------


def test_adaptive_surprise_threshold():
    """With adaptive_surprise=True and >10 history entries, the threshold
    should be the configured percentile of recent grad norms rather than the
    static surprise_threshold value."""
    static_threshold = 999.0  # absurdly high, should not be used after warmup
    percentile = 0.5  # median

    proc = _make_processor(
        surprise_threshold=static_threshold,
        adaptive_surprise=True,
        surprise_percentile=percentile,
    )

    # Seed history with 15 entries so adaptive kicks in
    for _ in range(15):
        x = torch.randn(DIM)
        proc.process(x)

    assert len(proc._history) >= 10

    # Compute expected threshold manually
    recent_norms = sorted(m.grad_norm for m in proc._history[-100:])
    idx = min(int(percentile * len(recent_norms)), len(recent_norms) - 1)
    expected_threshold = recent_norms[idx]

    # The internal method should match
    computed = proc._compute_threshold()
    assert computed == pytest.approx(expected_threshold, rel=1e-6)
    # And it should NOT be the absurd static value
    assert computed < static_threshold


def test_adaptive_surprise_falls_back_when_little_history():
    """With fewer than 10 history entries adaptive_surprise falls back to the
    static surprise_threshold."""
    static_threshold = 0.42

    proc = _make_processor(
        surprise_threshold=static_threshold,
        adaptive_surprise=True,
    )

    # Only 5 steps -- not enough for adaptive
    for _ in range(5):
        proc.process(torch.randn(DIM))

    assert len(proc._history) < 10
    assert proc._compute_threshold() == pytest.approx(static_threshold)


# ---------------------------------------------------------------------------
# 4. Non-finite loss returns updated=False
# ---------------------------------------------------------------------------


def test_nonfinite_loss_nan_returns_not_updated():
    """If the forward pass produces NaN loss, process() should return
    updated=False and loss=NaN without crashing."""
    proc = _make_processor()

    # Corrupt a weight to produce NaN output
    with torch.no_grad():
        for p in proc.field.parameters():
            p.fill_(float("nan"))
            break  # corrupt just one parameter

    x = torch.randn(DIM)
    metric = proc.process(x)

    assert metric.updated is False
    assert math.isnan(metric.loss)


def test_nonfinite_loss_inf_returns_not_updated():
    """If the forward pass produces Inf loss, process() should return
    updated=False."""
    proc = _make_processor()

    # Corrupt weights to produce Inf
    with torch.no_grad():
        for p in proc.field.parameters():
            p.fill_(float("inf"))
            break

    x = torch.randn(DIM)
    metric = proc.process(x)

    assert metric.updated is False
    assert math.isnan(metric.loss) or math.isinf(metric.loss)


# ---------------------------------------------------------------------------
# 5. Tier2 guidance loss increases total loss
# ---------------------------------------------------------------------------


def test_tier2_guidance_increases_loss():
    """Passing tier2_guidance should add an MSE term that increases the total
    loss compared to the same input without guidance."""
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    field = SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))

    proc_base = StreamingProcessor(
        field,
        streaming_config=StreamingConfig(
            surprise_threshold=0.0,
            adaptive_surprise=False,
            warmup_steps=0,
            tier2_guidance_weight=0.05,
        ),
    )

    x = torch.randn(DIM)
    # Baseline loss without guidance
    metric_base = proc_base.process(x.clone())

    # Now build a fresh processor with identical weights
    torch.manual_seed(42)
    field2 = SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))
    proc_guided = StreamingProcessor(
        field2,
        streaming_config=StreamingConfig(
            surprise_threshold=0.0,
            adaptive_surprise=False,
            warmup_steps=0,
            tier2_guidance_weight=0.05,
        ),
    )

    # A divergent guidance vector should add extra loss
    guidance = torch.randn(DIM) * 10.0
    metric_guided = proc_guided.process(x.clone(), tier2_guidance=guidance)

    assert metric_guided.loss > metric_base.loss


# ---------------------------------------------------------------------------
# 6. Axiom anchor loss increases total loss
# ---------------------------------------------------------------------------


def test_axiom_anchor_increases_loss():
    """Passing axiom_anchor should add an MSE term that increases the total
    loss compared to the same input without an anchor."""
    torch.manual_seed(99)
    field = SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))

    proc_base = StreamingProcessor(
        field,
        streaming_config=StreamingConfig(
            surprise_threshold=0.0,
            adaptive_surprise=False,
            warmup_steps=0,
            axiom_anchor_weight=0.02,
        ),
    )

    x = torch.randn(DIM)
    metric_base = proc_base.process(x.clone())

    # Fresh identical processor
    torch.manual_seed(99)
    field2 = SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))
    proc_anchored = StreamingProcessor(
        field2,
        streaming_config=StreamingConfig(
            surprise_threshold=0.0,
            adaptive_surprise=False,
            warmup_steps=0,
            axiom_anchor_weight=0.02,
        ),
    )

    anchor = torch.randn(DIM) * 10.0
    metric_anchored = proc_anchored.process(x.clone(), axiom_anchor=anchor)

    assert metric_anchored.loss > metric_base.loss


# ---------------------------------------------------------------------------
# 7. Salience gradient scaling
# ---------------------------------------------------------------------------


def test_salience_scaling_at_half():
    """salience_score=0.5 => scale = max(0.1, 0.5*2.0) = 1.0, so grads are
    unchanged compared to not passing salience_score."""
    torch.manual_seed(7)
    field = SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))
    proc = StreamingProcessor(
        field,
        streaming_config=StreamingConfig(
            surprise_threshold=0.0,
            adaptive_surprise=False,
            warmup_steps=0,
            salience_gradient_scaling=True,
            confidence_modulation_enabled=False,
        ),
    )

    x = torch.randn(DIM)

    # Snapshot weights before
    weights_before = [p.data.clone() for p in proc.field.parameters()]

    proc.process(x.clone(), salience_score=0.5)

    # Now repeat with a fresh identical processor but WITHOUT salience
    torch.manual_seed(7)
    field2 = SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))
    proc2 = StreamingProcessor(
        field2,
        streaming_config=StreamingConfig(
            surprise_threshold=0.0,
            adaptive_surprise=False,
            warmup_steps=0,
            salience_gradient_scaling=True,
            confidence_modulation_enabled=False,
        ),
    )

    proc2.process(x.clone(), salience_score=None)

    # With salience=0.5, scale = max(0.1, 0.5*2.0) = 1.0 which equals no-op.
    # Weight deltas should be identical.
    for p1, p2 in zip(proc.field.parameters(), proc2.field.parameters()):
        torch.testing.assert_close(p1.data, p2.data, atol=1e-6, rtol=1e-5)


def test_salience_scaling_at_low():
    """salience_score=0.1 => scale = max(0.1, 0.1*2.0) = 0.2, so grads are
    shrunk. Weight change should be smaller than with salience=0.5 (scale=1)."""
    torch.manual_seed(11)
    field_low = SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))
    proc_low = StreamingProcessor(
        field_low,
        streaming_config=StreamingConfig(
            surprise_threshold=0.0,
            adaptive_surprise=False,
            warmup_steps=0,
            salience_gradient_scaling=True,
            confidence_modulation_enabled=False,
        ),
    )

    x = torch.randn(DIM)
    weights_before_low = [p.data.clone() for p in proc_low.field.parameters()]
    proc_low.process(x.clone(), salience_score=0.1)
    delta_low = sum(
        (p.data - w).abs().sum().item()
        for p, w in zip(proc_low.field.parameters(), weights_before_low)
    )

    # Same setup but with salience=0.5 (scale=1.0)
    torch.manual_seed(11)
    field_mid = SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))
    proc_mid = StreamingProcessor(
        field_mid,
        streaming_config=StreamingConfig(
            surprise_threshold=0.0,
            adaptive_surprise=False,
            warmup_steps=0,
            salience_gradient_scaling=True,
            confidence_modulation_enabled=False,
        ),
    )

    weights_before_mid = [p.data.clone() for p in proc_mid.field.parameters()]
    proc_mid.process(x.clone(), salience_score=0.5)
    delta_mid = sum(
        (p.data - w).abs().sum().item()
        for p, w in zip(proc_mid.field.parameters(), weights_before_mid)
    )

    # Low salience (0.2 scale) should produce smaller weight changes
    assert delta_low > 0, "low-salience update should still change weights"
    assert delta_mid > 0, "mid-salience update should still change weights"
    assert delta_low < delta_mid
