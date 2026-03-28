"""Tests for LoRA multi-signal merge features."""

import torch
import torch.nn as nn

from sfp.config import FieldConfig, LoRAConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.core.lora import LoRALinear, LoRAMergeContext, OnlineLoRAManager

# A surprise history that has 120 flat entries.  With merge_threshold=2.0
# the ratio (1.0/1.0 = 1.0) is below threshold so Signal 1 never fires.
_FLAT_SURPRISE = [1.0] * 120


def _make_manager(
    dim: int = 32,
    n_layers: int = 2,
    rank: int = 4,
    alpha: float = 1.0,
    merge_threshold: float = 2.0,
    uncertainty_merge_threshold: float = 0.7,
    uncertainty_merge_window: int = 50,
    mood_merge_threshold: float = -0.3,
    mood_merge_window: int = 50,
    goal_stall_merge_steps: int = 100,
) -> OnlineLoRAManager:
    """Build a small SemanticFieldProcessor + OnlineLoRAManager for testing.

    Default merge_threshold is set high (2.0) so the surprise-ratio signal
    does not accidentally fire during multi-signal tests.
    """
    field_cfg = FieldConfig(dim=dim, n_layers=n_layers)
    field = SemanticFieldProcessor(field_cfg)
    lora_cfg = LoRAConfig(
        rank=rank,
        alpha=alpha,
        merge_threshold=merge_threshold,
        uncertainty_merge_threshold=uncertainty_merge_threshold,
        uncertainty_merge_window=uncertainty_merge_window,
        mood_merge_threshold=mood_merge_threshold,
        mood_merge_window=mood_merge_window,
        goal_stall_merge_steps=goal_stall_merge_steps,
    )
    return OnlineLoRAManager(field, lora_cfg)


class TestLoRAMergeContext:
    """LoRAMergeContext dataclass construction and field access."""

    def test_create_with_all_fields(self):
        ctx = LoRAMergeContext(
            prediction_uncertainty_history=[0.1, 0.5, 0.9],
            mood_history=[-0.2, 0.0, 0.3],
            goal_progress_history={0: [0.1, 0.2, 0.3], 1: [0.5, 0.5]},
        )
        assert ctx.prediction_uncertainty_history == [0.1, 0.5, 0.9]
        assert ctx.mood_history == [-0.2, 0.0, 0.3]
        assert ctx.goal_progress_history == {0: [0.1, 0.2, 0.3], 1: [0.5, 0.5]}

    def test_create_defaults_to_none(self):
        ctx = LoRAMergeContext()
        assert ctx.prediction_uncertainty_history is None
        assert ctx.mood_history is None
        assert ctx.goal_progress_history is None

    def test_create_with_partial_fields(self):
        ctx = LoRAMergeContext(mood_history=[-0.5, -0.6])
        assert ctx.prediction_uncertainty_history is None
        assert ctx.mood_history == [-0.5, -0.6]
        assert ctx.goal_progress_history is None


class TestUncertaintyTriggeredMerge:
    """Signal 2: Sustained high prediction uncertainty triggers LoRA merge."""

    def test_merge_when_sustained_above_threshold(self):
        window = 50
        threshold = 0.7
        manager = _make_manager(
            uncertainty_merge_threshold=threshold,
            uncertainty_merge_window=window,
        )

        # Build uncertainty history that exceeds threshold for the full window
        uncertainty_history = [0.8] * window  # all > 0.7

        ctx = LoRAMergeContext(prediction_uncertainty_history=uncertainty_history)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is True

    def test_no_merge_when_window_not_full(self):
        window = 50
        threshold = 0.7
        manager = _make_manager(
            uncertainty_merge_threshold=threshold,
            uncertainty_merge_window=window,
        )

        # Only 30 values -- less than the required window of 50
        uncertainty_history = [0.8] * 30

        ctx = LoRAMergeContext(prediction_uncertainty_history=uncertainty_history)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is False

    def test_no_merge_when_some_values_below_threshold(self):
        window = 50
        threshold = 0.7
        manager = _make_manager(
            uncertainty_merge_threshold=threshold,
            uncertainty_merge_window=window,
        )

        # 50 values but the last one dips below the threshold
        uncertainty_history = [0.8] * 49 + [0.5]

        ctx = LoRAMergeContext(prediction_uncertainty_history=uncertainty_history)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is False

    def test_merge_uses_last_window_slice(self):
        """Only the last `window` entries matter; earlier low values are ignored."""
        window = 50
        threshold = 0.7
        manager = _make_manager(
            uncertainty_merge_threshold=threshold,
            uncertainty_merge_window=window,
        )

        # 20 low values followed by 50 high values
        uncertainty_history = [0.1] * 20 + [0.8] * window

        ctx = LoRAMergeContext(prediction_uncertainty_history=uncertainty_history)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is True


class TestMoodTriggeredMerge:
    """Signal 3: Sustained negative mood triggers LoRA merge."""

    def test_merge_when_sustained_below_threshold(self):
        window = 50
        threshold = -0.3
        manager = _make_manager(
            mood_merge_threshold=threshold,
            mood_merge_window=window,
        )

        # All values below -0.3
        mood_history = [-0.5] * window

        ctx = LoRAMergeContext(mood_history=mood_history)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is True

    def test_no_merge_when_mood_window_not_full(self):
        window = 50
        threshold = -0.3
        manager = _make_manager(
            mood_merge_threshold=threshold,
            mood_merge_window=window,
        )

        # Only 20 entries -- below the required 50
        mood_history = [-0.5] * 20

        ctx = LoRAMergeContext(mood_history=mood_history)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is False

    def test_no_merge_when_some_mood_above_threshold(self):
        window = 50
        threshold = -0.3
        manager = _make_manager(
            mood_merge_threshold=threshold,
            mood_merge_window=window,
        )

        # Last value breaks above the threshold
        mood_history = [-0.5] * 49 + [0.0]

        ctx = LoRAMergeContext(mood_history=mood_history)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is False

    def test_merge_uses_last_mood_window_slice(self):
        """Only the last `window` entries matter; earlier non-negative values are ignored."""
        window = 50
        threshold = -0.3
        manager = _make_manager(
            mood_merge_threshold=threshold,
            mood_merge_window=window,
        )

        # 30 positive values then 50 negative values
        mood_history = [0.5] * 30 + [-0.5] * window

        ctx = LoRAMergeContext(mood_history=mood_history)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is True


class TestGoalStallTriggeredMerge:
    """Signal 4: Goal progress stall triggers LoRA merge."""

    def test_merge_when_goal_stalled(self):
        stall_steps = 100
        manager = _make_manager(goal_stall_merge_steps=stall_steps)

        # Goal 0 has been stalled at 0.5 for 100 steps (max-min < 0.01)
        goal_progress = {0: [0.5] * stall_steps}

        ctx = LoRAMergeContext(goal_progress_history=goal_progress)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is True

    def test_merge_when_any_single_goal_stalled(self):
        """Merge fires if *any* goal is stalled, even if others are progressing."""
        stall_steps = 100
        manager = _make_manager(goal_stall_merge_steps=stall_steps)

        goal_progress = {
            0: list(range(stall_steps)),  # progressing -- range is wide
            1: [0.3] * stall_steps,       # stalled -- max-min == 0
        }

        ctx = LoRAMergeContext(goal_progress_history=goal_progress)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is True

    def test_no_merge_when_goal_history_too_short(self):
        stall_steps = 100
        manager = _make_manager(goal_stall_merge_steps=stall_steps)

        # Only 50 steps of history -- less than required 100
        goal_progress = {0: [0.5] * 50}

        ctx = LoRAMergeContext(goal_progress_history=goal_progress)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is False

    def test_no_merge_when_goal_progressing(self):
        stall_steps = 100
        manager = _make_manager(goal_stall_merge_steps=stall_steps)

        # Progress has meaningful variation (max - min = 0.99 >> 0.01)
        goal_progress = {0: [i / stall_steps for i in range(stall_steps)]}

        ctx = LoRAMergeContext(goal_progress_history=goal_progress)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is False

    def test_stall_detected_with_tiny_variation(self):
        """Variation of exactly 0.009 (< 0.01) should still count as a stall."""
        stall_steps = 100
        manager = _make_manager(goal_stall_merge_steps=stall_steps)

        # Values oscillate between 0.500 and 0.509 -- range is 0.009 < 0.01
        goal_progress = {0: [0.500 + 0.009 * (i % 2) for i in range(stall_steps)]}

        ctx = LoRAMergeContext(goal_progress_history=goal_progress)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is True

    def test_no_stall_at_boundary(self):
        """Variation of exactly 0.01 should NOT be detected as a stall (< 0.01 is required)."""
        stall_steps = 100
        manager = _make_manager(goal_stall_merge_steps=stall_steps)

        # Values range from 0.50 to 0.51 -- range is exactly 0.01 (not < 0.01)
        goal_progress = {0: [0.50] * 50 + [0.51] * 50}

        ctx = LoRAMergeContext(goal_progress_history=goal_progress)
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is False


class TestNoMergeBelowThresholds:
    """Verify no merge when all signals are within normal ranges."""

    def test_no_merge_all_signals_normal(self):
        """No signal triggers when everything is within healthy ranges."""
        manager = _make_manager(
            uncertainty_merge_threshold=0.7,
            uncertainty_merge_window=50,
            mood_merge_threshold=-0.3,
            mood_merge_window=50,
            goal_stall_merge_steps=100,
        )

        ctx = LoRAMergeContext(
            prediction_uncertainty_history=[0.3] * 50,  # all below 0.7
            mood_history=[0.5] * 50,                     # all above -0.3
            goal_progress_history={0: [i * 0.01 for i in range(100)]},  # progressing
        )

        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is False

    def test_no_merge_empty_context(self):
        """A LoRAMergeContext with all-None fields triggers nothing extra."""
        manager = _make_manager()
        ctx = LoRAMergeContext()  # all None
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is False

    def test_no_merge_empty_goal_dict(self):
        """An empty goal_progress_history dict triggers no stall detection."""
        manager = _make_manager()
        ctx = LoRAMergeContext(goal_progress_history={})
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)
        assert result is False

    def test_no_merge_insufficient_surprise_history_and_no_context(self):
        """Less than 100 surprise entries and no context -> no merge."""
        manager = _make_manager()
        result = manager.check_and_merge([0.5] * 50, merge_context=None)
        assert result is False

    def test_no_merge_none_context(self):
        """Passing merge_context=None skips all multi-signal checks."""
        manager = _make_manager()
        result = manager.check_and_merge(_FLAT_SURPRISE, merge_context=None)
        assert result is False


class TestMergeActuallyModifiesWeights:
    """Confirm that a triggered merge actually changes the base weights."""

    def test_uncertainty_merge_modifies_base_weights(self):
        field_cfg = FieldConfig(dim=32, n_layers=2)
        field = SemanticFieldProcessor(field_cfg)
        lora_cfg = LoRAConfig(
            rank=4,
            alpha=1.0,
            merge_threshold=2.0,
            uncertainty_merge_threshold=0.7,
            uncertainty_merge_window=10,
        )
        manager = OnlineLoRAManager(field, lora_cfg)

        # Set LoRA A/B to non-trivial values so merge has visible effect
        for layer in manager.lora_layers:
            layer.A.data.fill_(0.1)
            layer.B.data.fill_(0.1)

        # Snapshot base weights before merge
        base_weights_before = [
            layer.base.weight.data.clone() for layer in manager.lora_layers
        ]

        ctx = LoRAMergeContext(prediction_uncertainty_history=[0.8] * 10)
        merged = manager.check_and_merge(_FLAT_SURPRISE, merge_context=ctx)

        assert merged is True

        # Base weights should have changed
        for before, layer in zip(base_weights_before, manager.lora_layers):
            assert not torch.allclose(layer.base.weight.data, before)

        # B should be reinitialized to zeros
        for layer in manager.lora_layers:
            assert torch.allclose(layer.B.data, torch.zeros_like(layer.B.data))


class TestSignalPriority:
    """Verify that signal 1 (surprise ratio) takes precedence over multi-signal checks."""

    def test_surprise_ratio_fires_before_multi_signal(self):
        manager = _make_manager(
            merge_threshold=0.5,  # low threshold so surprise ratio WILL trigger
            uncertainty_merge_threshold=0.7,
            uncertainty_merge_window=50,
        )

        # Craft surprise_history where recent_mean / previous_mean > 0.5
        # previous 50 entries are low (0.1), recent 50 entries are high (1.0)
        # ratio = 1.0 / 0.1 = 10.0 >> 0.5
        surprise_history = [0.1] * 50 + [1.0] * 50

        # Also provide an uncertainty signal that would trigger on its own
        ctx = LoRAMergeContext(prediction_uncertainty_history=[0.8] * 50)

        # Should return True (from signal 1 -- surprise ratio)
        result = manager.check_and_merge(surprise_history, merge_context=ctx)
        assert result is True
