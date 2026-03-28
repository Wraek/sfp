"""Tests for LoRA components."""

import torch
import pytest

from sfp.config import LoRAConfig
from sfp.core.lora import LoRALinear, OnlineLoRAManager


class TestLoRALinear:
    def test_forward(self):
        base = torch.nn.Linear(64, 64)
        config = LoRAConfig(rank=4, alpha=1.0)
        lora = LoRALinear(base, config)
        x = torch.randn(8, 64)
        out = lora(x)
        assert out.shape == (8, 64)

    def test_base_frozen(self):
        base = torch.nn.Linear(64, 64)
        config = LoRAConfig(rank=4)
        lora = LoRALinear(base, config)
        assert not lora.base.weight.requires_grad
        assert lora.A.requires_grad
        assert lora.B.requires_grad

    def test_merge_and_reinit(self):
        base = torch.nn.Linear(64, 64)
        config = LoRAConfig(rank=4)
        lora = LoRALinear(base, config)

        # Store original base weight
        orig_weight = base.weight.data.clone()

        # Make A/B nonzero
        lora.A.data.fill_(0.1)
        lora.B.data.fill_(0.1)

        lora.merge_and_reinit()

        # Base weight should have changed
        assert not torch.allclose(base.weight.data, orig_weight)
        # B should be zeros after reinit
        assert torch.allclose(lora.B.data, torch.zeros_like(lora.B.data))

    def test_lora_param_count(self):
        base = torch.nn.Linear(64, 64)
        config = LoRAConfig(rank=4)
        lora = LoRALinear(base, config)
        assert lora.lora_param_count == 64 * 4 + 4 * 64


class TestOnlineLoRAManager:
    def test_wrap_field(self, tiny_field):
        config = LoRAConfig(rank=4)
        manager = OnlineLoRAManager(tiny_field, config)
        assert len(manager.lora_layers) == tiny_field.config.n_layers

    def test_trainable_parameters(self, tiny_field):
        config = LoRAConfig(rank=4)
        manager = OnlineLoRAManager(tiny_field, config)
        params = list(manager.trainable_parameters())
        # 2 params (A, B) per layer
        assert len(params) == 2 * len(manager.lora_layers)

    def test_total_lora_params(self, tiny_field):
        config = LoRAConfig(rank=4)
        manager = OnlineLoRAManager(tiny_field, config)
        assert manager.total_lora_params > 0

    def test_check_and_merge_not_enough_history(self, tiny_field):
        config = LoRAConfig(rank=4, merge_threshold=0.5)
        manager = OnlineLoRAManager(tiny_field, config)
        # Less than 100 history points -> no merge
        assert not manager.check_and_merge([0.1] * 50)

    def test_merge_all(self, tiny_field):
        config = LoRAConfig(rank=4)
        manager = OnlineLoRAManager(tiny_field, config)
        # Should not raise
        manager.merge_all()
