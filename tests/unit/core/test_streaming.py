"""Tests for StreamingProcessor."""

import torch
import pytest

import sfp
from sfp.core.streaming import StreamingProcessor


class TestStreamingProcessor:
    def test_process_single(self, tiny_processor, random_input):
        metric = tiny_processor.process(random_input)
        assert metric.grad_norm >= 0
        assert metric.loss >= 0
        assert isinstance(metric.updated, bool)

    def test_process_with_target(self, tiny_processor, random_input):
        target = torch.randn_like(random_input)
        metric = tiny_processor.process(random_input, target=target)
        assert metric.loss >= 0

    def test_process_batch(self, tiny_processor, random_batch):
        metric = tiny_processor.process(random_batch)
        assert metric.grad_norm >= 0

    def test_process_stream(self, tiny_processor):
        stream = [torch.randn(256) for _ in range(5)]
        results = tiny_processor.process_stream(stream)
        assert len(results) == 5
        for r in results:
            assert r.grad_norm >= 0

    def test_process_stream_callback(self, tiny_processor):
        collected = []
        stream = [torch.randn(256) for _ in range(3)]
        tiny_processor.process_stream(stream, callback=collected.append)
        assert len(collected) == 3

    def test_surprise_history(self, tiny_processor, random_input):
        assert len(tiny_processor.surprise_history) == 0
        tiny_processor.process(random_input)
        assert len(tiny_processor.surprise_history) == 1

    def test_query(self, tiny_processor, random_input):
        result = tiny_processor.query(random_input)
        assert hasattr(result, "point")
        assert hasattr(result, "converged")

    def test_lora_manager_access(self, tiny_processor):
        assert tiny_processor.lora_manager is not None

    def test_ewc_strategy_access(self, tiny_processor):
        assert tiny_processor.ewc_strategy is not None

    def test_no_lora_no_ewc(self, tiny_field):
        sp = StreamingProcessor(tiny_field)
        assert sp.lora_manager is None
        assert sp.ewc_strategy is None
        metric = sp.process(torch.randn(256))
        assert metric.grad_norm >= 0

    def test_cosine_loss(self, tiny_field):
        sp = StreamingProcessor(
            tiny_field,
            streaming_config=sfp.StreamingConfig(loss_fn="cosine"),
        )
        metric = sp.process(torch.randn(256))
        assert metric.loss >= 0

    def test_reset_optimizer(self, tiny_processor):
        tiny_processor.reset_optimizer()
        metric = tiny_processor.process(torch.randn(256))
        assert metric.grad_norm >= 0
