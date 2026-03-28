"""Tests for serialization module."""

import tempfile
import os

import torch
import pytest

import sfp
from sfp.storage.serialization import ManifoldCheckpoint


class TestManifoldCheckpoint:
    def test_save_load_field(self, tiny_field):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.pt")
            ManifoldCheckpoint.save(path, tiny_field)
            loaded, sp, meta = ManifoldCheckpoint.load(path)
            assert loaded.config.dim == tiny_field.config.dim
            assert sp is None
            assert meta == {}

    def test_save_load_with_metadata(self, tiny_field):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.pt")
            ManifoldCheckpoint.save(path, tiny_field, metadata={"version": 1})
            _, _, meta = ManifoldCheckpoint.load(path)
            assert meta["version"] == 1

    def test_save_load_streaming_processor(self, tiny_processor):
        x = torch.randn(256)
        tiny_processor.process(x)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.pt")
            ManifoldCheckpoint.save(path, tiny_processor.field, tiny_processor)
            loaded_field, loaded_sp, _ = ManifoldCheckpoint.load(path)
            assert loaded_sp is not None
            assert len(loaded_sp.surprise_history) == 1

    def test_weights_preserved(self, tiny_field):
        x = torch.randn(256)
        orig_out = tiny_field(x)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.pt")
            ManifoldCheckpoint.save(path, tiny_field)
            loaded, _, _ = ManifoldCheckpoint.load(path)
            loaded_out = loaded(x)

        assert torch.allclose(orig_out, loaded_out, atol=1e-6)
