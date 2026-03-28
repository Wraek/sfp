"""Tests for byte-level encoder."""

import torch
import pytest

from sfp.input.bytelevel import (
    ByteLevelConfig,
    ByteLevelEncoder,
    BytePatcher,
    EntropyEstimator,
)


class TestEntropyEstimator:
    def test_fit_and_estimate(self):
        est = EntropyEstimator(ngram_order=2)
        est.fit(b"hello hello hello")
        h = est.estimate_entropy(b"ll")
        assert 0 <= h <= 8.0

    def test_unseen_context_returns_max(self):
        est = EntropyEstimator(ngram_order=2)
        est.fit(b"hello")
        h = est.estimate_entropy(b"zz")
        assert h == 8.0

    def test_fit_default(self):
        est = EntropyEstimator.fit_default()
        h = est.estimate_entropy(b"the")
        assert 0 <= h <= 8.0


class TestBytePatcher:
    def test_patches_nonempty(self):
        est = EntropyEstimator.fit_default()
        config = ByteLevelConfig(min_patch_size=2, max_patch_size=16)
        patcher = BytePatcher(est, config)
        patches = patcher.patch(b"hello world, this is a test string")
        assert len(patches) >= 1
        # Reconstruct
        assert b"".join(patches) == b"hello world, this is a test string"

    def test_empty_input(self):
        est = EntropyEstimator.fit_default()
        patcher = BytePatcher(est, ByteLevelConfig())
        assert patcher.patch(b"") == []

    def test_max_patch_size_respected(self):
        est = EntropyEstimator.fit_default()
        config = ByteLevelConfig(max_patch_size=8)
        patcher = BytePatcher(est, config)
        patches = patcher.patch(b"a" * 100)
        for p in patches:
            assert len(p) <= 8


class TestByteLevelEncoder:
    def test_encode_strings(self):
        enc = ByteLevelEncoder()
        result = enc.encode(["hello world", "test"])
        assert result.shape == (2, 256)

    def test_encode_bytes(self):
        enc = ByteLevelEncoder()
        result = enc.encode([b"\x00\x01\x02"])
        assert result.shape == (1, 256)

    def test_encode_patches(self):
        enc = ByteLevelEncoder()
        result = enc.encode_patches(["hello world"])
        assert len(result) == 1
        assert result[0].dim() == 2
        assert result[0].shape[1] == 256

    def test_custom_config(self):
        config = ByteLevelConfig(patch_dim=128, max_patch_size=16)
        enc = ByteLevelEncoder(config)
        result = enc.encode(["test"])
        assert result.shape == (1, 128)

    def test_fit_entropy_model(self):
        enc = ByteLevelEncoder()
        enc.fit_entropy_model(b"custom domain data " * 50)
        result = enc.encode(["test"])
        assert result.shape == (1, 256)
