"""Tests for defense.input_validation — InputSanitizer and EmbeddingAnomalyDetector."""

import torch
import pytest

from sfp.defense.input_validation import EmbeddingAnomalyDetector, InputSanitizer


class TestInputSanitizer:
    def test_norm_clamping_reduces_large_vectors(self):
        san = InputSanitizer(max_norm=5.0, smoothing_sigma=0.0)
        big = torch.randn(64) * 100  # norm >> 5
        out = san.sanitize(big)
        assert out.norm().item() <= 5.0 + 1e-3

    def test_small_vectors_unchanged_without_smoothing(self):
        san = InputSanitizer(max_norm=100.0, smoothing_sigma=0.0)
        small = torch.randn(64) * 0.01
        out = san.sanitize(small)
        assert torch.allclose(out, small, atol=1e-6)

    def test_smoothing_adds_noise(self):
        san = InputSanitizer(max_norm=100.0, smoothing_sigma=0.5)
        x = torch.ones(64)
        out = san.sanitize(x)
        # With sigma=0.5, output should differ from input
        assert not torch.allclose(out, x, atol=1e-3)

    def test_batch_input(self):
        san = InputSanitizer(max_norm=3.0, smoothing_sigma=0.0)
        batch = torch.randn(8, 64) * 50
        out = san.sanitize(batch)
        norms = out.norm(dim=-1)
        assert (norms <= 3.0 + 1e-3).all()

    def test_record_provenance_returns_hash(self):
        san = InputSanitizer()
        h = san.record_provenance(b"hello", "text", "user")
        assert isinstance(h, bytes)
        assert len(h) == 32  # SHA-256

    def test_provenance_log_grows(self):
        san = InputSanitizer(provenance_log_size=5)
        for i in range(7):
            san.record_provenance(f"data_{i}".encode(), "text")
        assert san.provenance_log_size == 5  # capped at maxlen

    def test_different_inputs_different_hashes(self):
        san = InputSanitizer()
        h1 = san.record_provenance(b"aaa", "text")
        h2 = san.record_provenance(b"bbb", "text")
        assert h1 != h2


class TestEmbeddingAnomalyDetector:
    def test_returns_false_during_warmup(self):
        det = EmbeddingAnomalyDetector(d_model=32, warmup_samples=10)
        emb = torch.randn(32)
        for _ in range(5):
            det.update_statistics(emb, "test")
        assert not det.is_anomalous(emb * 100, "test")

    def test_normal_embeddings_not_anomalous(self):
        det = EmbeddingAnomalyDetector(d_model=16, warmup_samples=20, threshold=15.0)
        for _ in range(200):
            det.update_statistics(torch.randn(16), "test")
        # A normal draw should not be anomalous with a generous threshold
        assert not det.is_anomalous(torch.randn(16), "test")

    def test_extreme_embedding_is_anomalous(self):
        det = EmbeddingAnomalyDetector(d_model=16, warmup_samples=20, threshold=3.0)
        for _ in range(50):
            det.update_statistics(torch.randn(16) * 0.1, "test")
        # An extreme outlier
        outlier = torch.ones(16) * 100.0
        assert det.is_anomalous(outlier, "test")

    def test_unknown_modality_not_anomalous(self):
        det = EmbeddingAnomalyDetector(d_model=16)
        assert not det.is_anomalous(torch.randn(16), "unknown_modality")

    def test_get_statistics_empty(self):
        det = EmbeddingAnomalyDetector(d_model=16)
        stats = det.get_statistics("nothing")
        assert stats["count"] == 0

    def test_get_statistics_after_updates(self):
        det = EmbeddingAnomalyDetector(d_model=16, warmup_samples=5)
        for _ in range(10):
            det.update_statistics(torch.randn(16), "audio")
        stats = det.get_statistics("audio")
        assert stats["count"] == 10
        assert stats["warmup_complete"] is True

    def test_per_modality_isolation(self):
        det = EmbeddingAnomalyDetector(d_model=16, warmup_samples=5, threshold=3.0)
        # Train on modality A with small embeddings
        for _ in range(20):
            det.update_statistics(torch.randn(16) * 0.01, "mod_a")
        # Modality B has no data — should not flag
        assert not det.is_anomalous(torch.ones(16) * 100, "mod_b")
