"""Integration tests: end-to-end workflows."""

import torch
import pytest

import sfp
from sfp.comms.protocol import CommEndpoint, Message
from sfp.comms.sync import ManifoldSynchronizer
from sfp.storage.serialization import ManifoldCheckpoint
from sfp.storage.quantization import ManifoldQuantizer


class TestEndToEnd:
    def test_stream_then_query(self):
        """Full pipeline: create field, stream data, query attractor."""
        processor = sfp.create_field("tiny", lora=False, ewc=False, device="cpu")

        # Stream some data
        stream = [torch.randn(256) for _ in range(20)]
        results = processor.process_stream(stream)
        assert len(results) == 20

        # Query
        result = processor.query(torch.randn(256))
        assert result.point.shape == (256,)

    def test_lora_ewc_stream(self):
        """Stream with LoRA and EWC enabled."""
        processor = sfp.create_field("tiny", lora=True, ewc=True, device="cpu")
        stream = [torch.randn(256) for _ in range(30)]
        results = processor.process_stream(stream)
        assert len(results) == 30
        updates = sum(1 for r in results if r.updated)
        assert updates > 0

    def test_save_load_roundtrip(self):
        """Create, train, save, load, verify."""
        import tempfile, os

        processor = sfp.create_field("tiny", device="cpu")
        x = torch.randn(256)
        processor.process(x)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            ManifoldCheckpoint.save(path, processor.field, processor)
            loaded_field, loaded_sp, _ = ManifoldCheckpoint.load(path)
            assert loaded_sp is not None

            # Outputs should match
            orig = processor.field(x)
            loaded = loaded_field(x)
            assert torch.allclose(orig, loaded, atol=1e-5)

    def test_quantize_roundtrip_fidelity(self):
        """INT8 quantization preserves outputs."""
        field = sfp.SemanticFieldProcessor(sfp.FieldConfig.from_preset(sfp.FieldSize.TINY))
        x = torch.randn(10, 256)
        orig = field(x)

        qstate = ManifoldQuantizer.quantize(field)
        restored = ManifoldQuantizer.dequantize(qstate)
        rest_out = restored(x)

        mse = (orig - rest_out).pow(2).mean().item()
        assert mse < 0.01

    def test_communication_l1_roundtrip(self):
        """Two endpoints communicate via L1."""
        config = sfp.FieldConfig(dim=64, n_layers=2)
        field_a = sfp.SemanticFieldProcessor(config)
        field_b = sfp.SemanticFieldProcessor(config)
        # Copy weights so they share initial state
        field_b.load_state_dict(field_a.state_dict())

        ep_a = CommEndpoint(field_a, "A")
        ep_b = CommEndpoint(field_b, "B")

        concept = torch.randn(64)
        msg = ep_a.encode(concept, layer=sfp.CommLayer.L1_EMBEDDING)
        decoded = ep_b.decode(msg)
        mse = (concept - decoded).pow(2).mean().item()
        assert mse < 0.01

    def test_communication_l2_roundtrip(self):
        """Two endpoints communicate via L2 manifold coordinates."""
        config = sfp.FieldConfig(dim=64, n_layers=2)
        field_a = sfp.SemanticFieldProcessor(config)
        field_b = sfp.SemanticFieldProcessor(config)
        field_b.load_state_dict(field_a.state_dict())

        ep_a = CommEndpoint(field_a, "A")
        ep_b = CommEndpoint(field_b, "B")

        # Build shared codebook
        ep_a.build_codebook(n_probes=50, merge_radius=0.5)
        ep_b._l2.set_codebook(ep_a.codebook)

        concept = torch.randn(64)
        msg = ep_a.encode(concept, layer=sfp.CommLayer.L2_MANIFOLD_COORD)
        decoded = ep_b.decode(msg)

        raw_size = 64 * 4
        assert msg.size_bytes() < raw_size

    def test_byte_level_to_field(self):
        """Byte-level input -> projection -> field processing."""
        from sfp.input.bytelevel import ByteLevelEncoder
        from sfp.input.projection import DimensionalityProjection

        # Encode text to 256-dim patch vectors
        enc = ByteLevelEncoder()
        vectors = enc.encode(["hello world", "test data"])

        # Project to field dim
        proj = DimensionalityProjection(256, 64)
        projected = proj(vectors)
        assert projected.shape == (2, 64)

        # Feed to field
        config = sfp.FieldConfig(dim=64, n_layers=2)
        field = sfp.SemanticFieldProcessor(config)
        out = field(projected)
        assert out.shape == (2, 64)
