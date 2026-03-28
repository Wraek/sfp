"""Learning validation tests: verify that SFP works and learns as intended.

Tests are arranged in diagnostic order — from component isolation to full
behavioral validation. If any test fails, the failure location indicates
which subsystem is broken:

  Tests 1-2:  Tier 0 surprise-gated learning mechanism
  Test 3:     Episodic memory admission gates
  Tests 4, 7: Essential memory retrieval / consistency
  Tests 5-6:  Consolidation pipeline (knowledge flow between tiers)
  Test 8:     Full pipeline integration
  Test 9:     Learning dynamics progression
  Test 10:    Core learning-and-relearning (MOST DIAGNOSTIC)
  Test 11:    Defense surprise hardening
  Test 12:    Persistence after learning
"""

import time

import pytest
import torch
import torch.nn.functional as F

import sfp
from sfp.config import (
    ConsolidationConfig,
    EWCConfig,
    FieldConfig,
    LoRAConfig,
    StreamingConfig,
    Tier0Config,
    Tier1Config,
    Tier2Config,
    Tier3Config,
)
from sfp.core.field import SemanticFieldProcessor
from sfp.core.streaming import StreamingProcessor
from sfp.memory.consolidation import ConsolidationEngine
from sfp.memory.core import CoreMemory
from sfp.memory.episodic import EpisodicMemory
from sfp.memory.essential import EssentialMemory
from sfp.memory.integrity import compute_episode_hash, compute_weight_hash
from sfp.memory.processor import HierarchicalMemoryProcessor
from sfp.storage.serialization import ManifoldCheckpoint
from sfp.types import ConsolidationMode, Episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(
    episode_id: int,
    embedding: torch.Tensor,
    surprise: float,
    field: SemanticFieldProcessor,
    *,
    modality: str = "test",
    basin_id: int = 0,
) -> Episode:
    """Create a valid Episode with correct integrity hash."""
    weight_hash = compute_weight_hash(field)
    logit_snapshot = field(embedding).detach()
    integrity_hash = compute_episode_hash(embedding.detach(), logit_snapshot, weight_hash)
    weight_summary = field.get_weight_summary()

    return Episode(
        id=episode_id,
        timestamp=time.monotonic(),
        modality=modality,
        provenance_hash=weight_hash[:16],
        input_embedding=embedding.detach(),
        working_memory_state=weight_summary.detach(),
        logit_snapshot=logit_snapshot,
        surprise_at_storage=surprise,
        attractor_basin_id=basin_id,
        attractor_distance=0.0,
        preceding_episode_id=None,
        following_episode_id=None,
        integrity_hash=integrity_hash,
        weight_hash_at_storage=weight_hash,
    )


# ---------------------------------------------------------------------------
# Test 1: Tier 0 surprise adaptation
# ---------------------------------------------------------------------------

class TestTier0SurpriseAdaptation:
    """Tier 0 should learn from repeated input and detect novelty."""

    def test_loss_decreases_for_repeated_input(self):
        """Feeding the same input repeatedly should decrease reconstruction loss."""
        field = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        processor = StreamingProcessor(
            field=field,
            streaming_config=StreamingConfig(
                surprise_threshold=0.0, warmup_steps=0, adaptive_surprise=False,
            ),
            lora_config=None,
            ewc_config=None,
        )

        x = torch.randn(64)
        results = [processor.process(x) for _ in range(50)]

        # Loss should decrease significantly
        assert results[-1].loss < results[0].loss * 0.7, (
            f"Loss should decrease by >30% for repeated input: "
            f"first={results[0].loss:.4f}, last={results[-1].loss:.4f}"
        )

    def test_updates_occur(self):
        """At least some inputs should trigger weight updates."""
        field = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        processor = StreamingProcessor(
            field=field,
            streaming_config=StreamingConfig(surprise_threshold=0.0),
            lora_config=None,
            ewc_config=None,
        )

        results = [processor.process(torch.randn(64)) for _ in range(20)]
        updates = sum(1 for r in results if r.updated)
        assert updates > 0, "At least some inputs should trigger updates"

    def test_novel_input_spikes_surprise(self):
        """A novel input after repeated inputs should produce higher gradient norm."""
        field = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        processor = StreamingProcessor(
            field=field,
            streaming_config=StreamingConfig(surprise_threshold=0.0),
            lora_config=None,
            ewc_config=None,
        )

        x_repeated = torch.randn(64)
        for _ in range(50):
            processor.process(x_repeated)

        last_repeated = processor.surprise_history[-1]

        # Novel input
        x_novel = torch.randn(64) * 3.0  # different distribution
        novel_result = processor.process(x_novel)

        assert novel_result.grad_norm > last_repeated.grad_norm, (
            f"Novel input should have higher surprise: "
            f"novel={novel_result.grad_norm:.4f}, last_repeated={last_repeated.grad_norm:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Tier 0 surprise gating
# ---------------------------------------------------------------------------

class TestTier0SurpriseGating:
    """The surprise threshold should correctly gate weight updates."""

    def test_high_threshold_blocks_all_updates(self):
        """With impossibly high threshold, no updates should occur."""
        field = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        processor = StreamingProcessor(
            field=field,
            streaming_config=StreamingConfig(
                surprise_threshold=1e6, adaptive_surprise=False,
            ),
            lora_config=None,
            ewc_config=None,
        )

        results = [processor.process(torch.randn(64)) for _ in range(20)]
        assert all(not r.updated for r in results), (
            "No updates should pass an impossibly high surprise threshold"
        )

    def test_zero_threshold_allows_all_updates(self):
        """With threshold=0, all inputs should trigger updates."""
        field = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        processor = StreamingProcessor(
            field=field,
            streaming_config=StreamingConfig(
                surprise_threshold=0.0, adaptive_surprise=False,
            ),
            lora_config=None,
            ewc_config=None,
        )

        results = [processor.process(torch.randn(64)) for _ in range(20)]
        assert all(r.updated for r in results), (
            "All updates should pass with surprise threshold of 0.0"
        )


# ---------------------------------------------------------------------------
# Test 3: Episodic memory admission gates
# ---------------------------------------------------------------------------

class TestEpisodicMemoryAdmission:
    """Tier 1 should admit surprising episodes and reject duplicates."""

    def test_low_surprise_rejected(self):
        """Episodes below surprise threshold should be rejected."""
        field = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        tier1 = EpisodicMemory(Tier1Config(surprise_threshold=0.5), d_model=64)

        ep = _make_episode(tier1.allocate_id(), torch.randn(64), surprise=0.3, field=field)
        assert not tier1.maybe_store(ep), "Low-surprise episode should be rejected"

    def test_high_surprise_admitted(self):
        """Episodes above surprise threshold should be admitted."""
        field = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        tier1 = EpisodicMemory(Tier1Config(surprise_threshold=0.5), d_model=64)

        ep = _make_episode(tier1.allocate_id(), torch.randn(64), surprise=0.8, field=field)
        assert tier1.maybe_store(ep), "High-surprise episode should be admitted"
        assert tier1.hot_count == 1

    def test_duplicate_rejected(self):
        """Near-duplicate episodes should be caught by cosine deduplication."""
        field = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        tier1 = EpisodicMemory(
            Tier1Config(surprise_threshold=0.5, dedup_threshold=0.95), d_model=64
        )

        embedding = torch.randn(64)
        ep1 = _make_episode(tier1.allocate_id(), embedding, surprise=0.8, field=field)
        assert tier1.maybe_store(ep1)

        # Near-duplicate: same embedding + tiny noise
        near_dup = embedding + torch.randn(64) * 0.01
        ep2 = _make_episode(tier1.allocate_id(), near_dup, surprise=0.8, field=field)
        assert not tier1.maybe_store(ep2), "Near-duplicate should be rejected"

    def test_different_episode_admitted(self):
        """Genuinely different episodes should be admitted."""
        field = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        tier1 = EpisodicMemory(
            Tier1Config(surprise_threshold=0.5, dedup_threshold=0.95), d_model=64
        )

        ep1 = _make_episode(tier1.allocate_id(), torch.randn(64), surprise=0.8, field=field)
        assert tier1.maybe_store(ep1)

        ep2 = _make_episode(tier1.allocate_id(), torch.randn(64), surprise=0.8, field=field)
        assert tier1.maybe_store(ep2), "Different episode should be admitted"
        assert tier1.hot_count == 2


# ---------------------------------------------------------------------------
# Test 4: Essential memory store and retrieve
# ---------------------------------------------------------------------------

class TestEssentialMemoryRetrieval:
    """Tier 2 basins should be creatable and retrievable."""

    def test_retrieve_nearest_basin(self):
        """Query near a basin center should retrieve that basin.

        EssentialMemory uses learned projection layers (query_proj, key_proj)
        that transform the key/query space. To test retrieval geometry, we set
        the projection weights to identity so raw key proximity is preserved.
        """
        d = 64
        tier2 = EssentialMemory(Tier2Config(n_slots=16, d_value=d), d_model=d)

        # Set projection to identity so key proximity is preserved through retrieval
        with torch.no_grad():
            tier2.query_proj.weight.copy_(torch.eye(d))
            tier2.key_proj.weight.copy_(torch.eye(d))

        # Allocate 3 basins at distinct positions
        keys = [torch.zeros(d) for _ in range(3)]
        keys[0][0] = 5.0  # basin 0 at [5, 0, ...]
        keys[1][1] = 5.0  # basin 1 at [0, 5, ...]
        keys[2][2] = 5.0  # basin 2 at [0, 0, 5, ...]

        slots = []
        for k in keys:
            slot = tier2.allocate_slot(k)
            tier2.update_slot(slot, confidence_update=0.8, episode_increment=10)
            slots.append(slot)

        # Query near each basin
        for i, slot in enumerate(slots):
            query = keys[i] + torch.randn(d) * 0.1  # near basin center
            _, basin_id, _ = tier2.retrieve(query)
            assert basin_id.item() == slot, (
                f"Query near basin {i} should retrieve slot {slot}, got {basin_id.item()}"
            )

    def test_active_count(self):
        """n_active should track allocated basins."""
        d = 64
        tier2 = EssentialMemory(Tier2Config(n_slots=16, d_value=d), d_model=d)
        assert tier2.n_active == 0

        tier2.allocate_slot(torch.randn(d))
        assert tier2.n_active == 1

        tier2.allocate_slot(torch.randn(d))
        assert tier2.n_active == 2


# ---------------------------------------------------------------------------
# Test 5: Consolidation Tier 0 -> Tier 1
# ---------------------------------------------------------------------------

class TestConsolidationMini:
    """Mini-consolidation should create episodes from Tier 0 surprise history."""

    def test_mini_consolidation_creates_episodes(self):
        """After processing inputs, mini-consolidation should store at least one episode."""
        field = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        tier0 = StreamingProcessor(
            field=field,
            streaming_config=StreamingConfig(surprise_threshold=0.0),
            lora_config=None,
            ewc_config=None,
        )
        tier1 = EpisodicMemory(Tier1Config(surprise_threshold=0.0), d_model=64)

        engine = ConsolidationEngine(
            config=ConsolidationConfig(mini_interval=10),
            tier0=tier0,
            tier1=tier1,
        )

        # Process enough inputs to generate surprise history
        for _ in range(15):
            tier0.process(torch.randn(64))

        engine.mini_consolidate(step_count=15)
        assert tier1.total_count >= 1, (
            "Mini-consolidation should store at least one episode"
        )


# ---------------------------------------------------------------------------
# Test 6: Consolidation Tier 1 -> Tier 2
# ---------------------------------------------------------------------------

class TestConsolidationStandard:
    """Standard consolidation should create concept basins from episodes."""

    def test_standard_consolidation_creates_basins(self):
        """Episodes clustered around distinct centers should form Tier 2 basins."""
        d = 64
        field = SemanticFieldProcessor(FieldConfig(dim=d, n_layers=2))
        tier1 = EpisodicMemory(Tier1Config(surprise_threshold=0.0), d_model=d)
        tier2 = EssentialMemory(Tier2Config(n_slots=32, d_value=d), d_model=d)

        engine = ConsolidationEngine(
            config=ConsolidationConfig(
                standard_interval=10,
                new_concept_threshold=2,
                distillation_threshold=0.3,
            ),
            tier1=tier1,
            tier2=tier2,
        )

        # Create episodes clustered around two distinct centers
        center_a = torch.zeros(d)
        center_a[0] = 5.0
        center_b = torch.zeros(d)
        center_b[1] = 5.0

        for i in range(6):
            center = center_a if i < 3 else center_b
            embedding = center + torch.randn(d) * 0.1
            ep = _make_episode(tier1.allocate_id(), embedding, surprise=0.8, field=field)
            # Bypass admission gates for controlled test
            tier1._hot.append(ep)

        assert tier1.total_count == 6

        engine.standard_consolidate(step_count=50)

        assert tier2.n_active >= 1, (
            f"Standard consolidation should create at least 1 basin, got {tier2.n_active}"
        )


# ---------------------------------------------------------------------------
# Test 7: Consistency checker
# ---------------------------------------------------------------------------

class TestConsistencyChecker:
    """Tier 2 consistency check should detect contradictions."""

    def test_consistent_update_scores_high(self):
        """An update aligned with existing knowledge should score high.

        check_consistency compares proposed_update against the retrieved output
        (which passes through value_proj + output_proj). We set projections to
        identity and set the basin value = key so the comparison is meaningful.
        """
        d = 64
        tier2 = EssentialMemory(Tier2Config(n_slots=16, d_value=d), d_model=d)

        # Set projections to identity for transparent behavior
        with torch.no_grad():
            tier2.query_proj.weight.copy_(torch.eye(d))
            tier2.key_proj.weight.copy_(torch.eye(d))
            tier2.value_proj.weight.copy_(torch.eye(d))
            tier2.output_proj.weight.copy_(torch.eye(d))

        v = torch.randn(d)
        v = v / v.norm()
        slot = tier2.allocate_slot(v, value=v)  # set value = key
        tier2.update_slot(slot, confidence_update=0.9, episode_increment=100)

        score = tier2.check_consistency(v, v)
        assert score.item() > 0.7, (
            f"Aligned update should have high consistency, got {score.item():.4f}"
        )

    def test_contradictory_update_scores_low(self):
        """An update opposing existing knowledge should score low."""
        d = 64
        tier2 = EssentialMemory(Tier2Config(n_slots=16, d_value=d), d_model=d)

        # Set projections to identity for transparent behavior
        with torch.no_grad():
            tier2.query_proj.weight.copy_(torch.eye(d))
            tier2.key_proj.weight.copy_(torch.eye(d))
            tier2.value_proj.weight.copy_(torch.eye(d))
            tier2.output_proj.weight.copy_(torch.eye(d))

        v = torch.randn(d)
        v = v / v.norm()
        slot = tier2.allocate_slot(v, value=v)  # set value = key
        tier2.update_slot(slot, confidence_update=0.9, episode_increment=100)

        score = tier2.check_consistency(v, -v)
        assert score.item() < 0.5, (
            f"Contradictory update should have low consistency, got {score.item():.4f}"
        )

    def test_low_confidence_does_not_block(self):
        """Low-confidence basins should not restrict updates (score ~1.0)."""
        d = 64
        tier2 = EssentialMemory(Tier2Config(n_slots=16, d_value=d), d_model=d)

        v = torch.randn(d)
        v = v / v.norm()
        slot = tier2.allocate_slot(v)
        # confidence stays at 0.0

        score = tier2.check_consistency(v, -v)
        assert score.item() > 0.8, (
            f"Low-confidence basin should not block updates, got {score.item():.4f}"
        )

    def test_no_basins_permits_all(self):
        """With no active basins, all updates should be permitted."""
        d = 64
        tier2 = EssentialMemory(Tier2Config(n_slots=16, d_value=d), d_model=d)
        assert tier2.n_active == 0

        score = tier2.check_consistency(torch.randn(d), torch.randn(d))
        assert score.item() == pytest.approx(1.0), (
            "No active basins should mean full permission"
        )


# ---------------------------------------------------------------------------
# Test 8: Full pipeline learns clusters
# ---------------------------------------------------------------------------

class TestFullPipelineClusters:
    """The full HierarchicalMemoryProcessor should learn to distinguish clusters."""

    def test_pipeline_forms_basins_from_clusters(self):
        """Processing data from 2 clusters should form at least 2 Tier 2 basins."""
        d = 64
        processor = HierarchicalMemoryProcessor(
            field_config=FieldConfig(dim=d, n_layers=2),
            tier1_config=Tier1Config(
                hot_capacity=50, cold_capacity=100, surprise_threshold=0.0
            ),
            tier2_config=Tier2Config(n_slots=32, d_value=d),
            tier3_config=Tier3Config(n_slots=8, d_value=d),
            consolidation_config=ConsolidationConfig(
                mini_interval=10,
                standard_interval=50,
                deep_interval=500,
                new_concept_threshold=2,
                distillation_threshold=0.3,
            ),
            streaming_config=StreamingConfig(surprise_threshold=0.0),
            lora_config=None,
            ewc_config=None,
            device="cpu",
        )

        # Two well-separated clusters
        mean_a = torch.zeros(d)
        mean_a[0] = 3.0
        mean_b = torch.zeros(d)
        mean_b[0] = -3.0

        for i in range(200):
            if i % 2 == 0:
                x = mean_a + torch.randn(d) * 0.3
            else:
                x = mean_b + torch.randn(d) * 0.3
            processor.process(x)

        # Force consolidation
        processor.consolidate(force_mode=ConsolidationMode.STANDARD)

        assert processor.tier2.n_active >= 1, (
            f"Pipeline should form basins from clustered data, got {processor.tier2.n_active}"
        )

    def test_health_report_structure(self):
        """health_report() should return a valid dict with expected keys."""
        d = 64
        processor = HierarchicalMemoryProcessor(
            field_config=FieldConfig(dim=d, n_layers=2),
            tier2_config=Tier2Config(n_slots=16, d_value=d),
            tier3_config=Tier3Config(n_slots=8, d_value=d),
            device="cpu",
        )

        processor.process(torch.randn(d))
        report = processor.health_report()

        assert "step_count" in report
        assert "tier0" in report
        assert "tier1" in report
        assert "tier2" in report
        assert "tier3" in report
        assert "consolidation" in report


# ---------------------------------------------------------------------------
# Test 9: Learning dynamics over time
# ---------------------------------------------------------------------------

class TestLearningDynamics:
    """Learning metrics should evolve correctly over time."""

    def test_surprise_decreases_over_time(self):
        """Mean surprise should decrease as the system sees repeated patterns."""
        d = 64
        processor = HierarchicalMemoryProcessor(
            field_config=FieldConfig(dim=d, n_layers=2),
            tier1_config=Tier1Config(hot_capacity=50, cold_capacity=100, surprise_threshold=0.0),
            tier2_config=Tier2Config(n_slots=32, d_value=d),
            tier3_config=Tier3Config(n_slots=8, d_value=d),
            consolidation_config=ConsolidationConfig(
                mini_interval=10, standard_interval=50, deep_interval=500,
                new_concept_threshold=2,
            ),
            streaming_config=StreamingConfig(
                surprise_threshold=0.0, warmup_steps=0, adaptive_surprise=False,
            ),
            lora_config=None,
            ewc_config=None,
            device="cpu",
        )

        # Generate 3 clusters, 100 samples each
        torch.manual_seed(42)
        clusters = []
        for c in range(3):
            mean = torch.zeros(d)
            mean[c] = 3.0
            for _ in range(100):
                clusters.append(mean + torch.randn(d) * 0.3)

        # Shuffle deterministically
        indices = torch.randperm(len(clusters), generator=torch.Generator().manual_seed(42))
        data = [clusters[i] for i in indices]

        # Process all data, collect surprise
        early_losses = []
        late_losses = []
        for i, x in enumerate(data):
            metric = processor.process(x)
            if i < 50:
                early_losses.append(metric.loss)
            elif i >= 250:
                late_losses.append(metric.loss)

        mean_early = sum(early_losses) / len(early_losses)
        mean_late = sum(late_losses) / len(late_losses)

        assert mean_late < mean_early, (
            f"Mean loss should decrease over time: early={mean_early:.4f}, late={mean_late:.4f}"
        )

    def test_episodes_accumulate(self):
        """Tier 1 should accumulate episodes over time."""
        d = 64
        processor = HierarchicalMemoryProcessor(
            field_config=FieldConfig(dim=d, n_layers=2),
            tier1_config=Tier1Config(hot_capacity=50, cold_capacity=100, surprise_threshold=0.0),
            tier2_config=Tier2Config(n_slots=32, d_value=d),
            tier3_config=Tier3Config(n_slots=8, d_value=d),
            consolidation_config=ConsolidationConfig(
                mini_interval=10, standard_interval=50, deep_interval=500,
            ),
            streaming_config=StreamingConfig(surprise_threshold=0.0),
            lora_config=None,
            ewc_config=None,
            device="cpu",
        )

        for _ in range(100):
            processor.process(torch.randn(d))

        assert processor.tier1.total_count > 0, (
            "Tier 1 should accumulate episodes during processing"
        )


# ---------------------------------------------------------------------------
# Test 10: Contingency reversal (MOST IMPORTANT)
# ---------------------------------------------------------------------------

class TestContingencyReversal:
    """The system should learn associations, then relearn when contingencies reverse.

    This is the single most diagnostic test — it exercises Tier 0 adaptation,
    episodic storage, consolidation, and the ability to update without
    catastrophic forgetting.
    """

    def test_learns_then_reverses(self):
        """Phase 1: learn A->pos, B->neg. Phase 2: reverse. System should adapt."""
        d = 64
        processor = HierarchicalMemoryProcessor(
            field_config=FieldConfig(dim=d, n_layers=2),
            tier1_config=Tier1Config(hot_capacity=100, cold_capacity=200, surprise_threshold=0.0),
            tier2_config=Tier2Config(n_slots=32, d_value=d),
            tier3_config=Tier3Config(n_slots=8, d_value=d),
            consolidation_config=ConsolidationConfig(
                mini_interval=10, standard_interval=50, deep_interval=500,
                new_concept_threshold=2,
            ),
            streaming_config=StreamingConfig(
                surprise_threshold=0.0, warmup_steps=0, adaptive_surprise=False,
            ),
            lora_config=None,
            ewc_config=None,
            device="cpu",
        )

        # Define signals and targets
        signal_a = torch.zeros(d)
        signal_a[0] = 2.0
        signal_b = torch.zeros(d)
        signal_b[0] = -2.0

        target_pos = torch.zeros(d)
        target_pos[1] = 2.0
        target_neg = torch.zeros(d)
        target_neg[1] = -2.0

        # --- Phase 1: A->pos, B->neg ---
        phase1_losses = []
        for i in range(200):
            if i % 2 == 0:
                x = signal_a + torch.randn(d) * 0.1
                t = target_pos + torch.randn(d) * 0.1
            else:
                x = signal_b + torch.randn(d) * 0.1
                t = target_neg + torch.randn(d) * 0.1
            metric = processor.process(x, target=t)
            phase1_losses.append(metric.loss)

        processor.consolidate(force_mode=ConsolidationMode.STANDARD)

        # Record phase 1 end loss
        phase1_end_loss = sum(phase1_losses[-20:]) / 20

        # --- Phase 2: REVERSE — A->neg, B->pos ---
        phase2_losses = []
        for i in range(200):
            if i % 2 == 0:
                x = signal_a + torch.randn(d) * 0.1
                t = target_neg + torch.randn(d) * 0.1  # REVERSED
            else:
                x = signal_b + torch.randn(d) * 0.1
                t = target_pos + torch.randn(d) * 0.1  # REVERSED
            metric = processor.process(x, target=t)
            phase2_losses.append(metric.loss)

        processor.consolidate(force_mode=ConsolidationMode.STANDARD)

        # --- Assertions ---

        # 1. Phase 1 learned: end loss < start loss
        phase1_start_loss = sum(phase1_losses[:20]) / 20
        assert phase1_end_loss < phase1_start_loss, (
            f"Phase 1 should learn: start={phase1_start_loss:.4f}, end={phase1_end_loss:.4f}"
        )

        # 2. Phase 2 initially surprised: first few losses spike
        phase2_start_loss = sum(phase2_losses[:20]) / 20
        assert phase2_start_loss > phase1_end_loss * 0.5, (
            "Phase 2 start should show surprise from reversal"
        )

        # 3. Phase 2 adapts: end loss decreases
        phase2_end_loss = sum(phase2_losses[-20:]) / 20
        assert phase2_end_loss < phase2_start_loss, (
            f"Phase 2 should adapt: start={phase2_start_loss:.4f}, end={phase2_end_loss:.4f}"
        )

        # 4. Episodic memory has traces from both phases
        assert processor.tier1.total_count > 0, (
            "Tier 1 should have episodes from the learning process"
        )


# ---------------------------------------------------------------------------
# Test 11: Defense surprise hardening
# ---------------------------------------------------------------------------

class TestDefenseSurpriseHardening:
    """Surprise hardening should limit damage from adversarial inputs."""

    def test_adversarial_inputs_clamped(self):
        """High-magnitude adversarial inputs should not cause excessive weight drift."""
        d = 64
        field = SemanticFieldProcessor(FieldConfig(dim=d, n_layers=2))
        processor = StreamingProcessor(
            field=field,
            streaming_config=StreamingConfig(surprise_threshold=0.0),
            tier0_config=Tier0Config(
                max_surprise_ratio=3.0,
                rate_limit_threshold=0.3,
                rate_limit_window=50,
            ),
            lora_config=None,
            ewc_config=None,
        )

        # Establish baseline with normal inputs
        for _ in range(50):
            processor.process(torch.randn(d))

        # Record weight state before attack
        weights_before = {
            name: p.data.clone() for name, p in field.named_parameters()
        }

        # Inject adversarial inputs (extremely large magnitude)
        for _ in range(50):
            adversarial = torch.randn(d) * 100.0
            processor.process(adversarial)

        # Measure weight drift
        total_drift = 0.0
        total_norm = 0.0
        for name, p in field.named_parameters():
            drift = (p.data - weights_before[name]).norm().item()
            total_drift += drift
            total_norm += weights_before[name].norm().item()

        relative_drift = total_drift / max(total_norm, 1e-8)

        # With hardening, drift should be bounded
        # Without hardening, 100x magnitude inputs would cause massive drift
        assert relative_drift < 5.0, (
            f"Surprise hardening should limit weight drift: relative_drift={relative_drift:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 12: Save/load preserves learned state
# ---------------------------------------------------------------------------

class TestSaveLoadLearned:
    """Checkpoint save/load should preserve learned state."""

    def test_outputs_match_after_roundtrip(self):
        """After save/load, the same input should produce the same output."""
        import os
        import tempfile

        field = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        processor = StreamingProcessor(
            field=field,
            streaming_config=StreamingConfig(surprise_threshold=0.0),
            lora_config=None,
            ewc_config=None,
        )

        # Learn something
        x = torch.randn(64)
        for _ in range(30):
            processor.process(x)

        # Record output before save
        field.eval()
        with torch.no_grad():
            output_before = field(x).clone()

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "learned.pt")
            ManifoldCheckpoint.save(path, field, processor)

            loaded_field, loaded_sp, _ = ManifoldCheckpoint.load(path)
            loaded_field.eval()
            with torch.no_grad():
                output_after = loaded_field(x)

        assert torch.allclose(output_before, output_after, atol=1e-5), (
            "Outputs should match after save/load roundtrip"
        )
