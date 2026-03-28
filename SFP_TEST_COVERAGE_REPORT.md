# SFP Test Coverage Report

Updated: 2026-02-20

## Overview

- **Total source modules:** 36
- **Modules with tests:** 36 (100%)
- **Total test functions:** 342 (all passing)
- **Test runtime:** ~2.5 min on CPU (fast suite ~9s, sustainability suite ~2.3 min)
- **Test framework:** pytest with benchmark and coverage plugins
- **Sustainability tests:** 16 long-running tests (2,000–12,000 steps) marked `@pytest.mark.slow`

---

## 1. Test Coverage by Category

### Unit Tests — Core (`tests/unit/core/`)

**test_field.py** (13 tests)
| Test | What it validates |
|------|-------------------|
| `TestFieldConfig::test_from_preset_tiny` | TINY preset: 256 dim, 4 layers |
| `TestFieldConfig::test_from_preset_small` | SMALL preset: 512 dim, 6 layers |
| `TestFieldConfig::test_from_preset_medium` | MEDIUM preset: 1024 dim, 8 layers |
| `TestFieldConfig::test_from_preset_large` | LARGE preset: 2048 dim, 8 layers |
| `TestSemanticFieldProcessor::test_forward_single` | Single input forward pass |
| `TestSemanticFieldProcessor::test_forward_batch` | Batch forward pass (8 samples) |
| `TestSemanticFieldProcessor::test_param_count` | Parameter counting accuracy |
| `TestSemanticFieldProcessor::test_memory_bytes` | FP32 memory estimation |
| `TestSemanticFieldProcessor::test_memory_bytes_fp16` | FP16 memory estimation |
| `TestSemanticFieldProcessor::test_linear_layers` | Linear layer extraction |
| `TestSemanticFieldProcessor::test_jacobian` | Jacobian computation |
| `TestSemanticFieldProcessor::test_residual_field` | Residual connections |
| `TestSemanticFieldProcessor::test_relu_activation` | ReLU activation |
| `TestSemanticFieldProcessor::test_no_layernorm` | Optional LayerNorm removal |

**test_attractors.py** (6 tests)
| Test | What it validates |
|------|-------------------|
| `TestAttractorQuery::test_query_single` | Single point attractor query |
| `TestAttractorQuery::test_query_with_trajectory` | Trajectory recording |
| `TestAttractorQuery::test_query_batch` | Batch queries (8 points) |
| `TestAttractorQuery::test_discover_attractors` | Automated attractor discovery (50 probes) |
| `TestAttractorQuery::test_map_basins` | Basin of attraction mapping |
| `TestAttractorQuery::test_convergence_tolerance` | Convergence with loose tolerance |

**test_streaming.py** (12 tests)
| Test | What it validates |
|------|-------------------|
| `TestStreamingProcessor::test_process_single` | Single sample streaming update |
| `TestStreamingProcessor::test_process_with_target` | Supervised learning with target |
| `TestStreamingProcessor::test_process_batch` | Batch processing (8 samples) |
| `TestStreamingProcessor::test_process_stream` | Sequential stream processing |
| `TestStreamingProcessor::test_process_stream_callback` | Callback invocation |
| `TestStreamingProcessor::test_surprise_history` | Surprise metric accumulation |
| `TestStreamingProcessor::test_query` | Query integration |
| `TestStreamingProcessor::test_lora_manager_access` | LoRA manager initialization |
| `TestStreamingProcessor::test_ewc_strategy_access` | EWC strategy initialization |
| `TestStreamingProcessor::test_no_lora_no_ewc` | Optional LoRA/EWC disabling |
| `TestStreamingProcessor::test_cosine_loss` | Cosine loss function |
| `TestStreamingProcessor::test_reset_optimizer` | Optimizer reset |

**test_lora.py** (8 tests)
| Test | What it validates |
|------|-------------------|
| `TestLoRALinear::test_forward` | LoRA linear layer forward pass |
| `TestLoRALinear::test_base_frozen` | Base weights frozen, LoRA trainable |
| `TestLoRALinear::test_merge_and_reinit` | Merge and reinitialize |
| `TestLoRALinear::test_lora_param_count` | Parameter counting (rank=4) |
| `TestOnlineLoRAManager::test_wrap_field` | Wrap field with LoRA |
| `TestOnlineLoRAManager::test_trainable_parameters` | Trainable param enumeration |
| `TestOnlineLoRAManager::test_total_lora_params` | Total LoRA param aggregation |
| `TestOnlineLoRAManager::test_check_and_merge_not_enough_history` | Merge threshold (100 min) |
| `TestOnlineLoRAManager::test_merge_all` | Merge all LoRA layers |

**test_forgetting.py** (5 tests)
| Test | What it validates |
|------|-------------------|
| `TestWeightDecayStrategy::test_penalty_is_zero` | Weight decay baseline penalty |
| `TestWeightDecayStrategy::test_update_importance_noop` | Update importance no-op |
| `TestEWCStrategy::test_initial_penalty_is_zero` | EWC initial penalty |
| `TestEWCStrategy::test_penalty_increases_after_update` | Penalty increases post-learning |
| `TestEWCStrategy::test_update_anchors` | Anchor update resets penalty |

### Unit Tests — Architecture (`tests/unit/architecture/`)

**test_backbone.py** (8 tests)
| Test | What it validates |
|------|-------------------|
| `TestBackboneTransformer::test_forward_shape` | Output shape (2, 8, 64) |
| `TestBackboneTransformer::test_single_batch` | Single batch forward |
| `TestBackboneTransformer::test_single_token` | Single token forward |
| `TestBackboneTransformer::test_output_differs_from_input` | Non-identity transform |
| `TestBackboneTransformer::test_param_count` | Parameter counting |
| `TestBackboneTransformer::test_memory_bytes` | FP32 memory estimation |
| `TestBackboneTransformer::test_gradient_flow` | Gradient backpropagation |
| `TestBackboneTransformer::test_different_inputs_different_outputs` | Input sensitivity |

**test_perceiver.py** (7 tests)
| Test | What it validates |
|------|-------------------|
| `TestPerceiverIO::test_forward_shape` | Output shape (2, 8, 64) |
| `TestPerceiverIO::test_variable_input_length` | Variable-length input handling |
| `TestPerceiverIO::test_decode` | Decode from latents via output cross-attention |
| `TestPerceiverIO::test_param_count` | Parameter counting |
| `TestPerceiverIO::test_memory_bytes` | FP32 memory estimation |
| `TestPerceiverIO::test_gradient_flow` | Gradient backpropagation |
| `TestPerceiverIO::test_different_inputs_different_latents` | Input sensitivity |

### Unit Tests — Defense (`tests/unit/defense/`)

**test_input_validation.py** (14 tests)
| Test | What it validates |
|------|-------------------|
| `TestInputSanitizer::test_norm_clamping_reduces_large_vectors` | L2 norm clamping |
| `TestInputSanitizer::test_small_vectors_unchanged_without_smoothing` | Passthrough for small vectors |
| `TestInputSanitizer::test_smoothing_adds_noise` | Gaussian smoothing defense |
| `TestInputSanitizer::test_batch_input` | Batch norm clamping |
| `TestInputSanitizer::test_record_provenance_returns_hash` | SHA-256 provenance hash |
| `TestInputSanitizer::test_provenance_log_grows` | Bounded provenance log |
| `TestInputSanitizer::test_different_inputs_different_hashes` | Hash uniqueness |
| `TestEmbeddingAnomalyDetector::test_returns_false_during_warmup` | Warmup gate |
| `TestEmbeddingAnomalyDetector::test_normal_embeddings_not_anomalous` | Normal distribution acceptance |
| `TestEmbeddingAnomalyDetector::test_extreme_embedding_is_anomalous` | Outlier detection via Mahalanobis |
| `TestEmbeddingAnomalyDetector::test_unknown_modality_not_anomalous` | Unknown modality passthrough |
| `TestEmbeddingAnomalyDetector::test_get_statistics_empty` | Empty statistics |
| `TestEmbeddingAnomalyDetector::test_get_statistics_after_updates` | Statistics after training |
| `TestEmbeddingAnomalyDetector::test_per_modality_isolation` | Cross-modality isolation |

**test_gradient_bounds.py** (9 tests)
| Test | What it validates |
|------|-------------------|
| `TestAdaptiveGradientClipper::test_ema_initialized_on_first_call` | EMA initialization |
| `TestAdaptiveGradientClipper::test_normal_gradients_not_clipped` | Normal gradient passthrough |
| `TestAdaptiveGradientClipper::test_spike_is_clipped` | Spike detection and clipping |
| `TestAdaptiveGradientClipper::test_no_grad_params_skipped` | No-grad parameter handling |
| `TestUpdateBudget::test_within_budget_no_rollback` | Normal update passthrough |
| `TestUpdateBudget::test_exceeds_budget_triggers_rollback` | Budget violation rollback |
| `TestUpdateBudget::test_scale_back_to_budget` | Proportional rollback |
| `TestUpdateBudget::test_snapshot_updates` | Weight snapshot management |
| `TestUpdateBudget::test_rollback_restores_weights` | Full weight restoration |

**test_anchor_verification.py** (5 tests)
| Test | What it validates |
|------|-------------------|
| `TestAnchorVerifier::test_no_violations_with_correct_basins` | Correct basin assignments |
| `TestAnchorVerifier::test_basin_shift_detected` | Basin shift detection |
| `TestAnchorVerifier::test_no_active_basins_violation` | Empty basin error |
| `TestAnchorVerifier::test_n_anchors_property` | Anchor count property |
| `TestAnchorVerifier::test_pairwise_distance_stability` | Pairwise distance monitoring |

**test_topology_monitor.py** (7 tests)
| Test | What it validates |
|------|-------------------|
| `TestManifoldIntegrityMonitor::test_distinct_basins_no_alerts` | Normal state produces no alerts |
| `TestManifoldIntegrityMonitor::test_merge_alert_similar_basins` | Basin merge detection |
| `TestManifoldIntegrityMonitor::test_component_change_detected` | Component count changes |
| `TestManifoldIntegrityMonitor::test_single_basin` | Single basin edge case |
| `TestManifoldIntegrityMonitor::test_reasoning_loops_empty_graph` | Empty graph cycle check |
| `TestManifoldIntegrityMonitor::test_reasoning_loops_cycle` | DFS cycle detection |
| `TestManifoldIntegrityMonitor::test_reasoning_loops_acyclic` | Acyclic graph verification |

**test_surprise_hardening.py** (8 tests)
| Test | What it validates |
|------|-------------------|
| `TestSurpriseHardener::test_surprise_ratio_clamped` | Max ratio clamping (3.0x) |
| `TestSurpriseHardener::test_normal_surprise_passes_through` | Normal passthrough |
| `TestSurpriseHardener::test_rate_limiting_activates` | Burst detection activation |
| `TestSurpriseHardener::test_rate_limiting_deactivates` | Rate limit recovery |
| `TestSurpriseHardener::test_dual_path_reduces_surprise_on_disagreement` | Dual-path verification |
| `TestSurpriseHardener::test_clip_gradients_reduces_outliers` | Adaptive gradient clipping |
| `TestSurpriseHardener::test_reset_clears_state` | State reset |

### Unit Tests — Memory (`tests/unit/memory/`)

**test_core_memory.py** (8 tests)
| Test | What it validates |
|------|-------------------|
| `TestCoreMemory::test_retrieve_empty_returns_zeros` | Empty retrieval |
| `TestCoreMemory::test_write_slot_and_retrieve` | Write + retrieve roundtrip |
| `TestCoreMemory::test_integrity_verification_passes` | SHA-256 integrity pass |
| `TestCoreMemory::test_integrity_verification_fails_on_tamper` | Tamper detection |
| `TestCoreMemory::test_promote_rejected_low_confidence` | Low confidence rejection |
| `TestCoreMemory::test_promote_rejected_by_event_system` | Event system denial |
| `TestCoreMemory::test_eviction_when_full` | Lowest-confidence eviction |
| `TestCoreMemory::test_batch_retrieve` | Batch retrieval |

**test_events.py** (12 tests)
| Test | What it validates |
|------|-------------------|
| `TestPromotionEventEmitter::test_default_deny` | Default deny policy |
| `TestPromotionEventEmitter::test_default_approve` | Default approve policy |
| `TestPromotionEventEmitter::test_handler_approve` | Handler approval |
| `TestPromotionEventEmitter::test_handler_deny` | Handler denial |
| `TestPromotionEventEmitter::test_handler_defer` | Handler deferral |
| `TestPromotionEventEmitter::test_handler_priority_order` | Priority-ordered handler chain |
| `TestPromotionEventEmitter::test_unregister` | Handler unregistration |
| `TestCriteriaAuthorizationHandler::test_all_criteria_met_approves` | Full criteria approval |
| `TestCriteriaAuthorizationHandler::test_low_confidence_defers` | Low confidence deferral |
| `TestCriteriaAuthorizationHandler::test_low_episodes_defers` | Low episode count deferral |
| `TestCriteriaAuthorizationHandler::test_low_modalities_defers` | Low modality count deferral |
| `TestCriteriaAuthorizationHandler::test_young_defers` | Age-based deferral |
| `TestCriteriaAuthorizationHandler::test_integrated_with_emitter` | Full emitter integration |

**test_replay.py** (15 tests)
| Test | What it validates |
|------|-------------------|
| `TestGenerativeReplayShouldGenerate::test_disabled_during_warmup` | Warmup phase gating |
| `TestGenerativeReplayShouldGenerate::test_enabled_after_warmup` | Post-warmup activation |
| `TestGenerativeReplayShouldGenerate::test_idle_daydreaming` | Idle phase generation |
| `TestGenerativeReplayValidation::test_validate_nan_rejected` | NaN rejection |
| `TestGenerativeReplayValidation::test_validate_inf_rejected` | Inf rejection |
| `TestGenerativeReplayValidation::test_validate_zero_norm_rejected` | Zero-norm rejection |
| `TestGenerativeReplayValidation::test_validate_normal_embedding_passes` | Normal embedding acceptance |
| `TestGenerativeReplayDriftMonitoring::test_drift_not_excessive_initially` | Initial drift baseline |
| `TestGenerativeReplayDriftMonitoring::test_excessive_drift_detection` | Excessive drift alerting |
| `TestGenerativeReplayDriftMonitoring::test_excluded_basins` | Excluded basin tracking |
| `TestGenerativeReplayStats::test_stats_initial` | Initial statistics |
| `TestGenerativeReplayStats::test_reset_stats` | Statistics reset |
| `TestGenerativeReplayStats::test_record_inference` | Inference recording |
| `TestGenerativeReplayInterpolation::test_interpolation_needs_min_basins` | Min basin requirement |
| `TestGenerativeReplayInterpolation::test_boundary_probe_needs_min_basins` | Boundary probe requirement |

### Unit Tests — Reasoning (`tests/unit/reasoning/`)

**test_transitions.py** (9 tests)
| Test | What it validates |
|------|-------------------|
| `TestTransitionStructure::test_add_edge` | Edge creation in COO format |
| `TestTransitionStructure::test_duplicate_edge_updates` | Duplicate edge weight update |
| `TestTransitionStructure::test_get_outgoing` | Outgoing edge query |
| `TestTransitionStructure::test_get_outgoing_empty` | Empty outgoing edges |
| `TestTransitionStructure::test_get_incoming` | Incoming edge query |
| `TestTransitionStructure::test_compute_transition_scores` | Transition scoring |
| `TestTransitionStructure::test_compute_transition_scores_empty` | Empty graph scoring |
| `TestTransitionStructure::test_get_edge_info` | Edge metadata retrieval |
| `TestTransitionStructure::test_eviction_at_capacity` | Lowest-confidence eviction |

**test_chain.py** (6 tests)
| Test | What it validates |
|------|-------------------|
| `TestAssociativeReasoningChain::test_empty_memory_returns_zero_knowledge` | Empty memory handling |
| `TestAssociativeReasoningChain::test_basic_chain_traversal` | Multi-hop traversal |
| `TestAssociativeReasoningChain::test_trace_returned_when_requested` | Trace recording |
| `TestAssociativeReasoningChain::test_cycle_detection` | Cycle termination |
| `TestAssociativeReasoningChain::test_dead_end_detection` | Dead end termination |
| `TestAssociativeReasoningChain::test_target_bias_influences_path` | Target-biased routing |

**test_learning.py** (8 tests)
| Test | What it validates |
|------|-------------------|
| `TestTransitionLearner::test_learn_from_empty_episodes` | Empty episode handling |
| `TestTransitionLearner::test_learn_temporal_relations` | Temporal co-occurrence mining |
| `TestTransitionLearner::test_learn_compositional_relations` | Compositional relation discovery |
| `TestTransitionLearner::test_learn_inhibitory_relations` | Inhibitory relation detection |
| `TestChainShortcutLearner::test_short_chain_ignored` | Short chain filtering |
| `TestChainShortcutLearner::test_shortcut_created_after_threshold` | Shortcut creation |
| `TestChainShortcutLearner::test_low_quality_chain_not_shortcut` | Quality gating |
| `TestChainShortcutLearner::test_counts_reset_after_shortcut` | Fragment count reset |

**test_router.py** (4 tests)
| Test | What it validates |
|------|-------------------|
| `TestReasoningRouter::test_empty_memory_single_hop` | Empty memory fallback |
| `TestReasoningRouter::test_single_hop_sufficient` | Single-hop path selection |
| `TestReasoningRouter::test_multi_hop_triggered` | Multi-hop escalation |
| `TestReasoningRouter::test_result_has_valid_structure` | Result structure validation |

### Unit Tests — Cognitive (`tests/unit/cognitive/`)

**test_world_model.py** (14 tests)
| Test | What it validates |
|------|-------------------|
| `TestPredictiveWorldModel::test_step_returns_state` | RSSM step output |
| `TestPredictiveWorldModel::test_multiple_steps_update_state` | State evolution |
| `TestPredictiveWorldModel::test_train_step_returns_losses` | Loss dictionary keys |
| `TestPredictiveWorldModel::test_train_step_decreases_loss` | Learning progression |
| `TestPredictiveWorldModel::test_enhanced_surprise` | 8-subspace directional error |
| `TestPredictiveWorldModel::test_cache_miss_returns_none` | Cache miss handling |
| `TestPredictiveWorldModel::test_cache_hit_returns_value` | Cache hit retrieval |
| `TestPredictiveWorldModel::test_cache_reset` | Cache clearing |
| `TestPredictiveWorldModel::test_imagination_trajectory` | Multi-step imagination |
| `TestPredictiveWorldModel::test_project_multi_step` | Projection shape |
| `TestPredictiveWorldModel::test_directional_prediction_error_shape` | Error shape validation |
| `TestPredictiveWorldModel::test_directional_error_zero_for_same` | Zero error for identical |
| `TestPredictiveWorldModel::test_current_state_shape` | State vector shape |
| `TestPredictiveWorldModel::test_reset_state` | State reset to zeros |

**test_goals.py** (16 tests)
| Test | What it validates |
|------|-------------------|
| `TestGoalCreation::test_create_basic_goal` | Basic goal creation |
| `TestGoalCreation::test_goal_id_increments` | Auto-incrementing IDs |
| `TestGoalCreation::test_create_parent_child` | Parent-child decomposition |
| `TestGoalCreation::test_eviction_when_full` | Full register eviction |
| `TestGoalProgress::test_update_progress` | Cosine progress tracking |
| `TestGoalProgress::test_auto_complete` | Auto-completion at threshold |
| `TestGoalPriority::test_priority_ordering` | Priority score computation |
| `TestGoalDeadlines::test_ttl_expiry` | TTL-based expiration |
| `TestGoalDeadlines::test_stalled_detection` | Stall detection |
| `TestGoalManagement::test_pause_resume` | Pause/resume lifecycle |
| `TestGoalManagement::test_remove_goal` | Goal removal |
| `TestGoalContext::test_context_vector_shape` | Context embedding shape |
| `TestGoalContext::test_empty_context` | Empty register context |
| `TestGoalContext::test_salience_modulation` | Salience score modulation |
| `TestGoalSerialization::test_save_load_roundtrip` | Save/load preservation |

**test_metacognition.py** (14 tests)
| Test | What it validates |
|------|-------------------|
| `TestMetacognitionEngine::test_retrieval_uncertainty_in_range` | Retrieval uncertainty [0,1] |
| `TestMetacognitionEngine::test_chain_uncertainty_in_range` | Chain uncertainty [0,1] |
| `TestMetacognitionEngine::test_prediction_uncertainty_in_range` | Prediction uncertainty [0,1] |
| `TestMetacognitionEngine::test_knowledge_uncertainty_in_range` | Knowledge uncertainty [0,1] |
| `TestMetacognitionEngine::test_uncertainty_composition` | Multi-source composition |
| `TestCalibration::test_calibration_update_and_ece` | ECE calibration metric |
| `TestCalibration::test_zero_calibration_insufficient_data` | Insufficient data handling |
| `TestCalibration::test_full_calibration_report` | Full report structure |
| `TestMemoryHealthMonitoring::test_empty_memory_health` | Empty memory baseline |
| `TestMemoryHealthMonitoring::test_record_activation` | Activation recording |
| `TestMemoryHealthMonitoring::test_dormant_basins_detected` | Dormancy detection |
| `TestInfoSeeking::test_confident_no_suggestions` | Confident state behavior |
| `TestInfoSeeking::test_uncertain_has_suggestions` | Uncertain state suggestions |

**test_valence.py** (16 tests)
| Test | What it validates |
|------|-------------------|
| `TestComputeValence::test_returns_valence_signal` | ValenceSignal output type |
| `TestComputeValence::test_positive_reward_positive_valence` | Reward → positive valence |
| `TestComputeValence::test_negative_feedback_negative_valence` | Negative feedback → negative valence |
| `TestComputeValence::test_embedding_valence_shape` | Embedding valence shape |
| `TestComputeValence::test_risk_tolerance_range` | Risk tolerance [0,1] |
| `TestComputeValence::test_vigilance_range` | Vigilance [0,1] |
| `TestAnnotation::test_basin_annotation` | Basin hedonic annotation |
| `TestAnnotation::test_edge_annotation` | Edge arousal annotation |
| `TestAnnotation::test_chain_valence` | Reasoning chain valence |
| `TestModulators::test_surprise_threshold_modulation` | Surprise threshold adjustment |
| `TestModulators::test_retention_priority` | Memory retention scoring |
| `TestModulators::test_reasoning_mode` | Approach/avoidance mode switching |
| `TestModulators::test_neutral_bias_near_zero` | Neutral mood baseline |
| `TestModulators::test_consolidation_weights` | Consolidation priority weighting |
| `TestModulators::test_mood_property` | Mood property access |

**test_salience.py** (11 tests)
| Test | What it validates |
|------|-------------------|
| `TestSalienceEvaluation::test_evaluate_returns_result` | Evaluation result type |
| `TestSalienceEvaluation::test_empty_inputs_skip` | Empty input → SKIP level |
| `TestSalienceEvaluation::test_with_context_vectors` | Context-modulated evaluation |
| `TestSalienceEvaluation::test_per_modality_scores` | Per-modality score breakdown |
| `TestInterrupts::test_no_interrupt_at_full` | No interrupt during FULL processing |
| `TestInterrupts::test_cross_modal_convergence` | Cross-modal interrupt escalation |
| `TestSkimProcessing::test_skim_buffers_input` | Skim buffer accumulation |
| `TestSkimProcessing::test_skim_summary_empty` | Empty skim summary |
| `TestSkimProcessing::test_skim_summary_after_inputs` | Populated skim summary |
| `TestHindsightTraining::test_train_hindsight_insufficient_data` | Hindsight with insufficient data |
| `TestHindsightTraining::test_record_hindsight` | Hindsight buffer recording |
| `TestGoalModulation::test_apply_goal_modulation` | Goal-based score adjustment |
| `TestGoalModulation::test_apply_expectation_modulation` | Prediction error modulation |

### Unit Tests — Topology (`tests/unit/topology/`)

**test_health.py** (4 tests)
| Test | What it validates |
|------|-------------------|
| `TestManifoldHealthMetrics::test_compute_returns_health_report` | Full health report |
| `TestManifoldHealthMetrics::test_compute_with_few_samples` | Small sample handling |
| `TestManifoldHealthMetrics::test_spectral_gap_reasonable` | Spectral gap positivity |
| `TestManifoldHealthMetrics::test_info_density_positive` | Information density positivity |

**test_betti.py** (6 tests, skipped if giotto-tda unavailable)
| Test | What it validates |
|------|-------------------|
| `TestBettiNumberMonitor::test_current_betti_empty` | Empty history → (0,0,0) |
| `TestBettiNumberMonitor::test_current_betti_with_history` | Latest Betti numbers |
| `TestBettiNumberMonitor::test_betti_series` | Time series extraction |
| `TestBettiNumberMonitor::test_is_stable_true` | Stability detection (constant) |
| `TestBettiNumberMonitor::test_is_stable_false` | Instability detection (changing) |
| `TestBettiNumberMonitor::test_is_stable_insufficient_data` | Insufficient data handling |

### Unit Tests — Communications (`tests/unit/comms/`)

**test_compression.py** (8 tests) — TopK/SignSGD compression, error accumulation
**test_layers.py** (9 tests) — L0-L4 communication layers
**test_sync.py** (6 tests) — Manifold synchronization, fingerprinting

### Unit Tests — Input (`tests/unit/input/`)

**test_bytelevel.py** (8 tests) — Byte-level encoding, entropy estimation, patching
**test_projection.py** (3 tests) — Dimensionality projection, alignment

### Unit Tests — Storage (`tests/unit/storage/`)

**test_quantization.py** (7 tests) — INT8 quantization, information content
**test_serialization.py** (5 tests) — Checkpoint save/load, weight preservation

---

## 2. Integration Tests (`tests/integration/`)

**test_end_to_end.py** (7 tests) — Full pipeline: stream→query, LoRA+EWC, save/load, quantization, communication layers, byte-level encoding

**test_learning_validation.py** (24 tests) — Learning validation suite covering all tiers:

| Test Class | Tests | What it validates |
|-----------|-------|-------------------|
| `TestTier0SurpriseAdaptation` | 3 | Loss decrease, update occurrence, novel input surprise spike |
| `TestTier0SurpriseGating` | 2 | High threshold blocks all; zero threshold allows all |
| `TestEpisodicMemoryAdmission` | 4 | Low-surprise rejection, high-surprise admission, dedup, different episode admission |
| `TestEssentialMemoryRetrieval` | 2 | Basin retrieval accuracy, active count |
| `TestConsolidationMini` | 1 | Tier 0 → Tier 1 episode creation |
| `TestConsolidationStandard` | 1 | Tier 1 → Tier 2 basin formation |
| `TestConsistencyChecker` | 4 | Consistent/contradictory/low-confidence/empty checks |
| `TestFullPipelineClusters` | 2 | Full pipeline basin formation, health report |
| `TestLearningDynamics` | 2 | Surprise decrease over time, episode accumulation |
| `TestContingencyReversal` | 1 | **Most diagnostic test** — learn, reverse, relearn |
| `TestDefenseSurpriseHardening` | 1 | Adversarial input clamping |
| `TestSaveLoadLearned` | 1 | Checkpoint roundtrip fidelity |

---

## 3. Sustainability Tests (`tests/sustainability/`) — `@pytest.mark.slow`

16 long-running tests (2,000–12,000 steps) that detect issues only visible under sustained operation. Run separately via `pytest -m slow`.

### Priority 1 — Unbounded Growth Detection

| Test | Steps | What it validates |
|------|-------|-------------------|
| `test_surprise_history_bounded` | 5,000 | `StreamingProcessor._history` growth is linear, not superlinear |
| `test_importance_scores_bounded` | 3,000 | `EssentialMemory.importance` doesn't diverge to infinity; new basins can form |
| `test_shortcut_learner_dict_growth` | 3,000 | `ChainShortcutLearner._chain_counts` stays within O(n_basins²) |
| `test_metacognition_dict_growth` | 100 | `MetacognitionEngine._activation_counts` tracks unique basin IDs correctly |

### Priority 2 — EMA Drift and Saturation

| Test | Steps | What it validates |
|------|-------|-------------------|
| `test_surprise_ema_recovery_after_regime_change` | 3,000 | SurpriseHardener re-adapts after regime shift |
| `test_gradient_clipper_recovery` | 3,000 | AdaptiveGradientClipper allows learning after long stable period |
| `test_world_model_adam_doesnt_stall` | 5,500 | World model can still learn new patterns after 5K steps |
| `test_valence_mood_recovery` | 6,000 | Valence system recovers from prolonged positive/negative bias |
| `test_world_model_ema_normalizer_stability` | 3,001 | EMA normalizers don't cause infinite surprise on outliers |

### Priority 3 — Full Pipeline Stress

| Test | Steps | What it validates |
|------|-------|-------------------|
| `test_deep_consolidation_cycle` | 12,000 | Deep consolidation fires, Tier 2→3 promotion, integrity passes |
| `test_episodic_capacity_limits` | 5,000 | Episodic hot/cold capacity limits respected under sustained load |
| `test_transition_graph_at_capacity` | 3,000 | Graph eviction works, edge type diversity maintained |
| `test_full_pipeline_all_modules` | 5,000 | Full cognitive pipeline (world model, valence, salience, replay) runs without crashes or NaN |

### Priority 4 — Cross-Module Interaction

| Test | Steps | What it validates |
|------|-------|-------------------|
| `test_replay_drift_monitoring` | 5,000 | Generative replay doesn't cause excessive basin key drift |
| `test_salience_threshold_stability` | 4,000 | Adaptive salience thresholds recover after regime change |
| `test_calibration_evolution` | 6,000 | ECE metric reflects current accuracy, not stale history |

### Source Bugs Found and Fixed by Sustainability Tests

| Bug | Source file | Fix |
|-----|-----------|-----|
| Episodic hot buffer off-by-one | `episodic.py:97` | `>` → `>=` for capacity check |
| Eviction skips protected basins | `episodic.py:199` | Iterate all candidates, not just top N |
| Cold buffer off-by-one | `episodic.py:237` | `>` → `>=` for cold capacity check |

### Known Source Issues Documented (Not Fixed)

| Issue | Source file | Impact |
|-------|-----------|--------|
| `attn_weights=None` always passed to metacognition | `processor.py:457` | MetacognitionEngine crashes if enabled in full pipeline |
| World model deterministic dim != d_model | `processor.py:320` | Salience gate shape mismatch when `d_deterministic != d_model` |

---

## 4. Diagnostic Failure Guide

| Test failure | Indicates |
|-------------|-----------|
| Tier0SurpriseAdaptation | Tier 0 surprise-gated learning mechanism broken |
| Tier0SurpriseGating | Surprise threshold not gating updates correctly |
| EpisodicMemoryAdmission | Tier 1 admission/dedup gates broken |
| EssentialMemoryRetrieval | Tier 2 Hopfield retrieval or confidence weighting broken |
| ConsolidationMini/Standard | Knowledge not flowing between tiers |
| ConsistencyChecker | Tier 2 consistency checking broken |
| FullPipelineClusters | Full pipeline integration issue |
| LearningDynamics | Learning metrics not progressing over time |
| **ContingencyReversal** | **Core learning-and-relearning capability broken** |
| DefenseSurpriseHardening | Defense mechanisms ineffective |
| SaveLoadLearned | Persistence/serialization loses learned state |
| BackboneTransformer | Shared contextual processing broken |
| PerceiverIO | Multi-modal bottleneck broken |
| InputSanitizer/AnomalyDetector | Input defense broken |
| GradientClipper/UpdateBudget | Gradient defense broken |
| AnchorVerifier | Weight anchoring/tamper detection broken |
| TopologyMonitor | Manifold integrity monitoring broken |
| SurpriseHardener | Surprise gate hardening broken |
| CoreMemory | Tier 3 cryptographic integrity broken |
| PromotionEvents | Tier 2→3 authorization broken |
| GenerativeReplay | Synthetic replay generation broken |
| TransitionStructure | Concept graph broken |
| ReasoningChain | Multi-hop reasoning broken |
| TransitionLearner | Relation discovery broken |
| ReasoningRouter | Single/multi-hop routing broken |
| WorldModel | RSSM prediction broken |
| GoalRegister | Goal management broken |
| MetacognitionEngine | Uncertainty estimation broken |
| ValenceSystem | Hedonic/mood system broken |
| SalienceGate | Pre-processing attention broken |
| Sustainability tests 1-4 | **Unbounded growth** — memory leak or dict accumulation over long runs |
| Sustainability tests 5-6 | **Defense numbing** — defense EMAs can't re-adapt after regime changes |
| Sustainability tests 7-9 | **Cognitive drift** — world model, valence, or normalizers saturate/collapse |
| Sustainability test 10 | **Deep consolidation broken** — Tier 2→3 promotion path doesn't work at scale |
| Sustainability test 11 | **Capacity enforcement broken** — episodic memory exceeds configured limits |
| Sustainability test 12 | **Graph churn** — transition graph eviction is wrong or biased |
| Sustainability test 13 | **Full pipeline instability** — crash, NaN, or latency blowup under full cognitive load |
| Sustainability tests 14-16 | **Cross-module interference** — replay causes drift, salience gets stuck, calibration goes stale |

---

## 5. Coverage Summary

```
342 tests across 36 modules — 100% module coverage

Category            Tests   Modules
─────────────────────────────────────
Core                44      5 (field, attractors, streaming, lora, forgetting)
Architecture        15      2 (backbone, perceiver)
Defense             43      5 (input_validation, gradient_bounds, anchor, topology_monitor, surprise_hardening)
Memory              35      3 (core, events, replay)
Reasoning           27      4 (transitions, chain, learning, router)
Cognitive           71      5 (world_model, goals, metacognition, valence, salience)
Topology            10      2 (health, betti)
Communications      23      3 (compression, layers, sync)
Input               11      2 (bytelevel, projection)
Storage             12      2 (quantization, serialization)
Integration         31      2 (end_to_end, learning_validation)
Shared fixtures      4      1 (conftest)
Sustainability      16      — (cross-cutting, 2K–12K steps each)
─────────────────────────────────────
Total              342     36
```

Run commands:
- **Fast suite only:** `pytest tests/ -m "not slow"` (~9s)
- **Sustainability only:** `pytest tests/sustainability/ -m slow` (~2.3 min)
- **Everything:** `pytest tests/` (~2.5 min)

---

## 6. Recommended Future Test Expansion

### Behavioral & Long-Running Tests
1. **Adversarial robustness suite** — Slow poisoning, surprise inflation, replay amplification
2. **Transfer learning test** — Train in env A, test adaptation speed in structurally similar env B
3. **Soak test harness** — Run for hours/days, monitor for regressions, resource leaks, calibration drift

### Infrastructure
4. Convert `examples/01_basic_field.py` through `examples/05_storage_and_input.py` into pytest assertions
5. Add property-based tests (hypothesis) for numerical stability of key operations
6. Add mutation testing to verify test quality
