from pathlib import Path

from auto_annotation.evaluation.config import load_evaluation_config


def test_wa2_profile_is_parse_only_and_has_confirmed_boundary_defaults():
    path = Path("examples/evaluation/wa2_pick_place_boundary_v1.yaml")
    config = load_evaluation_config(path)
    assert config.metrics.boundary_error.event_ids == ["transfer", "place"]
    assert config.metrics.boundary_error.thresholds_ms == [500, 1000, 2000]
    assert config.metrics.temporal_iou.thresholds == [0.3, 0.5, 0.7]
    assert config.metrics.segmentation_errors.substantial_overlap_threshold == 0.5
    assert config.uncertainty.resamples == 10000
    assert config.decision.mode == "report_only"
    assert config.metrics.semantic_quality.implementation == "deferred"

