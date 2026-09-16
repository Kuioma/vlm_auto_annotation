from auto_annotation.evaluation.metrics.boundary import evaluate_boundaries


def test_boundary_errors_and_inclusive_hit_threshold():
    result = evaluate_boundaries(
        {"transfer": 1000, "place": 2000},
        {"transfer": 900, "place": 2500},
        [100, 500],
    )
    assert result["transfer"]["signed_error_ms"] == -100
    assert result["transfer"]["absolute_error_ms"] == 100
    assert result["transfer"]["hits"] == {"100": True, "500": True}
    assert result["place"]["hits"]["500"] is True


def test_missing_boundary_is_not_observed_and_never_hits():
    detail = evaluate_boundaries({"transfer": 1000}, None, [1000])["transfer"]
    assert detail["absolute_error_ms"] is None
    assert detail["hits"]["1000"] is False
    assert detail["status"] == "missing"

