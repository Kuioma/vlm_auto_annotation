from auto_annotation.evaluation.metrics.labels import evaluate_labels
from auto_annotation.evaluation.metrics.segmentation import evaluate_segmentation
from auto_annotation.evaluation.metrics.sequence import evaluate_sequence, levenshtein
from auto_annotation.evaluation.models import EvalSegment


def make(segment_id, start, end, action=None):
    return EvalSegment(
        segment_id=segment_id,
        start_ms=start,
        end_ms=end,
        values={"atomic_action": action or segment_id, "object": "part"},
        sentence=segment_id,
    )


def test_sequence_edit_distance_and_normalization():
    assert levenshtein(["a", "b"], ["a", "c", "b"]) == 1
    result = evaluate_sequence([make("a", 0, 1), make("b", 1, 2)], [make("a", 0, 1)], "atomic_action")
    assert result["distance"] == 1
    assert result["normalized_distance"] == 0.5
    assert result["exact"] is False


def test_label_pairing_is_independent_of_label():
    gold = [make("s", 0, 10, "pick")]
    prediction = [make("s", 0, 10, "place"), make("extra", 10, 20)]
    result = evaluate_labels(gold, prediction, ["atomic_action"], [["atomic_action", "object"]])
    assert result["fields"]["atomic_action"]["accuracy"] == 0.0
    assert result["groups"]["atomic_action+object"]["accuracy"] == 0.0
    assert result["extra_predictions"] == 1


def test_segmentation_overlap_graph_counts_split_merge_and_missing():
    gold = [make("a", 0, 100), make("b", 100, 200), make("miss", 200, 300)]
    prediction = [make("p1", 0, 50), make("p2", 50, 150), make("extra", 400, 500)]
    result = evaluate_segmentation(gold, prediction, 0.5)
    assert result["over_split_count"] == 1
    assert result["incorrect_merge_count"] == 1
    assert result["missed_segment_count"] == 1
    assert result["extra_segment_count"] == 1
