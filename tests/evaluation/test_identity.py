from pathlib import Path

from auto_annotation.evaluation.config import load_evaluation_config
from auto_annotation.evaluation.identity import build_evaluation_identity
from auto_annotation.evaluation.inputs import load_and_prevalidate_inputs
from auto_annotation.evaluation.versions import EVALUATOR_VERSION, METRIC_VERSIONS, RESULT_SCHEMA_VERSION

from .conftest import write_fixture


def identity(path: Path):
    config = load_evaluation_config(path)
    loaded = load_and_prevalidate_inputs(config)
    return build_evaluation_identity(
        config=config,
        versions={"evaluator": EVALUATOR_VERSION, "result": RESULT_SCHEMA_VERSION, "metrics": METRIC_VERSIONS},
        gold_manifest_bytes=loaded.manifest_bytes,
        selected_annotation_bytes=loaded.selected_annotation_bytes,
        prediction_bytes=loaded.prediction_bytes,
    )[0]


def test_identity_is_stable_and_tracks_raw_prediction_bytes(tmp_path: Path):
    path = write_fixture(tmp_path)
    first = identity(path)
    assert first == identity(path)
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(prediction_path.read_text() + "\n", encoding="utf-8")
    assert identity(path) != first


def test_out_of_selection_annotation_does_not_change_identity(tmp_path: Path):
    path = write_fixture(tmp_path, indices=(1,), annotation_indices=(1, 99))
    first = identity(path)
    extra = tmp_path / "annotations" / "arbitrary-99.json"
    extra.write_text(extra.read_text() + " ", encoding="utf-8")
    assert identity(path) == first


def test_identity_tracks_behavior_versions(tmp_path: Path):
    path = write_fixture(tmp_path, indices=(1,))
    config = load_evaluation_config(path)
    loaded = load_and_prevalidate_inputs(config)

    def calculate(version):
        return build_evaluation_identity(
            config=config,
            versions={"evaluator": version, "result": RESULT_SCHEMA_VERSION, "metrics": METRIC_VERSIONS},
            gold_manifest_bytes=loaded.manifest_bytes,
            selected_annotation_bytes=loaded.selected_annotation_bytes,
            prediction_bytes=loaded.prediction_bytes,
        )[0]

    assert calculate("gold-evaluator-v1") != calculate("gold-evaluator-v2")
