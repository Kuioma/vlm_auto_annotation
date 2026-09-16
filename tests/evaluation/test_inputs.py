import json
from pathlib import Path

import pytest

from auto_annotation.evaluation.config import load_evaluation_config
from auto_annotation.evaluation.errors import EvaluationError
from auto_annotation.evaluation.inputs import load_and_prevalidate_inputs

from .conftest import annotation, prediction, write_fixture


def test_manifest_selection_and_json_identity_drive_loading(tmp_path: Path):
    path = write_fixture(tmp_path, indices=(1,), annotation_indices=(1, 99), predictions=[prediction(1), prediction(88)])
    loaded = load_and_prevalidate_inputs(load_evaluation_config(path))
    assert [item.human_annotation.episode_index for item in loaded.included] == [1]
    assert loaded.extra_prediction_ids == ("episode_000088_multiview",)
    assert {item.category for item in loaded.failures} >= {"extra_gold_annotation", "extra_prediction"}


def test_duplicate_prediction_identity_is_fatal(tmp_path: Path):
    path = write_fixture(tmp_path, indices=(1,), predictions=[prediction(1), prediction(1)])
    with pytest.raises(EvaluationError, match="duplicate prediction video_id"):
        load_and_prevalidate_inputs(load_evaluation_config(path))


def test_annotation_filename_is_not_identity(tmp_path: Path):
    path = write_fixture(tmp_path, indices=(3,))
    loaded = load_and_prevalidate_inputs(load_evaluation_config(path))
    assert loaded.included[0].human_annotation.episode_index == 3
    assert next((tmp_path / "annotations").iterdir()).name == "arbitrary-3.json"

