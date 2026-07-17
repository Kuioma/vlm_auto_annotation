from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_annotation.task import load_ordered_task


def _write_task(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "task.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_ordered_task_preserves_declared_step_order(tmp_path: Path) -> None:
    task = load_ordered_task(
        _write_task(
            tmp_path,
            """
version: 1
task_id: demo-v1
schema_id: schema-v1
ontology_id: ontology-v1
task_summary: 完成演示任务
steps:
  - step_id: approach
    values: {atomic_action: approach}
    sentence: 接近物体。
  - step_id: grasp
    values: {atomic_action: grasp}
    sentence: 抓住物体。
  - step_id: move
    values: {atomic_action: move}
    sentence: 移动物体。
  - step_id: release
    values: {atomic_action: release}
    sentence: 释放物体。
""",
        )
    )

    assert task.task_id == "demo-v1"
    assert [step.step_id for step in task.steps] == [
        "approach",
        "grasp",
        "move",
        "release",
    ]


@pytest.mark.parametrize(
    ("steps", "message"),
    (
        (
            "  - step_id: only\n"
            "    values: {atomic_action: only}\n"
            "    sentence: 唯一步骤。\n",
            "at least 2",
        ),
        (
            "  - step_id: duplicate\n"
            "    values: {atomic_action: duplicate}\n"
            "    sentence: 第一步。\n"
            "  - step_id: duplicate\n"
            "    values: {atomic_action: duplicate}\n"
            "    sentence: 第二步。\n",
            "step IDs must be unique",
        ),
    ),
)
def test_load_ordered_task_rejects_invalid_sequences(
    tmp_path: Path,
    steps: str,
    message: str,
) -> None:
    path = _write_task(
        tmp_path,
        "version: 1\n"
        "task_id: demo-v1\n"
        "schema_id: schema-v1\n"
        "ontology_id: ontology-v1\n"
        "task_summary: 完成演示任务\n"
        "steps:\n"
        f"{steps}",
    )

    with pytest.raises(ValidationError, match=message):
        load_ordered_task(path)


def test_load_ordered_task_forbids_unknown_fields(tmp_path: Path) -> None:
    path = _write_task(
        tmp_path,
        """
version: 1
task_id: demo-v1
schema_id: schema-v1
ontology_id: ontology-v1
task_summary: 完成演示任务
unexpected: true
steps:
  - step_id: approach
    values: {atomic_action: approach}
    sentence: 接近物体。
  - step_id: grasp
    values: {atomic_action: grasp}
    sentence: 抓住物体。
""",
    )

    with pytest.raises(ValidationError, match="Extra inputs"):
        load_ordered_task(path)
