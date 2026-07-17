from pathlib import Path

import pytest

from auto_annotation.ontology.registry import load_contract


FIXTURES = Path(__file__).parents[1] / "fixtures"


def _ontology_source_with_grasp_metadata(metadata: str) -> str:
    source = (FIXTURES / "ontology/kitchen-v1.yaml").read_text(
        encoding="utf-8"
    )
    definition = "      definition: 建立并保持对物体的控制"
    return source.replace(definition, f"{definition}\n{metadata}")


def test_contract_builds_dynamic_enums() -> None:
    contract = load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        FIXTURES / "ontology/kitchen-v1.yaml",
    )
    response_schema = contract.coarse_response_schema()
    values = response_schema["properties"]["segments"]["items"]["properties"][
        "values"
    ]
    assert values["properties"]["atomic_action"]["enum"] == ["grasp", "place"]
    assert values["properties"]["target"]["type"] == "string"
    assert contract.schema_hash.startswith("sha256:")
    assert contract.ontology_hash.startswith("sha256:")
    assert len(contract.schema_hash.removeprefix("sha256:")) == 64
    assert len(contract.ontology_hash.removeprefix("sha256:")) == 64


def test_legacy_ontology_omits_absent_temporal_metadata() -> None:
    contract = load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        FIXTURES / "ontology/kitchen-v1.yaml",
    )

    grasp = contract.entry_for("ontology.atomic_actions", "grasp")

    assert grasp.model_dump(mode="json") == {
        "id": "grasp",
        "name": "抓取",
        "definition": "建立并保持对物体的控制",
    }
    assert (
        contract.ontology_hash
        == "sha256:5a7a5c80df35b25ca84503a4005e5c244439515d595e9e88c33cd6da2379ed33"
    )


def test_contract_loads_and_normalizes_temporal_metadata(tmp_path: Path) -> None:
    ontology_path = tmp_path / "ontology.yaml"
    ontology_path.write_text(
        _ontology_source_with_grasp_metadata(
            '''      temporal_type: point_event
      boundary_rule: "  第一张能够确认稳定抓持的采样帧  "
      positive_cues:
        - "  物体随夹爪同步运动  "
      negative_cues:
        - "  仅接触物体  "'''
        ),
        encoding="utf-8",
    )

    contract = load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        ontology_path,
    )

    grasp = contract.entry_for("ontology.atomic_actions", "grasp")
    assert grasp.temporal_type == "point_event"
    assert grasp.boundary_rule == "第一张能够确认稳定抓持的采样帧"
    assert grasp.positive_cues == ["物体随夹爪同步运动"]
    assert grasp.negative_cues == ["仅接触物体"]


@pytest.mark.parametrize("field", ("id", "name", "definition"))
def test_contract_rejects_blank_closed_set_explanations(
    tmp_path: Path,
    field: str,
) -> None:
    source = (FIXTURES / "ontology/kitchen-v1.yaml").read_text(
        encoding="utf-8"
    )
    original = {
        "id": "    - id: grasp",
        "name": "      name: 抓取",
        "definition": "      definition: 建立并保持对物体的控制",
    }[field]
    ontology_path = tmp_path / "ontology.yaml"
    ontology_path.write_text(
        source.replace(original, f"{original.split(':', 1)[0]}: '   '", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least 1 character"):
        load_contract(FIXTURES / "schema/subtask-v1.yaml", ontology_path)


def test_temporal_metadata_participates_in_ontology_hash(tmp_path: Path) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_source = _ontology_source_with_grasp_metadata(
        """      temporal_type: point_event
      boundary_rule: 第一张能够确认稳定抓持的采样帧"""
    )
    first_path.write_text(first_source, encoding="utf-8")
    second_path.write_text(
        first_source.replace("第一张能够确认", "最后一张能够确认"),
        encoding="utf-8",
    )

    first = load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        first_path,
    )
    second = load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        second_path,
    )

    assert first.ontology_hash != second.ontology_hash


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            "      temporal_type: point_event",
            "temporal_type and boundary_rule must be provided together",
        ),
        (
            "      boundary_rule: 第一张能够确认稳定抓持的采样帧",
            "temporal_type and boundary_rule must be provided together",
        ),
        (
            """      temporal_type: duration
      boundary_rule: 第一张能够确认稳定抓持的采样帧""",
            "Input should be",
        ),
        (
            '''      temporal_type: point_event
      boundary_rule: "   "''',
            "at least 1 character",
        ),
        (
            "      positive_cues: []",
            "at least 1 item",
        ),
        (
            '''      positive_cues:
        - "   "''',
            "at least 1 character",
        ),
        (
            '''      negative_cues:
        - "   "''',
            "at least 1 character",
        ),
    ],
)
def test_contract_rejects_invalid_temporal_metadata(
    tmp_path: Path,
    metadata: str,
    message: str,
) -> None:
    ontology_path = tmp_path / "ontology.yaml"
    ontology_path.write_text(
        _ontology_source_with_grasp_metadata(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_contract(FIXTURES / "schema/subtask-v1.yaml", ontology_path)


def test_entry_for_reuses_ontology_source_validation(tmp_path: Path) -> None:
    ontology_path = tmp_path / "ontology.yaml"
    ontology_path.write_text(
        (FIXTURES / "ontology/kitchen-v1.yaml").read_text(encoding="utf-8")
        + "  unused: []\n",
        encoding="utf-8",
    )
    contract = load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        ontology_path,
    )

    assert contract.entry_for("ontology.interaction_objects", "cup").name == "杯子"
    with pytest.raises(ValueError, match="invalid ontology source"):
        contract.entry_for("external.atomic_actions", "grasp")
    with pytest.raises(ValueError, match="missing ontology vocabulary"):
        contract.entry_for("ontology.missing", "grasp")
    with pytest.raises(ValueError, match="empty ontology vocabulary"):
        contract.entry_for("ontology.unused", "grasp")
    with pytest.raises(ValueError, match="unknown ontology id"):
        contract.entry_for("ontology.atomic_actions", "missing")


def test_contract_rejects_unknown_enum_id() -> None:
    contract = load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        FIXTURES / "ontology/kitchen-v1.yaml",
    )
    with pytest.raises(ValueError, match="unknown atomic_action"):
        contract.validate_values(
            {
                "atomic_action": "throw",
                "interaction_object": "cup",
                "target": "托盘",
            }
        )


def test_contract_rejects_fields_missing_from_schema() -> None:
    contract = load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        FIXTURES / "ontology/kitchen-v1.yaml",
    )
    with pytest.raises(ValueError, match="unexpected segment fields"):
        contract.validate_values(
            {
                "atomic_action": "grasp",
                "interaction_object": "cup",
                "target": "杯柄",
                "tool": "夹爪",
            }
        )


def test_core_slice_requires_stable_sentence_key(tmp_path: Path) -> None:
    source = (FIXTURES / "schema/subtask-v1.yaml").read_text(encoding="utf-8")
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        source.replace("field: sentence", "field: instruction"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="text_output.field=sentence"):
        load_contract(schema_path, FIXTURES / "ontology/kitchen-v1.yaml")


def test_core_slice_requires_sentence_for_every_segment(tmp_path: Path) -> None:
    source = (FIXTURES / "schema/subtask-v1.yaml").read_text(encoding="utf-8")
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        source.replace(
            "  style: imperative\n  required: true",
            "  style: imperative\n  required: false",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="text_output.required=true"):
        load_contract(schema_path, FIXTURES / "ontology/kitchen-v1.yaml")


def test_contract_rejects_duplicate_ids_within_vocabulary(tmp_path: Path) -> None:
    ontology_source = (
        FIXTURES / "ontology/kitchen-v1.yaml"
    ).read_text(encoding="utf-8")
    ontology_path = tmp_path / "ontology.yaml"
    ontology_path.write_text(
        ontology_source.replace("- id: place", "- id: grasp"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate ontology id.*atomic_actions.*grasp"):
        load_contract(FIXTURES / "schema/subtask-v1.yaml", ontology_path)


def test_contract_rejects_empty_referenced_vocabulary(tmp_path: Path) -> None:
    ontology_source = (
        FIXTURES / "ontology/kitchen-v1.yaml"
    ).read_text(encoding="utf-8")
    ontology_path = tmp_path / "ontology.yaml"
    start = ontology_source.index("  atomic_actions:")
    end = ontology_source.index("  interaction_objects:")
    ontology_path.write_text(
        ontology_source[:start] + "  atomic_actions: []\n" + ontology_source[end:],
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"empty ontology vocabulary: atomic_actions"):
        load_contract(FIXTURES / "schema/subtask-v1.yaml", ontology_path)
