from pathlib import Path

import pytest

from auto_annotation.ontology.registry import load_contract


FIXTURES = Path(__file__).parents[1] / "fixtures"


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
