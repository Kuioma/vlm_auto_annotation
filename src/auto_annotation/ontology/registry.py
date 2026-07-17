import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from auto_annotation.ontology.models import (
    AnnotationSchema,
    Ontology,
    OntologyEntry,
)


class ResolvedContract:
    def __init__(self, schema: AnnotationSchema, ontology: Ontology) -> None:
        self.schema = schema
        self.ontology = ontology
        for vocabulary, entries in ontology.vocabularies.items():
            seen: set[str] = set()
            for entry in entries:
                if entry.id in seen:
                    raise ValueError(
                        "duplicate ontology id in vocabulary "
                        f"{vocabulary}: {entry.id}"
                    )
                seen.add(entry.id)
        schema_canonical = json.dumps(
            schema.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        ontology_canonical = json.dumps(
            ontology.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.schema_hash = f"sha256:{hashlib.sha256(schema_canonical).hexdigest()}"
        self.ontology_hash = (
            f"sha256:{hashlib.sha256(ontology_canonical).hexdigest()}"
        )

    def _entries(self, source: str) -> list[OntologyEntry]:
        prefix = "ontology."
        if not source.startswith(prefix):
            raise ValueError(f"invalid ontology source: {source}")
        vocabulary = source.removeprefix(prefix)
        if vocabulary not in self.ontology.vocabularies:
            raise ValueError(f"missing ontology vocabulary: {vocabulary}")
        entries = self.ontology.vocabularies[vocabulary]
        if not entries:
            raise ValueError(f"empty ontology vocabulary: {vocabulary}")
        return entries

    def _enum_ids(self, source: str) -> list[str]:
        entries = self._entries(source)
        return [entry.id for entry in entries]

    def entry_for(self, source: str, entry_id: str) -> OntologyEntry:
        for entry in self._entries(source):
            if entry.id == entry_id:
                return entry
        raise ValueError(f"unknown ontology id in {source}: {entry_id}")

    def validate_values(self, values: dict[str, Any]) -> None:
        unexpected = set(values) - set(self.schema.segment_values)
        if unexpected:
            raise ValueError(f"unexpected segment fields: {sorted(unexpected)}")
        for name, spec in self.schema.segment_values.items():
            if spec.required and name not in values:
                raise ValueError(f"missing required field: {name}")
            if name not in values:
                continue
            if spec.type == "enum":
                allowed = self._enum_ids(spec.source or "")
                if values[name] not in allowed:
                    raise ValueError(f"unknown {name}: {values[name]}")
            elif not isinstance(values[name], str):
                raise ValueError(f"{name} must be a string")

    def coarse_response_schema(self) -> dict[str, Any]:
        value_properties: dict[str, Any] = {}
        required_values: list[str] = []
        for name, spec in self.schema.segment_values.items():
            value_properties[name] = (
                {"type": "string", "enum": self._enum_ids(spec.source or "")}
                if spec.type == "enum"
                else {"type": "string"}
            )
            if spec.required:
                required_values.append(name)
        segment = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "segment_id": {"type": "string"},
                "start_ms": {"type": "integer", "minimum": 0},
                "end_ms": {"type": "integer", "minimum": 1},
                "values": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": value_properties,
                    "required": required_values,
                },
                self.schema.text_output.field: {"type": "string", "minLength": 1},
                "evidence_timestamps_ms": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                },
            },
            "required": [
                "segment_id",
                "start_ms",
                "end_ms",
                "values",
                self.schema.text_output.field,
                "evidence_timestamps_ms",
            ],
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task_summary": {"type": "string", "minLength": 1},
                "segments": {"type": "array", "items": segment},
            },
            "required": ["task_summary", "segments"],
        }


def load_contract(schema_path: Path, ontology_path: Path) -> ResolvedContract:
    schema = AnnotationSchema.model_validate(
        yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    )
    ontology = Ontology.model_validate(
        yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    )
    if schema.text_output.field != "sentence":
        raise ValueError("core slice requires text_output.field=sentence")
    if not schema.text_output.required:
        raise ValueError("core slice requires text_output.required=true")
    contract = ResolvedContract(schema, ontology)
    contract.coarse_response_schema()
    return contract
