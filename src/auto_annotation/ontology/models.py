from typing import Literal

from pydantic import BaseModel, ConfigDict


class RegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SegmentFieldSpec(RegistryModel):
    type: Literal["enum", "string"]
    source: str | None = None
    required: bool = True


class TextOutputSpec(RegistryModel):
    field: str
    language: str
    style: str
    required: bool = True


class AnnotationSchema(RegistryModel):
    schema_id: str
    segment_values: dict[str, SegmentFieldSpec]
    text_output: TextOutputSpec


class OntologyEntry(RegistryModel):
    id: str
    name: str
    definition: str


class Ontology(RegistryModel):
    ontology_id: str
    revision: int | str | None = None
    vocabularies: dict[str, list[OntologyEntry]]
