from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


NonBlankText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
TemporalType = Literal["point_event", "interval_action", "state"]


def _is_none(value: object) -> bool:
    return value is None


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
    id: NonBlankText
    name: NonBlankText
    definition: NonBlankText
    temporal_type: TemporalType | None = Field(
        default=None,
        exclude_if=_is_none,
    )
    boundary_rule: NonBlankText | None = Field(
        default=None,
        exclude_if=_is_none,
    )
    positive_cues: list[NonBlankText] | None = Field(
        default=None,
        min_length=1,
        exclude_if=_is_none,
    )
    negative_cues: list[NonBlankText] | None = Field(
        default=None,
        min_length=1,
        exclude_if=_is_none,
    )

    @model_validator(mode="after")
    def validate_temporal_semantics(self) -> "OntologyEntry":
        if (self.temporal_type is None) != (self.boundary_rule is None):
            raise ValueError(
                "temporal_type and boundary_rule must be provided together"
            )
        return self


class Ontology(RegistryModel):
    ontology_id: str
    revision: int | str | None = None
    vocabularies: dict[str, list[OntologyEntry]]
