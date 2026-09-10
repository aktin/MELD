"""Typed contract models backed by the canonical JSON Schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping, TextIO

from jsonschema import validate

from utils import load_yaml
from utils.config import PULL_WITH_DIGEST


ContractDataType = Literal[
    "Int64",
    "Float64",
    "boolean",
    "string",
    "datetime64[ns]",
]


@dataclass
class SchemaModel:
    """Base class that preserves schema extensions not yet modeled in Python."""

    _extra: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __getattr__(self, name: str) -> Any:
        extra = self.__dict__.get("_extra", {})
        if name in extra:
            return extra[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in fields(self):
            if item.name == "_extra":
                continue
            value = getattr(self, item.name)
            if value is not None:
                result[item.name] = _to_dict(value)
        result.update(_to_dict(self._extra))
        return result


@dataclass
class ContractMetadata(SchemaModel):
    name: str
    description: str
    version: str
    id: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContractMetadata":
        instance = cls(
            name=data["name"],
            description=data["description"],
            version=data["version"],
            id=data.get("id"),
        )
        instance._extra = _extra(data, {"name", "description", "version", "id"})
        return instance


@dataclass
class RuntimeImage(SchemaModel):
    name: str
    tag: str
    digest: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeImage":
        instance = cls(name=data["name"], tag=data["tag"], digest=data["digest"])
        instance._extra = _extra(data, {"name", "tag", "digest"})
        return instance


    def construct_image_ref(self) -> str:
        """
        Constructs a formatted image reference string based on the provided contract
        dictionary.

        Parameters:
        contract (Contract): Contract containing the runtime image configuration.

        Returns:
        str: A formatted image ref string in the format "<name>:<tag>@<digest>".
        """
        ref = f"{self.name}:{self.tag}"
        if PULL_WITH_DIGEST:
            return f"{ref}@{self.digest}"
        else:
            return ref


@dataclass
class RuntimeConfig(SchemaModel):
    framework: str
    image: RuntimeImage
    environment_variables: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeConfig":
        instance = cls(
            framework=data["framework"],
            image=RuntimeImage.from_dict(data["image"]),
            environment_variables=data.get("environment_variables"),
        )
        instance._extra = _extra(data, {"framework", "image", "environment_variables"})
        return instance


@dataclass
class TemporalScope(SchemaModel):
    type: Literal["relative", "absolute"] | None = None
    value: str | None = None
    anchor: str | None = None
    start: str | None = None
    end: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TemporalScope":
        instance = cls(
            type=data.get("type"),
            value=data.get("value"),
            anchor=data.get("anchor"),
            start=data.get("start"),
            end=data.get("end"),
        )
        instance._extra = _extra(data, {"type", "value", "anchor", "start", "end"})
        return instance


@dataclass
class Feature(SchemaModel):
    name: str
    datatype: ContractDataType
    required: bool | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Feature":
        instance = cls(
            name=data["name"],
            datatype=data["datatype"],
            required=data.get("required"),
        )
        instance._extra = _extra(data, {"name", "datatype", "required"})
        return instance


@dataclass
class Query(SchemaModel):
    type: Literal["sql"]
    statement: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Query":
        instance = cls(type=data["type"], statement=data["statement"])
        instance._extra = _extra(data, {"type", "statement"})
        return instance


@dataclass
class InputSchema(SchemaModel):
    temporal_scope: TemporalScope
    features: list[Feature]
    query: Query

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InputSchema":
        instance = cls(
            temporal_scope=TemporalScope.from_dict(data["temporal_scope"]),
            features=[Feature.from_dict(feature) for feature in data["features"]],
            query=Query.from_dict(data["query"]),
        )
        instance._extra = _extra(data, {"temporal_scope", "features", "query"})
        return instance


@dataclass
class OutputSchema(SchemaModel):
    type: Literal["csv"]
    predictor: list[Feature]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputSchema":
        instance = cls(
            type=data["type"],
            predictor=[Feature.from_dict(feature) for feature in data["predictor"]],
        )
        instance._extra = _extra(data, {"type", "predictor"})
        return instance


@dataclass
class Contract(SchemaModel):
    """A validated MELD contract with attribute-based access to all fields."""

    contract: ContractMetadata
    runtime: RuntimeConfig
    input_schema: InputSchema
    output_schema: OutputSchema
    schema_version: str | None = None

    schema_path: ClassVar[Path] = (
        Path(__file__).resolve().parents[1] / "resources" / "contract.schema.json"
    )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """Read the current schema so schema edits are picked up automatically."""
        with cls.schema_path.open(encoding="utf-8") as schema_file:
            return json.load(schema_file)

    @property
    def id(self) -> str:
        """Return a stable identifier derived from the contract contents."""
        identity = self.to_dict()
        identity.get("contract", {}).pop("id", None)
        canonical = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Contract":
        validate(data, cls.schema())
        instance = cls(
            contract=ContractMetadata.from_dict(data["contract"]),
            runtime=RuntimeConfig.from_dict(data["runtime"]),
            input_schema=InputSchema.from_dict(data["input_schema"]),
            output_schema=OutputSchema.from_dict(data["output_schema"]),
            schema_version=data.get("schema_version"),
        )
        instance._extra = _extra(
            data,
            {"schema_version", "contract", "runtime", "input_schema", "output_schema"},
        )
        return instance

    @classmethod
    def from_yaml(cls, source: str | TextIO) -> "Contract":
        return cls.from_dict(load_yaml(source))


def _extra(data: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in known}


def _to_dict(value: Any) -> Any:
    if isinstance(value, SchemaModel):
        return value.to_dict()
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_dict(item) for key, item in value.items()}
    return value


__all__ = [
    "Contract",
    "ContractDataType",
    "ContractMetadata",
    "Feature",
    "InputSchema",
    "OutputSchema",
    "Query",
    "RuntimeConfig",
    "RuntimeImage",
    "TemporalScope",
]
