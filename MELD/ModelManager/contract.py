from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TextIO

from jsonschema import validate

from utils import load_yaml
from utils.config import PULL_WITH_DIGEST

from .generated import (
    Contract as ContractMetadata,
    Feature,
    Image,
    InputSchema,
    MeldInferenceRuntimeContract,
    OutputSchema,
    Query,
    Runtime,
    Schedule,
    TemporalScope,
)


def _to_dict(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _to_dict(getattr(value, item.name))
            for item in fields(value)
            if getattr(value, item.name) is not None
        }
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_dict(item) for key, item in value.items()}
    return value


def _build(model: type[Any], data: Mapping[str, Any]) -> Any:
    nested = {
        ContractMetadata: {"contract": ContractMetadata},
        Runtime: {"image": Image},
        InputSchema: {
            "temporal_scope": TemporalScope,
            "features": Feature,
            "query": Query,
        },
        OutputSchema: {"labels": Feature},
        MeldInferenceRuntimeContract: {
            "contract": ContractMetadata,
            "runtime": Runtime,
            "input_schema": InputSchema,
            "output_schema": OutputSchema,
            "schedule": Schedule,
        },
    }.get(MeldInferenceRuntimeContract if issubclass(model, MeldInferenceRuntimeContract) else model, {})
    values: dict[str, Any] = {}
    field_names = {item.name for item in fields(model)}
    for name, value in data.items():
        if name not in field_names:
            continue
        child = nested.get(name)
        if child is Feature:
            values[name] = [_build(child, item) for item in value]
        elif child is not None and value is not None:
            values[name] = _build(child, value)
        else:
            values[name] = value
    return model(**values)


class Contract(MeldInferenceRuntimeContract):
    schema_path = Path(__file__).resolve().parents[1] / "resources" / "contract.schema.json"

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return json.loads(cls.schema_path.read_text(encoding="utf-8"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Contract":
        validate(data, cls.schema())
        instance = _build(cls, data)
        instance._extra = {
            key: value
            for key, value in data.items()
            if key not in {item.name for item in fields(cls)}
        }
        return instance

    @classmethod
    def from_yaml(cls, source: str | Path | TextIO) -> "Contract":
        return cls.from_dict(load_yaml(source))

    @property
    def id(self) -> str:
        value = f"{self.contract.name}-{self.contract.version}"
        self.contract.id = value
        return value

    def assign_id(self, force: bool = False) -> str:
        return self.id

    def to_dict(self) -> dict[str, Any]:
        result = _to_dict(self)
        result.pop("_extra", None)
        result.update(_to_dict(getattr(self, "_extra", {})))
        return result

    def __getattr__(self, name: str) -> Any:
        extra = self.__dict__.get("_extra", {})
        if name in extra:
            return extra[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MeldInferenceRuntimeContract):
            return NotImplemented
        return self.id == other.id


def construct_image_ref(image: Image) -> str:
    reference = f"{image.name}:{image.tag}"
    return f"{reference}@{image.digest}" if PULL_WITH_DIGEST else reference


Image.construct_image_ref = construct_image_ref


__all__ = ["Contract", "construct_image_ref"]
