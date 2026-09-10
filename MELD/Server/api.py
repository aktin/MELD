"""Shared Flask and Flask-RESTX API configuration."""

from flask import Blueprint
from flask_restx import Api, Namespace, fields
from werkzeug.utils import cached_property

from ModelManager import Contract

try:
    from __version__ import version as API_VERSION
except ImportError:  # pragma: no cover - supports importing the module standalone
    API_VERSION = "0.1.0"


bp = Blueprint("meld", __name__)


class MeldApi(Api):
    """API variant that preserves operation-specific request media types."""

    @cached_property
    def __schema__(self):
        schema = super().__schema__
        schema["definitions"]["ContractPayload"] = Contract.schema()
        schema["paths"]["/contracts"]["post"]["consumes"] = [
            "text/plain",
            "application/yaml",
        ]
        schema.pop("consumes", None)
        return schema


api = MeldApi(
    bp,
    version=API_VERSION,
    title="MELD API",
    description="Proposed API for managing contracts, inferences, and schedules.",
    doc="/swagger/",
)

contracts = Namespace("contracts", description="Contract operations")
inferences = Namespace("inferences", description="Inference operations")
schedules = Namespace("schedules", description="Schedule operations")

api.add_namespace(contracts, path="/contracts")
api.add_namespace(
    inferences,
    path="/contracts/<string:contractId>/inferences",
)
api.add_namespace(
    schedules,
    path="/contracts/<string:contractId>/schedules",
)


error_model = api.model(
    "ErrorResponse",
    {
        "error": fields.Nested(
            api.model(
                "Error",
                {
                    "code": fields.String(description="Application error code"),
                    "message": fields.String(
                        description="Human-readable error message"
                    ),
                    "details": fields.Raw(description="Additional error details"),
                    "requestId": fields.String(
                        description="Request correlation ID"
                    ),
                },
            )
        )
    },
)

contract_body = api.schema_model(
    "ContractTextPayload",
    {
        "type": "string",
        "description": "YAML-encoded MELD contract.",
    },
)
inference_body = {
    "type": "object",
    "description": "Inference request payload; schema TBD.",
}
schedule_body = {
    "type": "object",
    "description": "Schedule payload; schema TBD.",
}


def not_implemented():
    return {"message": "Endpoint stub; implementation pending."}, 501


def error_response(code: str, message: str, details=None):
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return {"error": error}
