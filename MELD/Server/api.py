"""Shared Flask and Flask-RESTX API configuration."""

from flask import Blueprint, render_template
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

    def render_root(self):
        return render_template("index.html")

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
    description="Proposed API for managing contracts and executions.",
    doc="/swagger/",
)

contracts = Namespace("contracts", description="Contract operations")
executions = Namespace("executions", description="Execution operations")

api.add_namespace(contracts, path="/contracts")
api.add_namespace(
    executions,
    path="/contracts/<string:contractId>/executions",
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
execution_status_model = api.model(
    "ExecutionStatus",
    {
        "status": fields.String(
            description="Current execution state.",
            enum=[
                "PENDING",
                "PREPARING",
                "START_QUERY",
                "QUERY_FINISHED",
                "CREATED",
                "RUNNING",
                "SUCCESS",
                "FAILED",
                "CANCELED",
                "TIMEOUT",
            ],
        ),
        "lastUpdated": fields.DateTime(
            description="Timestamp of the most recent status update.",
        ),
        "jobId": fields.String(description="Execution identifier."),
    },
)
execution_model = api.model(
    "Execution",
    {
        "execution_id": fields.String(description="Execution identifier."),
        "status": fields.Nested(execution_status_model),
    },
)


def not_implemented():
    return {"message": "Endpoint stub; implementation pending."}, 501


def error_response(code: str, message: str, details=None):
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return {"error": error}
