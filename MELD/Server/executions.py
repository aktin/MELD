"""Execution endpoints."""

from flask import current_app, Response, url_for, send_file
from flask_restx import Resource
from jsonschema import ValidationError
import yaml
from werkzeug.exceptions import BadRequest

from ModelEnvironment import ExecutionService, ExecutionStatus
from .api import (
    error_model,
    execution_model,
    execution_status_model,
    error_response,
    executions as executions_namespace,
)


def _execution_service() -> ExecutionService:
    service = current_app.extensions.get("execution_service")
    if service is None:
        service = ExecutionService()
        current_app.extensions["execution_service"] = service
    return service

def _location_response(contract_id: str, status: int) -> Response:
    return Response(
        status=status,
        headers={
            "Location": url_for(
                "meld.contracts_contract_resource",
                contract_id=contract_id,
            )
        },
    )


def _not_found(error: FileNotFoundError):
    return error_response(
        "EXECUTION_RESOURCE_NOT_FOUND",
        "Requested execution resource does not exist.",
        str(error),
    ), 404


def _storage_error(error: Exception, message: str):
    return error_response("STORAGE_ERROR", message, str(error)), 500


def _status_value(ctx) -> str | None:
    status = ctx.status
    if isinstance(status, dict):
        return status.get("status")
    if isinstance(status, ExecutionStatus):
        return status.value
    return status


FINAL_EXECUTION_STATES = {
    ExecutionStatus.SUCCESS.value,
    ExecutionStatus.FAILED.value,
    ExecutionStatus.CANCELED.value,
    ExecutionStatus.TIMEOUT.value,
}


@executions_namespace.route("", strict_slashes=False)
@executions_namespace.doc(params={"contractId": "Contract identifier"})
class ExecutionsResource(Resource):
    @executions_namespace.doc(description="List executions for a contract.")
    @executions_namespace.response(200, "List of executions", [execution_model])
    @executions_namespace.response(404, "Contract does not exist", error_model)
    @executions_namespace.response(500, "Could not retrieve executions", error_model)
    def get(self, contractId):
        try:
            return list(_execution_service().get_executions(contractId))
        except FileNotFoundError as error:
            return _not_found(error)
        except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as error:
            return _storage_error(error, "Could not retrieve executions.")

    @executions_namespace.doc(description="Start an execution asynchronously.")
    @executions_namespace.response(202, "Execution started")
    @executions_namespace.response(400, "Execution request is malformed", error_model)
    @executions_namespace.response(404, "Contract does not exist", error_model)
    @executions_namespace.response(500, "Could not start execution", error_model)
    def post(self, contractId):
        try:
            execution_id = _execution_service().start_execution(contractId)
        except FileNotFoundError as error:
            return _not_found(error)
        except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError, RuntimeError) as error:
            return _storage_error(error, "Could not start execution.")
        return Response(
            status=202,
            headers={
                "Location": url_for(
                    "meld.executions_execution_resource",
                    contractId=contractId,
                    executionId=execution_id,
                )
            },
        )


@executions_namespace.route("/<string:executionId>")
@executions_namespace.doc(
    params={
        "contractId": "Contract identifier",
        "executionId": "Execution identifier",
    }
)
class ExecutionResource(Resource):
    @executions_namespace.doc(
        description=(
            "Retrieve execution status while running or the completed result "
            "archive as a file object."
        )
    )
    @executions_namespace.doc(
        produces=["application/json", "application/zip"],
    )
    @executions_namespace.response(200, "Execution status or completed result archive", execution_status_model)
    @executions_namespace.response(404, "Contract or execution does not exist", error_model)
    @executions_namespace.response(500, "Could not retrieve execution", error_model)
    def get(self, contractId, executionId):
        try:
            ctx = _execution_service().get_execution(contractId, executionId)
            status = _status_value(ctx)
            if status == ExecutionStatus.SUCCESS.value:
                archive = _execution_service().get_result_archive(contractId, executionId)
                archive.seek(0)
                return send_file(archive, mimetype="application/zip")
            return ctx.status, 200
        except FileNotFoundError as error:
            return _not_found(error)
        except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as error:
            return _storage_error(error, "Could not retrieve execution.")

    @executions_namespace.doc(description="Cancel a running execution.")
    @executions_namespace.response(204, "Execution canceled")
    @executions_namespace.response(400, "Execution has already completed", error_model)
    @executions_namespace.response(404, "Contract or execution does not exist", error_model)
    @executions_namespace.response(500, "Could not cancel execution", error_model)
    def delete(self, contractId, executionId):
        try:
            _execution_service().cancel_execution(contractId, executionId)
        except BadRequest as error:
            return error_response(
                "EXECUTION_ALREADY_COMPLETED",
                error.description,
            ), 400
        except FileNotFoundError as error:
            return _not_found(error)
        except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as error:
            return _storage_error(error, "Could not cancel execution.")
        return Response(status=204)


@executions_namespace.route("/<string:executionId>/logs")
@executions_namespace.doc(
    params={
        "contractId": "Contract identifier",
        "executionId": "Execution identifier",
    }
)
class ExecutionLogsResource(Resource):
    @executions_namespace.doc(
        description="Retrieve the execution log stream.",
        produces=["text/plain", "application/octet-stream"],
    )
    @executions_namespace.response(200, "Execution log stream")
    @executions_namespace.response(404, "Contract or execution does not exist", error_model)
    @executions_namespace.response(500, "Could not retrieve execution logs", error_model)
    def get(self, contractId, executionId):
        try:
            ctx = _execution_service().get_execution(contractId, executionId)
            if _status_value(ctx) in FINAL_EXECUTION_STATES:
                return Response(
                    _execution_service().read_log(contractId, executionId),
                    mimetype="text/plain",
                )
            return Response(
                _execution_service().stream_log(contractId, executionId),
                mimetype="text/plain",
            )
        except FileNotFoundError as error:
            return _not_found(error)
        except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as error:
            return _storage_error(error, "Could not retrieve execution logs.")
