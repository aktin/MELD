"""Contract endpoints."""

import json

from flask import Response, current_app, request, url_for
from flask_restx import Resource
from jsonschema import ValidationError
import yaml

from ModelManager.contract_service import ContractService

from .api import (
    contract_body,
    contracts as contracts_namespace,
    error_model,
    error_response,
)


def _contract_service() -> ContractService:
    service = current_app.extensions.get("contract_service")
    if service is None:
        service = ContractService()
        current_app.extensions["contract_service"] = service
    return service


def _invalid_contract(error: Exception):
    return error_response(
        "INVALID_CONTRACT",
        "Contract is malformed.",
        str(error),
    ), 400


def _storage_error(error: Exception):
    return error_response(
        "STORAGE_ERROR",
        "Could not retrieve contract.",
        str(error),
    ), 500


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


def _supported_media_type() -> bool:
    return request.mimetype in (
        None,
        "text/plain",
        "application/yaml",
        "text/yaml",
        "application/x-yaml",
    )


@contracts_namespace.route("", strict_slashes=False)
class ContractsResource(Resource):
    @contracts_namespace.response(200, "List of contracts")
    def get(self):
        try:
            contracts = _contract_service().get_contracts_info()
        except (OSError, TypeError, ValueError, KeyError, ValidationError, yaml.YAMLError) as error:
            return _storage_error(error)
        return Response(json.dumps(contracts), status=200, mimetype="application/json")

    @contracts_namespace.doc(
        description="Create a contract and pull its referenced image asynchronously.",
        params={
            "X-Docker-Registry-API-Key": {
                "in": "header",
                "description": "Optional API key for Docker registry access.",
                "required": False,
            }
        },
    )
    @contracts_namespace.expect(contract_body, validate=False)
    @contracts_namespace.doc(consumes=["text/plain", "application/yaml"])
    @contracts_namespace.response(
        201,
        "Contract created; image is available; Location header returned",
    )
    @contracts_namespace.response(
        202,
        "Contract created; image pull initiated; Location header returned",
    )
    @contracts_namespace.response(400, "Contract is malformed", error_model)
    @contracts_namespace.response(409, "Contract already exists", error_model)
    @contracts_namespace.response(502, "Image pull failed", error_model)
    def post(self):
        if not _supported_media_type():
            return error_response(
                "UNSUPPORTED_MEDIA_TYPE",
                "Unsupported media type in request.",
            ), 415

        service = _contract_service()
        try:
            contract = service.parse_contract(request.get_data(as_text=True))
        except (ValidationError, TypeError, ValueError, KeyError, yaml.YAMLError) as error:
            return _invalid_contract(error)

        try:
            image_available = service.create_contract(
                contract,
                request.headers.get("X-Docker-Registry-API-Key"),
            )
        except FileExistsError:
            return error_response("CONTRACT_EXISTS", "Contract already exists."), 409
        except ConnectionError as error:
            return error_response(
                "IMAGE_UNAVAILABLE",
                "The runtime image could not be found.",
                str(error),
            ), 502
        except RuntimeError as error:
            return error_response(
                "IMAGE_PULL_FAILED",
                "The runtime image pull could not be started.",
                str(error),
            ), 502
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            return error_response(
                "STORAGE_ERROR",
                "Could not store contract",
                str(error),
            ), 500

        return _location_response(contract.id, 201 if image_available else 202)


@contracts_namespace.route("/<string:contract_id>")
@contracts_namespace.doc(params={"contract_id": "Contract identifier"})
class ContractResource(Resource):
    @contracts_namespace.response(
        200,
        "Contract JSON data, installation status, and progress",
    )
    @contracts_namespace.response(404, "Contract does not exist", error_model)
    def get(self, contract_id):
        try:
            contract = _contract_service().get_contract_info(contract_id)
        except FileNotFoundError:
            return error_response("CONTRACT_NOT_FOUND", "Contract does not exist."), 404
        except (OSError, ValidationError, TypeError, ValueError, KeyError, yaml.YAMLError) as error:
            return _storage_error(error)
        return Response(json.dumps(contract), status=200, mimetype="application/json")

    @contracts_namespace.response(204, "Contract deleted")
    @contracts_namespace.response(404, "Contract does not exist", error_model)
    def delete(self, contract_id):
        try:
            _contract_service().delete_contract(contract_id)
        except FileNotFoundError:
            return error_response("CONTRACT_NOT_FOUND", "Contract does not exist."), 404
        except (OSError, ValidationError, TypeError, ValueError, KeyError, yaml.YAMLError) as error:
            return _storage_error(error)
        return Response(status=204)


@contracts_namespace.route("/validate")
class ContractValidationResource(Resource):
    @contracts_namespace.expect(contract_body, validate=False)
    @contracts_namespace.doc(consumes=["text/plain", "application/yaml"])
    @contracts_namespace.response(200, "Contract if valid")
    @contracts_namespace.response(400, "Parsing errors if malformed", error_model)
    def post(self):
        if not _supported_media_type():
            return error_response(
                "UNSUPPORTED_MEDIA_TYPE",
                "Unsupported media type in request.",
            ), 415

        try:
            contract = _contract_service().parse_contract(
                request.get_data(as_text=True),
            )
        except (ValidationError, TypeError, ValueError, KeyError, yaml.YAMLError) as error:
            return _invalid_contract(error)

        return Response(
            json.dumps(contract.to_dict()),
            status=200,
            mimetype="application/json",
        )
