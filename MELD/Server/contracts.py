"""Contract upload and retrieval endpoints."""

import json
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, Literal

from flask import Response, request, url_for
from flask_restx import Resource
from jsonschema import ValidationError
import yaml
from six import StringIO

from ModelManager import Contract, remove_runtime
from utils.config import CONTRACT_DIRECTORY

from .api import (
    contract_body,
    contracts as contracts_namespace,
    error_model,
    error_response,
    not_implemented,
)

logger = logging.getLogger(__name__)


def _contract_directory() -> Path:
    return Path(os.environ.get("MELD_CONTRACT_DIRECTORY", CONTRACT_DIRECTORY))


def _contract_path(contract_id: str) -> Path:
    return _contract_directory() / f"{contract_id}.yaml"


def _progress_path(contract_id: str) -> Path:
    return _contract_directory() / f"{contract_id}.progress"


def _read_contract(contract_id: str) -> Contract:
    with _contract_path(contract_id).open(encoding="utf-8") as contract_file:
        return Contract.from_yaml(contract_file)


def _parse_contract() -> Contract:
    payload = request.get_data(as_text=True)
    if not payload.strip():
        raise ValueError("The request body must not be empty.")
    return Contract.from_yaml(StringIO(payload))


def _image_reference(contract: Contract) -> str:
    return contract.runtime.image.construct_image_ref()


def _image_available(image_ref: str) -> bool:
    from ModelEnvironment.docker_runtime import image_exists

    return image_exists(image_ref)


def _read_progress(contract_id: str) -> dict[str, Any] | str | None:
    path = _progress_path(contract_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as progress_file:
            content = progress_file.read()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
    except Exception as error:
        logger.exception("Unable to read progress for contract %s: %s", contract_id, error)
        return None


def _track_pull_progress(
    contract_id: str,
    progress: Literal["pulling", "available", "failed"] | dict[str, Any] | int | float | str,
    error: str | None = None,
) -> None:
    path = _progress_path(contract_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_data = _read_progress(contract_id)
        current_data: dict[str, Any] = (
            existing_data
            if isinstance(existing_data, dict)
            else {
                "contract_id": contract_id,
                "status": "pulling",
                "progress": 0,
                "layers": {},
            }
        )

        if isinstance(progress, str):
            current_data["status"] = progress
            if progress == "available":
                current_data["progress"] = 100
            elif progress == "failed":
                if error:
                    current_data["error"] = error
        elif isinstance(progress, (int, float)):
            current_data["status"] = "pulling"
            current_data["progress"] = progress
        elif isinstance(progress, dict):
            event_status = progress.get("status", "")
            layer_id = progress.get("id")
            progress_detail = progress.get("progressDetail", {})

            if layer_id:
                layer_info = current_data.setdefault("layers", {}).setdefault(layer_id, {})
                layer_info["status"] = event_status
                if "current" in progress_detail:
                    layer_info["current"] = progress_detail["current"]
                if "total" in progress_detail:
                    layer_info["total"] = progress_detail["total"]

            layers = current_data.get("layers", {})
            total_bytes = sum(
                l.get("total", 0) for l in layers.values() if isinstance(l, dict)
            )
            current_bytes = sum(
                l.get("current", 0) for l in layers.values() if isinstance(l, dict)
            )
            if total_bytes > 0:
                current_data["progress"] = round((current_bytes / total_bytes) * 100, 2)

            current_data["last_event"] = progress
            if event_status:
                current_data["status"] = event_status

        if error:
            current_data["status"] = "failed"
            current_data["error"] = error

        temp_path = path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as progress_file:
            json.dump(current_data, progress_file, indent=2)
        temp_path.replace(path)
    except Exception as e:
        logger.exception("Unable to write progress for contract %s: %s", contract_id, e)


def _pull_image(
    contract_id: str,
    image_ref: str,
    registry_api_key: str | None = None,
) -> None:
    # This function runs in the single pull worker. Docker invokes the callback
    # as layer progress arrives, allowing the polling GET to observe updates.
    try:
        from ModelEnvironment.docker_runtime import pull_image

        _track_pull_progress(contract_id, "pulling")
        pull_kwargs: dict[str, Any] = {
            "progress_callback": lambda event: _track_pull_progress(contract_id, event),
        }
        if registry_api_key:
            pull_kwargs["registry_api_key"] = registry_api_key
        pull_image(image_ref, **pull_kwargs)
    except Exception as error:
        # Keep the failure visible to polling clients; the worker must not
        # turn an image error into an unhandled executor exception.
        logger.exception("Unable to pull image %s", image_ref)
        _track_pull_progress(contract_id, "failed", error=str(error))
        return

    _track_pull_progress(contract_id, "available")


def _store_contract(contract: Contract) -> None:
    path = _contract_path(contract.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as contract_file:
            yaml.safe_dump(contract.to_dict(), contract_file, sort_keys=False)
    except FileExistsError:
        raise
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _delete_contract(contract: Contract) -> None:
    contract_path = _contract_path(contract.id)
    progress_path = _progress_path(contract.id)
    try:
        os.remove(contract_path)
        os.remove(progress_path)
    except Exception:
        raise


def _to_contract_info(contract: Contract) -> dict[str, Any] | None:
    return {
        "id": contract.id,
        "contract": _contract_path(contract.id).read_text(encoding="utf-8"),
        "contract_json": contract.to_dict(),
        "pull_status": _get_pull_status(contract.id),
    }

def _contract_info_response(contract_data: Contract | list[Contract], status: int, with_location: bool = False):
    """Return the stored YAML and parsed contract data for polling clients."""

    payload = _to_contract_info(contract_data) if isinstance(contract_data, Contract) else [_to_contract_info(c) for c in contract_data]
    return Response(
        json.dumps(payload),
        status=status,
        mimetype="application/json",
        headers={
            "Location": url_for(
                "meld.contracts_contract_resource",
                contract_id=contract_data.id,
            ),
        } if with_location and isinstance(contract_data, Contract) else None,
    )


_pull_executor = ThreadPoolExecutor(max_workers=1)
_contract_operation_lock = threading.Lock()


def _get_pull_status(contract_id: str) -> dict[str, Any] | None:
    progress = _read_progress(contract_id)
    if isinstance(progress, dict):
        return progress
    return None


def _start_image_pull(
    contract: Contract,
    image_ref: str,
    registry_api_key: str | None = None,
) -> None:
    # The executor deliberately has one worker, so pulls run one after another.
    # The request thread only queues the work and remains available for polling.
    _track_pull_progress(contract.id, "pulling")
    try:
        _pull_executor.submit(
            _pull_image,
            contract.id,
            image_ref,
            registry_api_key,
        )
    except Exception as error:
        _track_pull_progress(contract.id, "failed", error=str(error))
        raise


def _storage_error(error: Exception):
    return error_response(
        "STORAGE_ERROR",
        "Could not retrieve contract.",
        str(error),
    ), 500


@contracts_namespace.route("", strict_slashes=False)
class ContractsResource(Resource):
    @contracts_namespace.response(200, "List of contracts")
    def get(self):
        contracts = []
        directory = _contract_directory()
        if directory.exists():
            contracts = [
                _read_contract(path.stem)
                for path in sorted(directory.glob("*.yaml"))
            ]
        return _contract_info_response(contracts, 200, False)

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
    @contracts_namespace.response(201, "Contract created; image is available")
    @contracts_namespace.response(202, "Contract created; image pull initiated")
    @contracts_namespace.response(400, "Contract is malformed", error_model)
    @contracts_namespace.response(409, "Contract already exists", error_model)
    @contracts_namespace.response(502, "Image pull failed", error_model)
    def post(self):
        if request.mimetype and request.mimetype not in (
            "text/plain",
            "application/yaml",
            "text/yaml",
            "application/x-yaml",
        ):
            return error_response(
                "UNSUPPORTED_MEDIA_TYPE",
                "Unsupported media type in request.",
            ), 415

        try:
            contract = _parse_contract()
        except (
                ValidationError,
                TypeError,
                ValueError,
                KeyError,
                yaml.YAMLError,
        ) as error:
            return error_response("INVALID_CONTRACT", "Contract is malformed.", str(error)), 400

        registry_api_key = request.headers.get("X-Docker-Registry-API-Key")
        try:
            image_ref = _image_reference(contract)
        except Exception as error:
            return error_response(
                "IMAGE_UNAVAILABLE",
                "The runtime image could not be found.",
                str(error),
            ), 502

        with _contract_operation_lock:
            path_exists = _contract_path(contract.id).exists()
            pull_state = _get_pull_status(contract.id)
            pull_failed = (
                pull_state
                and (
                    pull_state.get("contract_id") == contract.id
                    or pull_state.get("reference") == image_ref
                )
                and pull_state.get("status") == "failed"
            )

            if path_exists:
                if not pull_failed:
                    return error_response("CONTRACT_EXISTS", "Contract already exists."), 409

                try:
                    _start_image_pull(contract, image_ref, registry_api_key)
                except Exception as error:
                    return error_response(
                        "IMAGE_PULL_FAILED",
                        "The runtime image pull could not be restarted.",
                        str(error),
                    ), 502
                return _contract_info_response(contract, 202)

            try:
                image_available = _image_available(image_ref)
            except Exception as error:
                return error_response(
                    "IMAGE_UNAVAILABLE",
                    "The runtime image could not be found.",
                    str(error),
                ), 502

            if image_available:
                _track_pull_progress(contract.id, "available")
            else:
                try:
                    _start_image_pull(contract, image_ref, registry_api_key)
                except Exception as error:
                    return error_response(
                        "IMAGE_PULL_FAILED",
                        "The runtime image pull could not be started.",
                        str(error),
                    ), 502

            try:
                _store_contract(contract)
            except FileExistsError:
                return error_response("CONTRACT_EXISTS", "Contract already exists."), 409
            except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
                return error_response(
                    "STORAGE_ERROR",
                    "Could not store contract",
                    str(error),
                ), 500

            return _contract_info_response(contract, 201 if image_available else 202, True)


@contracts_namespace.route("/<string:contract_id>")
@contracts_namespace.doc(params={"contract_id": "Contract identifier"})
class ContractResource(Resource):
    @contracts_namespace.response(200, "Contract YAML, JSON data, and pull status")
    @contracts_namespace.response(404, "Contract does not exist", error_model)
    def get(self, contract_id):
        try:
            contract = _read_contract(contract_id)
        except FileNotFoundError:
            return error_response("CONTRACT_NOT_FOUND", "Contract does not exist."), 404
        except (
                OSError,
                ValidationError,
                TypeError,
                ValueError,
                KeyError,
                yaml.YAMLError,
        ) as error:
            return _storage_error(error)

        if contract.id != contract_id:
            return error_response("CONTRACT_NOT_FOUND", "Contract does not exist."), 404
        return _contract_info_response(contract, 200)

    @contracts_namespace.response(204, "Contract deleted")
    @contracts_namespace.response(404, "Contract does not exist", error_model)
    def delete(self, contract_id):
        try:
            contract = _read_contract(contract_id)
        except FileNotFoundError:
            return error_response("CONTRACT_NOT_FOUND", "Contract does not exist."), 404
        except (
                OSError,
                ValidationError,
                TypeError,
                ValueError,
                KeyError,
                yaml.YAMLError,
        ) as error:
            return _storage_error(error)

        if contract.id != contract_id:
            return error_response("CONTRACT_NOT_FOUND", "Contract does not exist."), 404

        _delete_contract(contract)
        remove_runtime(contract)
        return Response(status=204, mimetype="application/json")


@contracts_namespace.route("/validate")
class ContractValidationResource(Resource):
    @contracts_namespace.expect(contract_body, validate=False)
    @contracts_namespace.doc(consumes=["text/plain", "application/yaml"])
    @contracts_namespace.response(200, "Contract if valid")
    @contracts_namespace.response(400, "Parsing errors if malformed", error_model)
    def post(self):
        if request.mimetype and request.mimetype not in (
                "text/plain",
                "application/yaml",
                "text/yaml",
                "application/x-yaml",
        ):
            return error_response(
                "UNSUPPORTED_MEDIA_TYPE",
                "Unsupported media type in request.",
            ), 415

        try:
            contract = _parse_contract()
        except (
                ValidationError,
                TypeError,
                ValueError,
                KeyError,
                yaml.YAMLError,
        ) as error:
            return error_response("INVALID_CONTRACT", "Contract is malformed.", str(error)), 400

        return Response(
            json.dumps(contract.to_dict()),
            status=200,
            mimetype="application/json",
        )
