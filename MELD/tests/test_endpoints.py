import json
import os
import sys
import tempfile
import unittest
import uuid
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import test_support  # noqa: F401
from ModelManager import Contract

# The application currently treats the MELD directory as its import root.
MELD_ROOT = Path(__file__).resolve().parents[1]
if str(MELD_ROOT) not in sys.path:
    sys.path.insert(0, str(MELD_ROOT))

from main import create_app, main  # noqa: E402


VALID_CONTRACT = """
contract:
  name: example-contract
  description: Example contract for endpoint tests
  version: 1.0.0
runtime:
  framework: sklearn
  image:
    name: example/runtime
    tag: 1.0.0
    digest: sha256:example
input_schema:
  temporal_scope:
    type: relative
    value: P1D
    anchor: "2020-01-01T00:00:00Z"
  features:
    - name: age
      datatype: Int64
  query:
    type: sql
    statement: SELECT age FROM patients
output_schema:
  type: csv
  predictor:
    - name: prediction
      datatype: Float64
"""

STORED_CONTRACT_ID = "12345678-1234-4234-8234-123456789abc"


def stored_contract():
    payload = VALID_CONTRACT.replace(
        "contract:\n",
        f"contract:\n  id: {STORED_CONTRACT_ID}\n",
        1,
    )
    return Contract.from_yaml(StringIO(payload)), payload


class EndpointStubTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = create_app()
        app.config["TESTING"] = True
        cls.app = app
        cls.client = app.test_client()
        cls.service = app.extensions["contract_service"]
        cls.execution_service = app.extensions["execution_service"]

    def assert_stub(self, method, path, **kwargs):
        response = self.client.open(path, method=method, **kwargs)
        self.assertEqual(response.status_code, 501, msg=f"{method} {path}")
        self.assertEqual(
            response.get_json(),
            {"message": "Endpoint stub; implementation pending."},
        )

    def assert_uuid(self, value):
        self.assertEqual(str(uuid.UUID(value)), value)

    def test_list_contracts(self):
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                response = self.client.get("/contracts")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")
        self.assertEqual(response.get_json(), [])

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "OK")

    def test_endpoint_tester_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/html")
        page = response.get_data(as_text=True)
        self.assertIn("MELD endpoint tester", page)
        self.assertIn('id="execution-list"', page)
        self.assertIn("setInterval(pollExecutionStatuses, 1000)", page)
        self.assertIn("SUCCESS', 'FAILED', 'CANCELED', 'TIMEOUT", page)

    def test_version(self):
        response = self.client.get("/version")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_data(as_text=True))

    def test_application_factory_registers_api_blueprint(self):
        app = create_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        self.assertIn("/contracts", rules)
        self.assertIn("/contracts/<string:contractId>/executions", rules)
        self.assertNotIn(
            "/contracts/<string:contractId>/schedules",
            rules,
        )
        self.assertNotIn(
            "/contracts/<string:contractId>/schedules/<string:scheduleId>",
            rules,
        )

    @patch("main.app.run")
    def test_serve_command_starts_flask_server(self, run):
        self.assertEqual(
            main(["serve", "--host", "127.0.0.1", "--port", "5050"]),
            0,
        )
        run.assert_called_once_with(host="127.0.0.1", port=5050, debug=False)

    def test_create_contract_rejects_malformed_payload(self):
        response = self.client.post(
            "/contracts",
            data="contract:\n  name: example\n",
            content_type="application/yaml",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()["error"]["code"],
            "INVALID_CONTRACT",
        )

    def test_create_contract_rejects_json_media_type(self):
        response = self.client.post(
            "/contracts",
            json={"contract": {}},
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(
            response.get_json()["error"]["code"],
            "UNSUPPORTED_MEDIA_TYPE",
        )

    def test_get_contract(self):
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                response = self.client.get("/contracts/contract-id")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.get_json()["error"]["code"],
            "CONTRACT_NOT_FOUND",
        )

    def test_get_contract_returns_json_contract_info(self):
        contract, payload = stored_contract()
        with tempfile.TemporaryDirectory() as contract_directory:
            contract_path = Path(contract_directory) / contract.id / "contract.yaml"
            contract_path.parent.mkdir()
            contract_path.write_text(payload, encoding="utf-8")
            (Path(contract_directory) / contract.id / "status.json").write_text(
                '{"status": "available", "progress": 100}',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                response = self.client.get(f"/contracts/{contract.id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")
        payload = response.get_json()
        self.assertEqual(payload["id"], contract.id)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["contract"], contract.to_dict())
        self.assertNotIn("progress", payload)

    def test_create_contract_pulls_before_storing(self):
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch.object(self.service, "_image_available", return_value=False):
                    with patch.object(self.service, "_start_image_pull") as start_pull:
                        response = self.client.post(
                            "/contracts",
                            data=VALID_CONTRACT,
                            content_type="application/yaml",
                        )
                contract_id = response.headers["Location"].rsplit("/", 1)[-1]
                self.assertTrue(
                    (Path(contract_directory) / contract_id / "contract.yaml").exists()
                )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_data(), b"")
        self.assert_uuid(contract_id)
        start_pull.assert_called_once_with(
            ANY,
            "example/runtime:1.0.0@sha256:example",
            None,
        )
        self.assertEqual(response.headers["Location"], f"/contracts/{contract_id}")

    def test_create_ready_contract_returns_empty_created_response(self):
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch.object(self.service, "_image_available", return_value=True):
                    response = self.client.post(
                        "/contracts",
                        data=VALID_CONTRACT,
                        content_type="application/yaml",
                    )
            contract_id = response.headers["Location"].rsplit("/", 1)[-1]
            contract_directory_path = Path(contract_directory) / contract_id
            self.assertTrue(contract_directory_path.is_dir())
            self.assertTrue((contract_directory_path / "contract.yaml").is_file())
            self.assertTrue((contract_directory_path / "status.json").is_file())
            stored = Contract.from_yaml(contract_directory_path / "contract.yaml")
            self.assertEqual(stored.id, contract_id)
            self.assertFalse((Path(contract_directory) / "contract.yaml").exists())
            self.assertFalse((Path(contract_directory) / "contract.progress").exists())

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_data(), b"")
        self.assert_uuid(contract_id)
        self.assertEqual(response.headers["Location"], f"/contracts/{contract_id}")

    def test_create_contract_replaces_submitted_id(self):
        payload = VALID_CONTRACT.replace(
            "contract:\n",
            "contract:\n  id: caller-provided-id\n",
            1,
        )
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch.object(self.service, "_image_available", return_value=True):
                    response = self.client.post(
                        "/contracts",
                        data=payload,
                        content_type="application/yaml",
                    )

        contract_id = response.headers["Location"].rsplit("/", 1)[-1]
        self.assertEqual(response.status_code, 201)
        self.assert_uuid(contract_id)
        self.assertNotEqual(contract_id, "caller-provided-id")

    def test_duplicate_contract_returns_conflict(self):
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch.object(self.service, "_image_available", return_value=True):
                    first_response = self.client.post(
                        "/contracts",
                        data=VALID_CONTRACT,
                        content_type="application/yaml",
                    )
                    second_response = self.client.post(
                        "/contracts",
                        data=VALID_CONTRACT,
                        content_type="application/yaml",
                    )
                contract_count = len(
                    list(Path(contract_directory).glob("*/contract.yaml"))
                )

        self.assertEqual(first_response.status_code, 201)
        self.assertEqual(second_response.status_code, 409)
        self.assertEqual(contract_count, 1)

    def test_fingerprint_collision_does_not_reject_different_contract(self):
        changed_contract = VALID_CONTRACT.replace(
            "Example contract for endpoint tests",
            "Different contract with the same fingerprint",
        )
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch.object(self.service, "_image_available", return_value=True):
                    with patch.object(
                        Contract,
                        "fingerprint",
                        new_callable=PropertyMock,
                        return_value="same-fingerprint",
                    ):
                        first_contract = self.service.parse_contract(VALID_CONTRACT)
                        self.service.create_contract(first_contract)
                        second_contract = self.service.parse_contract(changed_contract)
                        self.service.create_contract(second_contract)
                contract_count = len(
                    list(Path(contract_directory).glob("*/contract.yaml"))
                )

        self.assertEqual(contract_count, 2)
        self.assert_uuid(first_contract.id)
        self.assert_uuid(second_contract.id)
        self.assertNotEqual(first_contract.id, second_contract.id)

    def test_retry_existing_contract_returns_empty_accepted_response(self):
        contract, payload = stored_contract()
        with tempfile.TemporaryDirectory() as contract_directory:
            contract_path = Path(contract_directory, contract.id, "contract.yaml")
            contract_path.parent.mkdir()
            contract_path.write_text(
                payload,
                encoding="utf-8",
            )
            Path(contract_directory, contract.id, "status.json").write_text(
                json.dumps({"status": "failed", "progress": 0}),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch.object(self.service, "_start_image_pull") as start_pull:
                    response = self.client.post(
                        "/contracts",
                        data=VALID_CONTRACT,
                        content_type="application/yaml",
                    )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_data(), b"")
        self.assertEqual(response.headers["Location"], f"/contracts/{contract.id}")
        start_pull.assert_called_once_with(
            ANY,
            "example/runtime:1.0.0@sha256:example",
            None,
        )

    def test_create_contract_does_not_store_when_pull_cannot_start(self):
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch.object(self.service, "_image_available", return_value=False):
                    with patch.object(
                        self.service,
                        "_start_image_pull",
                        side_effect=RuntimeError("executor unavailable"),
                    ):
                        response = self.client.post(
                            "/contracts",
                            data=VALID_CONTRACT,
                            content_type="application/yaml",
                        )

            self.assertFalse(
                any(Path(contract_directory).glob("*/contract.yaml"))
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.get_json()["error"]["code"],
            "IMAGE_PULL_FAILED",
        )

    def test_create_contract_passes_registry_api_key_to_pull(self):
        contract = Contract.from_yaml(StringIO(VALID_CONTRACT))
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch.object(self.service, "_image_available", return_value=False):
                    with patch.object(self.service, "_start_image_pull") as start_pull:
                        response = self.client.post(
                            "/contracts",
                            data=VALID_CONTRACT,
                            content_type="application/yaml",
                            headers={
                                "X-Docker-Registry-API-Key": "registry-token",
                            },
                        )

        self.assertEqual(response.status_code, 202)
        start_pull.assert_called_once_with(
            ANY,
            "example/runtime:1.0.0@sha256:example",
            "registry-token",
        )

    def test_validate_contract_returns_json(self):
        contract = Contract.from_yaml(StringIO(VALID_CONTRACT))
        response = self.client.post(
            "/contracts/validate",
            data=VALID_CONTRACT,
            content_type="application/yaml",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")
        self.assertEqual(response.get_json(), contract.to_dict())

    def test_get_contract_includes_installation_progress(self):
        contract, payload = stored_contract()
        with tempfile.TemporaryDirectory() as contract_directory:
            contract_path = Path(contract_directory) / contract.id / "contract.yaml"
            contract_path.parent.mkdir()
            contract_path.write_text(payload, encoding="utf-8")
            progress_path = Path(contract_directory) / contract.id / "status.json"
            progress_path.write_text(
                '{"status": "pulling", "progress": 42.5}',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                response = self.client.get(f"/contracts/{contract.id}")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")
        self.assertEqual(payload["id"], contract.id)
        self.assertEqual(payload["status"], "installing")
        self.assertEqual(payload["progress"], 42.5)
        self.assertEqual(payload["contract"], contract.to_dict())
        self.assertEqual(set(payload), {"id", "status", "progress", "contract"})

    def test_get_contract_requeues_failed_installation(self):
        contract, payload = stored_contract()
        with tempfile.TemporaryDirectory() as contract_directory:
            contract_path = Path(contract_directory, contract.id, "contract.yaml")
            contract_path.parent.mkdir()
            contract_path.write_text(
                payload,
                encoding="utf-8",
            )
            Path(contract_directory, contract.id, "status.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "progress": 42.5,
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch.object(self.service, "_image_available", return_value=False):
                    with patch.object(self.service, "_start_image_pull") as start_pull:
                        response = self.client.get(f"/contracts/{contract.id}")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "installing")
        self.assertEqual(payload["progress"], 42.5)
        start_pull.assert_called_once_with(
            ANY,
            "example/runtime:1.0.0@sha256:example",
        )

    def test_list_contracts_omits_contract_body(self):
        contract, payload = stored_contract()
        with tempfile.TemporaryDirectory() as contract_directory:
            contract_path = Path(contract_directory, contract.id, "contract.yaml")
            contract_path.parent.mkdir()
            contract_path.write_text(
                payload,
                encoding="utf-8",
            )
            Path(contract_directory, contract.id, "status.json").write_text(
                '{"status": "pulling", "progress": 25}',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                response = self.client.get("/contracts")

        self.assertEqual(
            response.get_json(),
            [{"id": contract.id, "status": "installing", "progress": 25}],
        )

    def test_pull_retries_and_resumes_after_error(self):
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                pull_image = unittest.mock.Mock(
                    side_effect=[RuntimeError("connection reset"), None]
                )
                service = type(self.service)()
                with patch.object(service, "_pull_runtime_image", pull_image):
                    with patch.object(service, "_pull_retry_delay", return_value=0):
                        service._pull_image("retry-contract", "example/runtime:1.0.0")
                progress = service._read_progress("retry-contract")

        self.assertEqual(pull_image.call_count, 2)
        self.assertEqual(progress["status"], "available")
        self.assertEqual(progress["progress"], 100)

    def test_delete_contract(self):
        response = self.client.delete("/contracts/contract-id")
        self.assertEqual(response.status_code, 404)

    def test_delete_contract_removes_contract_directory(self):
        contract, payload = stored_contract()
        with tempfile.TemporaryDirectory() as contract_directory:
            contract_path = Path(contract_directory, contract.id, "contract.yaml")
            contract_path.parent.mkdir()
            contract_path.write_text(payload, encoding="utf-8")
            Path(contract_directory, contract.id, "status.json").write_text(
                '{"status": "available", "progress": 100}',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch("ModelManager.remove_runtime"):
                    response = self.client.delete(f"/contracts/{contract.id}")

            self.assertFalse(contract_path.parent.exists())

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.get_data(), b"")

    def test_validate_contract(self):
        response = self.client.post("/contracts/validate")
        self.assertEqual(response.status_code, 404)

    def test_list_executions(self):
        response = self.client.get("/contracts/contract-id/executions")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.get_json()["error"]["code"],
            "EXECUTION_RESOURCE_NOT_FOUND",
        )

    def test_start_execution(self):
        with patch.object(
            self.execution_service,
            "start_execution",
            return_value="execution-id",
        ) as start_execution:
            response = self.client.post(
                "/contracts/contract-id/executions",
                json={},
            )

        start_execution.assert_called_once_with("contract-id")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_data(), b"")
        self.assertEqual(
            response.headers["Location"],
            "/contracts/contract-id/executions/execution-id",
        )

    def test_get_execution(self):
        response = self.client.get(
            "/contracts/contract-id/executions/execution-id",
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.get_json()["error"]["code"],
            "EXECUTION_RESOURCE_NOT_FOUND",
        )

    def test_cancel_execution(self):
        response = self.client.delete(
            "/contracts/contract-id/executions/execution-id",
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.get_json()["error"]["code"],
            "EXECUTION_RESOURCE_NOT_FOUND",
        )

    def test_cancel_completed_execution_returns_bad_request(self):
        from werkzeug.exceptions import BadRequest

        with patch.object(
            self.execution_service,
            "cancel_execution",
            side_effect=BadRequest("Execution has already completed."),
        ):
            response = self.client.delete(
                "/contracts/contract-id/executions/execution-id",
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()["error"]["code"],
            "EXECUTION_ALREADY_COMPLETED",
        )

    def test_get_execution_logs(self):
        response = self.client.get(
            "/contracts/contract-id/executions/execution-id/logs",
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.get_json()["error"]["code"],
            "EXECUTION_RESOURCE_NOT_FOUND",
        )

    def test_final_execution_states_read_completed_logs(self):
        from ModelEnvironment import ExecutionStatus

        for state in (
            ExecutionStatus.SUCCESS,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELED,
            ExecutionStatus.TIMEOUT,
        ):
            with self.subTest(state=state):
                context = MagicMock(status={"status": state.value})
                with patch.object(
                    self.execution_service,
                    "get_execution",
                    return_value=context,
                ), patch.object(
                    self.execution_service,
                    "read_log",
                    return_value=f"{state.value}\n",
                ) as read_log, patch.object(
                    self.execution_service,
                    "stream_log",
                ) as stream_log:
                    response = self.client.get(
                        "/contracts/contract-id/executions/execution-id/logs",
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_data(as_text=True), f"{state.value}\n")
                read_log.assert_called_once_with("contract-id", "execution-id")
                stream_log.assert_not_called()

    def test_running_execution_streams_logs(self):
        from ModelEnvironment import ExecutionStatus

        context = MagicMock(status={"status": ExecutionStatus.RUNNING.value})
        with patch.object(
            self.execution_service,
            "get_execution",
            return_value=context,
        ), patch.object(
            self.execution_service,
            "read_log",
        ) as read_log, patch.object(
            self.execution_service,
            "stream_log",
            return_value=iter(["running\n"]),
        ) as stream_log:
            response = self.client.get(
                "/contracts/contract-id/executions/execution-id/logs",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "running\n")
        read_log.assert_not_called()
        stream_log.assert_called_once_with("contract-id", "execution-id")

    def test_successful_execution_returns_result_archive(self):
        from ModelEnvironment import ExecutionStatus

        context = MagicMock(status={"status": ExecutionStatus.SUCCESS.value})
        with patch.object(
            self.execution_service, "get_execution", return_value=context
        ), patch.object(
            self.execution_service,
            "get_result_archive",
            return_value=BytesIO(b"zip-data"),
        ) as get_archive:
            response = self.client.get(
                "/contracts/contract-id/executions/execution-id"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/zip")
        self.assertEqual(response.get_data(), b"zip-data")
        get_archive.assert_called_once_with("contract-id", "execution-id")

    def test_running_execution_returns_json_status(self):
        from ModelEnvironment import ExecutionStatus

        status = {"status": ExecutionStatus.RUNNING.value}
        with patch.object(
            self.execution_service,
            "get_execution",
            return_value=MagicMock(status=status),
        ):
            response = self.client.get(
                "/contracts/contract-id/executions/execution-id"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), status)

    def test_execution_route_maps_storage_errors(self):
        with patch.object(
            self.execution_service,
            "get_execution",
            side_effect=OSError("read failed"),
        ):
            response = self.client.get(
                "/contracts/contract-id/executions/execution-id"
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"]["code"], "STORAGE_ERROR")

    def test_contract_routes_map_storage_errors(self):
        with patch.object(
            self.service, "get_contract_info", side_effect=OSError("read failed")
        ):
            response = self.client.get("/contracts/contract-id")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"]["code"], "STORAGE_ERROR")

    def test_validate_contract_rejects_unsupported_media_type(self):
        response = self.client.post(
            "/contracts/validate",
            data=VALID_CONTRACT,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(
            response.get_json()["error"]["code"], "UNSUPPORTED_MEDIA_TYPE"
        )

    def test_swagger_documentation_is_available(self):
        response = self.client.get("/swagger.json")
        self.assertEqual(response.status_code, 200)

        paths = response.get_json()["paths"]
        self.assertIn("/contracts", paths)
        self.assertIn("/contracts/{contractId}/executions", paths)
        self.assertIn(
            "/contracts/{contractId}/executions/{executionId}",
            paths,
        )
        self.assertEqual(
            paths["/contracts/{contractId}/executions/{executionId}/logs"]["get"]["produces"],
            ["text/plain", "application/octet-stream"],
        )
        self.assertNotIn("/contracts/{contractId}/schedules", paths)
        self.assertNotIn(
            "/contracts/{contractId}/schedules/{scheduleId}",
            paths,
        )
        parameters = paths["/contracts"]["post"]["parameters"]
        self.assertEqual(
            paths["/contracts"]["post"]["consumes"],
            ["text/plain", "application/yaml"],
        )
        body_parameter = next(
            parameter
            for parameter in parameters
            if parameter.get("in") == "body"
        )
        self.assertEqual(
            body_parameter["schema"],
            {"$ref": "#/definitions/ContractTextPayload"},
        )
        contract_schema = response.get_json()["definitions"]["ContractPayload"]
        self.assertEqual(contract_schema["type"], "object")
        self.assertIn("contract", contract_schema["properties"])
        self.assertIn("runtime", contract_schema["properties"])
        self.assertTrue(
            any(
                parameter.get("name") == "X-Docker-Registry-API-Key"
                and parameter.get("in") == "header"
                for parameter in parameters
            )
        )
        responses = paths["/contracts"]["post"]["responses"]
        self.assertIn("202", responses)

    def test_track_and_read_pull_progress(self):
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                contract_id = "test-contract-123"

                # Initial status: None before tracking
                self.assertIsNone(self.service._read_progress(contract_id))

                # Step 1: Initiated
                self.service._track_pull_progress(contract_id, "pulling")
                progress = self.service._read_progress(contract_id)
                self.assertIsNotNone(progress)
                self.assertEqual(progress["status"], "pulling")
                self.assertEqual(progress["progress"], 0)
                self.assertEqual(set(progress), {"status", "progress"})

                # Step 2: Docker pull layer event
                self.service._track_pull_progress(
                    contract_id,
                    {
                        "status": "Downloading",
                        "id": "layer-1",
                        "progressDetail": {"current": 50, "total": 100},
                    },
                )
                progress = self.service._read_progress(contract_id)
                self.assertEqual(progress["status"], "Downloading")
                self.assertEqual(progress["progress"], 50.0)
                self.assertEqual(set(progress), {"status", "progress"})

                # Step 3: Second layer event
                self.service._track_pull_progress(
                    contract_id,
                    {
                        "status": "Downloading",
                        "id": "layer-2",
                        "progressDetail": {"current": 100, "total": 100},
                    },
                )
                progress = self.service._read_progress(contract_id)
                self.assertEqual(progress["progress"], 75.0)
                self.assertEqual(set(progress), {"status", "progress"})

                # Step 4: Finished successfully
                self.service._track_pull_progress(contract_id, "available")
                progress = self.service._read_progress(contract_id)
                self.assertEqual(progress["status"], "available")
                self.assertEqual(progress["progress"], 100)
                self.assertEqual(set(progress), {"status", "progress"})

                # Verify file existence on disk
                progress_file = self.service._progress_path(contract_id)
                self.assertTrue(progress_file.exists())

    def test_track_pull_progress_failure(self):
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                contract_id = "failed-contract-456"
                self.service._track_pull_progress(
                    contract_id,
                    "failed",
                    error="Image pull timed out",
                )
                progress = self.service._read_progress(contract_id)
                self.assertIsNotNone(progress)
                self.assertEqual(progress["status"], "failed")
                self.assertEqual(progress["error"], "Image pull timed out")
                self.assertEqual(set(progress), {"status", "progress", "error"})


if __name__ == "__main__":
    unittest.main()
