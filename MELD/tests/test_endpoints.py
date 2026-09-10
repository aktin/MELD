import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, patch

from flask import Flask
import yaml

from ModelManager import Contract

# The application currently treats the MELD directory as its import root.
MELD_ROOT = Path(__file__).resolve().parents[1]
if str(MELD_ROOT) not in sys.path:
    sys.path.insert(0, str(MELD_ROOT))

from Server.endpoints import bp  # noqa: E402
from main import create_app, main  # noqa: E402


VALID_CONTRACT = """
contract:
  name: Example contract
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


class EndpointStubTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(bp)
        cls.app = app
        cls.client = app.test_client()

    def assert_stub(self, method, path, **kwargs):
        response = self.client.open(path, method=method, **kwargs)
        self.assertEqual(response.status_code, 501, msg=f"{method} {path}")
        self.assertEqual(
            response.get_json(),
            {"message": "Endpoint stub; implementation pending."},
        )

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

    def test_version(self):
        response = self.client.get("/version")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_data(as_text=True))

    def test_application_factory_registers_api_blueprint(self):
        app = create_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        self.assertIn("/contracts", rules)
        self.assertIn("/contracts/<string:contractId>/inferences", rules)
        self.assertIn(
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
        contract = Contract.from_yaml(StringIO(VALID_CONTRACT))
        with tempfile.TemporaryDirectory() as contract_directory:
            contract_path = Path(contract_directory) / f"{contract.id}.yaml"
            contract_path.write_text(VALID_CONTRACT, encoding="utf-8")
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                response = self.client.get(f"/contracts/{contract.id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")
        payload = response.get_json()
        self.assertEqual(payload["id"], contract.id)
        self.assertEqual(payload["contract"], VALID_CONTRACT)
        self.assertEqual(payload["contract_json"], contract.to_dict())
        self.assertIsNone(payload["pull_status"])

    def test_create_contract_pulls_before_storing(self):
        contract = Contract.from_yaml(StringIO(VALID_CONTRACT))
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch("Server.contracts._image_available", return_value=False):
                    with patch("Server.contracts._start_image_pull") as start_pull:
                        response = self.client.post(
                            "/contracts",
                            data=VALID_CONTRACT,
                            content_type="application/yaml",
                        )
                self.assertTrue(
                    (Path(contract_directory) / f"{contract.id}.yaml").exists()
                )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.mimetype, "application/json")
        payload = response.get_json()
        self.assertEqual(payload["id"], contract.id)
        self.assertIsInstance(payload["contract"], str)
        self.assertEqual(yaml.safe_load(payload["contract"]), contract.to_dict())
        self.assertEqual(payload["contract_json"], contract.to_dict())
        self.assertIsNone(payload["pull_status"])
        start_pull.assert_called_once_with(
            ANY,
            "example/runtime:1.0.0@sha256:example",
            None,
        )
        self.assertEqual(response.headers["Location"], f"/contracts/{contract.id}")

    def test_create_contract_does_not_store_when_pull_cannot_start(self):
        contract = Contract.from_yaml(StringIO(VALID_CONTRACT))
        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                with patch("Server.contracts._image_available", return_value=False):
                    with patch(
                        "Server.contracts._start_image_pull",
                        side_effect=RuntimeError("executor unavailable"),
                    ):
                        response = self.client.post(
                            "/contracts",
                            data=VALID_CONTRACT,
                            content_type="application/yaml",
                        )

            self.assertFalse(
                (Path(contract_directory) / f"{contract.id}.yaml").exists()
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
                with patch("Server.contracts._image_available", return_value=False):
                    with patch("Server.contracts._start_image_pull") as start_pull:
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

    def test_get_contract_includes_pull_status_for_polling(self):
        contract = Contract.from_yaml(StringIO(VALID_CONTRACT))
        with tempfile.TemporaryDirectory() as contract_directory:
            contract_path = Path(contract_directory) / f"{contract.id}.yaml"
            contract_path.write_text(VALID_CONTRACT, encoding="utf-8")
            progress_path = Path(contract_directory) / f"{contract.id}.progress"
            progress_path.write_text(
                '{"contract_id": "' + contract.id + '", "status": "pulling", "progress": 42.5}',
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
        self.assertEqual(payload["contract"], VALID_CONTRACT)
        self.assertEqual(payload["contract_json"], contract.to_dict())
        self.assertEqual(payload["pull_status"]["status"], "pulling")
        self.assertEqual(payload["pull_status"]["progress"], 42.5)

    def test_delete_contract(self):
        self.assert_stub("DELETE", "/contracts/contract-id")

    def test_validate_contract(self):
        self.assert_stub("POST", "/contracts/contract-id/validate")

    def test_list_inferences(self):
        self.assert_stub("GET", "/contracts/contract-id/inferences")

    def test_start_inference(self):
        self.assert_stub(
            "POST",
            "/contracts/contract-id/inferences",
            json={},
        )

    def test_get_inference(self):
        self.assert_stub(
            "GET",
            "/contracts/contract-id/inferences/inference-id",
        )

    def test_cancel_inference(self):
        self.assert_stub(
            "DELETE",
            "/contracts/contract-id/inferences/inference-id",
        )

    def test_get_inference_logs(self):
        self.assert_stub(
            "GET",
            "/contracts/contract-id/inferences/inference-id/logs",
        )

    def test_list_schedules(self):
        self.assert_stub("GET", "/contracts/contract-id/schedules")

    def test_create_schedule(self):
        self.assert_stub(
            "POST",
            "/contracts/contract-id/schedules",
            json={},
        )

    def test_get_schedule(self):
        self.assert_stub(
            "GET",
            "/contracts/contract-id/schedules/schedule-id",
        )

    def test_delete_schedule(self):
        self.assert_stub(
            "DELETE",
            "/contracts/contract-id/schedules/schedule-id",
        )

    def test_swagger_documentation_is_available(self):
        response = self.client.get("/swagger.json")
        self.assertEqual(response.status_code, 200)

        paths = response.get_json()["paths"]
        self.assertIn("/contracts", paths)
        self.assertIn("/contracts/{contractId}/inferences", paths)
        self.assertIn(
            "/contracts/{contractId}/inferences/{inferenceId}",
            paths,
        )
        self.assertIn("/contracts/{contractId}/schedules", paths)
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
        from Server.contracts import (
            _progress_path,
            _read_progress,
            _track_pull_progress,
        )

        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                contract_id = "test-contract-123"

                # Initial status: None before tracking
                self.assertIsNone(_read_progress(contract_id))

                # Step 1: Initiated
                _track_pull_progress(contract_id, "pulling")
                progress = _read_progress(contract_id)
                self.assertIsNotNone(progress)
                self.assertEqual(progress["status"], "pulling")
                self.assertEqual(progress["progress"], 0)

                # Step 2: Docker pull layer event
                _track_pull_progress(
                    contract_id,
                    {
                        "status": "Downloading",
                        "id": "layer-1",
                        "progressDetail": {"current": 50, "total": 100},
                    },
                )
                progress = _read_progress(contract_id)
                self.assertEqual(progress["status"], "Downloading")
                self.assertEqual(progress["progress"], 50.0)
                self.assertIn("layer-1", progress["layers"])
                self.assertEqual(progress["layers"]["layer-1"]["current"], 50)

                # Step 3: Second layer event
                _track_pull_progress(
                    contract_id,
                    {
                        "status": "Downloading",
                        "id": "layer-2",
                        "progressDetail": {"current": 100, "total": 100},
                    },
                )
                progress = _read_progress(contract_id)
                self.assertEqual(progress["progress"], 75.0)

                # Step 4: Finished successfully
                _track_pull_progress(contract_id, "available")
                progress = _read_progress(contract_id)
                self.assertEqual(progress["status"], "available")
                self.assertEqual(progress["progress"], 100)

                # Verify file existence on disk
                progress_file = _progress_path(contract_id)
                self.assertTrue(progress_file.exists())

    def test_track_pull_progress_failure(self):
        from Server.contracts import (
            _read_progress,
            _track_pull_progress,
        )

        with tempfile.TemporaryDirectory() as contract_directory:
            with patch.dict(
                os.environ,
                {"MELD_CONTRACT_DIRECTORY": contract_directory},
            ):
                contract_id = "failed-contract-456"
                _track_pull_progress(
                    contract_id,
                    "failed",
                    error="Image pull timed out",
                )
                progress = _read_progress(contract_id)
                self.assertIsNotNone(progress)
                self.assertEqual(progress["status"], "failed")
                self.assertEqual(progress["error"], "Image pull timed out")


if __name__ == "__main__":
    unittest.main()
