import unittest
from importlib import reload
from unittest.mock import MagicMock, patch

import requests
import test_support  # noqa: F401
from docker.errors import APIError, ImageNotFound, NotFound

from ModelEnvironment import ExecutionStatus
from ModelEnvironment import docker_runtime


class DockerRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.context = MagicMock()
        self.context.execution_id = "execution-id"
        self.context.image_ref = "image:tag"
        self.context.contract.id = "contract-id"
        self.context.contract.runtime.environment_variables = {"MODE": "test"}
        docker_runtime.client = self.client

    def tearDown(self):
        docker_runtime.client = None

    def test_pull_image_supports_plain_and_authenticated_pulls(self):
        docker_runtime.pull_image("image:tag")
        docker_runtime.pull_image("private/image:tag", registry_api_key="token")

        self.client.images.pull.assert_any_call("image:tag")
        self.client.images.pull.assert_any_call(
            "private/image:tag", auth_config={"identitytoken": "token"}
        )

    def test_import_does_not_connect_to_docker(self):
        docker_runtime.client = None
        with patch("ModelEnvironment.docker_runtime.docker.from_env") as from_env:
            reload(docker_runtime)

        from_env.assert_not_called()
        self.assertIsNone(docker_runtime.client)
        docker_runtime.client = self.client

    def test_pull_image_streams_progress_and_rejects_error_events(self):
        callback = MagicMock()
        self.client.api.pull.return_value = [{"status": "Downloading"}, {"status": "Done"}]

        docker_runtime.pull_image("image", callback, "token")

        self.client.api.pull.assert_called_once_with(
            "image", stream=True, decode=True, auth_config={"identitytoken": "token"}
        )
        self.assertEqual(callback.call_count, 2)

        self.client.api.pull.return_value = [{"error": "denied"}]
        with self.assertRaisesRegex(RuntimeError, "denied"):
            docker_runtime.pull_image("image", callback)

    def test_pull_and_delete_translate_docker_errors(self):
        self.client.images.pull.side_effect = NotFound("missing")
        with self.assertRaisesRegex(RuntimeError, "not found"):
            docker_runtime.pull_image("image")

        self.client.images.remove.side_effect = APIError("failed")
        with self.assertRaisesRegex(RuntimeError, "Failed to delete image"):
            docker_runtime.delete_image("image")

    def test_image_queries_and_ensure_image_update_expected_status(self):
        self.client.images.get.return_value = object()
        self.assertTrue(docker_runtime.image_exists("image"))
        docker_runtime.ensure_image_exists(self.context)

        self.client.images.get.side_effect = ImageNotFound("missing")
        self.assertFalse(docker_runtime.image_exists("image"))
        with self.assertRaisesRegex(RuntimeError, "not found"):
            docker_runtime.ensure_image_exists(self.context)
        self.context.set_status.assert_called_with(ExecutionStatus.FAILED)

    def test_container_lifecycle_passes_environment_and_updates_status(self):
        container = MagicMock(name="runtime")
        container.name = "runtime"
        self.client.containers.create.return_value = container

        created = docker_runtime.create_container("image", self.context)
        docker_runtime.start_container(container, self.context)
        docker_runtime.stop_container(container, self.context)
        docker_runtime.destroy_container(container, self.context)

        self.assertIs(created, container)
        self.client.containers.create.assert_called_once_with(
            "image", environment={"MODE": "test"}, name="runtime_contract-id_execution-id"
        )
        self.context.set_status.assert_any_call(ExecutionStatus.CREATED)
        self.context.set_status.assert_any_call(ExecutionStatus.RUNNING)
        container.stop.assert_called_once_with()
        container.remove.assert_called_once_with()

    def test_start_and_stop_translate_api_errors(self):
        container = MagicMock(name="runtime")
        container.name = "runtime"
        container.start.side_effect = APIError("start failed")
        with self.assertRaisesRegex(RuntimeError, "Failed to start"):
            docker_runtime.start_container(container, self.context)

        container.stop.side_effect = APIError("stop failed")
        with self.assertRaisesRegex(RuntimeError, "Failed to stop"):
            docker_runtime.stop_container(container, self.context)

    def test_create_container_and_image_size_translate_errors(self):
        self.client.containers.create.side_effect = NotFound("missing")
        with self.assertRaisesRegex(RuntimeError, "not found"):
            docker_runtime.create_container("image", self.context)

        self.client.images.get.side_effect = APIError("failed")
        with self.assertRaisesRegex(RuntimeError, "Failed to get image"):
            docker_runtime.get_image_size("image", self.context)

    def test_get_image_size_returns_docker_size(self):
        self.client.images.get.return_value.attrs = {"Size": 123}

        self.assertEqual(docker_runtime.get_image_size("image", self.context), 123)

    def test_stream_container_logs_routes_stdout_and_stderr(self):
        container = MagicMock()
        container.attach.return_value = [(b"output\n", b"error\n")]
        logger = MagicMock()
        with patch.object(docker_runtime, "get_inference_logger", return_value=logger):
            docker_runtime.stream_container_logs(container, self.context)

        logger.info.assert_called_once_with("output")
        logger.error.assert_called_once_with("error")

    def test_wait_for_container_handles_success_failure_and_timeout(self):
        container = MagicMock()
        with patch.object(docker_runtime, "stream_container_logs"):
            container.wait.return_value = {"StatusCode": 0}
            self.assertEqual(docker_runtime.wait_for_container(container, self.context, 1), 0)
            self.context.set_status.assert_called_with(ExecutionStatus.SUCCESS)

            container.wait.return_value = {"StatusCode": 7}
            self.assertEqual(docker_runtime.wait_for_container(container, self.context, 1), 7)
            self.context.set_status.assert_called_with(ExecutionStatus.FAILED)

            container.wait.side_effect = requests.exceptions.ReadTimeout()
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                docker_runtime.wait_for_container(container, self.context, 1)
            container.kill.assert_called_once_with()
            self.context.set_status.assert_called_with(ExecutionStatus.TIMEOUT)


if __name__ == "__main__":
    unittest.main()
