import os
import tempfile
import unittest
from pathlib import Path


RUNTIME_IMAGE = os.environ.get("MELD_TEST_RUNTIME_IMAGE")


@unittest.skipUnless(
    os.environ.get("MELD_RUN_RUNTIME_INTEGRATION") == "1" and RUNTIME_IMAGE,
    "set MELD_RUN_RUNTIME_INTEGRATION=1 and MELD_TEST_RUNTIME_IMAGE to run runtime integration tests",
)
class LiveRuntimeIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import docker

        try:
            cls.client = docker.from_env()
            cls.client.ping()
        except Exception as error:
            raise unittest.SkipTest(f"Docker daemon unavailable: {error}")

    def test_runtime_image_can_start_and_exit_successfully(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input"
            output_path = Path(directory) / "output"
            input_path.mkdir()
            output_path.mkdir()
            (input_path / "input.csv").write_text("ignored\n", encoding="utf-8")
            container = self.client.containers.create(
                RUNTIME_IMAGE,
                environment={"WAIT_SECONDS": "0"},
                volumes={
                    str(input_path): {"bind": "/input", "mode": "ro"},
                    str(output_path): {"bind": "/output", "mode": "rw"},
                },
            )
            try:
                container.start()
                self.assertEqual(container.wait(timeout=120)["StatusCode"], 0)
                self.assertTrue((output_path / "output.csv").is_file())
            finally:
                container.remove(force=True)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()


if __name__ == "__main__":
    unittest.main()
