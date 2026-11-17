import os
import unittest


DOCKER_IMAGE = os.environ.get("MELD_TEST_DOCKER_IMAGE")


@unittest.skipUnless(
    os.environ.get("MELD_RUN_DOCKER_INTEGRATION") == "1" and DOCKER_IMAGE,
    "set MELD_RUN_DOCKER_INTEGRATION=1 and MELD_TEST_DOCKER_IMAGE to run Docker integration tests",
)
class DockerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import docker

        try:
            cls.client = docker.from_env()
            cls.client.ping()
        except Exception as error:
            raise unittest.SkipTest(f"Docker daemon unavailable: {error}")
        cls.created_containers = []

    @classmethod
    def tearDownClass(cls):
        for container in cls.created_containers:
            try:
                container.remove(force=True)
            except Exception:
                pass
        cls.client.close()

    def test_image_pull_and_inspection(self):
        image = self.client.images.pull(DOCKER_IMAGE)
        self.assertTrue(image.id)
        self.assertGreater(image.attrs["Size"], 0)

    def test_container_lifecycle_and_output(self):
        container = self.client.containers.create(
            DOCKER_IMAGE,
            command=["sh", "-c", "echo meld-integration; echo meld-error >&2"],
        )
        self.created_containers.append(container)
        container.start()
        result = container.wait(timeout=120)
        output = container.logs(stdout=True, stderr=True).decode("utf-8")

        self.assertEqual(result["StatusCode"], 0)
        self.assertIn("meld-integration", output)
        self.assertIn("meld-error", output)

    def test_nonzero_exit_is_observable(self):
        container = self.client.containers.create(DOCKER_IMAGE, command=["sh", "-c", "exit 7"])
        self.created_containers.append(container)
        container.start()

        self.assertEqual(container.wait(timeout=120)["StatusCode"], 7)


if __name__ == "__main__":
    unittest.main()
