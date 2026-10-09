import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import test_support  # noqa: F401
except ModuleNotFoundError:
    from MELD.tests import test_support  # noqa: F401


class DatabaseConfigurationTest(unittest.TestCase):
    def run_import(self, **environment):
        code = "import InternalDataLoader.db; print(InternalDataLoader.db.engine.url.render_as_string(hide_password=False))"
        env = os.environ.copy()
        env.update(environment)
        env["PYTHONPATH"] = str(Path(__file__).parents[1])
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).parents[2],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_engine_uses_environment_and_first_password_line(self):
        with tempfile.TemporaryDirectory() as directory:
            password_file = Path(directory) / "password"
            password_file.write_text("secret\nignored\n", encoding="utf-8")
            result = self.run_import(
                DB_HOST="db.example",
                DB_PORT="5434",
                DB_USER="meld",
                DB_SCHEMA="warehouse",
                DB_PASSWORD_FILE=str(password_file),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("postgresql+psycopg2://meld:secret@db.example:5434/warehouse", result.stdout)

    def test_missing_password_file_exits_with_error(self):
        result = self.run_import(DB_PASSWORD_FILE="/tmp/meld-password-file-does-not-exist")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Database password file could not be found", result.stderr)

    def test_unset_password_file_exits_with_error(self):
        environment = os.environ.copy()
        environment.pop("DB_PASSWORD_FILE", None)
        environment["PYTHONPATH"] = str(Path(__file__).parents[1])
        result = subprocess.run(
            [sys.executable, "-c", "import InternalDataLoader.db"],
            cwd=Path(__file__).parents[2],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DB_PASSWORD_FILE is not set", result.stderr)


if __name__ == "__main__":
    unittest.main()
