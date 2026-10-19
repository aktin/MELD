import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ENTRYPOINT = Path(__file__).resolve().parents[1] / "entrypoint.sh"


class DockerEntrypointTest(unittest.TestCase):
    def run_entrypoint(self, uid="0", group_exists=True, socket_exists=True):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            log = bin_dir / "commands.log"
            commands = {
                "id": f"echo {uid}",
                "stat": "echo 9876" if socket_exists else "exit 1",
                "getent": "exit 0" if group_exists else "exit 2",
                "groupadd": 'printf "groupadd %s\\n" "$*" >> "$COMMAND_LOG"',
                "usermod": 'printf "usermod %s\\n" "$*" >> "$COMMAND_LOG"',
                "gosu": 'printf "gosu <%s>\\n" "$@" >> "$COMMAND_LOG"; '
                        'printf "HOME=%s\\n" "$HOME" >> "$COMMAND_LOG"',
            }
            for name, body in commands.items():
                command = bin_dir / name
                command.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
                command.chmod(0o755)
            result = subprocess.run(
                ["/bin/bash", str(ENTRYPOINT), "/bin/echo", "hello world"],
                env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                     "COMMAND_LOG": str(log)},
                capture_output=True,
                text=True,
            )
            return result, log.read_text() if log.exists() else ""

    def test_drops_privileges_before_executing_command(self):
        for group_exists in (True, False):
            with self.subTest(group_exists=group_exists):
                result, log = self.run_entrypoint(group_exists=group_exists)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = "" if group_exists else "groupadd --gid 9876 docker-socket-9876\n"
                expected += (
                    "usermod --append --groups 9876 meld\n"
                    "gosu <meld>\ngosu </bin/echo>\ngosu <hello world>\n"
                    "HOME=/home/meld\n"
                )
                self.assertEqual(log, expected)

    def test_socket_errors_stop_startup(self):
        result, log = self.run_entrypoint(socket_exists=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(log, "")

    def test_non_root_executes_command_without_setup(self):
        result, log = self.run_entrypoint(uid="1000", socket_exists=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "hello world\n")
        self.assertEqual(log, "")


if __name__ == "__main__":
    unittest.main()
