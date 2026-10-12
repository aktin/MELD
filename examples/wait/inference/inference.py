import os
import time
from pathlib import Path


def main() -> None:
    with Path("/input/input.csv").open("rb"):
        pass

    wait_seconds = float(os.environ.get("WAIT_SECONDS", "10"))
    time.sleep(wait_seconds)

    Path("/output/output.csv").write_text("result\n", encoding="utf-8")


if __name__ == "__main__":
    main()
