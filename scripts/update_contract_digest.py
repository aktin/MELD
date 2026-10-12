#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path


def update_digest(contract_path: Path, digest: str) -> None:
    if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        raise SystemExit(f"Invalid image digest: {digest!r}")

    lines = contract_path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated_lines: list[str] = []
    in_runtime = False
    in_image = False
    replaced = False

    for line in lines:
        body = line.rstrip("\r\n")
        line_ending = line[len(body) :]
        stripped = body.strip()
        indent = len(body) - len(body.lstrip(" "))
        is_comment = stripped.startswith("#")

        if not is_comment and stripped == "runtime:" and indent == 0:
            in_runtime = True
            in_image = False
        elif in_runtime and not is_comment and stripped == "image:" and indent == 2:
            in_image = True
        elif in_runtime and in_image and indent == 4 and re.match(r"digest:\s*", stripped):
            prefix = body[: body.index("digest:")]
            comment = ""
            value = body[body.index("digest:") + len("digest:") :]
            comment_match = re.search(r"\s+#.*$", value)
            if comment_match:
                comment = comment_match.group(0)
            updated_lines.append(f'{prefix}digest: "{digest}"{comment}{line_ending}')
            replaced = True
            continue
        elif not is_comment and indent == 0 and stripped and stripped != "runtime:":
            in_runtime = False
            in_image = False
        elif not is_comment and indent == 2 and stripped and stripped != "image:":
            in_image = False

        updated_lines.append(line)

    if not replaced:
        raise SystemExit(f"Could not find runtime.image.digest in {contract_path}")

    contract_path.write_text("".join(updated_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update runtime.image.digest in a MELD contract.")
    parser.add_argument("--contract", required=True, type=Path, help="Path to the contract YAML file")
    parser.add_argument("--digest", required=True, help="OCI image digest, for example sha256:...")
    args = parser.parse_args()

    update_digest(args.contract, args.digest)


if __name__ == "__main__":
    main()
