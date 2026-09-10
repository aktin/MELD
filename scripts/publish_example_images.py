#!/usr/bin/env python3

"""Build affected example images, push them, and record their digests."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

try:
    from .update_contract_digest import update_digest
except ImportError:
    from update_contract_digest import update_digest


ROOT = Path(__file__).resolve().parents[1]
SHA256_DIGEST = re.compile(r"sha256:[0-9a-fA-F]{64}")


def git_paths(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [path for path in result.stdout.decode().split("\0") if path]


def changed_example_names(before: str | None, after: str | None) -> list[str]:
    """Return example directory names touched by a push range.

    An empty or all-zero ``before`` is used for the first push of a branch and
    means that every example currently in the repository should be considered.
    """

    if not before or set(before) == {"0"}:
        changed_paths = git_paths("ls-tree", "-r", "--name-only", "-z", after or "HEAD", "--", "examples")
    else:
        changed_paths = git_paths("diff", "--name-only", "-z", before, after or "HEAD", "--", "examples")

    names = {
        parts[1]
        for path in changed_paths
        if (parts := Path(path).parts) and len(parts) >= 3 and parts[0] == "examples"
    }

    return sorted(names)


def image_config(example_name: str) -> tuple[Path, Path, Path, str, str]:
    example_dir = ROOT / "examples" / example_name
    build_dir = example_dir / "build"
    dockerfile = build_dir / "Dockerfile"
    contract_path = example_dir / "resources" / "contract.yaml"
    artifact_dir = example_dir / "artifact"
    inference_dir = example_dir / "inference"

    missing = [
        str(path.relative_to(ROOT))
        for path in (build_dir, dockerfile, contract_path, artifact_dir, inference_dir)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(f"Example {example_name!r} is missing: {', '.join(missing)}")

    try:
        contract: dict[str, Any] = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        image = contract["runtime"]["image"]
        image_name = image["name"]
        image_tag = image["tag"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as error:
        raise SystemExit(f"Could not read image configuration from {contract_path}: {error}") from error

    if not isinstance(image_name, str) or not image_name:
        raise SystemExit(f"runtime.image.name must be a non-empty string in {contract_path}")
    if not isinstance(image_tag, str) or not image_tag:
        raise SystemExit(f"runtime.image.tag must be a non-empty string in {contract_path}")
    if not image_name.startswith("ghcr.io/"):
        raise SystemExit(f"runtime.image.name must target ghcr.io in {contract_path}: {image_name}")

    return example_dir, build_dir, contract_path, image_name, image_tag


@contextmanager
def staged_build_context(example_dir: Path, build_dir: Path) -> Iterator[None]:
    """Stage files outside ``build/`` into the Docker context temporarily."""

    context_dir = build_dir / ".context"
    if context_dir.exists():
        raise SystemExit(f"Build context already exists; another build may be running: {context_dir}")

    try:
        context_dir.mkdir()
        shutil.copytree(example_dir / "artifact", context_dir / "artifact")
        shutil.copytree(example_dir / "inference", context_dir / "inference")
        yield
    finally:
        shutil.rmtree(context_dir, ignore_errors=True)


def pushed_digest(image_ref: str, metadata_file: Path) -> str:
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read build metadata for {image_ref}: {error}") from error

    digest = metadata.get("containerimage.digest") if isinstance(metadata, dict) else None
    if not isinstance(digest, str) or not SHA256_DIGEST.fullmatch(digest):
        inspect = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", image_ref, "--format", "{{.Manifest.Digest}}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        digest = inspect.stdout.strip()

    if not SHA256_DIGEST.fullmatch(digest):
        raise SystemExit(f"Docker did not return an immutable digest for {image_ref}: {digest!r}")
    return digest


def publish_example(example_name: str) -> None:
    example_dir, build_dir, contract_path, image_name, image_tag = image_config(example_name)
    image_ref = f"{image_name}:{image_tag}"

    print(f"Building affected example {example_name}: {image_ref}", flush=True)
    with staged_build_context(example_dir, build_dir):
        with tempfile.NamedTemporaryFile(prefix="example-image-", suffix=".json") as metadata_file:
            subprocess.run(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--build-arg",
                    f"IMAGE_VERSION={image_tag}",
                    "--file",
                    str(build_dir / "Dockerfile"),
                    "--tag",
                    image_ref,
                    "--metadata-file",
                    metadata_file.name,
                    "--push",
                    str(build_dir),
                ],
                cwd=ROOT,
                check=True,
            )
            digest = pushed_digest(image_ref, Path(metadata_file.name))

    update_digest(contract_path, digest)
    print(f"Updated {contract_path.relative_to(ROOT)} to {digest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", help="Previous commit in the push range")
    parser.add_argument("--after", help="New commit in the push range")
    args = parser.parse_args()

    examples = changed_example_names(args.before, args.after)
    if not examples:
        print("No example directories changed; nothing to build.")
        return

    for example_name in examples:
        publish_example(example_name)


if __name__ == "__main__":
    main()
