#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-ghcr.io/simhue/meld/runtime_examples/sklearn}"
TAG="${TAG:-0.1.0}"

BUILD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(dirname -- "$BUILD_DIR")"
CONTEXT_DIR="$BUILD_DIR/.context"

if [[ -e "$CONTEXT_DIR" ]]; then
  echo "$CONTEXT_DIR already exists; remove it or finish the other build first" >&2
  exit 1
fi

cleanup() {
  rm -rf -- "$CONTEXT_DIR"
}
trap cleanup EXIT

mkdir -p "$CONTEXT_DIR"
cp -a "$EXAMPLE_DIR/artifact" "$CONTEXT_DIR/artifact"
cp -a "$EXAMPLE_DIR/inference" "$CONTEXT_DIR/inference"

docker build --label org.opencontainers.image.created="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
              --build-arg "IMAGE_VERSION=$TAG" \
              -t "$IMAGE_NAME:$TAG" \
              -t "$IMAGE_NAME:latest" \
              -f "$BUILD_DIR/Dockerfile" \
              "$BUILD_DIR"
