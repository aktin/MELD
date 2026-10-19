#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-ghcr.io/aktin/meld-runtime-examples-wait}"
TAG="${TAG:-0.1.0}"

BUILD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(dirname -- "$BUILD_DIR")"

docker build --label org.opencontainers.image.created="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
              --build-arg "IMAGE_VERSION=$TAG" \
              -t "$IMAGE_NAME:$TAG" \
              -t "$IMAGE_NAME:latest" \
              -f "$BUILD_DIR/Dockerfile" \
              "$EXAMPLE_DIR"
