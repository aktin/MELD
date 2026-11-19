#!/bin/sh

set -eu

image_name="meld-orchestrator"

cd "$(dirname "$0")/../src"
version=$(cat VERSION)
docker build -t "ghcr.io/aktin/$image_name:$version" \
        --build-arg "IMAGE_VERSION=$version" \
        -f Dockerfile .
