#!/bin/sh

image_name="meld-orchestrator"
version="0.2.0-alpha"

cd ../MELD
docker build -t "ghcr.io/simhue/$image_name:latest" \
        -t "ghcr.io/simhue/$image_name:$version" \
        -f Dockerfile .