#!/bin/sh

image_name="meld-orchestrator"
version="0.2.1-SNAPSHOT"

cd ../MELD
docker build -t "ghcr.io/simhue/$image_name:latest" \
        -t "ghcr.io/simhue/$image_name:$version" \
        -f Dockerfile .