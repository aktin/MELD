#!/bin/sh

cd ../MELD
docker build -t ghcr.io/simhue/meld:latest -t ghcr.io/simhue/meld:0.1.0 -f Dockerfile .