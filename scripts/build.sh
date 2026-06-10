#!/bin/sh

cd ../MELD
docker build -t ghcr.io/simhue/meld:latest -f Dockerfile .