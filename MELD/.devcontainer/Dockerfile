FROM python:3.14-slim

WORKDIR /

RUN mkdir -p /logs /jobs

RUN apt-get update
RUN apt-get install -y git docker-cli
RUN rm -rf /var/lib/apt/lists/*

COPY *.py ./
COPY requirements.txt ./requirements.txt
COPY InternalDataLoader ./InternalDataLoader
COPY ModelEnvironment ./ModelEnvironment
COPY ModelManager ./ModelManager
COPY utils ./utils
COPY Logger ./Logger
COPY Server ./Server

RUN pip install --no-cache-dir -r ./requirements.txt