FROM python:3.12-slim

LABEL org.opencontainers.image.version="0.2.1-SNAPSHOT"
LABEL org.opencontainers.image.title="MELD Orchestrator"
LABEL org.opencontainers.image.description="The MELD (Medical Learning and Deployment) Orchestrator enables clinicians and researchers\
to run their own trained ML models."
LABEL org.opencontainers.image.source="https://github.com/aktin/MELD"

WORKDIR /

RUN mkdir -p /logs /jobs /app

WORKDIR /app

COPY *.py ./
COPY requirements.txt ./requirements.txt
COPY InternalDataLoader ./InternalDataLoader
COPY ModelEnvironment ./ModelEnvironment
#COPY InferenceScheduler ./InferenceScheduler
COPY ModelManager ./ModelManager
COPY ExecutionMonitor ./ExecutionMonitor
COPY utils ./utils
COPY Logger ./Logger
COPY resources ./resources
COPY Server ./Server

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash meld \
    && pip install --no-cache-dir -r ./requirements.txt \
    && chown -R meld:meld /logs /jobs

ENV TZ=UTC

USER meld
CMD ["sleep", "infinity"]
