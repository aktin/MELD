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
COPY ModelManager ./ModelManager
COPY utils ./utils
COPY Logger ./Logger

RUN pip install --no-cache-dir -r ./requirements.txt

ENTRYPOINT ["python", "main.py"]