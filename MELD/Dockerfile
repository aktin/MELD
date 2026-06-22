FROM python:3.12-slim

LABEL org.opencontainers.image.version="0.1.0-alpha"
LABEL org.opencontainers.image.title="MELD Orchestrator"
LABEL org.opencontainers.image.description="The MELD (Medical Learning and Deployment) Orchestrator enables clinicians and researchers\
to run their own trained ML models."
LABEL org.opencontainers.image.source="https://github.com/aktin/MELD"

WORKDIR /

RUN mkdir -p /logs /jobs /app

COPY *.py /app/
COPY requirements.txt /app/requirements.txt
COPY InternalDataLoader /app/InternalDataLoader
COPY ModelEnvironment /app/ModelEnvironment
COPY ModelManager /app/ModelManager
COPY utils /app/utils
COPY Logger /app/Logger


WORKDIR /app

RUN pip install --no-cache-dir -r ./requirements.txt

ENTRYPOINT ["python", "main.py"]