FROM python:3.14-slim

LABEL org.opencontainers.image.version="0.2.1-SNAPSHOT"
LABEL org.opencontainers.image.title="MELD Orchestrator"
LABEL org.opencontainers.image.description="The MELD (Medical Learning and Deployment) Orchestrator enables clinicians and researchers\
to run their own trained ML models."
LABEL org.opencontainers.image.source="https://github.com/aktin/MELD"

WORKDIR /

RUN mkdir -p /logs /contracts /schedules /app

WORKDIR /app

COPY *.py ./
COPY entrypoint.sh ./
COPY requirements.txt ./requirements.txt
COPY InternalDataLoader ./InternalDataLoader
COPY ModelEnvironment ./ModelEnvironment
COPY ExecutionScheduler ./ExecutionScheduler
COPY ModelManager ./ModelManager
COPY ExecutionMonitor ./ExecutionMonitor
COPY utils ./utils
COPY Logger ./Logger
COPY resources ./resources
COPY Server ./Server
COPY templates ./templates

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata gosu \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash meld \
    && pip install --no-cache-dir -r ./requirements.txt gunicorn \
    && chown -R meld:meld /logs /contracts /schedules

ENV TZ=UTC
ENV MELD_API_HOST=0.0.0.0
ENV MELD_API_PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c 'import http.client, os, sys; connection = http.client.HTTPConnection("127.0.0.1", int(os.environ.get("MELD_API_PORT", "5000")), timeout=3); connection.request("GET", "/health"); sys.exit(0 if connection.getresponse().status == 200 else 1)'

USER root
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
CMD ["sh", "-c", "exec gunicorn --bind \"${MELD_API_HOST}:${MELD_API_PORT}\" --workers 1 --threads 4 --access-logfile - --error-logfile - 'app:create_app()'"]
