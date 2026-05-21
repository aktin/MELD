FROM python:3.12-slim

WORKDIR /

RUN mkdir -p /logs /jobs /app

COPY ./utils /utils
COPY ./Logger /Logger
COPY MELD/*.py /app/
COPY MELD/requirements.txt /app/requirements.txt
COPY MELD/InternalDataLoader /app/InternalDataLoader
COPY MELD/ModelEnvironment /app/ModelEnvironment
COPY ./MELD/ModelManager /app/ModelManager


WORKDIR /app

RUN pip install --no-cache-dir -r ./requirements.txt

CMD ["python", "inference.py"]