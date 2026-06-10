FROM python:3.12-slim

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

CMD ["python", "main.py"]