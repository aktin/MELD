FROM python:3-alpine

WORKDIR /app

COPY . /app

RUN mkdir -p /shared

RUN pip install -r requirements.txt

RUN apk add --no-cache docker-cli

ENTRYPOINT ["python", "main.py"]