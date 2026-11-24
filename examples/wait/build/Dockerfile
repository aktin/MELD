FROM python:3.12-slim

ARG IMAGE_VERSION=dev

LABEL org.opencontainers.image.version="${IMAGE_VERSION}"
LABEL org.opencontainers.image.title="MELD runtime wait example"
LABEL org.opencontainers.image.description="Test runtime that waits and produces an empty result for MELD."
LABEL org.opencontainers.image.source="https://github.com/aktin/MELD"
LABEL org.opencontainers.image.documentation="https://github.com/aktin/MELD/tree/main/examples/wait/MODEL_CARD.md"

ENV PYTHONUNBUFFERED=1

RUN mkdir -p /input /output

COPY .context/inference/ /inference/

WORKDIR /inference

ENTRYPOINT ["python", "inference.py"]
