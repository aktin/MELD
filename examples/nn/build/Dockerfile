FROM python:3.12-slim

ARG IMAGE_VERSION=dev

LABEL org.opencontainers.image.version="${IMAGE_VERSION}"
LABEL org.opencontainers.image.title="MELD runtime image example neural net"
LABEL org.opencontainers.image.description="This image is an example runtime inference container with a trained neural net for MELD. \
The included model was trained with artificially synthesized data."
LABEL org.opencontainers.image.source="https://github.com/aktin/MELD"
LABEL org.opencontainers.image.documentation="https://github.com/aktin/MELD/tree/main/examples/nn/MODEL_CARD.md"

RUN mkdir -p /input /output

COPY .context/artifact/ /artifact/
COPY .context/inference/ /inference/

WORKDIR /inference

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "inference.py"]
