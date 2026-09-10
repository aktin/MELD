FROM python:3.12-slim

LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.title="MELD runtime image example decision forest"
LABEL org.opencontainers.image.description="This image is an example runtime inference container with a trained decision forest for MELD. \
The included model was trained with artificially synthesized data."
LABEL org.opencontainers.image.source="https://github.com/aktin/MELD"
LABEL org.opencontainers.image.documentation="https://github.com/aktin/MELD/tree/main/examples/tfdf/MODEL_CARD.md"

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=""

#RUN mkdir -p /input /output

COPY ./artifact /artifact
COPY ./inference /inference

WORKDIR /inference

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "inference.py"]
