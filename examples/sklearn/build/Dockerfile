FROM python:3.12-slim

ARG IMAGE_VERSION=dev

LABEL org.opencontainers.image.version="${IMAGE_VERSION}"
LABEL org.opencontainers.image.title="MELD runtime image example scikit-learn"
LABEL org.opencontainers.image.description="This image is an example runtime inference container with a trained scikit-learn Random Forest model for MELD. \
The included model was trained with artificially synthesized data."
LABEL org.opencontainers.image.source="https://github.com/aktin/MELD"
LABEL org.opencontainers.image.documentation="https://github.com/aktin/MELD/tree/main/examples/sklearn/MODEL_CARD.md"

ENV PYTHONUNBUFFERED=1

COPY .context/artifact/ /artifact/
COPY .context/inference/ /inference/

WORKDIR /inference

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "inference.py"]
