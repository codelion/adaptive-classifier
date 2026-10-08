# Serve a saved AdaptiveClassifier over HTTP.
#
#   docker build -f docker/Dockerfile -t adaptive-classifier .
#   docker run --rm -p 8000:8000 -v "$PWD/my_classifier:/model:ro" adaptive-classifier
#   curl -s localhost:8000/predict -H 'content-type: application/json' -d '{"text": "where is my refund?"}'
#
# Updates are off by default. To allow them (and persist what is learned), mount the
# model read-write and pass flags, for example:
#   docker run --rm -p 8000:8000 -v "$PWD/my_classifier:/model" \
#     -e ADAPTIVE_CLASSIFIER_API_KEY=change-me adaptive-classifier \
#     /model --host 0.0.0.0 --allow-updates --save-dir /model
#
# A classifier saved with the default `save()` includes an ONNX copy of the encoder and the
# tokenizer, so it starts without reaching the Hugging Face Hub. One saved with
# `include_onnx=False` fetches its base model from the Hub on startup; mount a populated
# Hugging Face cache at /cache/huggingface (or bake the model in) for offline use.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# CPU-only PyTorch keeps the image a few GB smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[serve]"

RUN useradd --create-home --uid 1000 app && mkdir -p /cache/huggingface && chown -R app /cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["python", "-m", "adaptive_classifier.serving"]
CMD ["/model", "--host", "0.0.0.0", "--port", "8000"]
