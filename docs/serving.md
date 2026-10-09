# Serving a classifier over HTTP

```bash
pip install "adaptive-classifier[serve]"
python -m adaptive_classifier.serving ./my_classifier --port 8000
```

`./my_classifier` is a directory written by `classifier.save(...)`. It is self-contained: it holds the
encoder (ONNX), the tokenizer and the learned state, so the server starts without reaching the Hugging
Face Hub. (Directories saved with `include_onnx=False`, or by an older version, fetch the base model
from the Hub when they start.)

## Endpoints

| Method and path | Body | Returns |
|---|---|---|
| `GET /health` | | `{"status": "ok", "classes": 3}` |
| `GET /info` | | classes, examples per class, whether calibrated, whether updates are allowed |
| `POST /predict` | `{"text", "k"?, "abstain_below"?, "abstain_ood"?}` | `{"predictions": [{"label", "score"}], "abstained": false}` |
| `POST /predict_batch` | `{"texts": [...], "k"?}` | `{"predictions": [[...], ...]}` |
| `POST /predict_set` | `{"text", "alpha"?}` | conformal label set (needs `calibrate`) |
| `POST /ood` | `{"text", "threshold"?}` | `{"score", "threshold", "ood"}` |
| `POST /examples` | `{"texts": [...], "labels": [...]}` | add examples or new classes (updates only) |
| `POST /forget` | `{"labels": [...]}` | remove classes (updates only) |
| `POST /remove_examples` | `{"texts": [...], "label"?}` | remove examples (updates only) |

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
     -d '{"text": "where is my refund?", "k": 3}'
```

Bad input gets a 4xx with a message: 422 for a malformed request (empty text, wrong types, batch too
large) and 400 when the classifier rejects it (unknown label, calibration missing, ...). The interactive
API docs are at `/docs`.

## Updates

Learning at runtime is the point of the library, but it is off by default on a server:

```bash
export ADAPTIVE_CLASSIFIER_API_KEY=change-me
python -m adaptive_classifier.serving ./my_classifier --allow-updates --save-dir ./my_classifier
```

Update endpoints then require `Authorization: Bearer change-me`; prediction endpoints stay open.
`--save-dir` saves the classifier after every update, so a restart keeps what was learned.
Without an API key anyone who can reach the server can change the model, and the server warns about it.

## Concurrency and scaling

Every classifier operation takes a lock, so one process can safely serve concurrent requests while
learning. That means requests are processed one at a time inside a process. For more throughput run
several **processes** (for example several containers behind a load balancer); with a read-only
server this scales cleanly. Each process has its own copy of the model, so updates only reach the
worker that received them: run a single process if you enable updates.

From Python, the same pieces are available without HTTP: `await classifier.apredict(...)`,
`apredict_batch` and `aadd_examples` run in a worker thread so they do not block an event loop, and
`create_app(classifier)` returns an ASGI app you can mount in an existing FastAPI project.

## Docker

```bash
docker build -f docker/Dockerfile -t adaptive-classifier .
docker run --rm -p 8000:8000 -v "$PWD/my_classifier:/model:ro" adaptive-classifier
```

The image runs as a non-root user, has a health check on `/health`, and uses CPU-only PyTorch. CI builds it and queries it (`scripts/docker_smoke.py`, workflow `docker.yml`) whenever the Dockerfile, `pyproject.toml` or the server change, and weekly.
