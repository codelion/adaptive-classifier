"""Serve an AdaptiveClassifier over HTTP.

    pip install "adaptive-classifier[serve]"
    python -m adaptive_classifier.serving ./my_classifier --port 8000

or, from Python::

    from adaptive_classifier.serving import create_app
    app = create_app(classifier)             # any ASGI server can run it

Endpoints: ``GET /health``, ``GET /info``, ``POST /predict``,
``POST /predict_batch``, ``POST /predict_set``, ``POST /ood``, and, only when
updates are allowed, ``POST /examples``, ``POST /forget`` and
``POST /remove_examples``.

The classifier takes a lock around every operation, so one process is safe to
share across requests; scale throughput with ``--workers`` (separate
processes, each with its own copy and its own updates) rather than threads.
Updates therefore only reach the worker that handled them. Run a single
worker if you enable them, and pass ``--save-dir`` to persist what was learned.
"""

import argparse
import asyncio
import hmac
import logging
from typing import Any, Dict, List, Optional

from .classifier import AdaptiveClassifier

logger = logging.getLogger(__name__)


def create_app(
    classifier: AdaptiveClassifier,
    *,
    allow_updates: bool = False,
    api_key: Optional[str] = None,
    max_batch: int = 256,
    max_text_chars: int = 20_000,
    save_dir: Optional[str] = None,
):
    """Build a FastAPI app around `classifier`.

    Args:
        classifier: A trained (or loaded) classifier.
        allow_updates: Enable ``/examples``, ``/forget`` and ``/remove_examples``.
        api_key: If set, update endpoints require ``Authorization: Bearer <key>``.
            Strongly recommended with ``allow_updates``.
        max_batch: Largest accepted batch (texts per request).
        max_text_chars: Longest accepted text, in characters.
        save_dir: If set, the classifier is saved there after every update.
    """
    try:
        from fastapi import Depends, FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field
    except ImportError as error:          # pragma: no cover - depends on the environment
        raise ImportError(
            'Serving needs FastAPI: pip install "adaptive-classifier[serve]"'
        ) from error

    if allow_updates and not api_key:
        logger.warning("Updates are enabled without an api_key: anyone who can reach this "
                       "server can change the classifier")

    class Prediction(BaseModel):
        label: str
        score: float

    class PredictRequest(BaseModel):
        text: str = Field(min_length=1, max_length=max_text_chars)
        k: int = Field(default=5, ge=0, le=1000)
        abstain_below: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        abstain_ood: bool = False

    class PredictResponse(BaseModel):
        predictions: List[Prediction]
        abstained: bool

    class BatchRequest(BaseModel):
        texts: List[str] = Field(min_length=1, max_length=max_batch)
        k: int = Field(default=5, ge=0, le=1000)

    class BatchResponse(BaseModel):
        predictions: List[List[Prediction]]

    class SetRequest(BaseModel):
        text: str = Field(min_length=1, max_length=max_text_chars)
        alpha: float = Field(default=0.1, gt=0.0, lt=1.0)

    class OodRequest(BaseModel):
        text: str = Field(min_length=1, max_length=max_text_chars)
        threshold: Optional[float] = Field(default=None, gt=0.0)

    class ExamplesRequest(BaseModel):
        texts: List[str] = Field(min_length=1, max_length=max_batch)
        labels: List[str] = Field(min_length=1, max_length=max_batch)

    class ForgetRequest(BaseModel):
        labels: List[str] = Field(min_length=1, max_length=max_batch)

    class RemoveRequest(BaseModel):
        texts: List[str] = Field(min_length=1, max_length=max_batch)
        label: Optional[str] = None

    app = FastAPI(title="Adaptive Classifier", version=_version())

    @app.exception_handler(ValueError)
    async def bad_request(_: Request, error: ValueError):
        # The classifier raises ValueError for unusable input (unknown label, empty text, ...).
        return JSONResponse(status_code=400, content={"detail": str(error)})

    def pairs(items) -> List[Dict[str, Any]]:
        return [{"label": label, "score": float(score)} for label, score in items]

    def require_updates(request: Request):
        if not allow_updates:
            raise HTTPException(status_code=403, detail="Updates are disabled on this server")
        if api_key:
            header = request.headers.get("authorization", "")
            supplied = header[7:] if header.lower().startswith("bearer ") else ""
            if not hmac.compare_digest(supplied.encode(), api_key.encode()):
                raise HTTPException(status_code=401, detail="Missing or invalid API key",
                                    headers={"WWW-Authenticate": "Bearer"})

    def persist():
        if save_dir:
            classifier.save(save_dir)

    @app.get("/health")
    async def health():
        return {"status": "ok", "classes": len(classifier.label_to_id)}

    @app.get("/info")
    async def info():
        stats = classifier.get_memory_stats()
        return {
            "version": _version(),
            "model": getattr(classifier, "_model_name", None),
            "classes": sorted(classifier.label_to_id),
            "examples_per_class": stats.get("examples_per_class", {}),
            "calibrated": bool(classifier.calibration),
            "updates_allowed": allow_updates,
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict(body: PredictRequest):
        result = await classifier.apredict(
            body.text, body.k, abstain_below=body.abstain_below, abstain_ood=body.abstain_ood
        )
        asked_to_abstain = body.abstain_below is not None or body.abstain_ood
        abstained = asked_to_abstain and not result and body.k > 0
        return {"predictions": pairs(result), "abstained": abstained}

    @app.post("/predict_batch", response_model=BatchResponse)
    async def predict_batch(body: BatchRequest):
        if any(not t or len(t) > max_text_chars for t in body.texts):
            raise HTTPException(status_code=422, detail=f"Each text must be 1-{max_text_chars} characters")
        result = await classifier.apredict_batch(body.texts, body.k)
        return {"predictions": [pairs(r) for r in result]}

    @app.post("/predict_set")
    async def predict_set(body: SetRequest):
        result = await asyncio.to_thread(classifier.predict_set, body.text, body.alpha)
        return {"labels": pairs(result)}

    @app.post("/ood")
    async def ood(body: OodRequest):
        score = await asyncio.to_thread(classifier.ood_score, body.text)
        limit = classifier.config.ood_threshold if body.threshold is None else body.threshold
        return {"score": None if score == float("inf") else score, "threshold": limit,
                "ood": score > limit}

    @app.post("/examples", dependencies=[Depends(require_updates)])
    async def add_examples(body: ExamplesRequest):
        if any(not t or len(t) > max_text_chars for t in body.texts):
            raise HTTPException(status_code=422, detail=f"Each text must be 1-{max_text_chars} characters")
        await classifier.aadd_examples(body.texts, body.labels)
        persist()
        return {"added": len(body.texts), "classes": sorted(classifier.label_to_id)}

    @app.post("/forget", dependencies=[Depends(require_updates)])
    async def forget(body: ForgetRequest):
        await asyncio.to_thread(classifier.forget, body.labels)
        persist()
        return {"classes": sorted(classifier.label_to_id)}

    @app.post("/remove_examples", dependencies=[Depends(require_updates)])
    async def remove_examples(body: RemoveRequest):
        removed = await asyncio.to_thread(classifier.remove_examples, body.texts, body.label)
        persist()
        return {"removed": removed, "classes": sorted(classifier.label_to_id)}

    return app


def _version() -> str:
    from . import __version__
    return __version__


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m adaptive_classifier.serving",
        description="Serve a saved AdaptiveClassifier over HTTP.",
    )
    parser.add_argument("model", help="Directory from classifier.save(), or a Hub id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-updates", action="store_true",
                        help="Enable /examples, /forget and /remove_examples")
    parser.add_argument("--api-key", default=None,
                        help="Bearer token required for updates (or set ADAPTIVE_CLASSIFIER_API_KEY)")
    parser.add_argument("--save-dir", default=None, help="Save the classifier here after every update")
    parser.add_argument("--max-batch", type=int, default=256)
    parser.add_argument("--no-onnx", action="store_true", help="Use PyTorch even on CPU")
    args = parser.parse_args(argv)

    import os
    try:
        import uvicorn
    except ImportError as error:          # pragma: no cover
        raise SystemExit('uvicorn is required: pip install "adaptive-classifier[serve]"') from error

    classifier = AdaptiveClassifier.load(args.model, use_onnx=False if args.no_onnx else "auto")
    app = create_app(
        classifier,
        allow_updates=args.allow_updates,
        api_key=args.api_key or os.environ.get("ADAPTIVE_CLASSIFIER_API_KEY"),
        max_batch=args.max_batch,
        save_dir=args.save_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
