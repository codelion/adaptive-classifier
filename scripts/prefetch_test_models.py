"""Download every Hugging Face model the test suite uses into the local cache.

CI runs this once, with retries, and the test jobs then run with
HF_HUB_OFFLINE=1. That way a Hub outage or rate limit (HTTP 429) fails one
clearly-labelled step instead of dozens of unrelated tests, and the matrix
legs do not each hit the Hub.

    python scripts/prefetch_test_models.py
"""

import sys
import time

from huggingface_hub import snapshot_download

# Keep in sync with the model names used in tests/ (tests/test_ci_models.py
# fails when a test uses a model that is not listed here).
MODELS = [
    "prajjwal1/bert-tiny",
    "bert-base-uncased",
    "distilbert-base-uncased",
    "distilbert/distilbert-base-cased",
    "google-bert/bert-large-cased",
    "answerdotai/ModernBERT-base",
]

ATTEMPTS = 6
WEIGHTS = ["*.safetensors"]
FALLBACK_WEIGHTS = ["*.bin"]
SMALL_FILES = ["*.json", "*.txt", "*.model", "*.py"]


def fetch(model: str):
    """Download config, tokenizer and weights (safetensors, else .bin)."""
    path = snapshot_download(model, allow_patterns=SMALL_FILES + WEIGHTS)
    import glob
    import os
    if not glob.glob(os.path.join(path, "*.safetensors")):
        snapshot_download(model, allow_patterns=SMALL_FILES + FALLBACK_WEIGHTS)


def main() -> int:
    failed = []
    for model in MODELS:
        for attempt in range(1, ATTEMPTS + 1):
            try:
                fetch(model)
                print(f"ok    {model}")
                break
            except Exception as error:  # network, 429, 5xx
                wait = min(120, 5 * 2 ** attempt)
                print(f"retry {model} ({attempt}/{ATTEMPTS}): {type(error).__name__}: "
                      f"{str(error)[:200]} - waiting {wait}s", flush=True)
                if attempt == ATTEMPTS:
                    failed.append(model)
                else:
                    time.sleep(wait)
    if failed:
        print(f"Could not download: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
