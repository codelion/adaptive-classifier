"""Smoke-test the serving image: save a classifier, mount it, query the container.

    docker build -f docker/Dockerfile -t adaptive-classifier .
    python scripts/docker_smoke.py adaptive-classifier

Uses a tiny randomly initialised encoder, so it needs no Hugging Face access, and
starts the container from the saved directory alone (ONNX encoder + tokenizer).
"""

import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

WORDS = "great terrible okay product service love hate awful fine good bad slow fast cheap broke works the a is it very".split()
PORT = 18765


def build_model(directory: Path) -> Path:
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    from adaptive_classifier import AdaptiveClassifier

    base = directory / "base"
    base.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (base / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    torch.manual_seed(0)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=64)).save_pretrained(base)
    BertTokenizerFast(str(base / "vocab.txt"), do_lower_case=True).save_pretrained(base)

    clf = AdaptiveClassifier(str(base), device="cpu")
    clf.add_examples(
        ["great love good", "love fine good", "works great fast", "good great love",
         "terrible awful bad", "hate broke bad", "awful slow hate", "bad terrible broke"],
        ["positive"] * 4 + ["negative"] * 4,
    )
    model = directory / "model"
    clf.save(str(model))
    return model


def call(path: str, body: dict | None = None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read())


def main(image: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        model = build_model(Path(tmp))
        # The image runs as uid 1000; make the read-only mount readable by it.
        subprocess.run(["chmod", "-R", "a+rX", str(model)], check=True)
        container = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", f"127.0.0.1:{PORT}:8000",
             "-v", f"{model}:/model:ro", image],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        try:
            deadline = time.time() + 180
            while True:
                try:
                    status, health = call("/health")
                    break
                except (urllib.error.URLError, ConnectionError, OSError):
                    if time.time() > deadline:
                        raise SystemExit("server did not become healthy:\n" + subprocess.run(
                            ["docker", "logs", container], capture_output=True, text=True).stdout)
                    time.sleep(2)
            assert status == 200, health
            status, result = call("/predict", {"text": "great love"})
            assert status == 200 and result, result
            print("health:", health)
            print("predict:", result)

            # Updates must be refused unless the server was started with --allow-updates.
            try:
                call("/examples", {"texts": ["x"], "labels": ["y"]})
                raise SystemExit("update was accepted on a read-only server")
            except urllib.error.HTTPError as error:
                assert error.code == 403, error.code
            print("OK")
            return 0
        finally:
            logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
            print(logs.stdout[-2000:], logs.stderr[-2000:])
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "adaptive-classifier"))
