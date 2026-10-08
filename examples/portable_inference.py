"""Run a saved AdaptiveClassifier without PyTorch, FAISS or this library.

A saved classifier directory is plain files:

    config.json        label maps, per-class training counts, blend weights
    model.safetensors  class prototypes + the neural head weights
    onnx/model*.onnx   the transformer encoder (token embeddings out)

so inference outside Python only needs three pieces: an ONNX encoder, a
tokenizer, and the small amount of math below. This script is the reference
for that math. Port it to C#, Node, Go or anything else with an ONNX runtime;
docs/deployment.md explains each step.

Usage:
    pip install onnxruntime numpy safetensors tokenizers
    python examples/portable_inference.py ./my_classifier "text to classify"

It needs `tokenizer.json` next to the model. `AdaptiveClassifier.save` does not
write one, so create it once with:

    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained("<base model>").save_pretrained("./my_classifier")
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file
from tokenizers import Tokenizer


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class PortableClassifier:
    def __init__(self, directory, pooling=None, prefer_quantized=True):
        root = Path(directory)
        self.cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
        settings = self.cfg.get("config") or {}
        self.max_length = settings.get("max_length", 512)
        self.prototype_weight = settings.get("prototype_weight", 0.7)
        self.neural_weight = settings.get("neural_weight", 0.3)
        self.new_threshold = settings.get("new_class_example_threshold", 10)
        self.new_prototype_weight = settings.get("new_class_prototype_weight", 0.3)
        self.new_neural_weight = settings.get("new_class_neural_weight", 0.7)

        # 'auto' is resolved at run time in Python (it reads the base model's
        # sentence-transformers config), so it is never written to disk as
        # 'mean' or 'cls'. Pass `pooling` yourself when config.json says 'auto'
        # or has no pooling key: use what the base model was trained with
        # ('mean' for most sentence-transformers models, 'cls' for BGE). Classifiers
        # saved before 0.2.0 have no pooling key and behave as 'cls'.
        self.pooling = pooling or settings.get("pooling")
        if self.pooling not in ("mean", "cls"):
            raise ValueError(
                "config.json does not fix a pooling mode; pass pooling='mean' or 'cls' "
                "to match the base model"
            )

        tensors = load_file(str(root / "model.safetensors"))
        self.labels = [self.cfg["id_to_label"][str(i)] for i in range(len(self.cfg["id_to_label"]))]
        self.prototypes = {
            label: tensors[f"prototype_{label}"].astype(np.float32)
            for label in self.labels
            if f"prototype_{label}" in tensors
        }
        # AdaptiveHead is Linear-ReLU-Dropout x2 then Linear. Dropout is a no-op
        # at inference, leaving three affine layers at indices 0, 3 and 6.
        self.head = None
        if "adaptive_head_model.0.weight" in tensors:
            self.head = [
                (tensors[f"adaptive_head_model.{i}.weight"], tensors[f"adaptive_head_model.{i}.bias"])
                for i in (0, 3, 6)
            ]

        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding()

        onnx_dir = root / "onnx"
        quantized, full = onnx_dir / "model_quantized.onnx", onnx_dir / "model.onnx"
        model_file = quantized if (prefer_quantized and quantized.exists()) else full
        self.session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def embed(self, text):
        enc = self.tokenizer.encode(text)
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        hidden = self.session.run(None, feeds)[0]  # (1, tokens, dim)

        if self.pooling == "cls":
            vec = hidden[:, 0, :]
        else:
            mask = feeds["attention_mask"][..., None].astype(hidden.dtype)
            vec = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        vec = vec[0]
        return vec / max(np.linalg.norm(vec), 1e-12)  # L2-normalise

    def _head_probs(self, emb):
        x = emb
        for n, (w, b) in enumerate(self.head):
            x = w @ x + b
            if n < len(self.head) - 1:
                x = np.maximum(x, 0.0)
        return softmax(x)

    def predict(self, text, k=5):
        emb = self.embed(text)

        # Prototype score: exp(-squared L2 distance), softmaxed over classes.
        labels = list(self.prototypes)
        sims = np.array([np.exp(-np.sum((p - emb) ** 2)) for p in self.prototypes.values()])
        proto_scores = dict(zip(labels, softmax(sims)))
        head_scores = (
            dict(zip(self.labels, self._head_probs(emb))) if self.head is not None else {}
        )

        history = self.cfg.get("training_history", {})
        combined = {}
        for label in self.labels:
            if history.get(label, 0) < self.new_threshold:
                wp, wn = self.new_prototype_weight, self.new_neural_weight
            else:
                wp, wn = self.prototype_weight, self.neural_weight
            combined[label] = proto_scores.get(label, 0.0) * wp + head_scores.get(label, 0.0) * wn

        total = sum(combined.values())
        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        return [(label, score / total) for label, score in ranked[:k]]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    clf = PortableClassifier(sys.argv[1])
    for label, score in clf.predict(" ".join(sys.argv[2:])):
        print(f"{label}\t{score:.4f}")
