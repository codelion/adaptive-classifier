"""Few-shot text classification benchmark: adaptive-classifier vs SetFit vs logistic regression.

Everything uses the same sentence encoder (``--model``). For each dataset, number of
examples per class ("shots") and random seed, every method trains on the same
stratified sample from the training split and is scored on the same test subset.

Methods
    adaptive        this library with default settings
    adaptive-proto  this library, prototypes only (neural head switched off)
    logreg          logistic regression on the frozen, L2-normalised embeddings
    setfit          SetFit (contrastive fine-tuning of the encoder, then a logistic head)

Measured per run: accuracy, macro-F1, expected calibration error (ECE) of the reported
confidences, training time, whole-test-set prediction time, median single-query
latency, and the cost of adding one more labelled example (adaptive-classifier updates in
place; logistic regression refits on cached features; SetFit has to retrain from scratch,
so its training time is reported as the update cost).

    pip install -e ".[benchmark]"
    python scripts/benchmark_fewshot.py check ag_news emotion banking77   # quick schema check
    python scripts/benchmark_fewshot.py run --dataset ag_news --shots 4 16 --seeds 3 \\
        --out results/ag_news.json
    python scripts/benchmark_fewshot.py report results/*.json --out results/summary.md

``--dataset synthetic`` generates a tiny dataset with no network access; it exists to
smoke-test the harness, and its numbers mean nothing.
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

METHODS = ["adaptive", "adaptive-proto", "logreg", "setfit"]

# Hub id, text column, label column. Add entries here to benchmark another dataset.
DATASETS = {
    "ag_news": {"hub": "fancyzhx/ag_news", "text": "text", "label": "label", "test": "test"},
    "emotion": {"hub": "dair-ai/emotion", "text": "text", "label": "label", "test": "test"},
    # PolyAI/banking77 is a loading script, which current `datasets` versions refuse to run;
    # this parquet copy has the same data. label_text holds the intent names.
    "banking77": {"hub": "mteb/banking77", "text": "text", "label": "label_text", "test": "test"},
}


# --------------------------------------------------------------------------------------
# Pure helpers (no model, no network)
# --------------------------------------------------------------------------------------

def sample_few_shot(labels: Sequence[str], shots: int, seed: int) -> List[int]:
    """Indices of a stratified sample with up to `shots` examples per class.

    Each class is shuffled once per seed and its first `shots` examples are taken, so a
    smaller sample is always a subset of a larger one for the same seed. That keeps
    results across shot counts comparable, with less noise from which examples were drawn.
    """
    if shots < 1:
        raise ValueError(f"shots must be at least 1; got {shots}")
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)
    chosen: List[int] = []
    for label in sorted(by_class):
        members = by_class[label]
        order = rng.permutation(len(members))
        chosen.extend(members[i] for i in order[:shots])
    return sorted(chosen)


def subsample(texts: Sequence[str], labels: Sequence[str], limit: Optional[int], seed: int = 0):
    """A fixed random subset of a labelled split (all of it when `limit` is None or large)."""
    if limit is None or limit >= len(texts):
        return list(texts), list(labels)
    order = np.random.default_rng(seed).permutation(len(texts))[:limit]
    return [texts[i] for i in order], [labels[i] for i in order]


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def score_predictions(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Accuracy, macro-F1 and ECE for a (n_samples, n_classes) probability matrix."""
    from sklearn.metrics import f1_score

    predicted = probs.argmax(axis=1)
    return {
        "accuracy": float((predicted == targets).mean()),
        "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probs, targets),
    }


def _mean_std(values: List[float]):
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def aggregate(results: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """Mean and standard deviation over seeds, keyed by (dataset, method, shots)."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for row in results:
        groups.setdefault((row["dataset"], row["method"], row["shots"]), []).append(row)
    summary = {}
    for key, rows in groups.items():
        ok = [r for r in rows if not r.get("error")]
        entry: Dict[str, Any] = {"runs": len(rows), "failed": len(rows) - len(ok)}
        for metric in ("accuracy", "macro_f1", "ece", "train_s", "predict_s",
                       "latency_ms_p50", "update_s"):
            mean, std = _mean_std([r[metric] for r in ok if r.get(metric) is not None])
            entry[metric] = (mean, std)
        entry["update_is_retrain"] = any(r.get("update_is_retrain") for r in ok)
        summary[key] = entry
    return summary


def _cell(pair, digits=3, scale=1.0, bold=False):
    mean, std = pair
    if mean is None:
        return "failed"
    text = f"{mean * scale:.{digits}f}"
    if std is not None:
        text += f" ± {std * scale:.{digits}f}"
    return f"**{text}**" if bold else text


def _table(summary, dataset, methods, shots, metric, higher_is_better=True, digits=3, scale=1.0):
    header = "| method | " + " | ".join(f"{s}-shot" for s in shots) + " |"
    lines = [header, "|---|" + "---|" * len(shots)]
    best = {}
    for s in shots:
        values = [(m, summary[(dataset, m, s)][metric][0]) for m in methods
                  if (dataset, m, s) in summary and summary[(dataset, m, s)][metric][0] is not None]
        if values:
            pick = max if higher_is_better else min
            best[s] = pick(v for _, v in values)
    for m in methods:
        cells = []
        for s in shots:
            entry = summary.get((dataset, m, s))
            if entry is None:
                cells.append("–")
                continue
            pair = entry[metric]
            is_best = pair[0] is not None and s in best and abs(pair[0] - best[s]) < 1e-12
            cells.append(_cell(pair, digits, scale, bold=is_best))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(results: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> str:
    """A readable report: one section per dataset; the best value in each column is bold."""
    meta = meta or {}
    summary = aggregate(results)
    datasets = sorted({r["dataset"] for r in results})
    out = ["# Few-shot benchmark", ""]
    if meta:
        facts = [f"**{k}**: {v}" for k, v in meta.items() if v not in (None, "")]
        out += ["  \n".join(facts), ""]
    errors = [r for r in results if r.get("error")]
    if errors:
        out += [f"> **{len(errors)} run(s) failed.** Cells marked `failed` have no result; "
                "see the `error` field in the JSON.", ""]

    for dataset in datasets:
        methods = [m for m in METHODS if any(k[0] == dataset and k[1] == m for k in summary)]
        methods += sorted({k[1] for k in summary if k[0] == dataset} - set(methods))
        shots = sorted({k[2] for k in summary if k[0] == dataset})
        sample = next(r for r in results if r["dataset"] == dataset)
        out += [f"## {dataset}", "",
                f"{sample.get('n_classes', '?')} classes, "
                f"{sample.get('n_test', '?')} test examples; mean ± std over seeds. "
                "Best per column in bold.", ""]
        out += ["### Accuracy", "", _table(summary, dataset, methods, shots, "accuracy"), ""]
        out += ["### Macro-F1", "", _table(summary, dataset, methods, shots, "macro_f1"), ""]
        out += ["### Calibration error (ECE, lower is better)", "",
                _table(summary, dataset, methods, shots, "ece", higher_is_better=False), ""]
        out += ["### Cost", ""]
        cost_shots = [shots[-1]]
        out += [f"At {cost_shots[0]} examples per class.", "",
                "| method | train (s) | predict test set (s) | latency p50 (ms) | add one example (s) |",
                "|---|---|---|---|---|"]
        for m in methods:
            entry = summary.get((dataset, m, cost_shots[0]))
            if entry is None:
                continue
            update = _cell(entry["update_s"], 2)
            if entry["update_is_retrain"] and update != "failed":
                update += " (full retrain)"
            out.append(f"| {m} | {_cell(entry['train_s'], 1)} | {_cell(entry['predict_s'], 1)} | "
                       f"{_cell(entry['latency_ms_p50'], 1)} | {update} |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def synthetic_dataset(n_classes: int = 4, per_class_train: int = 40, n_test: int = 80, seed: int = 0):
    """Texts made of class-specific words plus shared noise words. For smoke tests only."""
    rng = np.random.default_rng(seed)
    class_words = [[f"w{c}x{i}" for i in range(6)] for c in range(n_classes)]
    shared = [f"s{i}" for i in range(10)]

    def make(c):
        words = list(rng.choice(class_words[c], 4)) + list(rng.choice(shared, 3))
        rng.shuffle(words)
        return " ".join(words)

    train = [(make(c), f"class_{c}") for c in range(n_classes) for _ in range(per_class_train)]
    test = [(make(c % n_classes), f"class_{c % n_classes}") for c in range(n_test)]
    return {
        "train_texts": [t for t, _ in train], "train_labels": [l for _, l in train],
        "test_texts": [t for t, _ in test], "test_labels": [l for _, l in test],
    }


def load_dataset_splits(name: str) -> Dict[str, List[str]]:
    if name == "synthetic":
        return synthetic_dataset()
    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset {name!r}; choose from {sorted(DATASETS) + ['synthetic']}")
    from datasets import load_dataset

    spec = DATASETS[name]
    data = load_dataset(spec["hub"])
    names = getattr(data["train"].features[spec["label"]], "names", None)

    def split(part):
        rows = data[part]
        labels = rows[spec["label"]]
        return list(rows[spec["text"]]), [names[i] if names else str(i) for i in labels]

    train_texts, train_labels = split("train")
    test_texts, test_labels = split(spec["test"])
    return {"train_texts": train_texts, "train_labels": train_labels,
            "test_texts": test_texts, "test_labels": test_labels}


def verify_dataset(name: str) -> str:
    """Check that a dataset can be loaded and has the expected columns, without downloading it.

    Streams the first example of the train and test splits. Returns a one-line description,
    or raises with a message naming what is wrong (unknown Hub id, missing column or split).
    """
    if name == "synthetic":
        return "synthetic: generated locally"
    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset {name!r}; choose from {sorted(DATASETS) + ['synthetic']}")
    from datasets import load_dataset

    spec = DATASETS[name]
    problems = []
    classes = None
    for part in ("train", spec["test"]):
        try:
            stream = load_dataset(spec["hub"], split=part, streaming=True)
            first = next(iter(stream))
        except Exception as error:
            problems.append(f"{spec['hub']} split {part!r}: {type(error).__name__}: {str(error)[:200]}")
            continue
        for column in (spec["text"], spec["label"]):
            if column not in first:
                problems.append(f"{spec['hub']} split {part!r} has no column {column!r} "
                                f"(columns: {sorted(first)})")
        if part == "train":
            feature = (stream.features or {}).get(spec["label"])
            classes = len(feature.names) if getattr(feature, "names", None) else None
    if problems:
        raise RuntimeError(f"{name}: " + "; ".join(problems))
    return f"{name}: {spec['hub']} ok" + (f", {classes} classes" if classes else "")


# --------------------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------------------

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def _latency_p50_ms(predict_one, texts: Sequence[str], n: int = 30) -> float:
    predict_one(texts[0])                                        # warm-up
    times = []
    for text in texts[:n]:
        with Timer() as t:
            predict_one(text)
        times.append(t.seconds * 1000)
    return float(statistics.median(times))


class AdaptiveMethod:
    """adaptive-classifier. `prototypes_only` switches the neural head off."""

    def __init__(self, model: str, prototypes_only: bool = False, device: Optional[str] = None):
        self.model, self.device = model, device
        self.config = ({"prototype_weight": 1.0, "neural_weight": 0.0,
                        "new_class_example_threshold": 0, "head_steps": 1}
                       if prototypes_only else {})

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from adaptive_classifier import AdaptiveClassifier

        classifier = AdaptiveClassifier(self.model, config=self.config, use_onnx=False,
                                        device=self.device)                # load time not counted
        with Timer() as train:
            classifier.add_examples(data["train_texts"], data["train_labels"])
        classes = sorted(set(data["train_labels"]))
        column = {label: i for i, label in enumerate(classes)}

        with Timer() as predict:
            batches = classifier.predict_batch(data["test_texts"], k=len(classes))
        probs = np.zeros((len(data["test_texts"]), len(classes)))
        for row, scores in enumerate(batches):
            for label, score in scores:
                probs[row, column[label]] = score

        latency = _latency_p50_ms(lambda t: classifier.predict(t, k=1), data["test_texts"])
        with Timer() as update:
            classifier.add_examples([data["extra_text"]], [data["extra_label"]])
        return {"probs": probs, "classes": classes, "train_s": train.seconds,
                "predict_s": predict.seconds, "latency_ms_p50": latency,
                "update_s": update.seconds, "update_is_retrain": False}


class LogRegMethod:
    """Logistic regression on frozen embeddings from the same encoder."""

    def __init__(self, model: str, device: Optional[str] = None):
        self.model, self.device = model, device

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.linear_model import LogisticRegression

        from adaptive_classifier import AdaptiveClassifier

        encoder = AdaptiveClassifier(self.model, use_onnx=False, device=self.device)

        def embed(texts):
            out = []
            for i in range(0, len(texts), 64):
                out += [e.numpy() for e in encoder._get_embeddings(list(texts[i:i + 64]))]
            return np.stack(out)

        with Timer() as train:
            features = embed(data["train_texts"])
            model = LogisticRegression(max_iter=1000, C=10.0).fit(features, data["train_labels"])
        classes = list(model.classes_)

        with Timer() as predict:
            probs = model.predict_proba(embed(data["test_texts"]))
        latency = _latency_p50_ms(
            lambda t: model.predict_proba(embed([t])), data["test_texts"])

        extra = embed([data["extra_text"]])
        with Timer() as update:                       # refit on cached features plus the new example
            LogisticRegression(max_iter=1000, C=10.0).fit(
                np.vstack([features, extra]), data["train_labels"] + [data["extra_label"]])
        return {"probs": probs, "classes": classes, "train_s": train.seconds,
                "predict_s": predict.seconds, "latency_ms_p50": latency,
                "update_s": update.seconds, "update_is_retrain": True}


class SetFitMethod:
    """SetFit: contrastive fine-tuning of the encoder followed by a logistic-regression head."""

    def __init__(self, model: str, device: Optional[str] = None):
        self.model = model

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        classes = sorted(set(data["train_labels"]))
        index = {label: i for i, label in enumerate(classes)}
        model = SetFitModel.from_pretrained(self.model)                      # load time not counted
        # SetFit writes checkpoints; keep them out of the working directory.
        scratch = tempfile.mkdtemp(prefix="setfit_checkpoints_")
        args = TrainingArguments(output_dir=scratch, batch_size=16, num_epochs=1, num_iterations=20,
                                 body_learning_rate=2e-5, seed=data["seed"],
                                 show_progress_bar=False, report_to="none")
        trainer = Trainer(
            model=model, args=args,
            train_dataset=Dataset.from_dict({
                "text": data["train_texts"],
                "label": [index[l] for l in data["train_labels"]]}))
        with Timer() as train:
            trainer.train()

        with Timer() as predict:
            probs = np.asarray(model.predict_proba(data["test_texts"]))
        latency = _latency_p50_ms(lambda t: model.predict_proba([t]), data["test_texts"])
        return {"probs": probs, "classes": classes, "train_s": train.seconds,
                "predict_s": predict.seconds, "latency_ms_p50": latency,
                "update_s": train.seconds, "update_is_retrain": True}


def build_method(name: str, model: str, device: Optional[str]):
    if name == "adaptive":
        return AdaptiveMethod(model, device=device)
    if name == "adaptive-proto":
        return AdaptiveMethod(model, prototypes_only=True, device=device)
    if name == "logreg":
        return LogRegMethod(model, device=device)
    if name == "setfit":
        return SetFitMethod(model, device=device)
    raise SystemExit(f"Unknown method {name!r}; choose from {METHODS}")


def plan(datasets: str, shots: str, seeds: str, methods: str, model: str, max_test: str) -> Dict[str, Any]:
    """Validate and normalise benchmark settings given as plain strings (workflow inputs).

    Raises ValueError naming the first bad value, so a typo fails in seconds rather than
    after the long jobs have started. Nothing here is ever interpolated into a shell command.
    """
    import re

    def words(text):
        return [w for w in re.split(r"[,\s]+", text.strip()) if w]

    chosen = words(datasets)
    unknown = [d for d in chosen if d not in DATASETS and d != "synthetic"]
    if not chosen or unknown:
        raise ValueError(f"datasets must be a non-empty list from {sorted(DATASETS) + ['synthetic']}; "
                         f"got {datasets!r}")
    try:
        shot_list = [int(w) for w in words(shots)]
    except ValueError:
        raise ValueError(f"shots must be whole numbers; got {shots!r}") from None
    if not shot_list or min(shot_list) < 1 or max(shot_list) > 512:
        raise ValueError(f"shots must be between 1 and 512; got {shots!r}")
    try:
        seed_count = int(seeds)
    except ValueError:
        raise ValueError(f"seeds must be a whole number; got {seeds!r}") from None
    if not 1 <= seed_count <= 10:
        raise ValueError(f"seeds must be between 1 and 10; got {seeds!r}")
    method_list = words(methods) or list(METHODS)
    bad = [m for m in method_list if m not in METHODS]
    if bad:
        raise ValueError(f"unknown methods {bad}; choose from {METHODS}")
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)?", model.strip()):
        raise ValueError(f"model must look like 'name' or 'owner/name'; got {model!r}")
    try:
        test_limit = int(max_test)
    except ValueError:
        raise ValueError(f"max_test must be a whole number; got {max_test!r}") from None
    if not 10 <= test_limit <= 100_000:
        raise ValueError(f"max_test must be between 10 and 100000; got {max_test!r}")
    return {
        "datasets": list(dict.fromkeys(chosen)),
        "shots": " ".join(str(k) for k in sorted(set(shot_list))),
        "seeds": seed_count,
        "methods": " ".join(dict.fromkeys(method_list)),
        "model": model.strip(),
        "max_test": test_limit,
    }


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------

def run_benchmark(dataset: str, shots: Sequence[int], seeds: int, methods: Sequence[str],
                  model: str, max_test: Optional[int], device: Optional[str] = None,
                  log=print) -> List[Dict[str, Any]]:
    splits = load_dataset_splits(dataset)
    test_texts, test_labels = subsample(splits["test_texts"], splits["test_labels"], max_test)
    train_texts, train_labels = splits["train_texts"], splits["train_labels"]
    classes = sorted(set(train_labels))
    class_index = {label: i for i, label in enumerate(classes)}
    targets = np.array([class_index[l] for l in test_labels])
    results: List[Dict[str, Any]] = []

    for seed in range(seeds):
        # One extra labelled example (from a class we know) to time the cost of an update.
        extra_rng = np.random.default_rng(10_000 + seed)
        extra_i = int(extra_rng.integers(len(train_texts)))
        for k in shots:
            picked = sample_few_shot(train_labels, k, seed)
            data = {
                "train_texts": [train_texts[i] for i in picked],
                "train_labels": [train_labels[i] for i in picked],
                "test_texts": test_texts, "seed": seed,
                "extra_text": train_texts[extra_i], "extra_label": train_labels[extra_i],
            }
            for name in methods:
                row = {"dataset": dataset, "method": name, "shots": k, "seed": seed,
                       "n_classes": len(classes), "n_train": len(picked), "n_test": len(test_texts)}
                try:
                    out = build_method(name, model, device).run(data)
                    order = [out["classes"].index(c) for c in classes]       # align columns
                    probs = np.asarray(out["probs"])[:, order]
                    row.update(score_predictions(probs, targets))
                    row.update({key: out[key] for key in
                                ("train_s", "predict_s", "latency_ms_p50", "update_s",
                                 "update_is_retrain")})
                    log(f"{dataset} {name:15s} shots={k:<3d} seed={seed} "
                        f"acc={row['accuracy']:.3f} f1={row['macro_f1']:.3f} "
                        f"ece={row['ece']:.3f} train={row['train_s']:.1f}s")
                except Exception as error:                                   # keep going; report at the end
                    row["error"] = f"{type(error).__name__}: {str(error)[:300]}"
                    log(f"{dataset} {name:15s} shots={k:<3d} seed={seed} FAILED {row['error']}")
                results.append(row)
    return results


def environment_meta(model: str, max_test: Optional[int], seeds: int) -> Dict[str, Any]:
    def version(module):
        try:
            return __import__(module).__version__
        except Exception:
            return None

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = None
    import adaptive_classifier
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": model, "seeds": seeds, "max test examples": max_test,
        "device": device, "commit": commit,
        "adaptive-classifier": adaptive_classifier.__version__,
        "setfit": version("setfit"), "torch": version("torch"),
        "python": platform.python_version(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the benchmark on one dataset")
    run.add_argument("--dataset", required=True, help=f"{sorted(DATASETS)} or synthetic")
    run.add_argument("--shots", type=int, nargs="+", default=[4, 16])
    run.add_argument("--seeds", type=int, default=3)
    run.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    run.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    run.add_argument("--max-test", type=int, default=1000)
    run.add_argument("--device", default=None)
    run.add_argument("--out", required=True, help="JSON file to write")

    planner = sub.add_parser("plan", help="validate workflow inputs and print them as JSON")
    planner.add_argument("--datasets", default="ag_news,emotion")
    planner.add_argument("--shots", default="4 16")
    planner.add_argument("--seeds", default="3")
    planner.add_argument("--methods", default=" ".join(METHODS))
    planner.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    planner.add_argument("--max-test", default="1000")

    check = sub.add_parser("check", help="verify that datasets load and have the expected columns")
    check.add_argument("datasets", nargs="+", help=f"{sorted(DATASETS)} or synthetic")

    report = sub.add_parser("report", help="turn result files into a Markdown report")
    report.add_argument("files", nargs="+")
    report.add_argument("--out", default=None, help="write here as well as printing")

    args = parser.parse_args(argv)

    if args.command == "plan":
        try:
            print(json.dumps(plan(args.datasets, args.shots, args.seeds, args.methods,
                                  args.model, args.max_test)))
        except ValueError as error:
            print(f"Invalid benchmark settings: {error}", file=sys.stderr)
            return 2
        return 0

    if args.command == "check":
        failures = 0
        for name in args.datasets:
            try:
                print(verify_dataset(name))
            except RuntimeError as error:
                failures += 1
                print(f"FAIL {error}", file=sys.stderr)
        return 1 if failures else 0

    if args.command == "run":
        results = run_benchmark(args.dataset, args.shots, args.seeds, args.methods, args.model,
                                args.max_test, args.device)
        meta = environment_meta(args.model, args.max_test, args.seeds)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
        failed = [r for r in results if r.get("error")]
        if failed:
            print(f"{len(failed)} of {len(results)} runs failed", file=sys.stderr)
        return 1 if failed else 0

    results, meta = [], {}
    for name in args.files:
        payload = json.loads(Path(name).read_text(encoding="utf-8"))
        results += payload["results"]
        meta = meta or payload.get("meta", {})
    text = render_markdown(results, meta)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
