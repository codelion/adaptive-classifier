# Changelog

## 0.3.1

### Fixed
- `MultiLabelAdaptiveClassifier`: `forget()` and `remove_examples()` no longer crash; `load(..., use_onnx=...)` works and the multi-label settings and per-label thresholds are saved and restored; a bare-string label (which was split into one class per character) is rejected; `max_labels=0` returns nothing and `max_labels` caps the `min_predictions` top-up; the head is trained for `head_steps` steps and retrained when labels are added, instead of ~10 steps with a softmax loss that did not apply; the public methods hold the instance lock.

- Strategic mode: `cost_coefficients` is validated (a dict of named features, as the README once showed, or a list of the wrong length used to disable strategic mode with a cryptic message; the reason now says what is wrong and how many entries are needed); the best response explored only the first few embedding dimensions and used unseeded randomness, so it now samples the cheapest and a random spread of dimensions with a fixed seed; `predict`, `predict_strategic` and `predict_robust` scores no longer change with `k`; `evaluate_strategic_robustness` no longer divides by zero, rejects unknown labels with `ValueError`, and is repeatable and thread-safe.
- Strategic-mode predictions are different from 0.3.0 because of the candidate and `k` fixes.

- Copying a classifier (`copy.deepcopy`, `pickle`) failed with "cannot pickle RLock" since the instance lock was added in 0.3.0; copies now get their own lock.
- `predict_batch` disagreed with `predict` in strategic mode (it skipped the strategic blend) and for `MultiLabelAdaptiveClassifier` (it softmaxed sigmoid scores). Both now return what `predict` returns, which also fixes the server's `/predict_batch` for those classifiers.

- `merge_classifiers`: classes taken from the other classifier were not searchable afterwards (the prototype index was not rebuilt, so the merged classifier got 0% on them); merging into an empty classifier raised; training counts were not combined, so merged classes got no head weight; stale calibration was kept; example objects were shared with the source. Merging a classifier into itself is now a no-op.

- `add_examples` now rejects blank texts and blank labels (with the index of the first offender) instead of learning them; a blank text carries no information but still counted as an example of its class and moved its prototype. **Behaviour change:** code that passes empty rows now gets a `ValueError`. `select_representative_examples(k=0)` returns an empty list instead of failing inside scikit-learn.

### Added
- Tests for strategic mode (`tests/test_strategic.py`); README and `docs/API.md` document it and `MultiLabelAdaptiveClassifier`, and the settings added in 0.3.0.
- `benchmark` and `scripts` extras pin `datasets<4` and `huggingface-hub<1.0`, which keeps `transformers` working.
- `examples/javascript/portable_inference.mjs`: a tested Node.js port of the portable inference, and a Docker smoke test (`scripts/docker_smoke.py`, `.github/workflows/docker.yml`) that builds the serving image and queries it.
- README: measured guidance on tuning `new_class_example_threshold` for new-class recall.

## 0.3.0

### Added
- `SklearnAdaptiveClassifier`: scikit-learn estimator (`fit`, `partial_fit`, `predict`, `predict_proba`, `score`) usable in `Pipeline`, `cross_val_score` and `GridSearchCV`.
- `forget(labels)` and `remove_examples(texts, label=None, retrain=True)` to delete classes or fix wrong labels.
- `ood_score()` / `is_ood()` out-of-distribution detection, and `predict(..., abstain_below=, abstain_ood=)`. New config keys `ood_threshold` and `ood_min_radius`.
- `docs/deployment.md` and `examples/portable_inference.py`: run a saved classifier without PyTorch or FAISS (multilingual models, .NET, JavaScript).
- `save()` now writes the tokenizer files and the resolved pooling mode, so a saved directory is self-contained.

- `calibrate(texts, labels)` fits one temperature on held-out labelled data so confidences match how often the model is right, and `calibration_report(...)` measures accuracy, expected calibration error and NLL. `predict`, `predict_batch` and `abstain_below` then use calibrated probabilities. Calibration is saved with the model and discarded when the set of classes changes.
- `predict_set(text, alpha=0.1)`: split-conformal prediction sets that contain the true label with probability at least `1 - alpha`.

- `suggest_labels(texts, n, strategy, diverse)`: active learning. Ranks unlabeled texts by how much a label would help (`margin`, `entropy`, `least_confidence`, or `ood` to discover unseen classes), optionally spreading the picks so near-duplicates are not all chosen.
- `drift_report(texts)`: tests whether a window of incoming texts has moved away from the known classes (share of out-of-distribution texts against an expected rate, one-sided binomial test).

- **Serving.** `python -m adaptive_classifier.serving ./model` runs a FastAPI server (`/predict`, `/predict_batch`, `/predict_set`, `/ood`, and optional authenticated `/examples`, `/forget`, `/remove_examples`); `create_app(classifier)` returns the app for embedding. Install with `pip install "adaptive-classifier[serve]"`. See `docs/serving.md`. `docker/Dockerfile` is provided but has not been built in the environment it was written in.
- Async methods `apredict`, `apredict_batch` and `aadd_examples` that run in a worker thread.
- A saved directory is self-contained for loading: the tokenizer saved next to the model is used instead of fetching the base model's repository, so an ONNX-saved classifier starts without Hub access.


### Changed (default behaviour: predictions change)
- **The neural head is now actually trained.** It used to get about ten optimiser steps and stop at the first loss plateau, so it could not learn even a 2-D XOR from 120 examples. It now trains for at least `head_steps` (default 300) steps with a cosine-decayed `head_learning_rate` (default 0.003). Small memories take roughly 1-2 s per `add_examples` call; lower `head_steps` if you add examples very frequently.
- **Prototype scores are sharper.** They are now `softmax(-distance² / prototype_temperature)` (default 0.25) instead of `softmax(exp(-distance))`, which barely separated the nearest class from the rest (about 0.45 vs 0.21), so a confident head could always outvote it. `prototype_temperature=None` restores the old scoring.
- **Classes with few examples lean on their prototypes.** Below `new_class_example_threshold` examples the head's weight ramps up from zero, instead of the old fixed 0.3 prototype / 0.7 head split. On separable synthetic data the old default scored 0.55-0.73 with fewer than 10 examples per class where prototypes alone scored 0.96-1.0; accuracy no longer drops off a cliff at the threshold. Trade-off: a class added with a single example is recognised less often than before, because the old split biased the head toward new classes at the expense of existing ones. Set both `new_class_prototype_weight` and `new_class_neural_weight` to restore a fixed split.
- Classifiers saved by an earlier version keep their old scoring when loaded, so their predictions do not change.
- **Python 3.10 or newer is required** (3.8 and 3.9 are end-of-life).
- Packaging moved to a `[project]` table in `pyproject.toml`; `setup.py` and `requirements.txt` are gone. Install extras: `[test]`, `[scripts]`. `tqdm` is no longer a core dependency.
- `clear_memory(labels=[...])` now removes those classes completely (it used to leave them in the label map and neural head, so they could still be predicted).

### Fixed
- **Thread safety.** Predicting while another thread added or forgot classes crashed with errors such as "selected index k out of range", and concurrent `add_examples` calls corrupted each other. Every public operation now holds a per-instance lock, so one classifier can be shared between threads. Throughput scales with processes, not threads.
- `predict_batch(k=...)` scored only `k` classes before blending, so with more than `k` classes its scores differed from `predict` for the same text. It now scores every class and trims to `k`.
- Accuracy on small training sets (see above): the default configuration got only 75-92% of its own training data right on perfectly separable data with fewer than 10 examples per class.
- Integer (or other non-string) labels corrupted a classifier on save/reload, duplicating classes. `add_examples` now rejects non-string labels with a clear error; convert with `str(label)`.
- `predict(k=-1)` returned a silently truncated list; negative or non-integer `k` now raises `ValueError`.
- `None` as a text or label, and mixed label types, now raise a clear `ValueError` before anything is changed.
- Loading a directory without `config.json` now says so instead of reporting an invalid Hub repo id.

