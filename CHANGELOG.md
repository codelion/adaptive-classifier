# Changelog

## 0.3.0

### Added
- `SklearnAdaptiveClassifier`: scikit-learn estimator (`fit`, `partial_fit`, `predict`, `predict_proba`, `score`) usable in `Pipeline`, `cross_val_score` and `GridSearchCV`.
- `forget(labels)` and `remove_examples(texts, label=None, retrain=True)` to delete classes or fix wrong labels.
- `ood_score()` / `is_ood()` out-of-distribution detection, and `predict(..., abstain_below=, abstain_ood=)`. New config keys `ood_threshold` and `ood_min_radius`.
- `docs/deployment.md` and `examples/portable_inference.py`: run a saved classifier without PyTorch or FAISS (multilingual models, .NET, JavaScript).
- `save()` now writes the tokenizer files and the resolved pooling mode, so a saved directory is self-contained.

- `calibrate(texts, labels)` fits one temperature on held-out labelled data so confidences match how often the model is right, and `calibration_report(...)` measures accuracy, expected calibration error and NLL. `predict`, `predict_batch` and `abstain_below` then use calibrated probabilities. Calibration is saved with the model and discarded when the set of classes changes.
- `predict_set(text, alpha=0.1)`: split-conformal prediction sets that contain the true label with probability at least `1 - alpha`.

### Changed (default behaviour: predictions change)
- **The neural head is now actually trained.** It used to get about ten optimiser steps and stop at the first loss plateau, so it could not learn even a 2-D XOR from 120 examples. It now trains for at least `head_steps` (default 300) steps with a cosine-decayed `head_learning_rate` (default 0.003). Small memories take roughly 1-2 s per `add_examples` call; lower `head_steps` if you add examples very frequently.
- **Prototype scores are sharper.** They are now `softmax(-distance² / prototype_temperature)` (default 0.25) instead of `softmax(exp(-distance))`, which barely separated the nearest class from the rest (about 0.45 vs 0.21), so a confident head could always outvote it. `prototype_temperature=None` restores the old scoring.
- **Classes with few examples lean on their prototypes.** Below `new_class_example_threshold` examples the head's weight ramps up from zero, instead of the old fixed 0.3 prototype / 0.7 head split. On separable synthetic data the old default scored 0.55-0.73 with fewer than 10 examples per class where prototypes alone scored 0.96-1.0; accuracy no longer drops off a cliff at the threshold. Trade-off: a class added with a single example is recognised less often than before, because the old split biased the head toward new classes at the expense of existing ones. Set both `new_class_prototype_weight` and `new_class_neural_weight` to restore a fixed split.
- Classifiers saved by an earlier version keep their old scoring when loaded, so their predictions do not change.
- **Python 3.10 or newer is required** (3.8 and 3.9 are end-of-life).
- Packaging moved to a `[project]` table in `pyproject.toml`; `setup.py` and `requirements.txt` are gone. Install extras: `[test]`, `[scripts]`. `tqdm` is no longer a core dependency.
- `clear_memory(labels=[...])` now removes those classes completely (it used to leave them in the label map and neural head, so they could still be predicted).

### Fixed
- `predict_batch(k=...)` scored only `k` classes before blending, so with more than `k` classes its scores differed from `predict` for the same text. It now scores every class and trims to `k`.
- Accuracy on small training sets (see above): the default configuration got only 75-92% of its own training data right on perfectly separable data with fewer than 10 examples per class.
- Integer (or other non-string) labels corrupted a classifier on save/reload, duplicating classes. `add_examples` now rejects non-string labels with a clear error; convert with `str(label)`.
- `predict(k=-1)` returned a silently truncated list; negative or non-integer `k` now raises `ValueError`.
- `None` as a text or label, and mixed label types, now raise a clear `ValueError` before anything is changed.
- Loading a directory without `config.json` now says so instead of reporting an invalid Hub repo id.

