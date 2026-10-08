# Changelog

## 0.3.0

### Added
- `SklearnAdaptiveClassifier`: scikit-learn estimator (`fit`, `partial_fit`, `predict`, `predict_proba`, `score`) usable in `Pipeline`, `cross_val_score` and `GridSearchCV`.
- `forget(labels)` and `remove_examples(texts, label=None, retrain=True)` to delete classes or fix wrong labels.
- `ood_score()` / `is_ood()` out-of-distribution detection, and `predict(..., abstain_below=, abstain_ood=)`. New config keys `ood_threshold` and `ood_min_radius`.
- `docs/deployment.md` and `examples/portable_inference.py`: run a saved classifier without PyTorch or FAISS (multilingual models, .NET, JavaScript).
- `save()` now writes the tokenizer files and the resolved pooling mode, so a saved directory is self-contained.

### Changed
- **Python 3.10 or newer is required** (3.8 and 3.9 are end-of-life).
- Packaging moved to a `[project]` table in `pyproject.toml`; `setup.py` and `requirements.txt` are gone. Install extras: `[test]`, `[scripts]`. `tqdm` is no longer a core dependency.
- `clear_memory(labels=[...])` now removes those classes completely (it used to leave them in the label map and neural head, so they could still be predicted).

### Fixed
- Integer (or other non-string) labels corrupted a classifier on save/reload, duplicating classes. `add_examples` now rejects non-string labels with a clear error; convert with `str(label)`.
- `predict(k=-1)` returned a silently truncated list; negative or non-integer `k` now raises `ValueError`.
- `None` as a text or label, and mixed label types, now raise a clear `ValueError` before anything is changed.
- Loading a directory without `config.json` now says so instead of reporting an invalid Hub repo id.

### Known issues
- With fewer than 10 examples per class the default blend gives the neural head 70% of the vote. A head trained on so few examples can outvote correct prototypes, so accuracy on small training sets is lower than prototype-only prediction. Tracked by a strict expected-failure test in `tests/test_robustness.py`.
