"""A saved classifier must be self-describing and runnable outside PyTorch.

These tests use a tiny randomly initialised BERT built on the fly, so they need
no network access.

Covered here:
  - `save()` persists the *resolved* pooling mode instead of the literal 'auto'
  - `save()` writes tokenizer files, including tokenizer.json
  - the dependency-light reference in examples/portable_inference.py reproduces
    `AdaptiveClassifier.predict` from the saved files alone
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from adaptive_classifier import AdaptiveClassifier

TEXTS = ["great product love it", "works very good fast", "terrible awful service",
         "hate it broke bad", "okay fine the product", "it is okay fine"] * 3
LABELS = ["pos", "pos", "neg", "neg", "neu", "neu"] * 3
QUERIES = ["great fast product", "awful slow broke", "okay it is fine", "love the service"]


def _trained(base_model, **config):
    clf = AdaptiveClassifier(base_model, config=config, use_onnx=False, device="cpu")
    clf.add_examples(TEXTS, LABELS)
    return clf


def _saved_config(directory):
    with open(Path(directory) / "config.json", encoding="utf-8") as handle:
        return json.load(handle)["config"]


def _load_reference():
    path = Path(__file__).resolve().parent.parent / "examples" / "portable_inference.py"
    spec = importlib.util.spec_from_file_location("portable_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PortableClassifier


def _onnx_available():
    try:
        import onnxruntime  # noqa: F401
        import optimum.onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


# --- resolved pooling is persisted -------------------------------------------

def test_auto_pooling_is_saved_as_the_resolved_mode(base_model, tmp_path):
    clf = _trained(base_model)  # default is 'auto'; a local model has no ST config -> mean
    clf.save(str(tmp_path), include_onnx=False)
    assert _saved_config(tmp_path)["pooling"] == "mean"


def test_auto_pooling_saves_whatever_the_model_resolves_to(base_model, tmp_path, monkeypatch):
    clf = _trained(base_model)
    monkeypatch.setattr(clf, "_resolve_pooling", lambda: "cls")
    clf.save(str(tmp_path), include_onnx=False)
    assert _saved_config(tmp_path)["pooling"] == "cls"


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_explicit_pooling_is_saved_unchanged(base_model, tmp_path, pooling):
    _trained(base_model, pooling=pooling).save(str(tmp_path), include_onnx=False)
    assert _saved_config(tmp_path)["pooling"] == pooling


def test_saving_does_not_change_the_live_config(base_model, tmp_path):
    clf = _trained(base_model)
    clf.save(str(tmp_path), include_onnx=False)
    assert clf.config.pooling == "auto"


def test_reload_uses_saved_pooling_without_a_hub_lookup(base_model, tmp_path, monkeypatch):
    clf = _trained(base_model, pooling="cls")
    clf.save(str(tmp_path), include_onnx=False)

    def no_network(*args, **kwargs):
        raise AssertionError("pooling must come from config.json, not the Hub")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", no_network)
    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")
    assert loaded._resolve_pooling() == "cls"


def test_reload_round_trips_predictions(base_model, tmp_path):
    clf = _trained(base_model)
    clf.save(str(tmp_path), include_onnx=False)
    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")
    for query in QUERIES:
        before, after = dict(clf.predict(query)), dict(loaded.predict(query))
        assert before.keys() == after.keys()
        for label in before:
            assert after[label] == pytest.approx(before[label], abs=1e-5)


def test_pre_0_2_0_directories_still_load_as_cls(base_model, tmp_path):
    """A config without a 'pooling' key predates 0.2.0 and means CLS pooling."""
    _trained(base_model, pooling="cls").save(str(tmp_path), include_onnx=False)
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    del data["config"]["pooling"]
    config_path.write_text(json.dumps(data), encoding="utf-8")

    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")
    assert loaded._resolve_pooling() == "cls"


# --- tokenizer ships with the model ------------------------------------------

def test_save_writes_tokenizer_files(base_model, tmp_path):
    _trained(base_model).save(str(tmp_path), include_onnx=False)
    assert (tmp_path / "tokenizer.json").exists()
    assert (tmp_path / "tokenizer_config.json").exists()


def test_saved_tokenizer_matches_the_original(base_model, tmp_path):
    from transformers import AutoTokenizer

    clf = _trained(base_model)
    clf.save(str(tmp_path), include_onnx=False)
    reloaded = AutoTokenizer.from_pretrained(str(tmp_path))
    for query in QUERIES:
        assert reloaded(query)["input_ids"] == clf.tokenizer(query)["input_ids"]


def test_save_survives_a_tokenizer_that_cannot_be_written(base_model, tmp_path, monkeypatch):
    clf = _trained(base_model)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(clf.tokenizer, "save_pretrained", broken)
    clf.save(str(tmp_path), include_onnx=False)  # must not raise
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model.safetensors").exists()


# --- reference implementation -------------------------------------------------

@pytest.mark.skipif(not _onnx_available(), reason="optimum[onnxruntime] not installed")
@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_portable_reference_matches_predict(base_model, tmp_path, pooling):
    pytest.importorskip("tokenizers")
    clf = _trained(base_model, pooling=pooling)
    clf.save(str(tmp_path), include_onnx=True, quantize_onnx=False)

    portable = _load_reference()(tmp_path, prefer_quantized=False)
    assert portable.pooling == pooling
    for query in QUERIES:
        expected = dict(clf.predict(query, k=3))
        actual = dict(portable.predict(query, k=3))
        assert actual.keys() == expected.keys()
        for label in expected:
            assert float(actual[label]) == pytest.approx(expected[label], abs=1e-4)


@pytest.mark.skipif(not _onnx_available(), reason="optimum[onnxruntime] not installed")
def test_portable_reference_ranks_like_predict(base_model, tmp_path):
    pytest.importorskip("tokenizers")
    clf = _trained(base_model)
    clf.save(str(tmp_path), include_onnx=True, quantize_onnx=False)
    portable = _load_reference()(tmp_path, prefer_quantized=False)
    for query in QUERIES:
        assert [label for label, _ in portable.predict(query)] == [label for label, _ in clf.predict(query)]


@pytest.mark.skipif(not _onnx_available(), reason="optimum[onnxruntime] not installed")
def test_portable_reference_needs_a_pooling_mode_for_legacy_configs(base_model, tmp_path):
    pytest.importorskip("tokenizers")
    clf = _trained(base_model, pooling="cls")
    clf.save(str(tmp_path), include_onnx=True, quantize_onnx=False)
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    del data["config"]["pooling"]
    config_path.write_text(json.dumps(data), encoding="utf-8")

    Portable = _load_reference()
    with pytest.raises(ValueError, match="pooling"):
        Portable(tmp_path, prefer_quantized=False)

    portable = Portable(tmp_path, pooling="cls", prefer_quantized=False)
    expected = dict(clf.predict(QUERIES[0], k=3))
    actual = dict(portable.predict(QUERIES[0], k=3))
    for label in expected:
        assert float(actual[label]) == pytest.approx(expected[label], abs=1e-4)


# --- the reference tracks every branch of the scoring math ---------------------------------

@pytest.mark.skipif(not _onnx_available(), reason="optimum[onnxruntime] not installed")
@pytest.mark.parametrize("config", [
    {"prototype_temperature": None},                                   # legacy scoring
    {"prototype_temperature": 0.1},
    {"new_class_prototype_weight": 0.3, "new_class_neural_weight": 0.7},  # fixed split below threshold
    {"new_class_example_threshold": 4},                                # some classes established
    {"prototype_weight": 0.5, "neural_weight": 0.5},
], ids=["legacy-scoring", "temperature-0.1", "fixed-new-class", "mixed-established", "even-blend"])
def test_portable_reference_matches_predict_for_every_scoring_branch(base_model, tmp_path, config):
    pytest.importorskip("tokenizers")
    clf = _trained(base_model, pooling="mean", **config)
    clf.add_examples(["great product love it"] * 6, ["pos"] * 6)      # 'pos' now has more examples than the rest
    clf.save(str(tmp_path), include_onnx=True, quantize_onnx=False)

    portable = _load_reference()(tmp_path, prefer_quantized=False)
    for query in QUERIES:
        expected = dict(clf.predict(query, k=3))
        actual = dict(portable.predict(query, k=3))
        assert actual.keys() == expected.keys()
        for label in expected:
            assert float(actual[label]) == pytest.approx(expected[label], abs=1e-4)
