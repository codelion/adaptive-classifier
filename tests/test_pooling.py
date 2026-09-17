"""Pooling selection for sentence embeddings.

Before 0.2.0 every model was embedded with its CLS token. That is wrong for most
encoders: `all-MiniLM-L6-v2` publishes `pooling_mode_mean_tokens`, and a masked
LM never trains CLS as a sentence representation. Since prototype memory is
cosine nearest-neighbour search over these vectors, bad pooling degrades the
library's central feature.
"""

import pytest
import torch

from adaptive_classifier import AdaptiveClassifier
from adaptive_classifier.models import ModelConfig

MODEL = "prajjwal1/bert-tiny"


def test_default_pooling_is_auto():
    assert ModelConfig({}).pooling == "auto"
    assert ModelConfig({}).to_dict()["pooling"] == "auto"


def test_explicit_pooling_is_respected():
    assert ModelConfig({"pooling": "cls"}).pooling == "cls"
    assert ModelConfig({"pooling": "mean"}).pooling == "mean"


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_pool_shapes(pooling):
    clf = AdaptiveClassifier(MODEL, device="cpu", config={"pooling": pooling})
    hidden = torch.randn(3, 7, clf.embedding_dim)
    mask = torch.ones(3, 7, dtype=torch.long)
    pooled = clf._pool(hidden, mask)
    assert pooled.shape == (3, clf.embedding_dim)


def test_cls_pooling_takes_the_first_token():
    clf = AdaptiveClassifier(MODEL, device="cpu", config={"pooling": "cls"})
    hidden = torch.randn(2, 5, clf.embedding_dim)
    mask = torch.ones(2, 5, dtype=torch.long)
    assert torch.allclose(clf._pool(hidden, mask), hidden[:, 0, :])


def test_mean_pooling_ignores_padding():
    """Padded positions must not drag the mean toward zero."""
    clf = AdaptiveClassifier(MODEL, device="cpu", config={"pooling": "mean"})
    hidden = torch.ones(1, 4, clf.embedding_dim)
    hidden[:, 2:, :] = 99.0                      # padding, should be excluded
    mask = torch.tensor([[1, 1, 0, 0]])
    pooled = clf._pool(hidden, mask)
    assert torch.allclose(pooled, torch.ones(1, clf.embedding_dim))


def test_mean_pooling_averages_real_tokens():
    clf = AdaptiveClassifier(MODEL, device="cpu", config={"pooling": "mean"})
    hidden = torch.zeros(1, 2, clf.embedding_dim)
    hidden[:, 0, :] = 1.0
    hidden[:, 1, :] = 3.0
    mask = torch.ones(1, 2, dtype=torch.long)
    assert torch.allclose(clf._pool(hidden, mask), torch.full((1, clf.embedding_dim), 2.0))


def test_pooling_changes_the_embedding():
    """The two strategies must actually produce different vectors."""
    texts = ["the cat sat on the mat", "quarterly revenue exceeded forecasts"]
    cls = AdaptiveClassifier(MODEL, device="cpu", config={"pooling": "cls"})
    mean = AdaptiveClassifier(MODEL, device="cpu", config={"pooling": "mean"})
    a = torch.stack(cls._get_embeddings(texts))
    b = torch.stack(mean._get_embeddings(texts))
    assert a.shape == b.shape
    assert not torch.allclose(a, b)


def test_embeddings_stay_unit_norm():
    clf = AdaptiveClassifier(MODEL, device="cpu", config={"pooling": "mean"})
    for emb in clf._get_embeddings(["one text", "another somewhat longer text"]):
        assert pytest.approx(1.0, abs=1e-4) == float(emb.norm())


def test_auto_falls_back_to_mean_without_a_pooling_config():
    """bert-tiny publishes no sentence-transformers pooling config."""
    clf = AdaptiveClassifier(MODEL, device="cpu", config={"pooling": "auto"})
    assert clf._resolve_pooling() == "mean"


def test_resolved_pooling_is_cached():
    clf = AdaptiveClassifier(MODEL, device="cpu", config={"pooling": "auto"})
    first = clf._resolve_pooling()
    clf._model_name = "this/does-not-exist"      # a second lookup would fail
    assert clf._resolve_pooling() == first


def test_saved_classifiers_keep_cls_pooling(tmp_path):
    """A model saved before 0.2.0 must not silently change its embeddings.

    Its stored config has no `pooling` key. Reloading it under the new default
    would re-embed every prototype differently, so the loader pins it to the
    behaviour it was built with.
    """
    clf = AdaptiveClassifier(MODEL, device="cpu")
    clf.add_examples(["good product", "great service", "awful", "terrible"],
                     ["pos", "pos", "neg", "neg"])
    target = tmp_path / "saved"
    clf.save(str(target), include_onnx=False)

    import json
    config_path = target / "config.json"
    payload = json.loads(config_path.read_text())
    payload["config"].pop("pooling", None)        # simulate a pre-0.2.0 save
    config_path.write_text(json.dumps(payload))

    reloaded = AdaptiveClassifier.load(str(target), device="cpu")
    assert reloaded.config.pooling == "cls"


def test_new_saves_round_trip_their_pooling(tmp_path):
    clf = AdaptiveClassifier(MODEL, device="cpu", config={"pooling": "mean"})
    clf.add_examples(["good product", "great service", "awful", "terrible"],
                     ["pos", "pos", "neg", "neg"])
    target = tmp_path / "saved"
    clf.save(str(target), include_onnx=False)
    assert AdaptiveClassifier.load(str(target), device="cpu").config.pooling == "mean"


# ---------------------------------------------------------------------------
# Prototype / neural blend weights
# ---------------------------------------------------------------------------

def test_blend_weight_defaults():
    config = ModelConfig({})
    assert config.prototype_weight == 0.7
    assert config.neural_weight == 0.3
    assert config.new_class_example_threshold == 10


def test_blend_weights_are_configurable():
    clf = AdaptiveClassifier(MODEL, device="cpu",
                             config={"prototype_weight": 0.2, "neural_weight": 0.8})
    clf.training_history = {"a": 50}
    assert clf._blend_weights("a") == (0.2, 0.8)


def test_new_classes_lean_on_the_neural_head():
    """A class with few examples has an unreliable prototype."""
    clf = AdaptiveClassifier(MODEL, device="cpu")
    clf.training_history = {"established": 50, "fresh": 2}
    assert clf._blend_weights("established") == (0.7, 0.3)
    assert clf._blend_weights("fresh") == (0.3, 0.7)


def test_new_class_threshold_is_configurable():
    clf = AdaptiveClassifier(MODEL, device="cpu",
                             config={"new_class_example_threshold": 100})
    clf.training_history = {"a": 50}
    assert clf._blend_weights("a") == (0.3, 0.7)      # still "new" at 50


def test_blend_weight_changes_the_prediction():
    """The knob must actually move the output; before 0.2.0 it did nothing.

    Six repeats so each class clears `new_class_example_threshold`. Below it
    both classifiers use the new-class weights and the configured values never
    apply, which is what this test caught the first time round.
    """
    texts = ["good product", "great service", "awful thing", "terrible item"] * 6
    labels = ["pos", "pos", "neg", "neg"] * 6

    proto_heavy = AdaptiveClassifier(MODEL, device="cpu",
                                     config={"prototype_weight": 1.0, "neural_weight": 0.0})
    neural_heavy = AdaptiveClassifier(MODEL, device="cpu",
                                      config={"prototype_weight": 0.0, "neural_weight": 1.0})
    for clf in (proto_heavy, neural_heavy):
        clf.add_examples(texts, labels)

    a = dict(proto_heavy.predict("good product"))
    b = dict(neural_heavy.predict("good product"))
    assert a != b


def test_predict_and_predict_batch_agree_on_weights():
    """Both paths must read the same config, not two hardcoded constants."""
    texts = ["good product", "great service", "awful thing", "terrible item"] * 6
    labels = ["pos", "pos", "neg", "neg"] * 6
    clf = AdaptiveClassifier(MODEL, device="cpu",
                             config={"prototype_weight": 0.1, "neural_weight": 0.9})
    clf.add_examples(texts, labels)

    single = dict(clf.predict("good product", k=2))
    batched = dict(clf.predict_batch(["good product"], k=2)[0])
    for label in single:
        assert single[label] == pytest.approx(batched[label], abs=1e-5)
