"""State must stay consistent when an operation fails, and operations must do what their names say."""

import threading

import pytest
import torch

from adaptive_classifier import AdaptiveClassifier
from adaptive_classifier.ewc import EWC

from .synthetic import Dataset


@pytest.fixture
def data(monkeypatch):
    return Dataset().patch(monkeypatch)


@pytest.fixture
def clf(new_classifier, data):
    X, y = data.classes(10, n_classes=3, noise=0.15, seed=1)
    c = new_classifier(head_steps=30)
    c.add_examples(X, y)
    return c


def state_of(c):
    return {
        "labels": dict(c.label_to_id),
        "ids": dict(c.id_to_label),
        "history": dict(c.training_history),
        "counts": {k: len(v) for k, v in c.memory.examples.items()},
        "head_out": c.adaptive_head.model[-1].out_features,
        "prototypes": {k: v.clone() for k, v in c.memory.prototypes.items()},
        "steps": c.train_steps,
    }


def assert_unchanged(c, before):
    after = state_of(c)
    assert {k: v for k, v in after.items() if k != "prototypes"} == {k: v for k, v in before.items() if k != "prototypes"}
    assert after["prototypes"].keys() == before["prototypes"].keys()
    assert all(torch.equal(after["prototypes"][k], before["prototypes"][k]) for k in before["prototypes"])


# --- a failed add_examples changes nothing -----------------------------------------------------

@pytest.mark.parametrize("failure", [RuntimeError("boom"), KeyboardInterrupt()], ids=["error", "interrupt"])
@pytest.mark.parametrize("where", ["_train_adaptive_head", "_train_new_classes"])
def test_a_failure_while_training_rolls_everything_back(clf, data, monkeypatch, failure, where):
    before = state_of(clf)
    reference = clf.predict("tr1_3", k=3)

    def explode(*args, **kwargs):
        raise failure

    monkeypatch.setattr(clf, where, explode)
    if where == "_train_adaptive_head":
        texts, labels = data.classes(4, 3, seed=5, prefix="x")                  # existing classes only
    else:
        texts, labels = ["nx0", "nx1"], ["NEW", "NEW"]                        # a new class
        data.add("nx0", 7, 0.1, 1), data.add("nx1", 7, 0.1, 2)
    with pytest.raises(type(failure)):
        clf.add_examples(texts, labels)

    assert_unchanged(clf, before)
    assert clf.predict("tr1_3", k=3) == reference
    assert clf.predict("tr0_0", k=1)[0][0] == "C0"                              # the search index is intact too

    monkeypatch.undo()                                                          # and the classifier still works
    data.patch(monkeypatch)
    clf.add_examples(texts, labels)
    assert set(labels) <= set(clf.label_to_id)


def test_a_failure_while_encoding_changes_nothing(clf, monkeypatch):
    before = state_of(clf)

    def explode(texts):
        raise RuntimeError("encoder down")

    monkeypatch.setattr(clf, "_get_embeddings", explode)
    with pytest.raises(RuntimeError):
        clf.add_examples(["a new text"], ["BRAND_NEW"])
    assert_unchanged(clf, before)


def test_a_failure_discards_nothing_from_calibration(clf, data, monkeypatch):
    X, y = data.classes(6, 3, seed=8, prefix="cal")
    clf.calibrate(X, y)
    monkeypatch.setattr(clf, "_train_new_classes", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    data.add("q0", 9, 0.1, 1)
    with pytest.raises(RuntimeError):
        clf.add_examples(["q0"], ["Q"])
    assert clf.calibration


# --- clear_memory without labels really clears --------------------------------------------------

def test_clear_memory_with_no_labels_empties_the_classifier(clf, data):
    X, y = data.classes(6, 3, seed=8, prefix="cal")
    clf.calibrate(X, y)

    clf.clear_memory()

    assert clf.label_to_id == {} and clf.id_to_label == {} and clf.training_history == {}
    assert clf.adaptive_head is None and clf.calibration is None
    assert clf.predict("tr0_0") == []
    assert clf.ood_score("tr0_0") == float("inf")
    assert clf.get_memory_stats()["num_classes"] == 0

    texts, labels = data.classes(8, 2, seed=3, prefix="again")                  # and it can start over
    clf.add_examples(texts, labels)
    assert clf.predict("again0_1", k=1)[0][0] == "C0"


# --- EWC ---------------------------------------------------------------------------------------------

def _small_head():
    from adaptive_classifier.models import AdaptiveHead
    torch.manual_seed(0)
    return AdaptiveHead(8, 3, hidden_dims=[8, 4])


def _dataset(n=24):
    g = torch.Generator().manual_seed(0)
    return torch.utils.data.TensorDataset(torch.randn(n, 8, generator=g), torch.randint(0, 3, (n,), generator=g))


def test_ewc_penalty_is_zero_at_the_stored_weights_and_grows_as_the_model_moves():
    head = _small_head()
    ewc = EWC(head, _dataset(), ewc_lambda=10.0)
    assert ewc.ewc_loss(model=head).item() == 0.0
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.1)
    small = ewc.ewc_loss(model=head).item()
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.1)
    assert 0 < small < ewc.ewc_loss(model=head).item()


def test_ewc_compares_only_the_rows_that_existed_before_new_classes_were_added():
    import copy
    old = _small_head()
    ewc = EWC(copy.deepcopy(old), _dataset())
    grown = copy.deepcopy(old)
    grown.update_num_classes(5)                                     # two new output rows
    assert ewc.ewc_loss(model=grown).item() == pytest.approx(0.0)   # unchanged old rows cost nothing
    with torch.no_grad():
        grown.model[-1].weight[:3] += 0.5
    assert ewc.ewc_loss(model=grown).item() > 0
    grown2 = copy.deepcopy(old)
    grown2.update_num_classes(5)
    with torch.no_grad():
        grown2.model[-1].weight[3:] += 5.0                          # changing only the new rows is free
    assert ewc.ewc_loss(model=grown2).item() == pytest.approx(0.0)


def test_adding_a_class_applies_a_real_ewc_penalty(clf, data, monkeypatch):
    import adaptive_classifier.classifier as module
    penalties = []
    real = module.EWC.ewc_loss

    def spy(self, batch_size=None, model=None):
        value = real(self, batch_size=batch_size, model=model)
        penalties.append(float(value))
        return value

    monkeypatch.setattr(module.EWC, "ewc_loss", spy)
    texts = [data.add(f"n{i}", 6, 0.1, i) for i in range(10)]
    clf.add_examples(texts, ["NEW"] * 10)
    assert penalties and max(penalties) > 0


# --- merging ------------------------------------------------------------------------------------------

def test_merging_two_classifiers_into_each_other_at_once_cannot_deadlock(new_classifier, data):
    a, b = new_classifier(head_steps=20), new_classifier(head_steps=20)
    x1, y1 = data.classes(6, 2, seed=1, prefix="a")
    x2, y2 = [data.add(f"b{i}", 4 + i % 2, 0.1, i) for i in range(12)], ["D0", "D1"] * 6
    a.add_examples(x1, y1)
    b.add_examples(x2, y2)
    threads = [threading.Thread(target=a.merge_classifiers, args=(b,), daemon=True),
               threading.Thread(target=b.merge_classifiers, args=(a,), daemon=True)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not any(t.is_alive() for t in threads), "the two merges are waiting on each other"


def test_merge_keeps_the_larger_recorded_class_spread(new_classifier, data):
    a, b = new_classifier(head_steps=20), new_classifier(head_steps=20)
    x, y = data.classes(8, 2, seed=1, prefix="a")
    a.add_examples(x, y)
    b.add_examples(x, y)
    b.memory.saved_radii["C0"] = 5.0                      # as if recorded from many more examples
    a.merge_classifiers(b)
    assert a.memory.class_radius("C0") >= 5.0


# --- a reloaded classifier keeps its prototypes where they were ----------------------------------------------

def test_adding_one_example_to_a_reloaded_class_barely_moves_its_prototype(new_classifier, data, tmp_path):
    texts, labels = data.classes(30, n_classes=3, noise=0.3, seed=1)
    live = new_classifier(head_steps=20)
    live.add_examples(texts, labels)
    live.save(str(tmp_path), include_onnx=False)
    loaded = AdaptiveClassifier.load(str(tmp_path), device="cpu", use_onnx=False)
    assert len(loaded.memory.examples["C0"]) < 30                    # only representative examples came back
    saved = loaded.memory.prototypes["C0"].clone()

    extra = data.add("extra", 0, 0.3, 999)
    live.add_examples([extra], ["C0"])
    loaded.add_examples([extra], ["C0"])

    assert torch.dist(loaded.memory.prototypes["C0"], saved) < 0.05              # 1 of 31 examples
    assert torch.dist(loaded.memory.prototypes["C0"], live.memory.prototypes["C0"]) < 0.01


def test_a_reloaded_prototype_follows_new_examples_in_proportion(new_classifier, data, tmp_path):
    texts, labels = data.classes(20, n_classes=2, noise=0.1, seed=1)
    live = new_classifier(head_steps=20)
    live.add_examples(texts, labels)
    live.save(str(tmp_path), include_onnx=False)
    loaded = AdaptiveClassifier.load(str(tmp_path), device="cpu", use_onnx=False)
    more = [data.add(f"m{i}", 0, 0.1, 500 + i) for i in range(20)]
    live.add_examples(more, ["C0"] * 20)
    loaded.add_examples(more, ["C0"] * 20)
    assert torch.dist(loaded.memory.prototypes["C0"], live.memory.prototypes["C0"]) < 0.01


def test_removing_examples_from_a_reloaded_class_recomputes_its_prototype_from_what_is_held(new_classifier, data, tmp_path):
    texts, labels = data.classes(20, n_classes=2, noise=0.1, seed=1)
    live = new_classifier(head_steps=20)
    live.add_examples(texts, labels)
    live.save(str(tmp_path), include_onnx=False)
    loaded = AdaptiveClassifier.load(str(tmp_path), device="cpu", use_onnx=False)
    held = [ex.text for ex in loaded.memory.examples["C0"]]
    loaded.remove_examples(held[:1], label="C0")
    assert "C0" not in loaded.memory.prototype_base
    assert loaded.predict("tr0_0", k=1)[0][0] == "C0"


# --- ONNX round trip and batching ----------------------------------------------------------------------

def test_resaving_an_onnx_loaded_classifier_keeps_the_original_model_name(base_model, tmp_path):
    pytest.importorskip("optimum.onnxruntime")
    import json
    clf = AdaptiveClassifier(base_model, config={"pooling": "mean"}, device="cpu", use_onnx=False)
    clf.add_examples(["great love", "good fast", "terrible hate", "awful bad"] * 3, ["p", "p", "n", "n"] * 3)
    first, second = tmp_path / "first", tmp_path / "second"
    clf.save(str(first), include_onnx=True, quantize_onnx=False)

    loaded = AdaptiveClassifier.load(str(first), use_onnx=True, prefer_quantized=False, device="cpu")
    loaded.save(str(second), include_onnx=False)

    assert json.loads((second / "config.json").read_text())["model_name"] == base_model


def test_texts_are_encoded_in_batches_of_the_configured_size(new_classifier):
    c = new_classifier(batch_size=4)
    sizes = []
    real = c.model.forward

    def spy(*args, **kwargs):
        sizes.append(kwargs["input_ids"].shape[0])
        return real(*args, **kwargs)

    c.model.forward = spy
    texts = ["great love", "good fast", "terrible hate"] * 4                  # 12 texts
    embeddings = c._get_embeddings(texts)
    assert sizes == [4, 4, 4] and len(embeddings) == 12

    c.config.batch_size = 100
    sizes.clear()
    whole = c._get_embeddings(texts)
    assert sizes == [12]
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(embeddings, whole))


def test_remove_examples_rejects_a_bare_string(clf):
    with pytest.raises(ValueError, match="not a single string"):
        clf.remove_examples("tr0_0")
