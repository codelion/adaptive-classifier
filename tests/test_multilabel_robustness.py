"""MultiLabelAdaptiveClassifier: learning quality, lifecycle operations and bad input.

The accuracy tests use hand-made embeddings (see synthetic.py) in which a text with
several labels sits between their axes, so a correct classifier should be near perfect.
"""

import threading

import pytest
import torch
import torch.nn.functional as F

from adaptive_classifier import MultiLabelAdaptiveClassifier
from .synthetic import DIM, Dataset, point

AXES = {"a": 0, "b": 1, "c": 2, "d": 3}


def _embed(labels, noise, seed):
    v = sum(point(AXES[label], 0.0, 0) for label in labels)
    g = torch.Generator().manual_seed(seed)
    return F.normalize(F.normalize(v, dim=0) + noise * torch.randn(DIM, generator=g), dim=0)


class MultiLabelData(Dataset):
    def sample(self, label_sets, n_per, noise=0.05, seed=0, prefix="t"):
        texts, labels = [], []
        for s, label_set in enumerate(label_sets):
            for i in range(n_per):
                name = f"{prefix}{'+'.join(label_set)}_{i}_{seed}"
                self.store[name] = _embed(label_set, noise, seed * 7919 + s * 131 + i)
                texts.append(name)
                labels.append(list(label_set))
        return texts, labels


@pytest.fixture
def data(monkeypatch, base_model):
    ds = MultiLabelData()
    ds.patch(monkeypatch)
    return ds


@pytest.fixture
def make(base_model):
    def build(**kwargs):
        return MultiLabelAdaptiveClassifier(base_model, device="cpu", use_onnx=False, **kwargs)
    return build


def _predicted(clf, text, **kwargs):
    return {label for label, _ in clf.predict_multilabel(text, **kwargs)}


def _exact_match(clf, texts, labels):
    return sum(_predicted(clf, t) == set(l) for t, l in zip(texts, labels)) / len(texts)


# --- learning ---------------------------------------------------------------

def test_learns_separable_multilabel_data(data, make):
    sets = [("a",), ("b",), ("c",), ("a", "b"), ("b", "c")]
    train_x, train_y = data.sample(sets, 12, seed=1)
    test_x, test_y = data.sample(sets, 6, seed=2, prefix="v")
    clf = make()
    clf.add_examples(train_x, train_y)
    assert _exact_match(clf, test_x, test_y) >= 0.9


def test_a_label_added_later_is_learned_without_forgetting_the_rest(data, make):
    sets = [("a",), ("b",), ("a", "b")]
    train_x, train_y = data.sample(sets, 12, seed=1)
    clf = make()
    clf.add_examples(train_x, train_y)

    new_x, new_y = data.sample([("c",), ("a", "c")], 12, seed=3, prefix="n")
    clf.add_examples(new_x, new_y)

    test_x, test_y = data.sample(sets + [("c",), ("a", "c")], 6, seed=4, prefix="v")
    assert _exact_match(clf, test_x, test_y) >= 0.85


def test_a_text_repeated_with_different_labels_is_one_training_text(data, make):
    texts, labels = data.sample([("a", "b")], 10, seed=1)
    clf = make()
    clf.add_examples(texts, labels)
    assert _predicted(clf, texts[0]) == {"a", "b"}


# --- lifecycle operations ---------------------------------------------------

def test_forget_removes_a_label_everywhere(data, make):
    sets = [("a",), ("b",), ("c",), ("a", "b")]
    train_x, train_y = data.sample(sets, 10, seed=1)
    clf = make()
    clf.add_examples(train_x, train_y)

    clf.forget("b")

    assert "b" not in clf.label_to_id and "b" not in clf.label_thresholds
    assert clf.adaptive_head.model[-1].out_features == len(clf.label_to_id) == 2
    assert all("b" not in _predicted(clf, t) for t in train_x)
    assert _predicted(clf, train_x[0]) == {"a"}                      # a-only text
    assert _predicted(clf, train_x[20]) == {"c"}


def test_remove_examples_retrains_and_can_drop_a_label(data, make):
    train_x, train_y = data.sample([("a",), ("b",), ("c",)], 10, seed=1)
    clf = make()
    clf.add_examples(train_x, train_y)

    only_c = [t for t, l in zip(train_x, train_y) if l == ["c"]]
    assert clf.remove_examples(only_c) == len(only_c)
    assert "c" not in clf.label_to_id
    assert clf.predict_multilabel(train_x[0])


def test_save_and_load_keep_settings_thresholds_and_predictions(data, make, tmp_path):
    sets = [("a",), ("b",), ("a", "b"), ("c",)]
    train_x, train_y = data.sample(sets, 10, seed=1)
    clf = make(default_threshold=0.45, min_predictions=2, max_predictions=3)
    clf.add_examples(train_x, train_y)
    clf.save(str(tmp_path), include_onnx=False)

    loaded = MultiLabelAdaptiveClassifier.load(str(tmp_path), device="cpu", use_onnx=False)

    assert isinstance(loaded, MultiLabelAdaptiveClassifier)
    assert (loaded.default_threshold, loaded.min_predictions, loaded.max_predictions) == (0.45, 2, 3)
    assert loaded.label_thresholds == clf.label_thresholds
    for text in train_x[::7]:
        a, b = dict(clf.predict_multilabel(text)), dict(loaded.predict_multilabel(text))
        assert a.keys() == b.keys()
        assert all(a[l] == pytest.approx(b[l], abs=1e-4) for l in a)


def test_a_plain_classifier_save_has_no_multilabel_block(data, base_model, tmp_path):
    from adaptive_classifier import AdaptiveClassifier
    clf = AdaptiveClassifier(base_model, device="cpu", use_onnx=False)
    texts, labels = data.sample([("a",), ("b",)], 5, seed=1)
    clf.add_examples(texts, [l[0] for l in labels])
    clf.save(str(tmp_path), include_onnx=False)
    assert "multilabel" not in (tmp_path / "config.json").read_text()


# --- limits and thresholds ----------------------------------------------------

def test_max_labels_is_a_hard_cap_even_with_min_predictions(data, make):
    train_x, train_y = data.sample([("a", "b", "c")], 10, seed=1)
    clf = make(min_predictions=3)
    clf.add_examples(train_x, train_y)
    assert len(clf.predict_multilabel(train_x[0], max_labels=1)) == 1
    assert clf.predict_multilabel(train_x[0], max_labels=0) == []


def test_min_predictions_returns_something_when_nothing_clears_the_threshold(data, make):
    train_x, train_y = data.sample([("a",), ("b",)], 10, seed=1)
    clf = make(min_predictions=1)
    clf.add_examples(train_x, train_y)
    assert len(clf.predict_multilabel(train_x[0], threshold=1.0)) >= 1


# --- bad input ----------------------------------------------------------------

@pytest.mark.parametrize("labels, message", [
    (["pos"], "list of label lists"),                  # a bare string used to become classes 'p', 'o', 's'
    ([b"pos"], "list of label lists"),
    ([5], "list of label lists"),
    ([[1, 2]], "labels must be strings"),
    ([[None]], "labels must be strings"),
])
def test_malformed_labels_are_rejected_before_anything_changes(make, labels, message):
    clf = make()
    with pytest.raises(ValueError, match=message):
        clf.add_examples(["great love"], labels)
    assert clf.label_to_id == {}


def test_non_string_texts_are_rejected(make):
    clf = make()
    with pytest.raises(ValueError, match="strings"):
        clf.add_examples([None], [["a"]])
    with pytest.raises(ValueError, match="strings"):
        clf.add_examples([7], [[]])           # also when no label would be stored


def test_empty_and_mismatched_inputs_are_rejected(make):
    clf = make()
    with pytest.raises(ValueError):
        clf.add_examples([], [])
    with pytest.raises(ValueError):
        clf.add_examples(["a b"], [["x"], ["y"]])


@pytest.mark.parametrize("kwargs", [
    {"threshold": -0.1}, {"threshold": 1.5}, {"max_labels": -1}, {"max_labels": 1.5}, {"max_labels": True},
])
def test_bad_prediction_arguments_are_rejected(data, make, kwargs):
    train_x, train_y = data.sample([("a",), ("b",)], 5, seed=1)
    clf = make()
    clf.add_examples(train_x, train_y)
    with pytest.raises(ValueError):
        clf.predict_multilabel(train_x[0], **kwargs)


def test_predict_validates_k_and_empty_text(data, make):
    train_x, train_y = data.sample([("a",), ("b",)], 5, seed=1)
    clf = make()
    clf.add_examples(train_x, train_y)
    with pytest.raises(ValueError):
        clf.predict(train_x[0], k=-1)
    with pytest.raises(ValueError):
        clf.predict_multilabel("")
    assert clf.predict(train_x[0], k=0) == []


@pytest.mark.parametrize("kwargs", [
    {"default_threshold": 1.5}, {"default_threshold": -0.2}, {"min_predictions": -1}, {"max_predictions": 0},
])
def test_bad_constructor_settings_are_rejected(make, kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


def test_an_untrained_classifier_predicts_nothing(make):
    assert make().predict_multilabel("anything") == []


# --- concurrency --------------------------------------------------------------

def test_predicting_while_labels_are_added_and_forgotten_does_not_crash(data, make):
    train_x, train_y = data.sample([("a",), ("b",), ("c",)], 8, seed=1)
    clf = make()
    clf.add_examples(train_x, train_y)
    extra_x, extra_y = data.sample([("d",)], 8, seed=2, prefix="x")
    stop, errors = threading.Event(), []

    def reader():
        while not stop.is_set():
            try:
                clf.predict_multilabel(train_x[0])
                clf.predict(train_x[9], k=3)
            except Exception as error:      # noqa: BLE001 - any failure is the bug
                errors.append(error)
                return

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    try:
        for _ in range(3):
            clf.add_examples(extra_x, extra_y)
            clf.forget("d")
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert not errors, errors[0]


def test_predict_batch_agrees_with_predict(data, make):
    train_x, train_y = data.sample([("a",), ("b",), ("a", "b")], 8, seed=1)
    clf = make()
    clf.add_examples(train_x, train_y)
    batch = clf.predict_batch(train_x[::5], k=3)
    assert batch == [clf.predict(text, k=3) for text in train_x[::5]]
    with pytest.raises(ValueError):
        clf.predict_batch([])


# --- thresholds, loading, unsupported combinations ---------------------------------------------------

def test_an_explicit_threshold_applies_to_every_label(data, make):
    train_x, train_y = data.sample([("a",), ("b",), ("c",)], 8, seed=1)
    clf = make(min_predictions=0)
    clf.add_examples(train_x, train_y)
    assert _predicted(clf, train_x[0], threshold=0.0) == {"a", "b", "c"}       # nothing is filtered out
    assert _predicted(clf, train_x[0], threshold=1.0) == set()                  # sigmoid never reaches 1.0
    low = _predicted(clf, train_x[0], threshold=0.001)
    high = _predicted(clf, train_x[0], threshold=0.9)
    assert high <= low


def test_without_a_threshold_the_learned_per_label_ones_are_used(data, make):
    train_x, train_y = data.sample([("a",), ("b",), ("c",)], 8, seed=1)
    clf = make()
    clf.add_examples(train_x, train_y)
    clf.label_thresholds["b"] = 0.0
    assert "b" in _predicted(clf, train_x[0])


def test_the_base_class_loader_returns_a_multilabel_classifier(data, make, tmp_path):
    from adaptive_classifier import AdaptiveClassifier

    train_x, train_y = data.sample([("a",), ("b",), ("a", "b")], 8, seed=1)
    clf = make(default_threshold=0.45)
    clf.add_examples(train_x, train_y)
    clf.save(str(tmp_path), include_onnx=False)

    loaded = AdaptiveClassifier.load(str(tmp_path), device="cpu", use_onnx=False)

    assert isinstance(loaded, MultiLabelAdaptiveClassifier)
    assert loaded.default_threshold == 0.45
    for text in train_x[::5]:
        assert dict(loaded.predict_multilabel(text)).keys() == dict(clf.predict_multilabel(text)).keys()


def test_predict_accepts_the_abstain_options_of_the_base_class(data, make):
    train_x, train_y = data.sample([("a",), ("b",)], 8, seed=1)
    clf = make()
    clf.add_examples(train_x, train_y)
    assert clf.predict(train_x[0], abstain_below=0.0)
    assert clf.predict(train_x[0], abstain_below=1.1) == []
    far = "far_away"
    data.store[far] = -torch.ones(32) / 32 ** 0.5
    assert clf.predict(far, abstain_ood=True) == []


@pytest.mark.parametrize("method", ["calibrate", "calibration_report", "predict_set"])
def test_confidence_calibration_is_refused_rather_than_silently_wrong(data, make, method):
    clf = make()
    with pytest.raises(NotImplementedError, match="one label per text"):
        getattr(clf, method)(["x"], ["a"])


def test_strategic_mode_is_refused_for_multilabel(make):
    with pytest.raises(ValueError, match="not supported"):
        make(config={"enable_strategic_mode": True, "cost_coefficients": [0.1] * 32})


def test_labels_given_as_a_dict_are_rejected(make):
    with pytest.raises(ValueError, match="list of label lists"):
        make().add_examples(["great love"], [{"a": 1}])


def test_thresholds_follow_merges_and_removals(data, make):
    a_x, a_y = data.sample([("a",)], 8, seed=1)
    b_x, b_y = data.sample([("b",), ("c",)], 8, seed=2, prefix="o")
    one, other = make(), make()
    one.add_examples(a_x, a_y)
    other.add_examples(b_x, b_y)
    one.merge_classifiers(other)
    assert set(one.label_thresholds) == {"a", "b", "c"}

    c_texts = [t for t, l in zip(b_x, b_y) if l == ["c"]]
    one.remove_examples(c_texts)
    assert set(one.label_thresholds) == {"a", "b"}
