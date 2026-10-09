"""merge_classifiers: the merged classifier should know both sets of classes."""

import pytest

from adaptive_classifier import AdaptiveClassifier

from .synthetic import Dataset


@pytest.fixture
def data(monkeypatch):
    return Dataset().patch(monkeypatch)


def _trained(new_classifier, data, axes, seed, prefix, n_per=12):
    clf = new_classifier()
    texts, labels = [], []
    for axis in axes:
        for i in range(n_per):
            name = data.add(f"{prefix}{axis}_{i}", axis, 0.1, seed * 1000 + axis * 50 + i)
            texts.append(name)
            labels.append(f"C{axis}")
    clf.add_examples(texts, labels)
    return clf, texts, labels


def _accuracy(clf, texts, labels):
    return sum(clf.predict(t)[0][0] == l for t, l in zip(texts, labels)) / len(texts)


def test_merged_classifier_recognises_both_sets_of_classes(new_classifier, data):
    a, a_x, a_y = _trained(new_classifier, data, [0, 1], 1, "a")
    b, b_x, b_y = _trained(new_classifier, data, [2, 3], 2, "b")

    a.merge_classifiers(b)

    assert set(a.label_to_id) == {"C0", "C1", "C2", "C3"}
    assert _accuracy(a, a_x, a_y) >= 0.9
    assert _accuracy(a, b_x, b_y) >= 0.9


def test_merging_into_an_empty_classifier_gives_a_working_classifier(new_classifier, data):
    empty = new_classifier()
    b, b_x, b_y = _trained(new_classifier, data, [2, 3], 2, "b")
    empty.merge_classifiers(b)
    assert _accuracy(empty, b_x, b_y) >= 0.9


def test_merging_classifiers_that_share_a_class_adds_up_their_training_counts(new_classifier, data):
    a, _, _ = _trained(new_classifier, data, [0, 1], 1, "a")
    b, _, _ = _trained(new_classifier, data, [1, 2], 2, "b")
    before = a.training_history["C1"]
    a.merge_classifiers(b)
    assert a.training_history["C1"] == before + b.training_history["C1"]
    assert a.training_history["C2"] == b.training_history["C2"]       # so the head gets its full weight


def test_merge_leaves_the_other_classifier_alone(new_classifier, data):
    a, _, _ = _trained(new_classifier, data, [0, 1], 1, "a")
    b, b_x, b_y = _trained(new_classifier, data, [2, 3], 2, "b")
    a.merge_classifiers(b)
    a.forget("C2")
    assert "C2" in b.label_to_id
    assert _accuracy(b, b_x, b_y) >= 0.9


def test_merge_discards_calibration_because_the_classes_changed(new_classifier, data):
    a, a_x, a_y = _trained(new_classifier, data, [0, 1], 1, "a")
    a.calibrate(a_x, a_y)
    assert a.calibration
    b, _, _ = _trained(new_classifier, data, [2, 3], 2, "b")
    a.merge_classifiers(b)
    assert not a.calibration


def test_merge_rejects_different_embedding_sizes(new_classifier, data):
    a, _, _ = _trained(new_classifier, data, [0, 1], 1, "a")
    b, _, _ = _trained(new_classifier, data, [2, 3], 2, "b")
    b.embedding_dim = 64
    with pytest.raises(ValueError, match="embedding"):
        a.merge_classifiers(b)


def test_merging_a_classifier_with_itself_changes_nothing_harmful(new_classifier, data):
    a, a_x, a_y = _trained(new_classifier, data, [0, 1], 1, "a")
    a.merge_classifiers(a)
    assert set(a.label_to_id) == {"C0", "C1"}
    assert _accuracy(a, a_x, a_y) >= 0.9
