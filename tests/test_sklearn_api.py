"""scikit-learn interface (offline, tiny local model).

A randomly initialised encoder carries no meaning, so tests that check
accuracy swap in hand-made embeddings: text "a3" sits near axis 0, "b2" near
axis 1, and so on. Classes are therefore perfectly separable.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from adaptive_classifier import AdaptiveClassifier, SklearnAdaptiveClassifier

DIM = 32
PROTO_ONLY = {"new_class_example_threshold": 0}   # always use prototype_weight / neural_weight
AXES = {"a": 0, "b": 1, "c": 2}


def geometry(self, texts):
    out = []
    for text in texts:
        v = torch.zeros(DIM)
        v[AXES[text[0]]] = 1.0
        g = torch.Generator().manual_seed(int(text[1:]))
        out.append(F.normalize(v + 0.1 * torch.randn(DIM, generator=g), dim=0))
    return out


@pytest.fixture
def separable(monkeypatch):
    monkeypatch.setattr(AdaptiveClassifier, "_get_embeddings", geometry)


def data(n=12, classes=("a", "b", "c")):
    X = [f"{k}{i}" for k in classes for i in range(1, n + 1)]
    y = [k.upper() for k in classes for _ in range(n)]
    return X, y


def est(base_model, **kw):
    return SklearnAdaptiveClassifier(base_model, use_onnx=False, device="cpu", **kw)


# --- estimator contract ----------------------------------------------------------

def test_params_round_trip_and_clone(base_model):
    e = est(base_model, config={"pooling": "mean"})
    assert e.get_params()["model_name"] == base_model
    assert e.get_params()["config"] == {"pooling": "mean"}
    e.set_params(config={"pooling": "cls"}, use_onnx=False)
    copy = clone(e)
    assert copy.get_params() == e.get_params()
    assert copy is not e


def test_unfitted_estimator_raises(base_model):
    with pytest.raises(NotFittedError):
        est(base_model).predict(["a1"])
    with pytest.raises(NotFittedError):
        est(base_model).predict_proba(["a1"])


def test_fit_returns_self_and_sets_classes(base_model, separable):
    e = est(base_model)
    X, y = data()
    assert e.fit(X, y) is e
    assert list(e.classes_) == ["A", "B", "C"]
    assert isinstance(e.classifier_, AdaptiveClassifier)


def test_predict_and_proba_agree(base_model, separable):
    e = est(base_model).fit(*data())
    X = ["a1", "b2", "c3", "a4"]
    proba = e.predict_proba(X)
    assert proba.shape == (4, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()
    assert list(e.classes_[proba.argmax(axis=1)]) == list(e.predict(X))
    assert list(e.predict(X)) == ["A", "B", "C", "A"]
    assert np.allclose(np.exp(e.predict_log_proba(X)), proba)


def test_proba_columns_follow_classes_order_not_insertion_order(base_model, separable):
    e = est(base_model).fit(*data(classes=("c", "a", "b")))   # inserted c, a, b
    assert list(e.classes_) == ["A", "B", "C"]
    assert e.predict_proba(["a1"])[0].argmax() == 0
    assert e.predict_proba(["c1"])[0].argmax() == 2


def test_score_is_accuracy(base_model, separable):
    e = est(base_model).fit(*data())
    assert e.score(["a1", "b1", "c1", "a2"], ["A", "B", "C", "B"]) == pytest.approx(0.75)


def test_fit_starts_from_scratch(base_model, separable):
    e = est(base_model).fit(*data())
    e.fit(*data(classes=("a", "b")))
    assert list(e.classes_) == ["A", "B"]


def test_integer_labels_come_back_as_integers(base_model, separable):
    X, y = data()
    codes = {"A": 10, "B": 2, "C": 7}
    e = est(base_model).fit(X, [codes[label] for label in y])
    assert list(e.classes_) == [2, 7, 10]
    predicted = e.predict(["a1", "b1", "c1"])
    assert list(predicted) == [10, 2, 7]
    assert np.issubdtype(predicted.dtype, np.integer)


def test_numpy_labels_are_accepted(base_model, separable):
    X, y = data()
    e = est(base_model).fit(np.array(X), np.array(y))
    assert list(e.predict(np.array(["a1", "b1"]))) == ["A", "B"]


# --- partial_fit ---------------------------------------------------------------------

def test_partial_fit_accumulates_and_adds_classes(base_model, separable):
    e = est(base_model)
    e.partial_fit(["a1", "a2", "b1", "b2"], ["A", "A", "B", "B"])
    assert list(e.classes_) == ["A", "B"]
    assert e.classifier_.training_history["A"] == 2

    e.partial_fit(["a3", "c1", "c2"], ["A", "C", "C"])
    assert list(e.classes_) == ["A", "B", "C"]
    assert e.classifier_.training_history == {"A": 3, "B": 2, "C": 2}
    assert e.predict(["c3"])[0] == "C"
    assert e.predict_proba(["c3"]).shape == (1, 3)


def test_partial_fit_validates_against_classes(base_model, separable):
    e = est(base_model)
    with pytest.raises(ValueError, match="not in classes"):
        e.partial_fit(["a1", "d1"], ["A", "D"], classes=["A", "B"])
    e.partial_fit(["a1", "b1"], ["A", "B"], classes=["A", "B"])
    assert list(e.classes_) == ["A", "B"]


# --- input validation ------------------------------------------------------------------

def test_single_column_input_is_accepted(base_model, separable):
    X, y = data(n=12)
    e = est(base_model).fit(np.array(X).reshape(-1, 1), y)
    assert list(e.predict(np.array(["a1"]).reshape(-1, 1))) == ["A"]


@pytest.mark.parametrize("X, message", [
    ([], "empty"),
    ([["a1", "a2"], ["b1", "b2"]], "1-D"),
    (["a1", 5], "strings"),
    (np.zeros((3, 2)), "1-D"),
])
def test_bad_x_is_rejected(base_model, X, message):
    with pytest.raises(ValueError, match=message):
        est(base_model).fit(X, ["A"] * len(X))


def test_mismatched_lengths_are_rejected(base_model):
    with pytest.raises(ValueError, match="inconsistent"):
        est(base_model).fit(["a1", "a2"], ["A"])


def test_predict_rejects_non_text(base_model, separable):
    e = est(base_model).fit(*data(n=12))
    with pytest.raises(ValueError, match="strings"):
        e.predict([1, 2])


# --- the point: it works with scikit-learn tooling ---------------------------------------

def test_cross_val_score(base_model, separable):
    X, y = data(n=12)
    # Folds see ~8 examples per class, under the default new-class threshold of 10;
    # that regime leans on a barely-trained head, which is not what this test is about.
    scores = cross_val_score(est(base_model, config=PROTO_ONLY), X, y, cv=3)
    assert scores.shape == (3,)
    assert scores.mean() > 0.9


def test_pipeline_with_a_text_step(base_model, separable):
    pipeline = Pipeline([
        ("clean", FunctionTransformer(lambda texts: [t.strip() for t in texts])),
        ("clf", est(base_model)),
    ])
    X, y = data(n=12)
    pipeline.fit([f"  {t} " for t in X], y)
    assert list(pipeline.predict(["  a1", "b1  "])) == ["A", "B"]
    assert pipeline.score([" c1 "], ["C"]) == 1.0


def test_grid_search_over_config(base_model, separable):
    X, y = data(n=12)
    search = GridSearchCV(
        est(base_model),
        {"config": [{**PROTO_ONLY, "prototype_weight": 0.9, "neural_weight": 0.1},
                    {**PROTO_ONLY, "prototype_weight": 0.6, "neural_weight": 0.4}]},
        cv=2,
    )
    search.fit(X, y)
    assert search.best_score_ > 0.9
    assert search.best_params_["config"]["prototype_weight"] in (0.9, 0.6)
    assert list(search.predict(["b1"])) == ["B"]


def test_underlying_classifier_features_stay_reachable(base_model, separable, tmp_path):
    e = est(base_model).fit(*data())
    e.classifier_.forget("C")
    assert list(e.classifier_.label_to_id) == ["A", "B"]
    e.classifier_.save(str(tmp_path), include_onnx=False)
    assert (tmp_path / "config.json").exists()


def test_string_input_tag_is_declared(base_model):
    tags = est(base_model).__sklearn_tags__()
    assert tags.input_tags.string


# --- classes declared up front and label identity ---------------------------------------------

def test_partial_fit_lists_declared_classes_even_before_they_are_seen(base_model, separable):
    e = est(base_model)
    e.partial_fit(["a1", "a2", "b1", "b2"], ["A", "A", "B", "B"], classes=["A", "B", "Z"])
    assert list(e.classes_) == ["A", "B", "Z"]
    proba = e.predict_proba(["a1", "b1"])
    assert proba.shape == (2, 3)
    assert proba[:, 2].tolist() == [0.0, 0.0]               # nothing known about Z yet
    assert proba.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert set(e.predict(["a1", "b1"])) <= {"A", "B"}

    e.partial_fit(["c1", "c2"], ["Z", "Z"])                  # when Z arrives it is learned normally
    assert list(e.classes_) == ["A", "B", "Z"]
    assert e.predict(["c1"])[0] == "Z"


@pytest.mark.parametrize("labels", [[1, "1"], [1, "1", 2], [True, "True"]])
def test_labels_that_would_merge_into_one_class_are_rejected(base_model, separable, labels):
    e = est(base_model)
    texts = [f"a{i}" for i in range(len(labels))]
    with pytest.raises(ValueError, match="both map to the class"):
        e.fit(texts, labels)


def test_a_clashing_label_in_a_later_partial_fit_leaves_the_estimator_unchanged(base_model, separable):
    e = est(base_model)
    e.partial_fit(["a1", "a2"], [1, 1])
    with pytest.raises(ValueError, match="both map to the class"):
        e.partial_fit(["b1"], ["1"])
    assert list(e.classes_) == [1]
    assert e.classifier_.memory.examples["1"].__len__() == 2


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_none_and_nan_labels_are_rejected(base_model, separable, bad):
    with pytest.raises(ValueError, match="None or NaN"):
        est(base_model).fit(["a1", "b1"], ["A", bad])


def test_a_fitted_estimator_can_be_pickled(base_model, separable):
    import pickle
    X, y = data(n=6)
    e = est(base_model).fit(X, y)
    again = pickle.loads(pickle.dumps(e))
    assert list(again.predict(X[:3])) == list(e.predict(X[:3]))
