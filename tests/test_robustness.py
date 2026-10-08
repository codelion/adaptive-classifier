"""End-to-end quality and invariant checks.

These do not need a benchmark run: the data is synthetic and perfectly
separable (see tests/synthetic.py), so a shortfall is the classifier's fault.
They are the tests that would have caught the weaknesses found while building
the scikit-learn wrapper, and they check what a user relies on:

  * it learns separable classes, at small and large sizes
  * it keeps old classes when new ones arrive (no catastrophic forgetting)
  * it survives class imbalance and label noise
  * its output is always a valid, ordered probability distribution
  * it is deterministic and independent of example order where documented

Known weaknesses are recorded as strict expected failures, so they stay
visible and the test fails (forcing the marker off) once someone fixes them.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from adaptive_classifier import AdaptiveClassifier

from .synthetic import Dataset, accuracy

PROTOTYPE_ONLY = {"new_class_example_threshold": 0, "prototype_weight": 1.0, "neural_weight": 0.0}
SEEDS = range(4)


@pytest.fixture
def data(monkeypatch):
    return Dataset().patch(monkeypatch)


def fit(new_classifier, texts, labels, **config):
    clf = new_classifier(**config)
    clf.add_examples(texts, labels)
    return clf


# --- it learns separable classes ------------------------------------------------------

@pytest.mark.parametrize("n_per", [12, 30])
@pytest.mark.parametrize("noise", [0.1, 0.2])
def test_learns_separable_classes_with_enough_examples(new_classifier, data, n_per, noise):
    train_acc, test_acc = [], []
    for seed in SEEDS:
        X, y = data.classes(n_per, noise=noise, seed=seed)
        clf = fit(new_classifier, X, y)
        Xt, yt = data.classes(15, noise=noise, seed=seed + 50, prefix="te")
        train_acc.append(accuracy(clf, X, y))
        test_acc.append(accuracy(clf, Xt, yt))
    assert np.mean(train_acc) >= 0.98
    assert np.mean(test_acc) >= 0.92


@pytest.mark.parametrize("n_per", [1, 2, 3, 5, 8, 12])
def test_prototype_only_is_accurate_at_every_size(new_classifier, data, n_per):
    """The prototype path alone separates this data, even from a single example."""
    for seed in SEEDS:
        X, y = data.classes(n_per, noise=0.15, seed=seed)
        clf = fit(new_classifier, X, y, **PROTOTYPE_ONLY)
        Xt, yt = data.classes(15, noise=0.15, seed=seed + 50, prefix="te")
        assert accuracy(clf, X, y) == 1.0
        assert accuracy(clf, Xt, yt) >= 0.95


@pytest.mark.xfail(
    strict=True,
    reason="Known weakness: with fewer than new_class_example_threshold (10) examples per "
           "class the default blend gives the neural head 70% of the vote, and a head trained "
           "on a handful of examples outvotes correct prototypes. Observed on separable data: "
           "~75-92% training accuracy, ~66-83% held out, versus 100% from prototypes alone. "
           "Remove this marker when fixed.",
)
@pytest.mark.parametrize("n_per", [2, 3, 5, 8])
def test_default_config_fits_its_own_training_data_with_few_examples(new_classifier, data, n_per):
    accs = []
    for seed in SEEDS:
        X, y = data.classes(n_per, noise=0.1, seed=seed)
        accs.append(accuracy(fit(new_classifier, X, y), X, y))
    assert np.mean(accs) >= 0.97, f"training accuracy {np.mean(accs):.2f} at {n_per} per class"


# --- no catastrophic forgetting ---------------------------------------------------------

def test_new_class_does_not_degrade_old_classes(new_classifier, data):
    for seed in SEEDS:
        X, y = data.classes(15, n_classes=2, noise=0.1, seed=seed)
        Xt, yt = data.classes(15, n_classes=2, noise=0.1, seed=seed + 50, prefix="te")
        clf = fit(new_classifier, X, y)
        before = accuracy(clf, Xt, yt)

        X3, y3 = data.classes(15, n_classes=3, noise=0.1, seed=seed + 7, prefix="new")
        clf.add_examples(
            [t for t, l in zip(X3, y3) if l == "C2"], [l for l in y3 if l == "C2"]
        )
        after = accuracy(clf, Xt, yt)
        assert before >= 0.95
        assert after >= before - 0.05, f"seed {seed}: {before:.2f} -> {after:.2f}"
        assert "C2" in clf.label_to_id


def test_new_class_is_itself_learned(new_classifier, data):
    X, y = data.classes(15, n_classes=2, noise=0.1)
    clf = fit(new_classifier, X, y)
    X3, y3 = data.classes(15, n_classes=3, noise=0.1, seed=3, prefix="new")
    new_texts = [t for t, l in zip(X3, y3) if l == "C2"]
    clf.add_examples(new_texts, ["C2"] * len(new_texts))
    Xt, yt = data.classes(15, n_classes=3, noise=0.1, seed=60, prefix="te")
    new_only = [(t, l) for t, l in zip(Xt, yt) if l == "C2"]
    recall = np.mean([clf.predict(t)[0][0] == "C2" for t, _ in new_only])
    assert recall >= 0.9


# --- imbalance and noise -------------------------------------------------------------------

def test_minority_class_is_not_swallowed_by_a_large_one(new_classifier, data):
    big, _ = data.classes(120, n_classes=1, noise=0.1, seed=1)
    small_texts = [data.add(f"min_{i}", 1, 0.1, 900 + i) for i in range(12)]
    clf = fit(new_classifier, big + small_texts, ["C0"] * len(big) + ["C1"] * len(small_texts))

    held_out = [data.add(f"min_te_{i}", 1, 0.1, 5000 + i) for i in range(20)]
    recall = np.mean([clf.predict(t)[0][0] == "C1" for t in held_out])
    assert recall >= 0.9, f"minority recall {recall:.2f}"


def test_a_few_wrong_labels_do_not_wreck_the_model(new_classifier, data):
    rng = np.random.default_rng(0)
    X, y = data.classes(30, noise=0.1, seed=2)
    noisy = list(y)
    for i in rng.choice(len(y), size=len(y) // 10, replace=False):    # flip 10%
        noisy[i] = f"C{(int(y[i][1:]) + 1) % 3}"
    clf = fit(new_classifier, X, noisy)
    Xt, yt = data.classes(15, noise=0.1, seed=70, prefix="te")
    assert accuracy(clf, Xt, yt) >= 0.85


def test_wildly_overlapping_classes_degrade_gracefully(new_classifier, data):
    """Indistinguishable classes must give near-uniform confidence, not a confident guess."""
    X, y = data.classes(20, n_classes=2, noise=0.0, seed=0)        # identical points per class
    same = [data.add(f"same_{i}", 0, 0.0, 0) for i in range(len(X))]
    clf = fit(new_classifier, same, [("A", "B")[i % 2] for i in range(len(same))])
    top = clf.predict(same[0])
    assert len(top) == 2
    assert abs(top[0][1] - top[1][1]) < 0.2


# --- determinism and order --------------------------------------------------------------------

def test_training_is_deterministic(new_classifier, data):
    X, y = data.classes(12, noise=0.15, seed=4)
    first = fit(new_classifier, X, y)
    second = fit(new_classifier, X, y)
    probe = data.add("probe", 0, 0.2, 77)
    assert first.predict(probe) == second.predict(probe)


def test_prototype_only_predictions_ignore_example_order(new_classifier, data):
    X, y = data.classes(10, noise=0.15, seed=5)
    order = np.random.default_rng(1).permutation(len(X))
    a = fit(new_classifier, X, y, **PROTOTYPE_ONLY)
    b = fit(new_classifier, [X[i] for i in order], [y[i] for i in order], **PROTOTYPE_ONLY)
    for q in [data.add(f"q{i}", i % 3, 0.2, 800 + i) for i in range(10)]:
        pa, pb = dict(a.predict(q)), dict(b.predict(q))
        assert pa.keys() == pb.keys()
        assert all(pa[k] == pytest.approx(pb[k], abs=1e-5) for k in pa)


def test_adding_in_one_batch_or_many_gives_the_same_prototypes(new_classifier, data):
    X, y = data.classes(10, noise=0.15, seed=6)
    whole = fit(new_classifier, X, y, **PROTOTYPE_ONLY)
    parts = new_classifier(**PROTOTYPE_ONLY)
    for i in range(0, len(X), 7):
        parts.add_examples(X[i:i + 7], y[i:i + 7])
    for label in whole.memory.prototypes:
        assert np.allclose(whole.memory.prototypes[label].numpy(),
                           parts.memory.prototypes[label].numpy(), atol=1e-6)


def test_predict_and_predict_batch_agree(new_classifier, data):
    X, y = data.classes(12, noise=0.15, seed=8)
    clf = fit(new_classifier, X, y)
    queries = [data.add(f"b{i}", i % 3, 0.25, 300 + i) for i in range(12)]
    singles = [clf.predict(q, k=3) for q in queries]
    batch = clf.predict_batch(queries, k=3)
    for one, many in zip(singles, batch):
        assert [l for l, _ in one] == [l for l, _ in many]
        assert all(a == pytest.approx(b, abs=1e-5) for (_, a), (_, b) in zip(one, many))


# --- output invariants (property-based) --------------------------------------------------------

WORDS = "great terrible okay product service love hate awful fine good bad slow fast cheap broke works".split()
texts_st = st.lists(st.sampled_from(WORDS), min_size=1, max_size=8).map(" ".join)
labels_st = st.sampled_from(["pos", "neg", "neu", "été", "类别", "a b", "x-1"])


def assert_valid_distribution(preds, known, k):
    assert len(preds) <= k
    labels = [l for l, _ in preds]
    scores = [s for _, s in preds]
    assert len(set(labels)) == len(labels), "a label appeared twice"
    assert set(labels) <= set(known)
    assert all(np.isfinite(scores)) and all(0.0 <= s <= 1.0 + 1e-6 for s in scores)
    assert scores == sorted(scores, reverse=True)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    train=st.lists(st.tuples(texts_st, labels_st), min_size=1, max_size=14),
    queries=st.lists(texts_st, min_size=1, max_size=4),
    k=st.integers(min_value=0, max_value=9),
)
def test_predictions_are_always_a_valid_ordered_distribution(new_classifier, train, queries, k):
    clf = new_classifier()
    texts, labels = zip(*train)
    clf.add_examples(list(texts), list(labels))
    known = set(labels)

    for q in queries:
        preds = clf.predict(q, k=k)
        assert_valid_distribution(preds, known, k)
        if k >= len(known) and preds:
            assert sum(s for _, s in preds) == pytest.approx(1.0, abs=1e-5)
    for preds in clf.predict_batch(queries, k=max(k, 1)):
        assert_valid_distribution(preds, known, max(k, 1))


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(train=st.lists(st.tuples(texts_st, labels_st), min_size=2, max_size=12), q=texts_st)
def test_scores_and_ood_are_finite_for_any_training_set(new_classifier, train, q):
    clf = new_classifier()
    texts, labels = zip(*train)
    clf.add_examples(list(texts), list(labels))
    score = clf.ood_score(q)
    assert np.isfinite(score) and score >= 0.0
