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

A weakness that is known but not yet fixed should be recorded here as a strict
expected failure (`xfail(strict=True)`), so it stays visible and the test fails,
forcing the marker off, once someone fixes it.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hypothesis import HealthCheck, given, settings, strategies as st

from adaptive_classifier import AdaptiveClassifier

from .synthetic import DIM, Dataset, accuracy

PROTOTYPE_ONLY = {"new_class_example_threshold": 0, "prototype_weight": 1.0, "neural_weight": 0.0,
                  "head_steps": 10}
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


@pytest.mark.parametrize("n_per", [2, 3, 5, 8])
def test_default_config_fits_its_own_training_data_with_few_examples(new_classifier, data, n_per):
    """Regression: below 10 examples per class the neural head (trained on a handful of
    points, and given 70% of the vote) used to outvote correct prototypes, so the model
    got only ~75-92% of its own training data right on perfectly separable data."""
    accs, held = [], []
    for seed in SEEDS:
        X, y = data.classes(n_per, noise=0.1, seed=seed)
        clf = fit(new_classifier, X, y)
        Xt, yt = data.classes(15, noise=0.1, seed=seed + 50, prefix="te")
        accs.append(accuracy(clf, X, y))
        held.append(accuracy(clf, Xt, yt))
    assert np.mean(accs) >= 0.97, f"training accuracy {np.mean(accs):.2f} at {n_per} per class"
    assert np.mean(held) >= 0.95, f"held-out accuracy {np.mean(held):.2f} at {n_per} per class"


def test_accuracy_does_not_fall_off_a_cliff_at_the_new_class_threshold(new_classifier, data):
    """Before 0.3.0 accuracy jumped from ~0.7 to 1.0 between 8 and 12 examples per class."""
    def mean_heldout(n_per):
        scores = []
        for seed in SEEDS:
            X, y = data.classes(n_per, noise=0.3, seed=seed)
            clf = fit(new_classifier, X, y)
            Xt, yt = data.classes(20, noise=0.3, seed=seed + 50, prefix="te")
            scores.append(accuracy(clf, Xt, yt))
        return np.mean(scores)

    below, above = mean_heldout(8), mean_heldout(12)
    assert below >= above - 0.08, f"8 per class: {below:.2f}, 12 per class: {above:.2f}"
    assert below >= 0.8


# --- the neural head is a real component, not decoration ------------------------------------------

def make_xor(data, n, seed, noise=0.12, tag="tr"):
    """Class A = (+,+) and (-,-) blobs, class B = (+,-) and (-,+): both class means are ~0,
    so prototypes carry no information and only a trained head can separate them."""
    g = torch.Generator().manual_seed(seed)
    texts, labels = [], []
    for i in range(n):
        for sx, sy, label in [(1, 1, "A"), (-1, -1, "A"), (1, -1, "B"), (-1, 1, "B")]:
            v = torch.zeros(DIM)
            v[0], v[1] = sx, sy
            name = f"xor_{tag}_{seed}_{i}_{sx}_{sy}"
            data.store[name] = F.normalize(v + noise * torch.randn(DIM, generator=g), dim=0)
            texts.append(name)
            labels.append(label)
    return texts, labels


@pytest.mark.parametrize("n_per", [8, 12, 30])
def test_default_blend_solves_a_problem_only_the_head_can(new_classifier, data, n_per):
    """Prototypes cannot separate XOR; the default blend must still succeed via the head.
    Before 0.3.0 the head was too undertrained to learn it (0.49-0.70)."""
    accs = []
    for seed in range(3):
        X, y = make_xor(data, n_per, seed)
        clf = fit(new_classifier, X, y)
        Xt, yt = make_xor(data, 15, seed + 50, tag="te")
        accs.append(accuracy(clf, Xt, yt))
    assert np.mean(accs) >= 0.95, f"XOR accuracy {np.mean(accs):.2f}"


def test_prototypes_alone_cannot_solve_xor(new_classifier, data):
    """Guards the test above: if this started passing, the XOR data would no longer
    prove that the head is doing the work."""
    X, y = make_xor(data, 12, 0)
    clf = fit(new_classifier, X, y, **PROTOTYPE_ONLY)
    Xt, yt = make_xor(data, 15, 50, tag="te")
    assert accuracy(clf, Xt, yt) <= 0.65


def test_head_alone_learns_separable_data(new_classifier, data):
    head_only = {"prototype_weight": 0.0, "neural_weight": 1.0, "new_class_example_threshold": 0}
    accs = []
    for seed in range(3):
        X, y = data.classes(30, noise=0.3, seed=seed)
        clf = fit(new_classifier, X, y, **head_only)
        Xt, yt = data.classes(20, noise=0.3, seed=seed + 50, prefix="te")
        accs.append(accuracy(clf, Xt, yt))
    assert np.mean(accs) >= 0.85, f"head-only accuracy {np.mean(accs):.2f}"


def test_head_steps_is_a_working_knob(new_classifier, data):
    X, y = data.classes(20, noise=0.3, seed=1)
    little = fit(new_classifier, X, y, head_steps=1, prototype_weight=0.0, neural_weight=1.0,
                 new_class_example_threshold=0)
    plenty = fit(new_classifier, X, y, head_steps=400, prototype_weight=0.0, neural_weight=1.0,
                 new_class_example_threshold=0)
    Xt, yt = data.classes(20, noise=0.3, seed=51, prefix="te")
    assert accuracy(plenty, Xt, yt) > accuracy(little, Xt, yt) + 0.1


# --- prototype scoring ------------------------------------------------------------------------------

def test_prototype_scores_clearly_favour_the_nearest_class(new_classifier, data):
    X, y = data.classes(10, noise=0.1, seed=3)
    sharp = fit(new_classifier, X, y, **PROTOTYPE_ONLY)
    legacy = fit(new_classifier, X, y, prototype_temperature=None, **PROTOTYPE_ONLY)
    query = data.add("center", 0, 0.0, 0)
    top_sharp = sharp.predict(query)[0][1]
    top_legacy = legacy.predict(query)[0][1]
    assert top_sharp > 0.7, f"sharp scoring gave the nearest class only {top_sharp:.2f}"
    assert top_legacy < 0.55, "legacy scoring is expected to be flat"


def test_temperature_changes_confidence_but_not_the_ranking_of_an_easy_case(new_classifier, data):
    X, y = data.classes(10, noise=0.1, seed=4)
    q = data.add("q", 1, 0.1, 5)
    for tau in (None, 0.05, 0.25, 1.0):
        clf = fit(new_classifier, X, y, prototype_temperature=tau, **PROTOTYPE_ONLY)
        assert clf.predict(q)[0][0] == "C1"


def test_prototype_scores_form_a_distribution_for_any_temperature(new_classifier, data):
    X, y = data.classes(6, noise=0.2, seed=6)
    for tau in (None, 0.01, 0.25, 5.0):
        clf = fit(new_classifier, X, y, prototype_temperature=tau, **PROTOTYPE_ONLY)
        scores = [s for _, s in clf.predict(data.add("p", 2, 0.3, 9))]
        assert sum(scores) == pytest.approx(1.0, abs=1e-5) and all(np.isfinite(scores))


# --- how the blend depends on how much a class has seen ----------------------------------------------

def test_blend_weights_ramp_up_without_a_jump(new_classifier):
    clf = new_classifier()
    weights = []
    for n in range(0, 14):
        clf.training_history = {"c": n}
        weights.append(clf._blend_weights("c"))
    neural = [w[1] for w in weights]
    assert neural[0] == 0.0 and weights[0][0] == 1.0          # unseen class: prototype only
    assert neural == sorted(neural)                           # never decreases
    assert max(b - a for a, b in zip(neural, neural[1:])) <= 0.31 / 10 + 1e-9   # smooth steps
    assert weights[10] == (0.7, 0.3) == weights[13]           # established at the threshold
    assert all(sum(w) == pytest.approx(1.0) for w in weights)


def test_fixed_new_class_weights_still_work(new_classifier):
    clf = new_classifier(new_class_prototype_weight=0.3, new_class_neural_weight=0.7)
    clf.training_history = {"fresh": 2, "established": 50}
    assert clf._blend_weights("fresh") == (0.3, 0.7)
    assert clf._blend_weights("established") == (0.7, 0.3)


def test_ramp_scales_with_the_configured_threshold_and_weights(new_classifier):
    clf = new_classifier(new_class_example_threshold=4, prototype_weight=0.5, neural_weight=0.5)
    clf.training_history = {"c": 2}
    assert clf._blend_weights("c") == pytest.approx((0.75, 0.25))     # halfway from (1, 0) to (.5, .5)
    clf.training_history = {"c": 4}
    assert clf._blend_weights("c") == (0.5, 0.5)


def test_zero_threshold_disables_the_ramp(new_classifier):
    clf = new_classifier(new_class_example_threshold=0)
    clf.training_history = {"c": 0}
    assert clf._blend_weights("c") == (0.7, 0.3)


# --- models saved before 0.3.0 must not change ------------------------------------------------------------

def test_a_pre_0_3_0_save_keeps_its_scoring(new_classifier, base_model, data, tmp_path):
    """Old configs have no prototype_temperature and explicit new-class weights; loading one must
    reproduce its old predictions exactly rather than silently switching to the new scoring."""
    import json
    from adaptive_classifier import AdaptiveClassifier

    X, y = data.classes(6, noise=0.2, seed=7)
    legacy = fit(new_classifier, X, y, prototype_temperature=None,
                 new_class_prototype_weight=0.3, new_class_neural_weight=0.7)
    expected = {t: legacy.predict(t, k=5) for t in X[:6]}
    legacy.save(str(tmp_path), include_onnx=False)

    path = tmp_path / "config.json"
    data_json = json.loads(path.read_text(encoding="utf-8"))
    del data_json["config"]["prototype_temperature"]               # what a 0.2.x save looks like
    for key in ("head_steps", "head_learning_rate"):
        data_json["config"].pop(key, None)
    path.write_text(json.dumps(data_json), encoding="utf-8")

    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")
    assert loaded.config.prototype_temperature is None
    for text, preds in expected.items():
        got = loaded.predict(text, k=5)
        assert [l for l, _ in got] == [l for l, _ in preds]
        assert all(a == pytest.approx(b, abs=1e-4) for (_, a), (_, b) in zip(got, preds))


def test_new_saves_record_the_new_settings(new_classifier, data, tmp_path):
    import json
    X, y = data.classes(4, seed=8)
    fit(new_classifier, X, y).save(str(tmp_path), include_onnx=False)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["config"]
    assert config["prototype_temperature"] == 0.25
    assert config["head_steps"] == 300
    assert config["new_class_prototype_weight"] is None
    assert config["new_class_neural_weight"] is None


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
    clf = new_classifier(head_steps=20)
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
    clf = new_classifier(head_steps=20)
    texts, labels = zip(*train)
    clf.add_examples(list(texts), list(labels))
    score = clf.ood_score(q)
    assert np.isfinite(score) and score >= 0.0
