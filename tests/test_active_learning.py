"""suggest_labels (what to label next) and drift_report (has the data moved?).

What is checked is the contract: the ranking is the one each strategy defines,
it is deterministic, and it does the job it exists for (surface the confusing
and the unfamiliar). It is deliberately NOT asserted that uncertainty sampling
beats random labelling on every dataset: on heavily overlapping classes, where
the confusion is irreducible noise, a synthetic experiment showed it does not.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from .synthetic import Dataset, point

STRATEGIES = ["margin", "entropy", "least_confidence", "ood"]


@pytest.fixture
def data(monkeypatch):
    return Dataset().patch(monkeypatch)


@pytest.fixture
def clf(new_classifier, data):
    X, y = data.classes(20, n_classes=3, noise=0.15, seed=1)
    c = new_classifier(head_steps=60)
    c.add_examples(X, y)
    return c


def add_between(data, name, a, b, noise=0.05, seed=0):
    """A point halfway between the axes of classes a and b."""
    data.store[name] = F.normalize(point(a, noise, seed) + point(b, noise, seed + 1), dim=0)
    return name


# --- ranking ---------------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["margin", "entropy", "least_confidence"])
def test_boundary_texts_outrank_clear_ones(clf, data, strategy):
    clear = [data.add(f"clear{i}", i % 3, 0.05, 500 + i) for i in range(6)]
    boundary = [add_between(data, f"edge{i}", 0, 1, seed=600 + 2 * i) for i in range(3)]
    pool = clear + boundary
    ranked = [i for i, _ in clf.suggest_labels(pool, n=3, strategy=strategy)]
    assert set(ranked) == {6, 7, 8}, f"{strategy} picked {ranked}"


def test_ood_strategy_surfaces_a_class_the_model_has_never_seen(clf, data):
    known = [data.add(f"k{i}", i % 3, 0.15, 700 + i) for i in range(18)]
    unseen = [data.add(f"u{i}", 9, 0.15, 800 + i) for i in range(2)]        # axis 9: no such class
    pool = known + unseen
    picks = {i for i, _ in clf.suggest_labels(pool, n=2, strategy="ood")}
    assert picks == {18, 19}


def test_most_of_a_hidden_class_is_found_within_a_small_budget(clf, data):
    """10% of the pool belongs to a class the model has never seen; random labelling would
    find ~1 in 10, the ood strategy should find almost all of them first."""
    known = [data.add(f"k{i}", i % 3, 0.15, 900 + i) for i in range(45)]
    hidden = [data.add(f"h{i}", 7, 0.15, 950 + i) for i in range(5)]
    picks = [i for i, _ in clf.suggest_labels(known + hidden, n=5, strategy="ood")]
    assert sum(i >= 45 for i in picks) >= 4


def reference_scores(clf, texts, strategy):
    """Recompute each strategy's score independently from predict()."""
    out = []
    for text in texts:
        p = sorted((s for _, s in clf.predict(text, k=len(clf.id_to_label))), reverse=True)
        if strategy == "least_confidence":
            out.append(1 - p[0])
        elif strategy == "margin":
            out.append(1 - (p[0] - p[1]))
        else:
            q = np.array([x for x in p if x > 0])
            out.append(float(-(q * np.log(q)).sum() / np.log(len(p))))
    return out


@pytest.mark.parametrize("strategy", ["margin", "entropy", "least_confidence"])
def test_scores_are_what_the_strategy_defines(clf, data, strategy):
    pool = [data.add(f"s{i}", i % 3, 0.4, 1000 + i) for i in range(12)]
    got = dict(clf.suggest_labels(pool, n=12, strategy=strategy))
    expected = reference_scores(clf, pool, strategy)
    for i, score in got.items():
        assert score == pytest.approx(expected[i], abs=1e-4)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_results_are_sorted_unique_and_in_range(clf, data, strategy):
    pool = [data.add(f"r{i}", i % 3, 0.3, 1100 + i) for i in range(15)]
    out = clf.suggest_labels(pool, n=8, strategy=strategy)
    indices = [i for i, _ in out]
    scores = [s for _, s in out]
    assert len(out) == 8 and len(set(indices)) == 8
    assert all(0 <= i < 15 for i in indices)
    assert scores == sorted(scores, reverse=True)
    assert all(np.isfinite(scores))


def test_n_larger_than_the_pool_returns_everything(clf, data):
    pool = [data.add(f"p{i}", i % 3, 0.3, 1200 + i) for i in range(4)]
    assert sorted(i for i, _ in clf.suggest_labels(pool, n=50)) == [0, 1, 2, 3]


def test_suggestions_are_deterministic(clf, data):
    pool = [data.add(f"d{i}", i % 3, 0.35, 1300 + i) for i in range(10)]
    assert clf.suggest_labels(pool, n=5) == clf.suggest_labels(pool, n=5)


def test_calibration_changes_scores_but_not_the_most_ambiguous_pick(clf, data):
    pool = [data.add(f"c{i}", i % 3, 0.05, 1400 + i) for i in range(6)] + [add_between(data, "mid", 1, 2)]
    before = clf.suggest_labels(pool, n=1)[0]
    Xc, yc = data.classes(30, n_classes=3, noise=0.45, seed=9, prefix="cal")
    clf.calibrate(Xc, yc)
    after = clf.suggest_labels(pool, n=1)[0]
    assert before[0] == after[0] == 6


# --- diversity ----------------------------------------------------------------------------------

def test_diverse_selection_avoids_near_duplicates(clf, data):
    # Candidates between several class pairs; the most ambiguous one is then duplicated five times.
    candidates = [add_between(data, f"cand{i}", i % 3, (i + 1) % 3, noise=0.15, seed=1500 + 2 * i)
                  for i in range(8)]
    top = clf.suggest_labels(candidates, n=1, strategy="margin")[0][0]
    for i in range(5):
        data.store[f"twin{i}"] = data.store[candidates[top]]
    twins = [f"twin{i}" for i in range(5)]
    others = [c for j, c in enumerate(candidates) if j != top]
    pool = twins + others

    plain = [i for i, _ in clf.suggest_labels(pool, n=3, strategy="margin")]
    spread = [i for i, _ in clf.suggest_labels(pool, n=3, strategy="margin", diverse=True)]
    assert sum(i < 5 for i in plain) == 3, f"the twins should fill the plain top 3: {plain}"
    assert sum(i < 5 for i in spread) == 1, f"diverse selection should take one twin: {spread}"
    assert len(set(spread)) == 3


def test_diverse_selection_with_only_duplicates_still_returns_distinct_indices(clf, data):
    base = add_between(data, "dup_base", 0, 1)
    pool = []
    for i in range(4):
        data.store[f"dup{i}"] = data.store[base]
        pool.append(f"dup{i}")
    picks = [i for i, _ in clf.suggest_labels(pool, n=3, strategy="margin", diverse=True)]
    assert len(picks) == 3 and len(set(picks)) == 3


@pytest.mark.parametrize("diverse", [False, True])
def test_n_larger_than_the_pool_returns_each_text_once(clf, data, diverse):
    pool = [data.add(f"big{i}", i % 3, 0.3, 1700 + i) for i in range(4)]
    picks = [i for i, _ in clf.suggest_labels(pool, n=10, diverse=diverse)]
    assert sorted(picks) == [0, 1, 2, 3]


# --- validation -----------------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({"texts": []}, "Empty"),
    ({"texts": ["tr0_0"], "n": 0}, "positive integer"),
    ({"texts": ["tr0_0"], "n": -3}, "positive integer"),
    ({"texts": ["tr0_0"], "n": 1.5}, "positive integer"),
    ({"texts": ["tr0_0"], "strategy": "vibes"}, "strategy"),
    ({"texts": ["tr0_0", ""]}, "non-empty"),
    ({"texts": ["tr0_0", None]}, "non-empty"),
])
def test_bad_arguments_are_rejected(clf, kwargs, message):
    with pytest.raises(ValueError, match=message):
        clf.suggest_labels(**kwargs)


def test_needs_a_trained_classifier(new_classifier):
    with pytest.raises(ValueError, match="trained"):
        new_classifier().suggest_labels(["a"])
    with pytest.raises(ValueError, match="trained"):
        new_classifier().drift_report(["a"])


# --- drift ------------------------------------------------------------------------------------------

def test_data_like_the_training_set_is_not_drift(clf, data):
    fresh, _ = data.classes(40, n_classes=3, noise=0.15, seed=21, prefix="live")
    report = clf.drift_report(fresh)
    assert report["drifted"] is False and report["p_value"] > 0.01
    assert report["n"] == 120 and report["ood_rate"] < 0.1


def test_a_shifted_stream_is_flagged(clf, data):
    live = [data.add(f"ok{i}", i % 3, 0.15, 2000 + i) for i in range(30)]
    live += [data.add(f"new{i}", 9, 0.15, 2100 + i) for i in range(30)]          # half the traffic is unfamiliar
    report = clf.drift_report(live)
    assert report["drifted"] is True and report["p_value"] < 1e-6
    assert report["ood_rate"] > 0.4


def test_mean_score_grows_with_the_amount_of_shift(clf, data):
    def stream(shift_fraction, tag):
        total = 60
        moved = int(total * shift_fraction)
        return ([data.add(f"{tag}ok{i}", i % 3, 0.15, 2200 + i) for i in range(total - moved)]
                + [data.add(f"{tag}mv{i}", 9, 0.15, 2300 + i) for i in range(moved)])

    scores = [clf.drift_report(stream(f, f"f{int(f * 10)}"))["mean_ood_score"] for f in (0.0, 0.25, 0.5, 1.0)]
    assert scores == sorted(scores) and scores[-1] > 1.5 * scores[0]


def test_a_tiny_window_cannot_prove_drift(clf, data):
    report = clf.drift_report([data.add("lone", 9, 0.15, 2400)])
    assert report["ood_rate"] == 1.0 and report["drifted"] is False       # one sample: p = 0.05, above alpha


def test_expected_rate_and_alpha_control_the_verdict(clf, data):
    live = [data.add(f"x{i}", i % 3, 0.15, 2500 + i) for i in range(45)]
    live += [data.add(f"y{i}", 9, 0.15, 2600 + i) for i in range(5)]              # 10% unfamiliar
    assert clf.drift_report(live, expected_ood_rate=0.01)["drifted"] is True
    assert clf.drift_report(live, expected_ood_rate=0.3)["drifted"] is False
    assert clf.drift_report(live, expected_ood_rate=0.01, alpha=1e-9)["drifted"] is False


def test_threshold_override_changes_what_counts_as_unfamiliar(clf, data):
    live = [data.add(f"t{i}", i % 3, 0.3, 2700 + i) for i in range(30)]
    strict = clf.drift_report(live, threshold=0.2)
    lenient = clf.drift_report(live, threshold=50.0)
    assert strict["ood_rate"] > lenient["ood_rate"] == 0.0


@pytest.mark.parametrize("kwargs, message", [
    ({"texts": []}, "Empty"),
    ({"texts": ["tr0_0"], "expected_ood_rate": 1.0}, "expected_ood_rate"),
    ({"texts": ["tr0_0"], "expected_ood_rate": -0.1}, "expected_ood_rate"),
    ({"texts": ["tr0_0"], "alpha": 0}, "alpha"),
    ({"texts": ["tr0_0", ""]}, "non-empty"),
])
def test_drift_report_rejects_bad_arguments(clf, kwargs, message):
    with pytest.raises(ValueError, match=message):
        clf.drift_report(**kwargs)
