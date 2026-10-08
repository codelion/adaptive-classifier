"""Confidence calibration and conformal prediction sets.

Two layers: the pure maths in `adaptive_classifier.calibration` is checked
against cases with known answers, and the classifier integration is checked on
synthetic overlapping classes (so the model really is uncertain, which is what
makes calibration matter) with separate train / calibration / test splits.
"""

import json
import logging

import numpy as np
import pytest

from adaptive_classifier import AdaptiveClassifier
from adaptive_classifier import calibration as cal

from .synthetic import Dataset

NOISE = 0.45          # classes overlap, so confidence has to be earned


# --- the maths ------------------------------------------------------------------------------

def test_temperature_one_is_the_identity_and_large_temperature_is_uniform():
    p = np.array([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])
    assert cal.apply_temperature(p, 1.0) == pytest.approx(p)
    assert cal.apply_temperature(p, 1e6) == pytest.approx(np.full_like(p, 1 / 3), abs=1e-4)


def test_temperature_never_changes_the_ranking_and_rows_still_sum_to_one():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5), size=50)
    for t in (0.2, 0.7, 1.5, 6.0):
        out = cal.apply_temperature(p, t)
        assert out.sum(axis=1) == pytest.approx(np.ones(50))
        assert (out.argmax(axis=1) == p.argmax(axis=1)).all()
        assert (np.argsort(-out, axis=1) == np.argsort(-p, axis=1)).all()


def test_low_temperature_sharpens_and_high_temperature_softens():
    p = np.array([[0.6, 0.3, 0.1]])
    assert cal.apply_temperature(p, 0.5).max() > 0.6 > cal.apply_temperature(p, 2.0).max()


def test_zero_probabilities_do_not_produce_nan():
    out = cal.apply_temperature(np.array([[1.0, 0.0, 0.0]]), 0.5)
    assert np.isfinite(out).all() and out.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("true_t", [0.5, 2.0, 4.0])
def test_fit_temperature_recovers_a_known_miscalibration(true_t):
    """Labels drawn from the true probabilities; the model reports them rescaled by true_t."""
    rng = np.random.default_rng(1)
    truth = rng.dirichlet(np.ones(4) * 0.6, size=4000)
    targets = np.array([rng.choice(4, p=row) for row in truth])
    reported = cal.apply_temperature(truth, 1.0 / true_t)         # model is wrong by factor true_t
    assert cal.fit_temperature(reported, targets) == pytest.approx(true_t, rel=0.12)


def test_fit_temperature_leaves_a_calibrated_model_alone():
    rng = np.random.default_rng(2)
    truth = rng.dirichlet(np.ones(3), size=4000)
    targets = np.array([rng.choice(3, p=row) for row in truth])
    assert cal.fit_temperature(truth, targets) == pytest.approx(1.0, abs=0.1)


def test_fit_temperature_validates_input():
    with pytest.raises(ValueError):
        cal.fit_temperature(np.array([[0.5, 0.5]]), [0])           # one example
    with pytest.raises(ValueError):
        cal.fit_temperature(np.array([[0.5, 0.5], [0.5, 0.5]]), [0])   # length mismatch
    with pytest.raises(ValueError):
        cal.fit_temperature(np.array([0.5, 0.5, 0.5]), [0, 1, 0])      # not a matrix


def test_conformal_threshold_picks_the_right_order_statistic():
    scores = np.arange(1, 10) / 10                                 # 0.1 ... 0.9, n = 9
    assert cal.conformal_threshold(scores, 0.1) == pytest.approx(0.9)    # ceil(10 * 0.9) = 9th
    assert cal.conformal_threshold(scores, 0.5) == pytest.approx(0.5)    # ceil(10 * 0.5) = 5th
    assert cal.conformal_threshold(scores, 0.05) == float("inf")        # too few points for 95%
    assert cal.conformal_threshold([], 0.1) == float("inf")


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_conformal_threshold_rejects_an_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        cal.conformal_threshold([0.1, 0.2], alpha)


def test_expected_calibration_error_known_cases():
    perfect = np.array([[1.0, 0.0]] * 10)
    assert cal.expected_calibration_error(perfect, [0] * 10) == pytest.approx(0.0)
    # Always 100% sure, right half the time -> off by 0.5
    sure = np.array([[1.0, 0.0]] * 10)
    assert cal.expected_calibration_error(sure, [0] * 5 + [1] * 5) == pytest.approx(0.5)
    # Says 0.6 and is right 60% of the time -> calibrated
    hedged = np.array([[0.6, 0.4]] * 10)
    assert cal.expected_calibration_error(hedged, [0] * 6 + [1] * 4) == pytest.approx(0.0, abs=1e-9)


# --- the classifier -----------------------------------------------------------------------------

@pytest.fixture
def data(monkeypatch):
    return Dataset().patch(monkeypatch)


@pytest.fixture
def splits(data):
    """Train / calibration / test, all drawn from the same overlapping classes."""
    train = data.classes(30, n_classes=4, noise=NOISE, seed=1)
    calib = data.classes(60, n_classes=4, noise=NOISE, seed=2, prefix="cal")
    test = data.classes(150, n_classes=4, noise=NOISE, seed=3, prefix="te")
    return train, calib, test


@pytest.fixture
def fitted(new_classifier, splits):
    (X, y), _, _ = splits
    clf = new_classifier()
    clf.add_examples(X, y)
    return clf


def test_calibration_makes_confidence_honest_on_unseen_data(fitted, splits):
    _, (Xc, yc), (Xt, yt) = splits
    before = fitted.calibration_report(Xt, yt, calibrated=False)
    fitted.calibrate(Xc, yc)
    after = fitted.calibration_report(Xt, yt)

    assert before["ece"] > 0.12, "the test needs a model that starts out miscalibrated"
    assert after["ece"] < before["ece"] / 2
    assert after["ece"] < 0.08
    assert abs(after["mean_confidence"] - after["accuracy"]) < 0.06


def test_calibration_never_changes_which_class_wins(fitted, splits):
    (Xtr, _), (Xc, yc), (Xt, _) = splits
    top_before = [fitted.predict(t)[0][0] for t in Xt[:60]]
    fitted.calibrate(Xc, yc)
    assert [fitted.predict(t)[0][0] for t in Xt[:60]] == top_before


def test_calibrate_returns_a_summary_that_shows_the_improvement(fitted, splits):
    _, (Xc, yc), _ = splits
    info = fitted.calibrate(Xc, yc)
    assert set(info) >= {"temperature", "n", "ece_before", "ece_after", "nll_before", "nll_after", "accuracy"}
    assert info["n"] == len(Xc)
    assert info["ece_after"] < info["ece_before"] and info["nll_after"] < info["nll_before"]
    assert fitted.calibration["temperature"] == pytest.approx(info["temperature"])


def test_calibrated_predictions_are_still_a_valid_distribution(fitted, splits):
    _, (Xc, yc), (Xt, _) = splits
    fitted.calibrate(Xc, yc)
    for t in Xt[:20]:
        preds = fitted.predict(t, k=10)
        scores = [s for _, s in preds]
        assert sum(scores) == pytest.approx(1.0, abs=1e-5)
        assert scores == sorted(scores, reverse=True)
        assert all(0 <= s <= 1 for s in scores)


def test_k_truncation_does_not_change_the_calibrated_scores(fitted, splits):
    _, (Xc, yc), (Xt, _) = splits
    fitted.calibrate(Xc, yc)
    full = dict(fitted.predict(Xt[0], k=4))
    top2 = fitted.predict(Xt[0], k=2)
    assert all(full[label] == pytest.approx(score, abs=1e-6) for label, score in top2)


def test_abstain_below_applies_to_the_calibrated_confidence(fitted, splits):
    _, (Xc, yc), (Xt, _) = splits
    fitted.calibrate(Xc, yc)
    top = fitted.predict(Xt[0])[0][1]
    assert fitted.predict(Xt[0], abstain_below=top + 0.001) == []
    assert fitted.predict(Xt[0], abstain_below=top - 0.001) != []


def test_predict_batch_matches_predict_after_calibration(fitted, splits):
    _, (Xc, yc), (Xt, _) = splits
    fitted.calibrate(Xc, yc)
    batch = fitted.predict_batch(Xt[:15], k=2)
    for text, many in zip(Xt[:15], batch):
        one = fitted.predict(text, k=2)
        assert [l for l, _ in one] == [l for l, _ in many]
        assert all(a == pytest.approx(b, abs=1e-5) for (_, a), (_, b) in zip(one, many))


def test_predict_batch_scores_do_not_depend_on_k(new_classifier, data):
    """Regression: predict_batch used to look up only k prototypes, so with more than k
    classes its scores differed from predict() for the same text."""
    X, y = data.classes(12, n_classes=5, noise=0.3, seed=4)
    clf = new_classifier()
    clf.add_examples(X, y)
    q = data.add("q", 1, 0.3, 77)
    single = dict(clf.predict(q, k=2))
    batched = dict(clf.predict_batch([q], k=2)[0])
    assert single.keys() == batched.keys()
    for label in single:
        assert batched[label] == pytest.approx(single[label], abs=1e-5)


# --- conformal sets -------------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.2, 0.1])
def test_prediction_sets_cover_the_truth_at_roughly_the_requested_rate(fitted, splits, alpha):
    _, (Xc, yc), (Xt, yt) = splits
    fitted.calibrate(Xc, yc)
    covered = [label in {l for l, _ in fitted.predict_set(t, alpha=alpha)} for t, label in zip(Xt, yt)]
    # Guarantee is 1 - alpha on average; allow for sampling noise on 600 test points.
    assert np.mean(covered) >= 1 - alpha - 0.05


def test_stricter_alpha_gives_bigger_sets(fitted, splits):
    _, (Xc, yc), (Xt, _) = splits
    fitted.calibrate(Xc, yc)
    size = lambda alpha: np.mean([len(fitted.predict_set(t, alpha=alpha)) for t in Xt[:120]])
    assert size(0.02) >= size(0.1) >= size(0.4)
    assert size(0.02) > size(0.4)


def test_sets_are_small_for_easy_inputs_and_larger_for_ambiguous_ones(fitted, splits, data):
    """Queries sitting between two class axes are genuinely ambiguous; those on an axis are not."""
    import torch
    from .synthetic import point

    _, (Xc, yc), _ = splits
    fitted.calibrate(Xc, yc)
    easy = [data.add(f"easy{i}", i % 4, 0.1, 100 + i) for i in range(40)]
    ambiguous = []
    for i in range(40):
        name = f"between{i}"
        data.store[name] = torch.nn.functional.normalize(
            point(i % 3, 0.1, 200 + i) + point(i % 3 + 1, 0.1, 300 + i), dim=0)
        ambiguous.append(name)

    mean_size = lambda texts: np.mean([len(fitted.predict_set(t, alpha=0.1)) for t in texts])
    assert mean_size(easy) < 1.1
    assert mean_size(ambiguous) > mean_size(easy) + 0.3


def test_a_prediction_set_is_never_empty_and_starts_with_the_top_label(fitted, splits):
    _, (Xc, yc), (Xt, _) = splits
    fitted.calibrate(Xc, yc)
    for t in Xt[:60]:
        chosen = fitted.predict_set(t, alpha=0.5)
        assert chosen and chosen[0][0] == fitted.predict(t)[0][0]


def test_too_little_calibration_data_gives_the_whole_label_set(new_classifier, data):
    X, y = data.classes(10, n_classes=3, noise=0.3, seed=5)
    clf = new_classifier()
    clf.add_examples(X, y)
    Xc, yc = data.classes(3, n_classes=3, noise=0.3, seed=6, prefix="cal")     # 9 points
    clf.calibrate(Xc, yc)
    assert len(clf.predict_set(Xc[0], alpha=0.01)) == 3                         # cannot promise 99% with 9 points


# --- validation and state ----------------------------------------------------------------------------

def test_predict_set_requires_calibration(fitted, splits):
    with pytest.raises(ValueError, match="calibrate"):
        fitted.predict_set(splits[2][0][0])


def test_predict_set_validates_alpha_and_text(fitted, splits):
    _, (Xc, yc), (Xt, _) = splits
    fitted.calibrate(Xc, yc)
    with pytest.raises(ValueError, match="alpha"):
        fitted.predict_set(Xt[0], alpha=0)
    with pytest.raises(ValueError, match="Empty"):
        fitted.predict_set("")


@pytest.mark.parametrize("texts, labels, message", [
    ([], [], "Empty"),
    (["a"], ["C0", "C1"], "Mismatched"),
    (["x"], ["not-a-class"], "Unknown"),
    ([None], ["C0"], "strings"),
    (["x"], [3], "strings"),
])
def test_calibrate_rejects_bad_input(fitted, texts, labels, message):
    with pytest.raises(ValueError, match=message):
        fitted.calibrate(texts, labels)
    assert fitted.calibration is None


def test_calibration_needs_two_classes(new_classifier, data):
    X, y = data.classes(10, n_classes=1, noise=0.3, seed=7)
    clf = new_classifier()
    clf.add_examples(X, y)
    with pytest.raises(ValueError, match="2 classes"):
        clf.calibrate(X[:5], y[:5])


def test_a_small_calibration_set_warns(fitted, splits, caplog):
    _, (Xc, yc), _ = splits
    with caplog.at_level(logging.WARNING):
        fitted.calibrate(Xc[:10], yc[:10])
    assert "noisy" in caplog.text


def test_uncalibrated_predictions_are_unchanged(new_classifier, splits):
    (X, y), _, (Xt, _) = splits
    a, b = new_classifier(), new_classifier()
    a.add_examples(X, y)
    b.add_examples(X, y)
    assert a.calibration is None
    assert a.predict(Xt[0]) == b.predict(Xt[0])


def test_adding_a_new_class_discards_the_calibration(fitted, splits, data, caplog):
    _, (Xc, yc), _ = splits
    fitted.calibrate(Xc, yc)
    new_text = data.add("novel", 7, 0.1, 5)
    with caplog.at_level(logging.WARNING):
        fitted.add_examples([new_text], ["C-new"])
    assert fitted.calibration is None and "Discarding" in caplog.text


def test_adding_examples_to_known_classes_keeps_the_calibration(fitted, splits, data):
    _, (Xc, yc), _ = splits
    fitted.calibrate(Xc, yc)
    fitted.add_examples([data.add("more", 0, 0.3, 11)], ["C0"])
    assert fitted.calibration is not None


def test_forgetting_a_class_discards_the_calibration(fitted, splits):
    _, (Xc, yc), _ = splits
    fitted.calibrate(Xc, yc)
    fitted.forget("C3")
    assert fitted.calibration is None
    with pytest.raises(ValueError, match="calibrate"):
        fitted.predict_set("tr0_0")


def test_calibration_survives_save_and_reload(fitted, splits, tmp_path, data):
    _, (Xc, yc), (Xt, yt) = splits
    fitted.calibrate(Xc, yc)
    fitted.save(str(tmp_path), include_onnx=False)
    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")

    assert loaded.calibration["temperature"] == pytest.approx(fitted.calibration["temperature"])
    assert loaded.calibration["scores"] == pytest.approx(fitted.calibration["scores"])
    for t in Xt[:10]:
        a, b = fitted.predict(t, k=4), loaded.predict(t, k=4)
        assert [l for l, _ in a] == [l for l, _ in b]
        assert all(x == pytest.approx(y, abs=1e-4) for (_, x), (_, y) in zip(a, b))
        assert {l for l, _ in fitted.predict_set(t)} == {l for l, _ in loaded.predict_set(t)}


def test_saves_without_calibration_load_as_uncalibrated(fitted, tmp_path):
    fitted.save(str(tmp_path), include_onnx=False)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.pop("calibration", None)                                   # an older save
    path.write_text(json.dumps(config), encoding="utf-8")
    assert AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu").calibration is None


def test_sklearn_wrapper_uses_calibrated_probabilities(base_model, splits, data):
    from adaptive_classifier import SklearnAdaptiveClassifier
    (X, y), (Xc, yc), (Xt, yt) = splits
    est = SklearnAdaptiveClassifier(base_model, use_onnx=False, device="cpu").fit(X, y)
    before = est.predict_proba(Xt[:40]).max(axis=1).mean()
    est.classifier_.calibrate(Xc, yc)
    after = est.predict_proba(Xt[:40]).max(axis=1).mean()
    assert after != pytest.approx(before, abs=1e-3)
    assert est.predict_proba(Xt[:5]).sum(axis=1) == pytest.approx(np.ones(5), abs=1e-5)
