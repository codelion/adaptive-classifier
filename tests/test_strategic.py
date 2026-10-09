"""Strategic (anti-gaming) mode: set-up, prediction invariants, determinism and persistence.

Uses hand-made embeddings (synthetic.py, 32 dimensions to match the tiny test encoder).
"""

import logging
import threading

import pytest
import torch

from adaptive_classifier import AdaptiveClassifier
from adaptive_classifier.strategic import (
    CostFunctionFactory, LinearCostFunction, SeparableCostFunction, StrategicEvaluator,
)
from .synthetic import DIM, Dataset


def _config(cost=0.3, **extra):
    return {
        "enable_strategic_mode": True,
        "cost_function_type": "linear",
        "cost_coefficients": [cost] * DIM,
        "strategic_training_frequency": 1000,        # keep fitting fast; training is tested separately
        **extra,
    }


@pytest.fixture
def data(monkeypatch, base_model):
    return Dataset().patch(monkeypatch)


@pytest.fixture
def trained(data, base_model):
    def build(config=None, n_per=10, n_classes=3, noise=0.1):
        clf = AdaptiveClassifier(base_model, config=config or _config(), device="cpu", use_onnx=False)
        texts, labels = data.classes(n_per, n_classes, noise=noise, seed=1)
        clf.add_examples(texts, labels)
        return clf, texts, labels
    return build


# --- set-up -------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["linear", "separable"])
def test_list_coefficients_enable_strategic_mode(base_model, kind):
    clf = AdaptiveClassifier(base_model, config=_config(cost_function_type=kind), device="cpu", use_onnx=False)
    assert clf.strategic_mode


def test_strategic_mode_is_off_by_default(base_model):
    assert not AdaptiveClassifier(base_model, device="cpu", use_onnx=False).strategic_mode


@pytest.mark.parametrize("coefficients, why", [
    ({"sentiment_words": 0.5, "length_change": 0.1}, "dict"),     # the form the README once showed
    ([0.3] * 5, "length"),                                        # not one per embedding dimension
    ([], "empty"),
])
def test_unusable_coefficients_leave_strategic_mode_off_and_say_why(base_model, caplog, coefficients, why):
    config = _config()
    config["cost_coefficients"] = coefficients
    with caplog.at_level(logging.WARNING):
        clf = AdaptiveClassifier(base_model, config=config, device="cpu", use_onnx=False)
    assert not clf.strategic_mode
    assert caplog.records, "no explanation was logged"
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "cost_coefficients" in message


def test_unknown_cost_function_type_is_reported(base_model, caplog):
    with caplog.at_level(logging.WARNING):
        clf = AdaptiveClassifier(base_model, config=_config(cost_function_type="quadratic"),
                                 device="cpu", use_onnx=False)
    assert not clf.strategic_mode
    assert "quadratic" in " ".join(r.getMessage() for r in caplog.records)


def test_factory_rejects_unknown_types_and_dicts_without_names():
    with pytest.raises(ValueError):
        CostFunctionFactory.create_cost_function("nope", [0.1] * 4)
    with pytest.raises(ValueError):
        LinearCostFunction({"a": 1.0})


# --- cost functions -----------------------------------------------------------------

def test_linear_cost_is_zero_for_moves_that_do_not_raise_the_score():
    cost = LinearCostFunction(torch.tensor([1.0, 0.0, 0.0]))
    x = torch.tensor([0.5, 0.0, 0.0])
    assert cost.compute_cost(x, torch.tensor([0.2, 0.0, 0.0])).item() == 0.0     # cheaper direction
    assert cost.compute_cost(x, torch.tensor([0.9, 0.0, 0.0])).item() == pytest.approx(0.4)
    assert cost.compute_cost(x, x).item() == 0.0


def test_best_response_can_use_any_dimension_not_just_the_first_few():
    """Moving along the only free dimension (the last of 32) is the cheapest way to raise the score."""
    alpha = torch.ones(DIM)
    alpha[-1] = 0.0                                               # free to change
    cost = SeparableCostFunction(alpha, alpha)
    x = torch.zeros(DIM)

    def score(batch):                                              # the classifier likes a high last feature
        return torch.sigmoid(batch[:, -1:] * 4)

    response = cost.compute_best_response(x, score)
    assert response[-1] > 0, "best response never explored the last dimension"


def test_best_response_is_deterministic():
    cost = LinearCostFunction(torch.full((DIM,), 0.2))
    x = torch.linspace(-1, 1, DIM)

    def score(batch):
        return torch.softmax(batch[:, :3], dim=-1)

    assert torch.equal(cost.compute_best_response(x, score), cost.compute_best_response(x, score))


def test_a_prohibitive_cost_stops_the_moves_that_would_raise_the_score():
    """Costs only apply to moves that raise <alpha, x>; with a huge alpha no such move pays off."""
    cost = LinearCostFunction(torch.full((DIM,), 1000.0))
    x = torch.zeros(DIM)

    def likes_a_high_sum(batch):                    # the classifier is happier the larger the sum
        return torch.sigmoid(batch.sum(dim=-1, keepdim=True))

    assert cost.compute_best_response(x, likes_a_high_sum).sum().item() <= 1e-6


def test_a_free_direction_is_taken():
    cost = LinearCostFunction(torch.zeros(DIM))
    x = torch.zeros(DIM)
    response = cost.compute_best_response(x, lambda b: torch.sigmoid(b.sum(dim=-1, keepdim=True)))
    assert response.sum().item() > 0


# --- predictions ----------------------------------------------------------------------

@pytest.mark.parametrize("method", ["predict", "predict_strategic", "predict_robust"])
def test_predictions_are_ranked_probability_distributions(trained, method):
    clf, texts, _ = trained()
    result = getattr(clf, method)(texts[0], k=3)
    scores = [s for _, s in result]
    assert {label for label, _ in result} <= set(clf.label_to_id)
    assert scores == sorted(scores, reverse=True)
    assert sum(scores) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.parametrize("method", ["predict", "predict_strategic", "predict_robust"])
def test_scores_do_not_depend_on_how_many_results_are_requested(trained, method):
    clf, texts, _ = trained()
    full = dict(getattr(clf, method)(texts[0], k=3))
    top = getattr(clf, method)(texts[0], k=1)
    assert top[0][0] == max(full, key=full.get)
    assert top[0][1] == pytest.approx(full[top[0][0]], abs=1e-5)


@pytest.mark.parametrize("method", ["predict", "predict_strategic", "predict_robust"])
def test_k_is_respected(trained, method):
    clf, texts, _ = trained()
    assert len(getattr(clf, method)(texts[0], k=2)) == 2
    assert len(getattr(clf, method)(texts[0], k=100)) == 3


def test_predictions_are_repeatable(trained):
    clf, texts, _ = trained()
    assert clf.predict(texts[0], k=3) == clf.predict(texts[0], k=3)
    assert clf.predict_strategic(texts[0], k=3) == clf.predict_strategic(texts[0], k=3)


def test_separable_data_is_still_classified_correctly(trained, data):
    clf, _, _ = trained()
    texts, labels = data.classes(8, 3, noise=0.1, seed=9, prefix="v")
    hits = sum(clf.predict(t)[0][0] == l for t, l in zip(texts, labels))
    assert hits / len(texts) >= 0.9


def test_with_strategic_mode_off_every_method_matches_regular_prediction(data, base_model):
    clf = AdaptiveClassifier(base_model, device="cpu", use_onnx=False)
    texts, labels = data.classes(8, 3, seed=1)
    clf.add_examples(texts, labels)
    regular = clf.predict(texts[0], k=3)
    assert clf.predict_strategic(texts[0], k=3) == regular
    assert clf.predict_robust(texts[0], k=3) == regular


def test_robust_prediction_follows_the_prototypes_when_the_head_has_no_weight(trained):
    clf, texts, _ = trained(_config(strategic_robust_proto_weight=1.0, strategic_robust_head_weight=0.0))
    embedding = clf._get_embeddings([texts[0]])[0]
    proto = [label for label, _ in clf.memory.get_nearest_prototypes(embedding, k=3)]
    assert [label for label, _ in clf.predict_robust(texts[0], k=3)] == proto


def test_dual_prediction_blends_the_regular_and_strategic_ones(trained):
    config = _config(strategic_blend_regular_weight=1.0, strategic_blend_strategic_weight=0.0)
    clf, texts, _ = trained(config)
    dual = dict(clf.predict(texts[0], k=3))
    regular = dict(clf._predict_regular(texts[0], 3))
    assert dual.keys() == regular.keys()
    assert all(dual[l] == pytest.approx(regular[l], abs=1e-5) for l in dual)


def test_bad_input_is_rejected_like_regular_prediction(trained):
    clf, texts, _ = trained()
    with pytest.raises(ValueError):
        clf.predict("")
    with pytest.raises(ValueError):
        clf.predict(texts[0], k=-1)


# --- training with the strategic loss -------------------------------------------------------

def test_strategic_training_steps_run_and_keep_the_classifier_accurate(data, base_model):
    config = _config(strategic_training_frequency=1)              # every update trains strategically
    clf = AdaptiveClassifier(base_model, config=config, device="cpu", use_onnx=False)
    texts, labels = data.classes(6, 3, noise=0.1, seed=1)
    clf.add_examples(texts, labels)
    clf.add_examples(*data.classes(6, 3, noise=0.1, seed=2, prefix="u"))
    test_x, test_y = data.classes(8, 3, noise=0.1, seed=9, prefix="v")
    hits = sum(clf.predict(t)[0][0] == l for t, l in zip(test_x, test_y))
    assert hits / len(test_x) >= 0.9


def test_a_new_class_can_be_added_in_strategic_mode(trained, data):
    clf, _, _ = trained(n_classes=2)
    new_x, new_y = data.classes(10, 3, seed=5, prefix="n")
    clf.add_examples([t for t, l in zip(new_x, new_y) if l == "C2"], ["C2"] * 10)
    assert "C2" in clf.label_to_id
    assert clf.predict_strategic(new_x[-1], k=3)


# --- robustness evaluation --------------------------------------------------------------------

def test_evaluate_strategic_robustness_reports_sane_numbers(trained, data):
    clf, _, _ = trained()
    texts, labels = data.classes(6, 3, seed=9, prefix="e")
    report = clf.evaluate_strategic_robustness(texts, labels)
    for level in (0.0, 0.5, 1.0):
        assert 0.0 <= report[f"accuracy_gaming_{level}"] <= 1.0
    assert report["accuracy_gaming_0.0"] >= 0.9
    assert report["robustness_score"] == pytest.approx(report["accuracy_gaming_0.0"] - report["accuracy_gaming_1.0"])


def test_evaluate_strategic_robustness_is_repeatable(trained, data):
    clf, _, _ = trained()
    texts, labels = data.classes(6, 3, seed=9, prefix="e")
    assert clf.evaluate_strategic_robustness(texts, labels) == clf.evaluate_strategic_robustness(texts, labels)


def test_evaluate_strategic_robustness_needs_strategic_mode(data, base_model):
    clf = AdaptiveClassifier(base_model, device="cpu", use_onnx=False)
    with pytest.raises(ValueError):
        clf.evaluate_strategic_robustness(["a"], ["b"])


def test_evaluator_survives_a_classifier_that_gets_everything_wrong():
    """Accuracy of zero at no gaming used to divide by zero."""
    evaluator = StrategicEvaluator(LinearCostFunction(torch.full((4,), 0.1)))
    always_class_1 = lambda x: torch.tensor([[0.0, 1.0]]).expand(len(x), 2)
    report = evaluator.evaluate_robustness(always_class_1, torch.zeros(5, 4), torch.zeros(5, dtype=torch.long))
    assert report["accuracy_gaming_0.0"] == 0.0
    assert report["relative_robustness"] != report["relative_robustness"]    # nan, not an exception


def test_evaluate_strategic_robustness_rejects_unknown_labels(trained, data):
    clf, _, _ = trained()
    with pytest.raises(ValueError):
        clf.evaluate_strategic_robustness(["C0_0"], ["not-a-class"])


# --- persistence and concurrency ----------------------------------------------------------------

def test_save_and_load_keep_strategic_mode_and_predictions(trained, tmp_path):
    clf, texts, _ = trained(_config(strategic_robust_proto_weight=0.9, strategic_robust_head_weight=0.1))
    clf.save(str(tmp_path), include_onnx=False)
    loaded = AdaptiveClassifier.load(str(tmp_path), device="cpu", use_onnx=False)

    assert loaded.strategic_mode
    assert loaded.config.strategic_robust_proto_weight == 0.9
    for method in ("predict", "predict_strategic", "predict_robust"):
        a, b = dict(getattr(clf, method)(texts[0], k=3)), dict(getattr(loaded, method)(texts[0], k=3))
        assert a.keys() == b.keys()
        assert all(a[l] == pytest.approx(b[l], abs=1e-4) for l in a)


def test_strategic_predictions_survive_concurrent_updates(trained, data):
    clf, texts, _ = trained()
    extra_x, extra_y = data.classes(6, 4, seed=3, prefix="x")
    extra = [(t, l) for t, l in zip(extra_x, extra_y) if l == "C3"]
    stop, errors = threading.Event(), []

    def reader():
        while not stop.is_set():
            try:
                clf.predict(texts[0], k=3)
                clf.predict_robust(texts[1], k=3)
            except Exception as error:      # noqa: BLE001
                errors.append(error)
                return

    threads = [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    try:
        for _ in range(3):
            clf.add_examples([t for t, _ in extra], [l for _, l in extra])
            clf.forget("C3")
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert not errors, errors[0]
