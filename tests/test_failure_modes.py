"""What happens when things go wrong.

Positive-path tests show the library works when used as intended. These check
the other half: bad input is rejected with a clear error (not silently
accepted, not a cryptic crash), damaged files fail loudly, and a Hub outage
degrades to a working classifier instead of a broken one.
"""

import json
import logging
import random

import numpy as np
import pytest
import torch

from adaptive_classifier import AdaptiveClassifier

TEXTS = ["great love", "good fast", "terrible hate", "awful bad"] * 3
LABELS = ["p", "p", "n", "n"] * 3


@pytest.fixture
def trained(new_classifier):
    clf = new_classifier()
    clf.add_examples(TEXTS, LABELS)
    return clf


# --- bad prediction input -------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", None])
def test_empty_prediction_text_is_rejected(trained, bad):
    with pytest.raises(ValueError, match="Empty"):
        trained.predict(bad)


@pytest.mark.parametrize("bad", [123, 4.5, ["list"], {"a": 1}, b"bytes"])
def test_non_string_prediction_text_is_rejected(trained, bad):
    with pytest.raises((ValueError, TypeError)):
        trained.predict(bad)


@pytest.mark.parametrize("k", [-1, -100, 1.5, "3", None])
def test_invalid_k_is_rejected(trained, k):
    with pytest.raises(ValueError, match="k must be"):
        trained.predict("great", k=k)
    with pytest.raises(ValueError, match="k must be"):
        trained.predict_batch(["great"], k=k)


def test_k_zero_returns_nothing_and_huge_k_returns_every_class(trained):
    assert trained.predict("great", k=0) == []
    assert len(trained.predict("great", k=10_000)) == 2


def test_numpy_integers_are_accepted_for_k(trained):
    assert len(trained.predict("great", k=np.int64(1))) == 1


def test_empty_batch_is_rejected(trained):
    with pytest.raises(ValueError, match="Empty"):
        trained.predict_batch([])


def test_untrained_classifier_predicts_nothing_instead_of_crashing(new_classifier):
    clf = new_classifier()
    assert clf.predict("great") == []
    assert clf.predict_batch(["great", "bad"]) == [[], []]


# --- hostile but legal text -----------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "   ",
    "\n\t",
    "😀🎉 emoji only",
    "مرحبا بالعالم",
    "你好，世界",
    "mixed ‮ right-to-left ​ zero-width",
    "null\x00byte",
    "a" * 50_000,
    "great " * 20_000,
    "<script>alert(1)</script> '; DROP TABLE x;--",
])
def test_unusual_text_gives_a_finite_valid_prediction(trained, text):
    preds = trained.predict(text)
    assert preds and {l for l, _ in preds} <= {"p", "n"}
    assert all(np.isfinite(s) and 0 <= s <= 1 for _, s in preds)
    assert sum(s for _, s in preds) == pytest.approx(1.0, abs=1e-5)


def test_text_beyond_max_length_matches_its_truncation(new_classifier):
    clf = new_classifier(max_length=16, pooling="mean")
    clf.add_examples(TEXTS, LABELS)
    long = "great love " * 200
    head = clf.tokenizer.decode(clf.tokenizer(long, max_length=16, truncation=True)["input_ids"],
                                skip_special_tokens=True)
    a, b = dict(clf.predict(long)), dict(clf.predict(head))
    assert all(a[l] == pytest.approx(b[l], abs=1e-4) for l in a)


# --- bad training input ----------------------------------------------------------------------

@pytest.mark.parametrize("texts, labels, message", [
    ([], [], "Empty"),
    (["a", "b"], ["x"], "Mismatched"),
    ([None], ["x"], "strings"),
    ([5], ["x"], "strings"),
    (["a"], [None], "strings"),
    (["a", "b"], [1, 2], "strings"),
    (["a", "b"], ["x", 2], "strings"),
    (["a"], [("x",)], "strings"),
])
def test_bad_training_input_is_rejected_before_anything_changes(new_classifier, texts, labels, message):
    clf = new_classifier()
    with pytest.raises(ValueError, match=message):
        clf.add_examples(texts, labels)
    assert clf.label_to_id == {} and clf.memory.prototypes == {}


def test_rejected_input_leaves_a_trained_classifier_untouched(trained):
    before = (dict(trained.label_to_id), dict(trained.training_history),
              {l: len(v) for l, v in trained.memory.examples.items()})
    with pytest.raises(ValueError):
        trained.add_examples(["fine text", "another"], ["p", None])
    after = (dict(trained.label_to_id), dict(trained.training_history),
             {l: len(v) for l, v in trained.memory.examples.items()})
    assert before == after


def test_integer_labels_cannot_corrupt_a_saved_model(new_classifier, tmp_path):
    """Integer labels used to be accepted, then reloaded as a duplicated set of
    string and integer classes (JSON turns the keys into strings)."""
    clf = new_classifier()
    with pytest.raises(ValueError, match="strings"):
        clf.add_examples(TEXTS, [1, 1, 2, 2] * 3)

    clf.add_examples(TEXTS, [str(l) for l in [1, 1, 2, 2] * 3])
    clf.save(str(tmp_path), include_onnx=False)
    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")
    assert {l for l, _ in loaded.predict("great", k=10)} == {"1", "2"}


def test_empty_string_text_is_allowed_and_embeds(new_classifier):
    clf = new_classifier()
    clf.add_examples(["", "good"], ["blank", "ok"])
    assert {l for l, _ in clf.predict("good", k=5)} == {"blank", "ok"}


def test_identical_text_with_conflicting_labels_stays_consistent(new_classifier):
    clf = new_classifier()
    clf.add_examples(["great"] * 4, ["p", "n", "p", "n"])
    preds = clf.predict("great")
    assert {l for l, _ in preds} == {"p", "n"}
    assert abs(preds[0][1] - preds[1][1]) < 0.3


def test_single_class_classifier_is_certain(new_classifier):
    clf = new_classifier()
    clf.add_examples(["great", "good"], ["only", "only"])
    assert clf.predict("anything at all") == [("only", pytest.approx(1.0))]


def test_memory_stays_bounded(new_classifier):
    clf = new_classifier(max_examples_per_class=5)
    clf.add_examples(["great"] * 40 + ["bad"] * 40, ["p"] * 40 + ["n"] * 40)
    assert all(len(v) <= 5 for v in clf.memory.examples.values())
    assert {l for l, _ in clf.predict("great")} == {"p", "n"}


def test_forget_and_remove_on_unknown_things_fail_clearly(trained, new_classifier):
    with pytest.raises(ValueError, match="Unknown"):
        trained.forget("never-existed")
    with pytest.raises(ValueError, match="Unknown"):
        trained.remove_examples(["x"], label="never-existed")
    assert new_classifier().remove_examples(["x"]) == 0     # nothing to remove is not an error


# --- damaged or missing files ------------------------------------------------------------------

@pytest.fixture
def saved(trained, tmp_path):
    trained.save(str(tmp_path), include_onnx=False)
    return tmp_path


def load(path):
    return AdaptiveClassifier.load(str(path), use_onnx=False, device="cpu")


def test_a_clean_save_loads_and_predicts_the_same(trained, saved):
    assert load(saved).predict("great") == pytest.approx(trained.predict("great"), abs=1e-5) \
        or [l for l, _ in load(saved).predict("great")] == [l for l, _ in trained.predict("great")]


def test_directory_without_config_says_so(saved):
    (saved / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="config.json is missing"):
        load(saved)


def test_empty_directory_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json is missing"):
        load(tmp_path)


def test_nonexistent_path_fails_loudly():
    with pytest.raises(ValueError, match="Error loading model"):
        load("/definitely/not/a/real/dir")


@pytest.mark.parametrize("name", ["examples.json", "model.safetensors"])
def test_missing_state_file_is_an_error_not_a_silent_empty_model(saved, name):
    (saved / name).unlink()
    with pytest.raises(FileNotFoundError):
        load(saved)


def test_truncated_weights_fail_loudly(saved):
    data = (saved / "model.safetensors").read_bytes()
    (saved / "model.safetensors").write_bytes(data[: len(data) // 2])
    with pytest.raises(Exception, match="(?i)header|metadata|deserial|covered|safetensor"):
        load(saved)


def test_corrupt_config_json_fails_loudly(saved):
    (saved / "config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load(saved)


def test_config_that_disagrees_with_the_weights_fails_loudly(saved):
    cfg = json.loads((saved / "config.json").read_text(encoding="utf-8"))
    cfg["label_to_id"]["ghost"] = 2
    cfg["id_to_label"]["2"] = "ghost"
    (saved / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(RuntimeError, match="size mismatch|loading state_dict"):
        load(saved)


# --- the Hub misbehaving -------------------------------------------------------------------------

@pytest.mark.parametrize("error", [
    OSError("429 Client Error: Too Many Requests"),
    ConnectionError("Tunnel connection failed: 403 Forbidden"),
    RuntimeError("export failed"),
])
def test_onnx_failure_falls_back_to_a_working_pytorch_model(base_model, monkeypatch, caplog, error):
    """This is what happened in CI when the Hub rate-limited: the classifier must
    keep working on PyTorch and say why, not fail or pretend ONNX loaded."""
    pytest.importorskip("optimum.onnxruntime")
    from optimum.onnxruntime import ORTModelForFeatureExtraction

    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(ORTModelForFeatureExtraction, "from_pretrained", broken)
    with caplog.at_level(logging.WARNING):
        clf = AdaptiveClassifier(base_model, use_onnx=True, device="cpu")

    assert clf.use_onnx is False
    assert "Falling back to PyTorch" in caplog.text
    clf.add_examples(TEXTS, LABELS)
    assert clf.predict("great")


def test_missing_optimum_falls_back_with_install_hint(base_model, monkeypatch, caplog):
    import builtins
    real_import = builtins.__import__

    def no_optimum(name, *args, **kwargs):
        if name.startswith("optimum"):
            raise ImportError("No module named 'optimum'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_optimum)
    with caplog.at_level(logging.WARNING):
        clf = AdaptiveClassifier(base_model, use_onnx=True, device="cpu")
    assert clf.use_onnx is False
    assert "pip install optimum" in caplog.text


def test_pooling_lookup_failure_falls_back_to_mean(trained, monkeypatch):
    import huggingface_hub

    def offline(*args, **kwargs):
        raise OSError("Hub unreachable")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    trained._resolved_pooling = None
    assert trained._resolve_pooling() == "mean"


# --- randomized sequences of operations ------------------------------------------------------------

def check_consistent(clf):
    """Every structure that tracks classes must agree after any operation."""
    labels = set(clf.label_to_id)
    n = len(labels)
    assert sorted(clf.label_to_id.values()) == list(range(n)), "ids not contiguous"
    assert {clf.id_to_label[i] for i in range(n)} == labels
    assert all(clf.id_to_label[i] == l for l, i in clf.label_to_id.items())
    assert set(clf.memory.prototypes) == labels
    assert set(k for k, v in clf.memory.examples.items() if v) == labels
    assert clf.memory.index.ntotal == n
    assert set(clf.training_history) >= labels or n == 0
    if clf.adaptive_head is not None:
        assert clf.adaptive_head.model[-1].out_features == n
    for text in ("great", "bad"):
        preds = clf.predict(text, k=50)
        assert {l for l, _ in preds} <= labels
        assert all(np.isfinite(s) for _, s in preds)
        if preds:
            assert sum(s for _, s in preds) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("seed", range(6))
def test_random_operation_sequences_keep_every_structure_consistent(new_classifier, seed):
    rng = random.Random(seed)
    pool = ["great love", "good fast", "terrible hate", "awful bad", "okay fine", "cheap works",
            "slow broke", "very good"]
    names = ["a", "b", "c", "d", "e"]
    clf = new_classifier()
    added = []                                  # (text, label) pairs currently believed present

    for _ in range(25):
        op = rng.choice(["add", "add", "add", "forget", "remove", "remove_all_of_class", "predict"])
        known = sorted(clf.label_to_id)
        if op == "add" or not known:
            texts = [rng.choice(pool) + f" {rng.randint(0, 3)}" for _ in range(rng.randint(1, 6))]
            labels = [rng.choice(names) for _ in texts]
            clf.add_examples(texts, labels)
            added += list(zip(texts, labels))
        elif op == "forget":
            victim = rng.choice(known)
            clf.forget(victim)
            added = [(t, l) for t, l in added if l != victim]
        elif op == "remove" and added:
            text, label = rng.choice(added)
            clf.remove_examples([text], label=label, retrain=rng.random() < 0.5)
            added = [(t, l) for t, l in added if not (t == text and l == label)]
        elif op == "remove_all_of_class":
            victim = rng.choice(known)
            doomed = sorted({t for t, l in added if l == victim})
            if doomed:
                clf.remove_examples(doomed, label=victim, retrain=False)
                added = [(t, l) for t, l in added if l != victim]
        else:
            clf.predict(rng.choice(pool))
        check_consistent(clf)


def test_state_after_random_operations_survives_save_and_reload(new_classifier, tmp_path):
    rng = random.Random(99)
    clf = new_classifier(pooling="mean")
    clf.add_examples(TEXTS, LABELS)
    clf.add_examples(["okay fine", "cheap works"] * 6, ["c", "c"] * 6)
    clf.forget("p")
    clf.add_examples(["great love"] * 5, ["z"] * 5)
    clf.remove_examples(["okay fine"], label="c", retrain=False)

    clf.save(str(tmp_path), include_onnx=False)
    loaded = load(tmp_path)
    check_consistent(loaded)
    assert set(loaded.label_to_id) == set(clf.label_to_id)
    for text in ("great love", "terrible hate", "okay fine"):
        assert [l for l, _ in loaded.predict(text, k=9)] == [l for l, _ in clf.predict(text, k=9)]
