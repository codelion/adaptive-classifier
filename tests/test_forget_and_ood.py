"""Removing classes/examples, and telling when an input is out of distribution.

Uses the tiny offline model from conftest.py. Tests that depend on embedding
geometry replace `_get_embeddings` with hand-made unit vectors, because a
randomly initialised encoder places every text in nearly the same spot.
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from adaptive_classifier import AdaptiveClassifier

DIM = 32


def unit(axis, noise=0.0, seed=0):
    """A unit vector near axis `axis`, jittered by `noise`."""
    v = torch.zeros(DIM)
    v[axis] = 1.0
    if noise:
        g = torch.Generator().manual_seed(seed)
        v = v + noise * torch.randn(DIM, generator=g)
    return F.normalize(v, dim=0)


class Geometry:
    """Maps text to a fixed embedding: 'a3' -> axis 0 jitter 3, 'b1' -> axis 1, 'z' -> axis 9."""

    AXES = {"a": 0, "b": 1, "c": 2, "z": 9}

    def __init__(self, noise=0.15):
        self.noise = noise

    def __call__(self, texts):
        out = []
        for text in texts:
            axis = self.AXES[text[0]]
            seed = int(text[1:] or 0)
            out.append(unit(axis, self.noise if seed else 0.0, seed))
        return out


def make(base_model, geometry=None, **config):
    clf = AdaptiveClassifier(base_model, config=config, use_onnx=False, device="cpu")
    if geometry is not False:
        clf._get_embeddings = geometry or Geometry()
    return clf


def trained(base_model, n=8, geometry=None, **config):
    clf = make(base_model, geometry, **config)
    texts, labels = [], []
    for label, key in (("A", "a"), ("B", "b"), ("C", "c")):
        for i in range(1, n + 1):
            texts.append(f"{key}{i}")
            labels.append(label)
    clf.add_examples(texts, labels)
    return clf


def head_logits(clf, embedding):
    clf.adaptive_head.eval()
    with torch.no_grad():
        return clf.adaptive_head(embedding.unsqueeze(0))[0]


# --- forget -------------------------------------------------------------------

def test_forget_removes_class_everywhere(base_model):
    clf = trained(base_model)
    clf.forget("B")

    assert "B" not in clf.label_to_id
    assert "B" not in clf.memory.examples
    assert "B" not in clf.memory.prototypes
    assert "B" not in clf.training_history
    assert clf.memory.index.ntotal == 2
    assert sorted(clf.id_to_label.values()) == ["A", "C"]


def test_forget_leaves_contiguous_ids_and_a_matching_head(base_model):
    clf = trained(base_model)
    clf.forget("A")

    assert sorted(clf.label_to_id.values()) == [0, 1]
    assert {clf.id_to_label[i] for i in (0, 1)} == {"B", "C"}
    assert clf.adaptive_head.model[-1].out_features == 2


def test_forgotten_class_never_appears_in_predictions(base_model):
    clf = trained(base_model)
    clf.forget("B")
    for text in ("a1", "b1", "c1", "z"):
        assert "B" not in [label for label, _ in clf.predict(text, k=10)]
        batch = clf.predict_batch([text], k=10)[0]
        assert "B" not in [label for label, _ in batch]


def test_forget_preserves_what_the_head_learned_about_other_classes(base_model):
    clf = trained(base_model)
    probe = unit(0, 0.15, 1)
    old = head_logits(clf, probe)
    old_ids = dict(clf.label_to_id)

    clf.forget("B")

    new = head_logits(clf, probe)
    for label, new_id in clf.label_to_id.items():
        assert new[new_id].item() == pytest.approx(old[old_ids[label]].item(), abs=1e-6)


def test_forget_accepts_a_list_and_checks_labels(base_model):
    clf = trained(base_model)
    with pytest.raises(ValueError, match="Unknown"):
        clf.forget("nope")
    assert set(clf.label_to_id) == {"A", "B", "C"}   # a bad call changes nothing

    clf.forget(["A", "C"])
    assert list(clf.label_to_id) == ["B"]
    assert clf.id_to_label == {0: "B"}


def test_forgetting_every_class_gives_a_usable_empty_classifier(base_model):
    clf = trained(base_model)
    clf.forget(["A", "B", "C"])

    assert clf.label_to_id == {} and clf.adaptive_head is None
    assert clf.predict("a1") == []
    clf.add_examples(["a1", "a2", "b1", "b2"], ["A", "A", "B", "B"])
    assert {label for label, _ in clf.predict("a1")} == {"A", "B"}


def test_classes_can_be_added_after_forgetting(base_model):
    clf = trained(base_model)
    clf.forget("A")
    clf.add_examples(["z", "z"], ["Z", "Z"])

    assert sorted(clf.label_to_id.values()) == [0, 1, 2]
    assert clf.adaptive_head.model[-1].out_features == 3
    assert "Z" in [label for label, _ in clf.predict("z", k=10)]


def test_forget_survives_save_and_reload(base_model, tmp_path):
    clf = trained(base_model)
    clf.forget("B")
    clf.save(str(tmp_path), include_onnx=False)

    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")
    loaded._get_embeddings = Geometry()
    assert set(loaded.label_to_id) == {"A", "C"}
    assert "B" not in [label for label, _ in loaded.predict("b1", k=10)]


def test_clear_memory_with_labels_now_forgets_them(base_model):
    clf = trained(base_model)
    clf.clear_memory(["B", "not-a-class"])
    assert set(clf.label_to_id) == {"A", "C"}
    assert "B" not in [label for label, _ in clf.predict("b1", k=10)]


# --- remove_examples -----------------------------------------------------------

def test_remove_examples_recomputes_the_prototype(base_model):
    clf = trained(base_model)
    assert clf.remove_examples(["a1", "a2"]) == 2

    remaining = torch.stack([ex.embedding for ex in clf.memory.examples["A"]])
    assert len(remaining) == 6
    assert torch.allclose(clf.memory.prototypes["A"], remaining.mean(dim=0), atol=1e-6)
    assert clf.training_history["A"] == 6


def test_remove_examples_can_be_limited_to_one_label(base_model):
    clf = trained(base_model)
    clf.add_examples(["a1"], ["B"])        # same text under another label
    assert clf.remove_examples(["a1"], label="B") == 1
    assert len(clf.memory.examples["A"]) == 8   # untouched


def test_remove_examples_reports_zero_for_unknown_text(base_model):
    clf = trained(base_model)
    assert clf.remove_examples(["never seen"]) == 0
    assert clf.training_history["A"] == 8


def test_remove_examples_validates_arguments(base_model):
    clf = trained(base_model)
    with pytest.raises(ValueError):
        clf.remove_examples([])
    with pytest.raises(ValueError, match="Unknown"):
        clf.remove_examples(["a1"], label="nope")


def test_removing_the_last_example_drops_the_class(base_model):
    clf = make(base_model)
    clf.add_examples(["a1", "a2", "b1", "b2", "c1"], ["A", "A", "B", "B", "C"])
    assert clf.remove_examples(["c1"]) == 1

    assert "C" not in clf.label_to_id
    assert clf.adaptive_head.model[-1].out_features == 2
    assert "C" not in [label for label, _ in clf.predict("c1", k=10)]


def test_removing_a_mislabelled_example_repairs_the_prototype(base_model):
    clf = trained(base_model)
    clean = clf.memory.prototypes["A"].clone()

    clf.add_examples(["z"], ["A"])                      # a bad label
    assert not torch.allclose(clf.memory.prototypes["A"], clean, atol=1e-3)

    clf.remove_examples(["z"], label="A")
    assert torch.allclose(clf.memory.prototypes["A"], clean, atol=1e-5)


def test_retrain_false_leaves_the_head_alone(base_model):
    clf = trained(base_model)
    before = copy.deepcopy(clf.adaptive_head.state_dict())
    clf.remove_examples(["a1"], retrain=False)
    after = clf.adaptive_head.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_retrain_true_finetunes_the_head(base_model):
    clf = trained(base_model)
    before = copy.deepcopy(clf.adaptive_head.state_dict())
    clf.remove_examples(["a1"], retrain=True)
    after = clf.adaptive_head.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_remove_examples_after_reload_warns_about_partial_memory(base_model, tmp_path, caplog):
    clf = trained(base_model, n=12)
    clf.save(str(tmp_path), include_onnx=False)
    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")

    held = loaded.memory.examples["A"][0].text
    with caplog.at_level("WARNING"):
        loaded.remove_examples([held], label="A", retrain=False)
    assert "recomputed from the examples that remain" in caplog.text


# --- out-of-distribution detection ----------------------------------------------

def test_in_distribution_text_scores_low_and_far_text_scores_high(base_model):
    clf = trained(base_model)
    assert clf.ood_score("a3") < 1.0                  # a training example
    assert clf.ood_score("z") > clf.config.ood_threshold   # an unrelated direction
    assert clf.ood_score("z") > 1.5 * clf.ood_score("a3")
    assert not clf.is_ood("a3")
    assert clf.is_ood("z")


def test_score_is_relative_to_each_class_spread(base_model):
    """The same absolute distance is normal for a diffuse class and alien for a tight one."""
    geometry = Geometry(noise=0.0)

    def wide(axis):                                    # spread-out class
        return [unit(axis, 0.6, s) for s in range(1, 9)]

    def tight(axis):
        return [unit(axis, 0.02, s) for s in range(1, 9)]

    clf = make(base_model, geometry=False)
    vectors = {}
    for label, axis, maker in (("wide", 0, wide), ("tight", 1, tight)):
        for i, v in enumerate(maker(axis)):
            vectors[f"{label}{i}"] = v
    clf._get_embeddings = lambda texts: [vectors[t] for t in texts]
    clf.add_examples(list(vectors), [t.rstrip("0123456789") for t in vectors])

    wide_proto, tight_proto = clf.memory.prototypes["wide"], clf.memory.prototypes["tight"]
    offset = unit(5)
    probe_wide = F.normalize(wide_proto + 0.35 * offset, dim=0)
    probe_tight = F.normalize(tight_proto + 0.35 * offset, dim=0)
    clf._get_embeddings = lambda texts: [probe_wide if t == "w" else probe_tight for t in texts]

    assert clf.ood_score("w") < clf.ood_score("t")


def test_threshold_override_and_config(base_model):
    clf = trained(base_model)
    score = clf.ood_score("a3")
    assert clf.is_ood("a3", threshold=score / 2)
    assert not clf.is_ood("a3", threshold=score * 2)

    strict = trained(base_model, ood_threshold=0.0)
    assert strict.is_ood("a3")


def test_ood_score_of_an_empty_classifier_is_infinite(base_model):
    clf = make(base_model)
    assert clf.ood_score("a1") == float("inf")
    assert clf.is_ood("a1")


def test_single_example_class_uses_the_minimum_radius(base_model):
    clf = make(base_model)
    clf.add_examples(["a1", "b1", "b2"], ["A", "B", "B"])
    assert clf.memory.class_radius("A") == 0.0
    assert clf.ood_score("a1") == pytest.approx(0.0)
    assert 0 < clf.ood_score("a2") < float("inf")       # no ZeroDivisionError


def test_abstain_ood_returns_nothing_for_foreign_text(base_model):
    clf = trained(base_model)
    assert clf.predict("z", abstain_ood=True) == []
    assert clf.predict("a3", abstain_ood=True) != []


def test_abstain_below_uses_the_top_confidence(base_model):
    clf = trained(base_model)
    top = clf.predict("a3")[0][1]
    assert clf.predict("a3", abstain_below=top + 0.01) == []
    assert clf.predict("a3", abstain_below=top - 0.01) != []


def test_predict_is_unchanged_without_abstention_arguments(base_model):
    clf = trained(base_model)
    assert clf.predict("z") == clf.predict("z", abstain_below=None, abstain_ood=False)
    assert clf.predict("z") != []


def test_removing_examples_shrinks_the_class_radius(base_model):
    clf = trained(base_model)
    before = clf.memory.class_radius("A")
    farthest = max(clf.memory.examples["A"],
                   key=lambda ex: float(torch.norm(ex.embedding - clf.memory.prototypes["A"])))
    clf.remove_examples([farthest.text], label="A", retrain=False)
    assert clf.memory.class_radius("A") < before


def test_class_radius_survives_save_and_reload(base_model, tmp_path):
    """A reloaded memory holds few examples, so the spread must be persisted."""
    clf = trained(base_model, n=20)
    original = clf.memory.class_radius("A")
    clf.save(str(tmp_path), include_onnx=False)

    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")
    loaded._get_embeddings = Geometry()
    assert len(loaded.memory.examples["A"]) < 20
    assert loaded.memory.class_radius("A") == pytest.approx(original, abs=1e-5)
    assert loaded.ood_score("a3") == pytest.approx(clf.ood_score("a3"), rel=1e-3, abs=1e-3)
    assert loaded.is_ood("z")


def test_ood_settings_round_trip_in_the_saved_config(base_model, tmp_path):
    clf = trained(base_model, ood_threshold=2.0, ood_min_radius=0.2)
    clf.save(str(tmp_path), include_onnx=False)
    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")
    assert loaded.config.ood_threshold == 2.0
    assert loaded.config.ood_min_radius == 0.2


def test_old_saves_without_radii_still_load(base_model, tmp_path):
    import json
    clf = trained(base_model)
    clf.save(str(tmp_path), include_onnx=False)
    path = tmp_path / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["ood_radii"]
    del data["config"]["ood_threshold"]
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")
    loaded._get_embeddings = Geometry()
    assert loaded.config.ood_threshold == 1.25
    assert loaded.ood_score("a3") < float("inf")
