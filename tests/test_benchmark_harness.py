"""The few-shot benchmark harness (scripts/benchmark_fewshot.py).

The benchmark itself needs network access and takes a long time, so it runs as a manual
workflow. These tests check everything that can be checked offline, so that when it does
run, a surprising number is the library's and not the harness's: sampling, metrics,
aggregation, the report, the command line, and a smoke run of each method on a tiny model.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_classifier import calibration

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_fewshot.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("benchmark_fewshot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def labels_of(n_per_class, classes=("a", "b", "c")):
    return [c for c in classes for _ in range(n_per_class)]


# --- sampling ------------------------------------------------------------------------------

def test_sample_takes_exactly_k_per_class(bench):
    labels = labels_of(20)
    picked = bench.sample_few_shot(labels, 5, seed=0)
    assert len(picked) == 15
    for c in "abc":
        assert sum(labels[i] == c for i in picked) == 5


def test_sample_is_sorted_unique_and_in_range(bench):
    labels = labels_of(10)
    picked = bench.sample_few_shot(labels, 4, seed=3)
    assert picked == sorted(set(picked)) and all(0 <= i < len(labels) for i in picked)


def test_sample_is_deterministic_and_seed_dependent(bench):
    labels = labels_of(50)
    assert bench.sample_few_shot(labels, 8, 1) == bench.sample_few_shot(labels, 8, 1)
    assert bench.sample_few_shot(labels, 8, 1) != bench.sample_few_shot(labels, 8, 2)


def test_smaller_samples_are_subsets_of_larger_ones(bench):
    labels = labels_of(50)
    four, sixteen = set(bench.sample_few_shot(labels, 4, 7)), set(bench.sample_few_shot(labels, 16, 7))
    assert four < sixteen


def test_a_class_with_fewer_examples_than_shots_contributes_all_of_them(bench):
    labels = ["a"] * 10 + ["b"] * 2
    picked = bench.sample_few_shot(labels, 5, 0)
    assert sum(labels[i] == "b" for i in picked) == 2 and sum(labels[i] == "a" for i in picked) == 5


def test_sample_does_not_depend_on_row_order_within_labels(bench):
    labels = labels_of(12)
    assert len(bench.sample_few_shot(list(reversed(labels)), 3, 0)) == 9


@pytest.mark.parametrize("shots", [0, -1])
def test_sample_rejects_nonpositive_shots(bench, shots):
    with pytest.raises(ValueError, match="shots"):
        bench.sample_few_shot(["a", "b"], shots, 0)


def test_subsample(bench):
    texts = [f"t{i}" for i in range(100)]
    labels = [str(i % 3) for i in range(100)]
    all_t, all_l = bench.subsample(texts, labels, None)
    assert (all_t, all_l) == (texts, labels)
    assert bench.subsample(texts, labels, 500)[0] == texts
    sub_t, sub_l = bench.subsample(texts, labels, 20)
    assert len(sub_t) == 20 and len(set(sub_t)) == 20
    assert all(labels[texts.index(t)] == l for t, l in zip(sub_t, sub_l))        # still aligned
    assert bench.subsample(texts, labels, 20) == (sub_t, sub_l)                  # fixed subset


# --- metrics -------------------------------------------------------------------------------------

def test_perfect_predictions_score_perfectly(bench):
    probs = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    scores = bench.score_predictions(probs, np.array([0, 1, 2, 0, 1, 2]))
    assert scores == {"accuracy": 1.0, "macro_f1": 1.0, "ece": pytest.approx(0.0)}


def test_known_accuracy_and_f1(bench):
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    targets = np.array([0, 0, 1, 1])                      # last one wrong
    scores = bench.score_predictions(probs, targets)
    assert scores["accuracy"] == pytest.approx(0.75)
    assert scores["macro_f1"] == pytest.approx(0.7333, abs=1e-3)


def test_ece_matches_the_library_implementation(bench):
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=200)
    targets = rng.integers(0, 4, size=200)
    assert bench.expected_calibration_error(probs, targets) == pytest.approx(
        calibration.expected_calibration_error(probs, targets))


def test_ece_of_an_overconfident_model(bench):
    probs = np.array([[1.0, 0.0]] * 10)
    assert bench.expected_calibration_error(probs, np.array([0] * 5 + [1] * 5)) == pytest.approx(0.5)


# --- aggregation and report ------------------------------------------------------------------------

def row(dataset="ds", method="adaptive", shots=4, seed=0, accuracy=0.5, **extra):
    base = {"dataset": dataset, "method": method, "shots": shots, "seed": seed, "n_classes": 3,
            "n_train": 12, "n_test": 100, "accuracy": accuracy, "macro_f1": accuracy - 0.05,
            "ece": 0.1, "train_s": 2.0, "predict_s": 1.0, "latency_ms_p50": 5.0, "update_s": 0.5,
            "update_is_retrain": False}
    base.update(extra)
    return base


def test_aggregate_computes_mean_and_sample_std(bench):
    rows = [row(seed=s, accuracy=a) for s, a in enumerate([0.6, 0.7, 0.8])]
    summary = bench.aggregate(rows)[("ds", "adaptive", 4)]
    mean, std = summary["accuracy"]
    assert mean == pytest.approx(0.7) and std == pytest.approx(0.1)
    assert summary["runs"] == 3 and summary["failed"] == 0


def test_aggregate_single_seed_has_no_std(bench):
    assert bench.aggregate([row()])[("ds", "adaptive", 4)]["accuracy"] == (0.5, None)


def test_aggregate_excludes_failed_runs_but_counts_them(bench):
    rows = [row(seed=0, accuracy=0.6), {"dataset": "ds", "method": "adaptive", "shots": 4,
                                         "seed": 1, "error": "RuntimeError: boom"}]
    summary = bench.aggregate(rows)[("ds", "adaptive", 4)]
    assert summary["accuracy"][0] == pytest.approx(0.6)
    assert (summary["runs"], summary["failed"]) == (2, 1)


def test_aggregate_keeps_methods_datasets_and_shots_apart(bench):
    rows = [row(method="a", accuracy=0.1), row(method="b", accuracy=0.9),
            row(dataset="other", accuracy=0.4), row(shots=16, accuracy=0.7)]
    summary = bench.aggregate(rows)
    assert len(summary) == 4 and summary[("ds", "b", 4)]["accuracy"][0] == 0.9


def test_report_has_a_section_per_dataset_and_bolds_the_best(bench):
    rows = [row(dataset="alpha", method="adaptive", accuracy=0.80),
            row(dataset="alpha", method="logreg", accuracy=0.70),
            row(dataset="beta", method="adaptive", accuracy=0.60)]
    text = bench.render_markdown(rows, {"model": "tiny", "seeds": 1})
    assert "## alpha" in text and "## beta" in text
    assert "**model**: tiny" in text
    assert "**0.800**" in text and "0.700" in text and "**0.700**" not in text


def test_report_flags_failures_and_marks_gaps(bench):
    rows = [row(method="adaptive", shots=4), row(method="adaptive", shots=16),
            row(method="setfit", shots=4),
            {"dataset": "ds", "method": "setfit", "shots": 16, "seed": 0, "error": "OOM"}]
    text = bench.render_markdown(rows)
    assert "1 run(s) failed" in text and "failed" in text.split("### Accuracy")[1]
    only_one = bench.render_markdown([row(method="adaptive", shots=4), row(method="logreg", shots=16)])
    assert "–" in only_one


def test_report_cost_table_marks_retraining(bench):
    rows = [row(method="adaptive", shots=16), row(method="setfit", shots=16, update_is_retrain=True,
                                                    update_s=90.0)]
    text = bench.render_markdown(rows)
    cost = text.split("### Cost")[1]
    assert "90.00 (full retrain)" in cost and "0.50 (full retrain)" not in cost


def test_report_with_std_shows_plus_minus(bench):
    rows = [row(seed=0, accuracy=0.6), row(seed=1, accuracy=0.8)]
    assert "±" in bench.render_markdown(rows)


def test_report_orders_known_methods_first(bench):
    rows = [row(method="zzz-custom"), row(method="setfit"), row(method="adaptive")]
    text = bench.render_markdown(rows).split("### Accuracy")[1]
    assert text.index("| adaptive") < text.index("| setfit") < text.index("| zzz-custom")


# --- the synthetic dataset -----------------------------------------------------------------------------

def test_synthetic_dataset_is_deterministic_and_balanced(bench):
    a, b = bench.synthetic_dataset(), bench.synthetic_dataset()
    assert a == b
    assert sorted(set(a["train_labels"])) == [f"class_{c}" for c in range(4)]
    assert {a["train_labels"].count(l) for l in set(a["train_labels"])} == {40}
    assert set(a["test_labels"]) <= set(a["train_labels"])


def test_unknown_dataset_and_method_are_rejected(bench):
    with pytest.raises(SystemExit):
        bench.load_dataset_splits("no-such-dataset")
    with pytest.raises(SystemExit):
        bench.build_method("no-such-method", "model", None)


# --- running each method for real, on a tiny local model ---------------------------------------------------

@pytest.fixture
def smoke(bench, synthetic_text_model):
    def go(methods, shots=(5,), seeds=1):
        return bench.run_benchmark("synthetic", list(shots), seeds, methods, synthetic_text_model,
                                   max_test=60, log=lambda *_: None)
    return go


def test_adaptive_methods_and_logreg_run_end_to_end(smoke):
    results = smoke(["adaptive", "adaptive-proto", "logreg"])
    assert len(results) == 3 and not [r for r in results if r.get("error")]
    for r in results:
        assert 0.25 < r["accuracy"] <= 1.0, f"{r['method']} is no better than chance: {r['accuracy']}"
        assert 0.0 <= r["macro_f1"] <= 1.0 and 0.0 <= r["ece"] <= 1.0
        assert r["train_s"] > 0 and r["predict_s"] > 0 and r["latency_ms_p50"] > 0 and r["update_s"] > 0
        assert r["n_classes"] == 4 and r["n_test"] == 60 and r["n_train"] == 20


def test_adding_an_example_is_an_update_for_adaptive_and_a_refit_for_logreg(smoke):
    by_method = {r["method"]: r for r in smoke(["adaptive", "logreg"])}
    assert by_method["adaptive"]["update_is_retrain"] is False
    assert by_method["logreg"]["update_is_retrain"] is True


def test_results_are_reproducible_across_runs(smoke):
    first, second = smoke(["adaptive-proto", "logreg"]), smoke(["adaptive-proto", "logreg"])
    for a, b in zip(first, second):
        assert a["accuracy"] == b["accuracy"] and a["macro_f1"] == pytest.approx(b["macro_f1"])


def test_every_seed_and_shot_count_gets_a_row_per_method(smoke):
    results = smoke(["adaptive-proto", "logreg"], shots=(3, 6), seeds=2)
    assert len(results) == 2 * 2 * 2
    assert {(r["shots"], r["seed"], r["method"]) for r in results} == {
        (k, s, m) for k in (3, 6) for s in (0, 1) for m in ("adaptive-proto", "logreg")}


def test_a_failing_method_is_recorded_and_the_others_still_run(bench, synthetic_text_model, monkeypatch):
    real = bench.build_method

    def build(name, model, device):
        if name == "logreg":
            raise RuntimeError("boom")
        return real(name, model, device)

    monkeypatch.setattr(bench, "build_method", build)
    results = bench.run_benchmark("synthetic", [4], 1, ["logreg", "adaptive-proto"],
                                  synthetic_text_model, 40, log=lambda *_: None)
    by = {r["method"]: r for r in results}
    assert "boom" in by["logreg"]["error"] and "accuracy" not in by["logreg"]
    assert by["adaptive-proto"]["accuracy"] > 0


def test_setfit_does_not_write_checkpoints_into_the_working_directory(bench, synthetic_text_model,
                                                                      tmp_path, monkeypatch):
    try:
        import datasets  # noqa: F401
        import setfit  # noqa: F401
    except ImportError as error:
        pytest.skip(f"SetFit is not usable in this environment: {error}")
    monkeypatch.chdir(tmp_path)
    results = bench.run_benchmark("synthetic", [4], 1, ["setfit"], synthetic_text_model, 40,
                                  log=lambda *_: None)
    assert not results[0].get("error"), results[0].get("error")
    assert list(tmp_path.iterdir()) == [], "SetFit left files in the working directory"


def test_setfit_runs_end_to_end(bench, synthetic_text_model):
    try:
        import datasets  # noqa: F401
        import setfit  # noqa: F401
    except ImportError as error:        # absent, or installed against an incompatible stack
        pytest.skip(f"SetFit is not usable in this environment: {error}")
    results = bench.run_benchmark("synthetic", [4], 1, ["setfit"], synthetic_text_model, 40,
                                  log=lambda *_: None)
    assert not results[0].get("error"), results[0].get("error")
    assert 0.0 < results[0]["accuracy"] <= 1.0
    assert results[0]["update_is_retrain"] is True and results[0]["update_s"] == results[0]["train_s"]


# --- the command line ----------------------------------------------------------------------------------------

def test_run_writes_a_results_file_and_report_renders_it(bench, synthetic_text_model, tmp_path, capsys):
    out = tmp_path / "nested" / "synthetic.json"
    code = bench.main(["run", "--dataset", "synthetic", "--shots", "4", "--seeds", "2",
                       "--methods", "adaptive-proto", "logreg", "--model", synthetic_text_model,
                       "--max-test", "40", "--out", str(out)])
    assert code == 0 and out.exists()
    payload = json.loads(out.read_text())
    assert payload["meta"]["model"] == synthetic_text_model and payload["meta"]["seeds"] == 2
    assert len(payload["results"]) == 4

    summary = tmp_path / "summary.md"
    assert bench.main(["report", str(out), "--out", str(summary)]) == 0
    text = summary.read_text()
    assert "## synthetic" in text and "adaptive-proto" in text and "logreg" in text
    assert "# Few-shot benchmark" in capsys.readouterr().out


def test_run_exits_nonzero_but_still_writes_results_when_a_method_fails(bench, synthetic_text_model,
                                                                         tmp_path, monkeypatch):
    real = bench.build_method
    monkeypatch.setattr(bench, "build_method",
                        lambda n, m, d: (_ for _ in ()).throw(RuntimeError("down")) if n == "logreg"
                        else real(n, m, d))
    out = tmp_path / "r.json"
    code = bench.main(["run", "--dataset", "synthetic", "--shots", "4", "--seeds", "1",
                       "--methods", "adaptive-proto", "logreg", "--model", synthetic_text_model,
                       "--max-test", "40", "--out", str(out)])
    assert code == 1
    errors = [r for r in json.loads(out.read_text())["results"] if r.get("error")]
    assert len(errors) == 1 and "down" in errors[0]["error"]


def test_report_merges_several_result_files(bench, tmp_path):
    for name, dataset in (("a.json", "alpha"), ("b.json", "beta")):
        (tmp_path / name).write_text(json.dumps({"meta": {"model": "m"}, "results": [row(dataset=dataset)]}))
    out = tmp_path / "s.md"
    assert bench.main(["report", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--out", str(out)]) == 0
    assert "## alpha" in out.read_text() and "## beta" in out.read_text()


def test_command_line_rejects_unknown_methods(bench, tmp_path):
    with pytest.raises(SystemExit):
        bench.main(["run", "--dataset", "synthetic", "--methods", "magic", "--out", str(tmp_path / "x")])


# --- checking datasets before a long run ------------------------------------------------------------------------

class FakeStream:
    def __init__(self, rows, features=None):
        self.rows, self.features = rows, features

    def __iter__(self):
        return iter(self.rows)


class FakeFeature:
    names = ["a", "b", "c"]


def install_fake_datasets(monkeypatch, loader):
    import sys
    import types
    fake = types.ModuleType("datasets")
    fake.load_dataset = loader
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_verify_accepts_a_well_formed_dataset(bench, monkeypatch):
    def loader(hub, split, streaming):
        assert streaming is True, "verification must not download the whole dataset"
        return FakeStream([{"text": "hello", "label": 1}], {"label": FakeFeature()})

    install_fake_datasets(monkeypatch, loader)
    assert bench.verify_dataset("ag_news") == "ag_news: fancyzhx/ag_news ok, 3 classes"


def test_verify_names_the_hub_id_and_split_when_loading_fails(bench, monkeypatch):
    def loader(hub, split, streaming):
        raise FileNotFoundError(f"Dataset '{hub}' doesn't exist on the Hub")

    install_fake_datasets(monkeypatch, loader)
    with pytest.raises(RuntimeError) as info:
        bench.verify_dataset("emotion")
    assert "dair-ai/emotion" in str(info.value) and "'train'" in str(info.value)
    assert "doesn't exist" in str(info.value)


def test_verify_reports_a_missing_column(bench, monkeypatch):
    install_fake_datasets(monkeypatch, lambda hub, split, streaming:
                          FakeStream([{"sentence": "x", "label": 0}], {}))
    with pytest.raises(RuntimeError, match="no column 'text'"):
        bench.verify_dataset("banking77")


def test_no_default_dataset_depends_on_a_loading_script(bench):
    """Recent `datasets` releases refuse to run Hub loading scripts, so a dataset that only
    has one fails at load time. PolyAI/banking77 was one; the parquet copy replaces it."""
    assert all(spec["hub"] != "PolyAI/banking77" for spec in bench.DATASETS.values())


def test_verify_reports_every_problem_not_just_the_first(bench, monkeypatch):
    def loader(hub, split, streaming):
        if split == "test":
            raise ValueError("no test split")
        return FakeStream([{"text": "x"}], {})

    install_fake_datasets(monkeypatch, loader)
    with pytest.raises(RuntimeError) as info:
        bench.verify_dataset("ag_news")
    assert "'test'" in str(info.value) and "no column 'label'" in str(info.value)


def test_check_command_exit_codes(bench, monkeypatch, capsys):
    install_fake_datasets(monkeypatch, lambda hub, split, streaming:
                          FakeStream([{"text": "x", "label": 0}], {"label": FakeFeature()}))
    assert bench.main(["check", "synthetic", "ag_news"]) == 0
    assert "synthetic" in capsys.readouterr().out

    install_fake_datasets(monkeypatch, lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert bench.main(["check", "ag_news", "emotion"]) == 1
    assert capsys.readouterr().err.count("FAIL") == 2


def test_loading_works_for_datasets_without_class_names(bench, monkeypatch):
    class Split(dict):
        features = {"label": object()}                      # a plain value: no .names

    train = Split(text=["a", "b", "c"], label=[0, 1, 0])
    test = Split(text=["d"], label=[1])
    install_fake_datasets(monkeypatch, lambda hub: {"train": train, "test": test})
    loaded = bench.load_dataset_splits("ag_news")
    assert loaded["train_labels"] == ["0", "1", "0"] and loaded["test_labels"] == ["1"]


def test_loading_maps_class_ids_to_names(bench, monkeypatch):
    class Split(dict):
        features = {"label": FakeFeature()}

    train = Split(text=["a", "b"], label=[2, 0])
    install_fake_datasets(monkeypatch, lambda hub: {"train": train, "test": Split(text=["c"], label=[1])})
    loaded = bench.load_dataset_splits("emotion")
    assert loaded["train_labels"] == ["c", "a"] and loaded["test_labels"] == ["b"]


# --- validating workflow inputs -----------------------------------------------------------------------------------

def good_plan(bench, **overrides):
    args = dict(datasets="ag_news,emotion", shots="4 16", seeds="3", methods="adaptive logreg",
                model="acme/encoder-small", max_test="1000")
    args.update(overrides)
    return bench.plan(**args)


def test_plan_normalises_valid_input(bench):
    p = good_plan(bench, datasets=" ag_news , emotion,ag_news ", shots="16,4 4", methods="logreg  adaptive")
    assert p == {"datasets": ["ag_news", "emotion"], "shots": "4 16", "seeds": 3,
                 "methods": "logreg adaptive", "model": "acme/encoder-small",
                 "max_test": 1000}


def test_plan_defaults_to_every_method(bench):
    assert good_plan(bench, methods="")["methods"] == " ".join(bench.METHODS)


@pytest.mark.parametrize("field, value", [
    ("datasets", ""), ("datasets", "ag_news,imaginary"), ("datasets", "ag_news; rm -rf /"),
    ("shots", ""), ("shots", "0"), ("shots", "4 many"), ("shots", "1000"), ("shots", "-3"),
    ("seeds", "0"), ("seeds", "11"), ("seeds", "three"), ("seeds", ""),
    ("methods", "adaptive magic"), ("methods", "adaptive;ls"),
    ("model", ""), ("model", "a b"), ("model", "x; echo hi"), ("model", "$(whoami)"),
    ("model", "../../etc/passwd"), ("model", "a/b/c"),
    ("max_test", "5"), ("max_test", "lots"), ("max_test", "10000000"),
])
def test_plan_rejects_bad_values_and_names_the_setting(bench, field, value):
    with pytest.raises(ValueError) as info:
        good_plan(bench, **{field: value})
    assert field.replace("_", " ") in str(info.value) or field in str(info.value)


def test_plan_command_prints_json_or_fails_with_a_message(bench, capsys):
    assert bench.main(["plan", "--datasets", "emotion", "--shots", "8", "--seeds", "2"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["datasets"] == ["emotion"] and printed["shots"] == "8" and printed["seeds"] == 2

    assert bench.main(["plan", "--datasets", "nope"]) == 2
    assert "Invalid benchmark settings" in capsys.readouterr().err
