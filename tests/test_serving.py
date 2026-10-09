"""The HTTP server (adaptive_classifier.serving), exercised through FastAPI's TestClient."""

import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from adaptive_classifier import AdaptiveClassifier
from adaptive_classifier import serving
from adaptive_classifier.serving import create_app

from .synthetic import Dataset


@pytest.fixture
def data(monkeypatch):
    return Dataset().patch(monkeypatch)


@pytest.fixture
def clf(new_classifier, data):
    X, y = data.classes(12, n_classes=3, noise=0.2, seed=1)
    c = new_classifier(head_steps=30)
    c.add_examples(X, y)
    return c


@pytest.fixture
def client(clf):
    return TestClient(create_app(clf))


def updatable(clf, **kwargs):
    return TestClient(create_app(clf, allow_updates=True, **kwargs))


# --- reading ---------------------------------------------------------------------------------

def test_health_and_info(client, clf):
    assert client.get("/health").json() == {"status": "ok", "classes": 3}
    info = client.get("/info").json()
    assert info["classes"] == ["C0", "C1", "C2"]
    assert info["examples_per_class"] == {"C0": 12, "C1": 12, "C2": 12}
    assert info["calibrated"] is False and info["updates_allowed"] is False


def test_predict_returns_what_the_classifier_returns(client, clf):
    body = client.post("/predict", json={"text": "tr1_3", "k": 3}).json()
    expected = clf.predict("tr1_3", k=3)
    assert body["abstained"] is False
    assert [(p["label"], pytest.approx(p["score"])) for p in body["predictions"]] == expected


def test_predict_respects_k(client):
    assert len(client.post("/predict", json={"text": "tr0_0", "k": 1}).json()["predictions"]) == 1
    assert client.post("/predict", json={"text": "tr0_0", "k": 0}).json()["predictions"] == []


def test_abstention_is_reported(client, clf):
    top = clf.predict("tr0_0")[0][1]
    r = client.post("/predict", json={"text": "tr0_0", "abstain_below": min(top + 0.01, 1.0)}).json()
    assert r == {"predictions": [], "abstained": True}
    ok = client.post("/predict", json={"text": "tr0_0", "abstain_below": top - 0.01}).json()
    assert ok["abstained"] is False and ok["predictions"]


def test_abstain_ood_flags_foreign_text(client, data):
    foreign = data.add("far", 9, 0.1, 5)
    assert client.post("/predict", json={"text": foreign, "abstain_ood": True}).json()["abstained"] is True
    assert client.post("/predict", json={"text": "tr0_0", "abstain_ood": True}).json()["abstained"] is False


def test_predict_batch_matches_the_classifier(client, clf):
    texts = ["tr0_0", "tr1_1", "tr2_2"]
    body = client.post("/predict_batch", json={"texts": texts, "k": 2}).json()["predictions"]
    expected = clf.predict_batch(texts, k=2)
    for got, want in zip(body, expected):
        assert [(p["label"], pytest.approx(p["score"])) for p in got] == want


def test_ood_endpoint(client, data):
    near = client.post("/ood", json={"text": "tr0_0"}).json()
    far = client.post("/ood", json={"text": data.add("far2", 9, 0.1, 6)}).json()
    assert near["ood"] is False and far["ood"] is True and far["score"] > near["score"]
    strict = client.post("/ood", json={"text": "tr0_0", "threshold": 0.001}).json()
    assert strict["ood"] is True and strict["threshold"] == 0.001


def test_predict_set_needs_calibration_then_works(client, clf, data):
    r = client.post("/predict_set", json={"text": "tr0_0"})
    assert r.status_code == 400 and "calibrate" in r.json()["detail"]

    Xc, yc = data.classes(30, n_classes=3, noise=0.4, seed=8, prefix="cal")
    clf.calibrate(Xc, yc)
    r = client.post("/predict_set", json={"text": "tr0_0", "alpha": 0.2})
    assert r.status_code == 200 and r.json()["labels"][0]["label"] == "C0"
    assert client.get("/info").json()["calibrated"] is True


# --- input validation ------------------------------------------------------------------------------

@pytest.mark.parametrize("path, body", [
    ("/predict", {"text": ""}),
    ("/predict", {}),
    ("/predict", {"text": "x", "k": -1}),
    ("/predict", {"text": "x", "k": "many"}),
    ("/predict", {"text": "x", "abstain_below": 2}),
    ("/predict", {"text": "x" * 20_001}),
    ("/predict_batch", {"texts": []}),
    ("/predict_batch", {"texts": ["a", ""]}),
    ("/predict_set", {"text": "x", "alpha": 0}),
    ("/predict_set", {"text": "x", "alpha": 1}),
    ("/ood", {"text": ""}),
])
def test_invalid_requests_are_rejected(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_batch_size_is_capped(clf):
    small = TestClient(create_app(clf, max_batch=3))
    assert small.post("/predict_batch", json={"texts": ["tr0_0"] * 3}).status_code == 200
    assert small.post("/predict_batch", json={"texts": ["tr0_0"] * 4}).status_code == 422


def test_text_length_is_capped(clf):
    small = TestClient(create_app(clf, max_text_chars=5))
    assert small.post("/predict", json={"text": "tr0_0"}).status_code == 200      # exactly 5 characters
    assert small.post("/predict", json={"text": "tr0_10"}).status_code == 422     # 6 characters
    assert small.post("/predict_batch", json={"texts": ["tr0_10"]}).status_code == 422


def test_malformed_json_is_a_client_error(client):
    r = client.post("/predict", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


# --- updates ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path, body", [
    ("/examples", {"texts": ["a"], "labels": ["C0"]}),
    ("/forget", {"labels": ["C0"]}),
    ("/remove_examples", {"texts": ["tr0_0"]}),
])
def test_updates_are_off_by_default(client, clf, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 403
    assert sorted(clf.label_to_id) == ["C0", "C1", "C2"]


def test_adding_examples_and_a_new_class(clf, data):
    http = updatable(clf)
    novel = [data.add(f"nv{i}", 5, 0.1, 40 + i) for i in range(6)]
    r = http.post("/examples", json={"texts": novel, "labels": ["C5"] * 6})
    assert r.status_code == 200
    assert r.json() == {"added": 6, "classes": ["C0", "C1", "C2", "C5"]}
    assert http.post("/predict", json={"text": novel[0]}).json()["predictions"][0]["label"] == "C5"


def test_forgetting_and_removing(clf):
    http = updatable(clf)
    r = http.post("/remove_examples", json={"texts": ["tr0_0", "tr0_1"], "label": "C0"})
    assert r.json()["removed"] == 2 and clf.training_history["C0"] == 10
    r = http.post("/forget", json={"labels": ["C2"]})
    assert r.json()["classes"] == ["C0", "C1"]
    assert "C2" not in {p["label"] for p in http.post("/predict", json={"text": "tr2_0"}).json()["predictions"]}


def test_classifier_errors_become_400_with_the_message(clf):
    http = updatable(clf)
    r = http.post("/forget", json={"labels": ["nope"]})
    assert r.status_code == 400 and "Unknown labels" in r.json()["detail"]
    r = http.post("/examples", json={"texts": ["a", "b"], "labels": ["C0"]})
    assert r.status_code == 400 and "Mismatched" in r.json()["detail"]
    assert sorted(clf.label_to_id) == ["C0", "C1", "C2"]


def test_non_string_labels_are_rejected(clf):
    r = updatable(clf).post("/examples", json={"texts": ["a"], "labels": [3]})
    assert r.status_code == 422


@pytest.mark.parametrize("headers, status", [
    ({}, 401),
    ({"Authorization": "Bearer wrong"}, 401),
    ({"Authorization": "wrong-scheme secret"}, 401),
    ({"Authorization": "Bearer "}, 401),
    ({"Authorization": "Bearer secret"}, 200),
    ({"authorization": "bearer secret"}, 200),
])
def test_api_key_protects_updates(clf, headers, status):
    http = updatable(clf, api_key="secret")
    r = http.post("/remove_examples", json={"texts": ["nothing"]}, headers=headers)
    assert r.status_code == status
    if status == 401:
        assert r.headers["www-authenticate"] == "Bearer"


def test_api_key_is_not_needed_to_read(clf):
    http = updatable(clf, api_key="secret")
    assert http.post("/predict", json={"text": "tr0_0"}).status_code == 200
    assert http.get("/health").status_code == 200


def test_updates_without_a_key_log_a_warning(clf, caplog):
    with caplog.at_level("WARNING"):
        create_app(clf, allow_updates=True)
    assert "without an api_key" in caplog.text


def test_updates_are_saved_when_a_save_dir_is_given(clf, data, tmp_path):
    http = updatable(clf, save_dir=str(tmp_path))
    novel = [data.add(f"sv{i}", 6, 0.1, 80 + i) for i in range(5)]
    assert http.post("/examples", json={"texts": novel, "labels": ["C6"] * 5}).status_code == 200
    reloaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")
    assert "C6" in reloaded.label_to_id


def test_requests_in_parallel_do_not_corrupt_the_classifier(clf, data):
    http = updatable(clf)
    statuses = []

    def read():
        for _ in range(15):
            statuses.append(http.post("/predict", json={"text": "tr0_0", "k": 10}).status_code)

    def write():
        for i in range(4):
            texts = [data.add(f"par{i}_{j}", 4 + i, 0.1, 500 + 10 * i + j) for j in range(3)]
            statuses.append(http.post("/examples", json={"texts": texts, "labels": [f"P{i}"] * 3}).status_code)

    threads = [threading.Thread(target=read) for _ in range(3)] + [threading.Thread(target=write)]
    [t.start() for t in threads]
    [t.join(timeout=120) for t in threads]
    assert statuses and set(statuses) == {200}
    assert sorted(clf.label_to_id.values()) == list(range(len(clf.label_to_id)))


# --- command line -------------------------------------------------------------------------------------

@pytest.fixture
def saved(clf, tmp_path):
    clf.save(str(tmp_path), include_onnx=False)
    return str(tmp_path)


@pytest.fixture
def launched(monkeypatch):
    calls = {}
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: calls.update(app=app, host=host, port=port))
    return calls


def test_cli_loads_a_saved_classifier_and_serves_it(saved, launched):
    serving.main([saved, "--port", "9001", "--host", "0.0.0.0", "--no-onnx"])
    assert (launched["host"], launched["port"]) == ("0.0.0.0", 9001)
    client = TestClient(launched["app"])
    assert client.get("/health").json()["classes"] == 3
    assert client.post("/forget", json={"labels": ["C0"]}).status_code == 403


def test_cli_flags_enable_updates_and_a_key(saved, launched):
    serving.main([saved, "--allow-updates", "--api-key", "k1", "--no-onnx"])
    client = TestClient(launched["app"])
    assert client.post("/remove_examples", json={"texts": ["z"]}).status_code == 401
    assert client.post("/remove_examples", json={"texts": ["z"]},
                       headers={"Authorization": "Bearer k1"}).status_code == 200


def test_cli_reads_the_key_from_the_environment(saved, launched, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_CLASSIFIER_API_KEY", "from-env")
    serving.main([saved, "--allow-updates", "--no-onnx"])
    client = TestClient(launched["app"])
    assert client.post("/forget", json={"labels": ["C0"]}).status_code == 401
    assert client.post("/forget", json={"labels": ["C0"]},
                       headers={"Authorization": "Bearer from-env"}).status_code == 200


def test_cli_fails_clearly_for_a_missing_model(launched, tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        serving.main([str(tmp_path), "--no-onnx"])


# --- unsupported classifiers and a busy event loop -------------------------------------------------------

def test_a_multilabel_classifier_is_refused_with_a_clear_message(base_model):
    from adaptive_classifier import MultiLabelAdaptiveClassifier

    multi = MultiLabelAdaptiveClassifier(base_model, device="cpu", use_onnx=False)
    with pytest.raises(ValueError, match="MultiLabelAdaptiveClassifier"):
        create_app(multi)


def test_saving_after_an_update_does_not_block_other_requests(clf, tmp_path, monkeypatch):
    """A slow save (ONNX export takes seconds) used to freeze /health and /info while it ran."""
    import time

    real_save = clf.save

    def slow_save(path, *args, **kwargs):
        time.sleep(1.5)
        return real_save(path, *args, **kwargs)

    monkeypatch.setattr(clf, "save", slow_save)
    client = updatable(clf, save_dir=str(tmp_path / "saved"))
    done = {}

    def update():
        done["response"] = client.post("/examples", json={"texts": ["tr0_0"], "labels": ["C0"]})

    worker = threading.Thread(target=update)
    worker.start()
    time.sleep(0.3)                                   # the update is now inside the slow save
    started = time.perf_counter()
    assert client.get("/health").status_code == 200
    assert client.get("/info").status_code == 200
    elapsed = time.perf_counter() - started
    worker.join()

    assert done["response"].status_code == 200
    assert elapsed < 1.0, f"/health and /info waited {elapsed:.2f}s behind a save"
