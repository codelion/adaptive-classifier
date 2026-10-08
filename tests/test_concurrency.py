"""A classifier shared between threads (a web server) must stay consistent.

Predictions read the label maps, the prototypes and the neural head, which
`add_examples`, `forget` and friends rewrite together. Without a lock a reader
could see them half-updated and crash ("selected index k out of range"); the
stress test below reproduces that reliably when the lock is removed.
"""

import asyncio
import threading
import time

import pytest

from adaptive_classifier import AdaptiveClassifier

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


def run_threads(targets):
    errors = []

    def guarded(fn):
        def run():
            try:
                fn()
            except Exception as error:               # noqa: BLE001 - collecting for the assertion
                errors.append(f"{type(error).__name__}: {str(error)[:100]}")
        return run

    threads = [threading.Thread(target=guarded(fn)) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads), "a thread is stuck (deadlock?)"
    return errors


def test_predictions_survive_concurrent_class_changes(clf, data):
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            preds = clf.predict("tr0_0", k=10)
            assert all(score == score for _, score in preds)
            clf.predict_batch(["tr1_0", "tr2_1"], k=10)

    def writer():
        try:
            for round_ in range(8):
                extra, labels = data.classes(3, n_classes=3, noise=0.2, seed=60 + round_, prefix=f"w{round_}")
                clf.add_examples(extra, [f"N{round_}" if label == "C2" else label for label in labels])
                if round_ % 2:
                    clf.forget(f"N{round_}")
        finally:
            stop.set()

    errors = run_threads([reader, reader, reader, reader, writer])
    assert errors == []
    # The final state is internally consistent.
    n = len(clf.label_to_id)
    assert sorted(clf.label_to_id.values()) == list(range(n))
    assert clf.adaptive_head.model[-1].out_features == n
    assert set(clf.memory.prototypes) == set(clf.label_to_id)


def test_concurrent_adds_lose_no_examples(new_classifier, data):
    clf = new_classifier(head_steps=10)
    batches = []
    for t in range(6):
        X, y = data.classes(4, n_classes=2, noise=0.2, seed=200 + t, prefix=f"t{t}")
        batches.append((X, y))
    errors = run_threads([lambda b=b: clf.add_examples(*b) for b in batches])
    assert errors == []
    assert clf.training_history == {"C0": 24, "C1": 24}
    assert sum(len(v) for v in clf.memory.examples.values()) == 48


def test_the_lock_is_reentrant_so_methods_can_call_each_other(clf, data):
    Xc, yc = data.classes(20, n_classes=3, noise=0.3, seed=7, prefix="cal")
    clf.calibrate(Xc, yc)                          # calibrate -> predict, all under the lock
    assert clf.predict_set("tr0_0")                # predict_set -> predict


def test_each_classifier_has_its_own_lock(clf, new_classifier):
    other = new_classifier()
    assert clf._lock is clf._lock and clf._lock is not other._lock


def test_a_loaded_classifier_has_a_working_lock(clf, tmp_path):
    clf.save(str(tmp_path), include_onnx=False)
    loaded = AdaptiveClassifier.load(str(tmp_path), use_onnx=False, device="cpu")
    with loaded._lock:                             # built without __init__
        pass
    assert loaded.predict("tr0_0")


def test_a_failing_operation_releases_the_lock(clf):
    with pytest.raises(ValueError):
        clf.add_examples(["x"], [None])
    result = []
    t = threading.Thread(target=lambda: result.append(clf.predict("tr0_0")))
    t.start()
    t.join(timeout=10)
    assert result, "the lock was left held after an exception"


# --- async API ----------------------------------------------------------------------------------

def test_async_methods_match_the_sync_ones(clf, data):
    async def go():
        return (
            await clf.apredict("tr0_0", k=3),
            await clf.apredict_batch(["tr1_0", "tr2_0"], k=3),
        )
    single, batch = asyncio.run(go())
    assert single == clf.predict("tr0_0", k=3)
    assert batch == clf.predict_batch(["tr1_0", "tr2_0"], k=3)


def test_async_methods_pass_keyword_arguments_through(clf):
    top = clf.predict("tr0_0")[0][1]
    assert asyncio.run(clf.apredict("tr0_0", abstain_below=top + 0.01)) == []
    assert asyncio.run(clf.apredict("tr0_0", abstain_below=top - 0.01)) != []


def test_aadd_examples_adds_and_validates(clf, data):
    extra, labels = data.classes(3, n_classes=1, noise=0.2, seed=300, prefix="ax")
    asyncio.run(clf.aadd_examples(extra, ["C0"] * 3))
    assert clf.training_history["C0"] == 15
    with pytest.raises(ValueError, match="strings"):
        asyncio.run(clf.aadd_examples(["x"], [3]))


def test_async_calls_do_not_block_the_event_loop(clf, monkeypatch):
    original = clf.predict

    def slow(*args, **kwargs):
        time.sleep(0.4)
        return original(*args, **kwargs)

    monkeypatch.setattr(clf, "predict", slow)

    async def go():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        await clf.apredict("tr0_0")
        task.cancel()
        return ticks

    assert asyncio.run(go()) >= 8          # the loop kept running during the 0.4s call


def test_many_concurrent_async_predictions_all_complete(clf):
    async def go():
        return await asyncio.gather(*[clf.apredict(f"tr{i % 3}_{i % 5}") for i in range(40)])
    results = asyncio.run(go())
    assert len(results) == 40 and all(r for r in results)
