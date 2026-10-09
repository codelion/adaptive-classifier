"""The Node.js port in examples/javascript must agree with `AdaptiveClassifier.predict`.

Skipped unless Node and the example's npm dependencies are installed
(`cd examples/javascript && npm install`; CI does this).
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from adaptive_classifier import AdaptiveClassifier

JS_DIR = Path(__file__).resolve().parent.parent / "examples" / "javascript"
TEXTS = ["great product love it", "works very good fast", "terrible awful service",
         "hate it broke bad", "okay fine the product", "it is okay fine"] * 3
LABELS = ["pos", "pos", "neg", "neg", "neu", "neu"] * 3
QUERIES = ["great fast product", "awful slow broke", "okay it is fine", "love the service",
           "great " * 40]  # longer than max_length below: exercises truncation

SCRIPT = """
import { PortableClassifier } from "./portable_inference.mjs";
const [dir, queries] = [process.argv[1], JSON.parse(process.argv[2])];
const classifier = await PortableClassifier.load(dir, { preferQuantized: false });
const out = [];
for (const q of queries) out.push(await classifier.predict(q, 3));
console.log(JSON.stringify(out));
"""

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (JS_DIR / "node_modules").exists(),
    reason="Node.js or examples/javascript/node_modules not available",
)


def _onnx_available():
    try:
        import optimum.onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _onnx_available(), reason="optimum[onnxruntime] not installed")
@pytest.mark.parametrize("pooling", ["mean", "cls"])
@pytest.mark.parametrize("config", [
    {},
    {"prototype_temperature": None},
    {"new_class_prototype_weight": 0.3, "new_class_neural_weight": 0.7},
    {"new_class_example_threshold": 4},
], ids=["default", "legacy-scoring", "fixed-new-class", "mixed-established"])
def test_node_port_matches_predict(base_model, tmp_path, pooling, config):
    clf = AdaptiveClassifier(base_model, config={"pooling": pooling, "max_length": 32, **config},
                             use_onnx=False, device="cpu")
    clf.add_examples(TEXTS, LABELS)
    clf.add_examples(["great product love it"] * 6, ["pos"] * 6)
    clf.save(str(tmp_path), include_onnx=True, quantize_onnx=False)

    result = subprocess.run(
        ["node", "--input-type=module", "-e", SCRIPT, str(tmp_path), json.dumps(QUERIES)],
        cwd=JS_DIR, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    got = json.loads(result.stdout.strip().splitlines()[-1])

    for query, ranked in zip(QUERIES, got):
        expected = dict(clf.predict(query, k=3))
        actual = {label: score for label, score in ranked}
        assert actual.keys() == expected.keys()
        for label in expected:
            assert actual[label] == pytest.approx(expected[label], abs=1e-4)
