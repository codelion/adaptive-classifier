"""Guard the CI model prefetch list against drift.

CI downloads models once (scripts/prefetch_test_models.py) and runs the suite
offline. A test that names a Hub model missing from that list would pass
locally and fail in CI with an obscure offline error, so catch it here.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HUB_NAME = re.compile(r"""["']((?:[\w.-]+/)?[\w.-]*(?:bert|BERT|MiniLM)[\w.-]*)["']""")
# Local fixtures and names that are not Hub repos.
IGNORED = {"tiny_bert"}


def _prefetch_models():
    spec = importlib.util.spec_from_file_location(
        "prefetch_test_models", ROOT / "scripts" / "prefetch_test_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.MODELS)


def _models_used_by_tests():
    used = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        if path.name == Path(__file__).name:
            continue
        used |= {m for m in HUB_NAME.findall(path.read_text(encoding="utf-8"))}
    return used - IGNORED


def test_every_model_named_in_tests_is_prefetched():
    missing = _models_used_by_tests() - _prefetch_models()
    assert not missing, (
        f"Add {sorted(missing)} to MODELS in scripts/prefetch_test_models.py, "
        "or CI (which runs offline) cannot load them"
    )


def test_prefetch_list_has_no_unused_models():
    unused = _prefetch_models() - _models_used_by_tests()
    assert not unused, f"Remove {sorted(unused)} from scripts/prefetch_test_models.py"


def test_tests_use_canonical_model_ids():
    """Short aliases such as "bert-base-uncased" cannot be prefetched: the Hub's
    download-token endpoint returns 404 for them."""
    aliases = {m for m in _models_used_by_tests() if "/" not in m}
    assert not aliases, f"Use the canonical owner/name id instead of {sorted(aliases)}"
