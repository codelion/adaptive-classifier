from .classifier import AdaptiveClassifier
from .models import Example, AdaptiveHead, ModelConfig
from .memory import PrototypeMemory

import os
import re

def get_version_from_setup():
    try:
        setup_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'setup.py')
        with open(setup_path, 'r') as f:
            content = f.read()
            version_match = re.search(r'version=["\']([^"\']+)["\']', content)
            if version_match:
                return version_match.group(1)
    except Exception:
        pass
    return "unknown"

__version__ = get_version_from_setup()

__all__ = [
    "AdaptiveClassifier",
    "Example",
    "AdaptiveHead",
    "ModelConfig",
    "PrototypeMemory"
]