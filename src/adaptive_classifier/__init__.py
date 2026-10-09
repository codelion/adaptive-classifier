from .classifier import AdaptiveClassifier
from .models import Example, AdaptiveHead, ModelConfig
from .memory import PrototypeMemory
from .multilabel import MultiLabelAdaptiveClassifier, MultiLabelAdaptiveHead
from .sklearn_api import SklearnAdaptiveClassifier
from huggingface_hub import ModelHubMixin

__version__ = "0.3.1"

__all__ = [
    "AdaptiveClassifier",
    "MultiLabelAdaptiveClassifier",
    "MultiLabelAdaptiveHead",
    "SklearnAdaptiveClassifier",
    "Example",
    "AdaptiveHead",
    "ModelConfig",
    "PrototypeMemory"
]