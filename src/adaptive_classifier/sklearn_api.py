"""scikit-learn compatible interface to AdaptiveClassifier.

`SklearnAdaptiveClassifier` takes a sequence of strings as `X`, so it drops into
`Pipeline`, `cross_val_score` and `GridSearchCV`, and its `partial_fit` maps onto
the library's strength: adding examples and new classes without retraining.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .classifier import AdaptiveClassifier


class SklearnAdaptiveClassifier(ClassifierMixin, BaseEstimator):
    """Text classifier with the scikit-learn estimator API.

    Parameters
    ----------
    model_name : str
        Hugging Face model (or local directory) used as the encoder.
    config : dict, optional
        Passed to `AdaptiveClassifier` (pooling, prototype_weight, ...).
    device : str, optional
        "cpu" or "cuda"; chosen automatically when None.
    use_onnx : bool or "auto", default="auto"
        Whether to use ONNX Runtime for the encoder.
    trust_remote_code : bool, default=False
        Forwarded to the Hugging Face loaders.

    Attributes
    ----------
    classes_ : ndarray
        Sorted class labels seen so far. Labels may be strings or integers and
        are returned as given.
    classifier_ : AdaptiveClassifier
        The fitted underlying classifier. Use it for features that have no
        scikit-learn equivalent: `save`, `push_to_hub`, `forget`,
        `remove_examples`, `is_ood`.

    Notes
    -----
    `fit` starts from scratch each time (it reloads the encoder, so
    cross-validation pays that cost per fold). `partial_fit` adds to what is
    already learned. A fitted estimator can be pickled, but that copies the
    whole encoder; persist `classifier_` with `classifier_.save(...)` instead.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        use_onnx: Union[bool, str] = "auto",
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.config = config
        self.device = device
        self.use_onnx = use_onnx
        self.trust_remote_code = trust_remote_code

    # --- scikit-learn plumbing -------------------------------------------------

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        return tags

    def _more_tags(self):  # scikit-learn < 1.6
        return {"X_types": ["string"]}

    # --- input handling ------------------------------------------------------------

    @staticmethod
    def _as_texts(X) -> List[str]:
        array = np.asarray(X, dtype=object)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError(
                "X must be a 1-D sequence of strings (or a single column); "
                f"got an array of shape {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError("X is empty")
        texts = array.tolist()
        bad = next((t for t in texts if not isinstance(t, str)), None)
        if bad is not None:
            raise ValueError(f"X must contain only strings; found {type(bad).__name__}")
        return texts

    def _as_labels(self, y, n_samples: int) -> List[Any]:
        labels = np.asarray(y, dtype=object)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {labels.shape}")
        if labels.shape[0] != n_samples:
            raise ValueError(
                f"X and y have inconsistent lengths: {n_samples} != {labels.shape[0]}"
            )
        return [label.item() if isinstance(label, np.generic) else label
                for label in labels.tolist()]

    def _new_classifier(self) -> AdaptiveClassifier:
        return AdaptiveClassifier(
            self.model_name,
            device=self.device,
            config=self.config,
            use_onnx=self.use_onnx,
            trust_remote_code=self.trust_remote_code,
        )

    def _refresh_classes(self):
        # The wrapped classifier keys everything by str(label); keep the
        # caller's original label objects so predict returns what was passed in.
        # Classes declared through partial_fit(classes=...) are listed even
        # before an example of them has been seen (scikit-learn's contract).
        originals = list({**self._declared_labels, **self._original_labels}.values())
        try:
            ordered = sorted(originals)
        except TypeError:  # mixed label types
            ordered = sorted(originals, key=str)
        self.classes_ = np.array(ordered)

    def _register_labels(self, labels: List[Any], known: Dict[str, Any]):
        """Check labels can be told apart once turned into strings, and record them in `known`."""
        for label in labels:
            if label is None or (isinstance(label, float) and label != label):
                raise ValueError("y must not contain None or NaN labels")
            key = str(label)
            previous = known.get(key, label)
            if (type(previous), previous) != (type(label), label):
                raise ValueError(
                    f"labels {previous!r} and {label!r} are different but both map to the class "
                    f"{key!r}; the classifier identifies classes by their string form"
                )
            known[key] = label

    # --- fitting -------------------------------------------------------------------

    def fit(self, X, y):
        """Fit from scratch on texts `X` and labels `y`."""
        texts = self._as_texts(X)
        labels = self._as_labels(y, len(texts))
        known: Dict[str, Any] = {}
        self._register_labels(labels, known)
        self.classifier_ = self._new_classifier()
        self._original_labels = {}
        self._declared_labels = {}
        self._add(texts, labels)
        return self

    def partial_fit(self, X, y, classes=None):
        """Add examples (and any new classes) to what is already learned.

        Unlike most scikit-learn estimators, new classes may appear in any
        call. If `classes` is given, every label in `y` must be one of them.
        """
        texts = self._as_texts(X)
        labels = self._as_labels(y, len(texts))
        declared: Dict[str, Any] = {}
        if classes is not None:
            declared_list = np.asarray(classes, dtype=object).tolist()
            self._register_labels(declared_list, declared)
            unexpected = set(labels) - set(declared_list)
            if unexpected:
                raise ValueError(f"y contains labels not in classes: {sorted(map(str, unexpected))}")
        if not hasattr(self, "classifier_"):
            self.classifier_ = self._new_classifier()
            self._original_labels = {}
            self._declared_labels = {}
        self._declared_labels.update(declared)
        self._add(texts, labels)
        return self

    def _add(self, texts: List[str], labels: List[Any]):
        # Validate against everything known so far before changing anything.
        known = {**self._declared_labels, **self._original_labels}
        self._register_labels(labels, known)
        self.classifier_.add_examples(texts, [str(label) for label in labels])
        self._original_labels.update({str(label): label for label in labels})
        self._refresh_classes()

    # --- prediction ----------------------------------------------------------------

    def predict_proba(self, X):
        """Class probabilities, columns ordered as `classes_`."""
        check_is_fitted(self, "classifier_")
        texts = self._as_texts(X)
        keys = [str(label) for label in self.classes_]
        column = {key: i for i, key in enumerate(keys)}
        proba = np.zeros((len(texts), len(keys)))
        batches = self.classifier_.predict_batch(texts, k=len(keys))
        for row, scores in enumerate(batches):
            for key, score in scores:
                if key in column:
                    proba[row, column[key]] = score
        totals = proba.sum(axis=1, keepdims=True)
        return np.divide(proba, totals, out=np.full_like(proba, 1.0 / len(keys)), where=totals > 0)

    def predict_log_proba(self, X):
        with np.errstate(divide="ignore"):
            return np.log(self.predict_proba(X))

    def predict(self, X):
        """Most likely class for each text."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
