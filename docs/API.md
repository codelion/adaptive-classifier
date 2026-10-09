# API Documentation

## AdaptiveClassifier

The main class that provides the adaptive classification functionality.

### Constructor

```python
AdaptiveClassifier(
    model_name: str,
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    use_onnx: Union[bool, str, None] = "auto",
    trust_remote_code: bool = False
)
```

Creates a new adaptive classifier instance.

Parameters:
- `model_name`: Name of the HuggingFace transformer model to use (e.g., "bert-base-uncased")
- `device`: Device to run the model on ("cuda" or "cpu"). If None, automatically detects GPU availability
- `config`: Optional configuration dictionary (see ModelConfig for details)
- `seed`: Random seed for initialization (default: 42)
- `use_onnx`: Run the encoder with ONNX Runtime: `"auto"` (default) uses ONNX on CPU and PyTorch on GPU, `True`/`False` force it. With ONNX, `classifier.model` is an ONNX Runtime model, not a `torch.nn.Module`
- `trust_remote_code`: Allow models that ship custom code (default: False)

### Methods

#### add_examples

```python
def add_examples(texts: List[str], labels: List[str])
```

Add new examples to the classifier, automatically handling new classes.

Parameters:
- `texts`: List of text examples
- `labels`: List of corresponding labels

Raises:
- `ValueError`: If input lists are empty or have mismatched lengths

#### predict

```python
def predict(text: str, k: int = 5, abstain_below: Optional[float] = None,
            abstain_ood: bool = False) -> List[Tuple[str, float]]
```

Predict labels for a single text input.

Parameters:
- `text`: Input text to classify
- `k`: Number of top predictions to return (default: 5)
- `abstain_below`: Return an empty list when the top confidence is below this value
- `abstain_ood`: Return an empty list when the text is out of distribution (see `is_ood`)

Returns:
- List of (label, confidence) tuples, sorted by confidence

Raises:
- `ValueError`: If input text is empty

#### predict_batch

```python
def predict_batch(
    texts: List[str],
    k: int = 5,
    batch_size: int = 32
) -> List[List[Tuple[str, float]]]
```

Predict labels for multiple texts efficiently.

Parameters:
- `texts`: List of input texts
- `k`: Number of top predictions per text (default: 5)
- `batch_size`: Batch size for processing (default: 32)

Returns:
- List of prediction lists, where each prediction list contains (label, confidence) tuples

#### save

```python
def save(save_dir: str, include_onnx: bool = True, quantize_onnx: bool = True)
```

Save the classifier state to disk.

Parameters:
- `save_dir`: Directory to save the model state
- `include_onnx`: Also export the encoder as ONNX (needed to load without PyTorch and for non-Python runtimes)
- `quantize_onnx`: Also write an INT8 copy (`onnx/model_quantized.onnx`)

What a saved classifier keeps: every class prototype (averaged over all the examples the class was trained on), the neural head, label maps, training counts, class spreads and calibration, plus `num_representative_examples` examples per class (default 5). Not the full training set.

Consequences when you load and keep learning:
- Prototypes keep their saved position. New examples are folded in with the weight of the examples the prototype was saved from, so adding one example to a class trained on 30 barely moves it.
- The neural head is retrained from the examples held in memory, which after a load are only the representative ones. If the head matters for your data, keep the original training data and re-add it, or raise `num_representative_examples` before saving.
- `remove_examples` on a loaded class recomputes its prototype from the examples that are held.

#### load

```python
@classmethod
def load(cls, save_dir: str, device: Optional[str] = None, use_onnx: Union[bool, str, None] = "auto",
         prefer_quantized: bool = True, trust_remote_code: bool = False) -> 'AdaptiveClassifier'
```

Load a saved classifier from disk or from the Hugging Face Hub (`save_dir` may be a repo id). A saved `MultiLabelAdaptiveClassifier` loads as one.

Parameters:
- `save_dir`: Directory (or Hub repo id) containing the saved model state
- `device`: Optional device to load the model onto
- `use_onnx`: `"auto"` (default), `True` or `False`
- `prefer_quantized`: Use the INT8 ONNX copy when there is one (default: True)
- `trust_remote_code`: Allow models that ship custom code (default: False)

Returns:
- Loaded AdaptiveClassifier instance

#### to

```python
def to(device: str) -> 'AdaptiveClassifier'
```

Move the model to a specific device.

Parameters:
- `device`: Target device ("cuda" or "cpu")

Returns:
- Self for method chaining

#### clear_memory

```python
def clear_memory(labels: Optional[List[str]] = None)
```

Clear stored examples and prototypes.

Parameters:
- `labels`: Optional list of labels to clear. If given, this is the same as `forget(labels)` (unknown labels are ignored). If None, clears all stored examples and prototypes

#### calibrate

```python
def calibrate(texts: List[str], labels: List[str]) -> Dict[str, float]
```

Fit confidence calibration (temperature scaling) on labelled examples the model was not trained on. Afterwards `predict`, `predict_batch` and `abstain_below` use calibrated probabilities; the winning class never changes. Returns `temperature` (above 1 means the model was over-confident), `n`, `accuracy`, and `ece_before` / `ece_after` / `nll_before` / `nll_after`. Adding a new class or forgetting one discards the calibration.

#### calibration_report

```python
def calibration_report(texts, labels, calibrated=True, bins=10) -> Dict[str, float]
```

Accuracy, mean confidence, expected calibration error (`ece`) and `nll` on labelled data. Run it on a test set separate from the calibration set.

#### predict_set

```python
def predict_set(text: str, alpha: float = 0.1) -> List[Tuple[str, float]]
```

Split-conformal prediction set: contains the true label with probability at least `1 - alpha` for inputs that look like the calibration data (an average guarantee, not per input). Never empty. Requires `calibrate` first.

#### suggest_labels

```python
def suggest_labels(texts, n=10, strategy="margin", diverse=False) -> List[Tuple[int, float]]
```

Rank unlabeled texts by how useful a label would be. Strategies: `margin` (top two classes close together), `entropy`, `least_confidence`, and `ood` (far from every known class, for discovering new classes). With `diverse=True` texts similar to an earlier pick are discounted so the picks are spread out. Returns up to `n` `(index into texts, score)` pairs, best first; uses calibrated probabilities if `calibrate` was run.

#### drift_report

```python
def drift_report(texts, expected_ood_rate=0.05, alpha=0.01, threshold=None) -> Dict[str, Any]
```

Tests whether the share of out-of-distribution texts in a window of incoming data is higher than `expected_ood_rate` (one-sided binomial test). Returns `n`, `ood_rate`, `mean_ood_score`, `p_value` and `drifted` (`p_value < alpha`). A window of a few dozen texts cannot detect small shifts; use a stricter `alpha` when checking repeatedly.

#### forget

```python
def forget(labels: Union[str, List[str]])
```

Remove whole classes: their examples, prototype, label ids, training history and neural-head output. The remaining classes keep their learned head weights and no retraining happens. Raises `ValueError` for an unknown label.

#### remove_examples

```python
def remove_examples(texts: List[str], label: Optional[str] = None, retrain: bool = True) -> int
```

Remove training examples by exact text, for example to fix a wrong label. Prototypes are recomputed; a class left with no examples is removed. Returns the number of examples removed.

Parameters:
- `texts`: Texts to remove
- `label`: Restrict removal to one class; searches all classes if None
- `retrain`: Fine-tune the neural head on the remaining examples. This is fine-tuning, not retraining from scratch, so the head may keep some influence of the removed text

A classifier loaded from disk keeps only a few representative examples per class, so removing from it recomputes the prototype from those and shifts it more than removing from a live classifier (a warning is logged).

#### ood_score

```python
def ood_score(text: str) -> float
```

How far `text` is from the known classes: the distance to the nearest class prototype divided by that class's radius (the distance from its prototype to its farthest training example). Around 1 or below looks like known data; larger is further out. Returns `inf` for a classifier with no classes. Class radii are saved with the model.

#### is_ood

```python
def is_ood(text: str, threshold: Optional[float] = None) -> bool
```

True when `ood_score(text)` exceeds `threshold`, which defaults to the `ood_threshold` config value (1.25). `ood_min_radius` (default 0.05) sets a floor on class radii so single-example classes do not divide by zero. The right threshold depends on your encoder and data, so check it on held-out examples.

#### merge_classifiers

```python
def merge_classifiers(other: 'AdaptiveClassifier') -> 'AdaptiveClassifier'
```

Merge another classifier into this one.

Parameters:
- `other`: Another AdaptiveClassifier instance to merge

Returns:
- Self with merged data

Raises:
- `ValueError`: If classifiers have incompatible embedding dimensions

#### get_memory_stats

```python
def get_memory_stats() -> Dict[str, Any]
```

Get statistics about the memory system.

Returns:
- Dictionary containing memory statistics (number of examples, prototypes, etc.)

#### get_example_statistics

```python
def get_example_statistics() -> Dict[str, Any]
```

Get detailed statistics about stored examples and model state.

Returns:
- Dictionary containing comprehensive statistics (total examples, examples per class, memory usage, etc.)

## ModelConfig

Configuration class for the adaptive classifier.

### Constructor

```python
ModelConfig(config: Optional[Dict[str, Any]] = None)
```

Parameters:
- `config`: Optional configuration dictionary

### Attributes

Model Settings:
- `max_length`: Maximum sequence length (default: 512)
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Learning rate for training (default: 0.001)
- `warmup_steps`: Number of warmup steps (default: 0)

Memory Settings:
- `max_examples_per_class`: Maximum examples to store per class (default: 1000)
- `prototype_update_frequency`: Update frequency for prototypes (default: 100)
- `similarity_threshold`: Similarity threshold for matching (default: 0.6)

EWC Settings:
- `ewc_lambda`: Importance of old tasks (default: 100.0)
- `num_representative_examples`: Number of examples to keep for each class (default: 5)

Training Settings:
- `epochs`: Number of training epochs (default: 10)
- `early_stopping_patience`: Patience for early stopping (default: 3)
- `min_examples_per_class`: Minimum examples required per class (default: 3)

General Settings:
- `pooling`: How token embeddings become one vector: `"auto"` (default; the model's own sentence-transformers setting, falling back to mean), `"mean"` or `"cls"`. The resolved value is saved with the classifier

Prediction Settings:
- `prototype_weight`: Weight for prototype predictions (default: 0.7)
- `neural_weight`: Weight for neural network predictions (default: 0.3)
- `min_confidence`: Minimum confidence threshold (default: 0.1)
- `prototype_temperature`: Prototype scores are `softmax(-distance² / temperature)` (default: 0.25; `None` restores the scoring used before 0.3.0)
- `new_class_example_threshold`: Examples a class needs before the head has its full weight (default: 10)
- `new_class_prototype_weight` / `new_class_neural_weight`: Fixed split below that threshold (default: `None`, meaning the head's weight ramps up linearly; set both to use a fixed split)

Head Training Settings:
- `head_steps`: Minimum optimiser steps each time the head is trained (default: 300)
- `head_learning_rate`: Peak learning rate of the cosine schedule (default: 0.003)

Out-of-distribution Settings:
- `ood_threshold`: Distance-to-radius ratio above which `is_ood` is true (default: 1.25)
- `ood_min_radius`: Smallest class radius used for that ratio (default: 0.05)

Strategic Mode Settings (see Strategic mode below): `enable_strategic_mode`, `cost_function_type`, `cost_coefficients`, `strategic_lambda`, `strategic_training_frequency`, `strategic_blend_regular_weight`, `strategic_blend_strategic_weight`, `strategic_robust_proto_weight`, `strategic_robust_head_weight`, `strategic_prediction_proto_weight`, `strategic_prediction_head_weight`.

Device Settings:
- `device_map`: Device mapping strategy (default: 'auto')
- `quantization`: Quantization settings (default: None)
- `gradient_checkpointing`: Whether to use gradient checkpointing (default: False)

### Methods

#### update

```python
def update(**kwargs)
```

Update configuration parameters.

Parameters:
- `**kwargs`: Keyword arguments with new parameter values

#### to_dict

```python
def to_dict() -> Dict[str, Any]
```

Convert configuration to dictionary.

Returns:
- Dictionary containing all configuration parameters

## Example

Basic usage example:

```python
from adaptive_classifier import AdaptiveClassifier

# Initialize
classifier = AdaptiveClassifier("bert-base-uncased")

# Add examples
texts = [
    "Great product!",
    "Terrible experience",
    "Average performance"
]
labels = ["positive", "negative", "neutral"]
classifier.add_examples(texts, labels)

# Make predictions
prediction = classifier.predict("This is amazing!")
print(prediction)  # [('positive', 0.8), ('neutral', 0.15), ('negative', 0.05)]

# Save and load
classifier.save("./my_classifier")
loaded = AdaptiveClassifier.load("./my_classifier")
```

## Advanced Features

### Batch Processing

```python
# Process multiple texts efficiently
texts = ["Text 1", "Text 2", "Text 3"]
predictions = classifier.predict_batch(texts, k=3)
```

### Continuous Learning

```python
# Add new examples over time
new_texts = ["New example 1", "New example 2"]
new_labels = ["positive", "negative"]
classifier.add_examples(new_texts, new_labels)
```

### Dynamic Class Addition

```python
# Add completely new class
technical_texts = ["Error 404", "System crash"]
technical_labels = ["technical", "technical"]
classifier.add_examples(technical_texts, technical_labels)
```

### Memory Management

```python
# Get memory statistics
stats = classifier.get_memory_stats()
print(stats)

# Clear specific classes
classifier.clear_memory(labels=["technical"])
```

### Classifier Merging

```python
# Merge two classifiers
classifier1 = AdaptiveClassifier("bert-base-uncased")
classifier2 = AdaptiveClassifier("bert-base-uncased")
# ... train both classifiers ...
classifier1.merge_classifiers(classifier2)
```

## SklearnAdaptiveClassifier

```python
SklearnAdaptiveClassifier(model_name="sentence-transformers/all-MiniLM-L6-v2",
                          config=None, device=None, use_onnx="auto",
                          trust_remote_code=False)
```

scikit-learn estimator wrapping `AdaptiveClassifier`. `X` is a 1-D sequence of strings (a single column is also accepted); `y` holds string or integer labels, which `predict` returns unchanged.

| Method / attribute | Description |
|---|---|
| `fit(X, y)` | Fit from scratch; returns `self` |
| `partial_fit(X, y, classes=None)` | Add examples and any new classes to what is already learned. If `classes` is given, `y` must stay within it |
| `predict(X)` | Most likely class per text |
| `predict_proba(X)` | Probabilities, columns ordered as `classes_` |
| `predict_log_proba(X)` | Log of `predict_proba` |
| `score(X, y)` | Accuracy |
| `classes_` | Sorted labels seen so far |
| `classifier_` | The underlying fitted `AdaptiveClassifier` (`save`, `push_to_hub`, `forget`, `remove_examples`, `is_ood`, ...) |

Hyperparameters such as `prototype_weight` go inside `config`, so grid searches take `{"config": [{...}, {...}]}`. Fitted estimators can be pickled, but that copies the whole encoder; persist `classifier_` with `save` instead. `partial_fit(X, y, classes=[...])` lists every declared class in `classes_` even before an example of it has been seen, and labels that would become the same class name (`1` and `"1"`), `None` and `NaN` are rejected.

## MultiLabelAdaptiveClassifier

A text can have several labels. Subclass of `AdaptiveClassifier`; the head uses sigmoid outputs and is trained with binary cross-entropy.

```python
from adaptive_classifier import MultiLabelAdaptiveClassifier

clf = MultiLabelAdaptiveClassifier(
    "sentence-transformers/all-MiniLM-L6-v2",
    default_threshold=0.5,   # between 0 and 1; lowered automatically as the number of labels grows
    min_predictions=1,       # return at least this many labels, even below the threshold
    max_predictions=None,    # hard cap on labels returned (None = no cap)
)
clf.add_examples(["new GPU benchmarks", "vaccine trial results"],
                 [["technology", "hardware"], ["health", "science"]])   # one list of labels per text
clf.predict_multilabel("a study of AI in medicine", threshold=0.3, max_labels=5)
```

- `add_examples(texts, labels)`: `labels` holds one list of strings per text. A bare string instead of a list raises `ValueError`; a label repeated within a text counts once; a text with an empty list is skipped.
- `predict_multilabel(text, threshold=None, max_labels=None)`: labels whose score reaches their threshold, best first. `max_labels` is a hard cap (0 returns nothing) and also bounds the `min_predictions` top-up. Per-label thresholds are adjusted for how common each label is.
- `predict(text, k=5)`: `predict_multilabel` limited to `k` labels.
- `forget`, `remove_examples`, `save` and `load` work as on `AdaptiveClassifier`; the settings above and the per-label thresholds are saved and restored.
- Adding a new label retrains the head on all stored examples, so a classifier loaded from disk (which keeps only representative examples per class) retrains on those.

## Strategic mode

Optional defence against inputs that are edited to game the classifier. Enable it with a config where `cost_coefficients` has one entry per embedding dimension:

```python
config = {
    "enable_strategic_mode": True,
    "cost_function_type": "linear",          # or "separable"
    "cost_coefficients": [0.3] * 768,        # embedding size of the encoder
    "strategic_blend_regular_weight": 0.6,
    "strategic_blend_strategic_weight": 0.4,
}
clf = AdaptiveClassifier("bert-base-uncased", config=config)
clf.strategic_mode   # False if the coefficients were unusable; the reason is logged
```

- `predict(text)`: blends the regular and strategic predictions with the two blend weights.
- `predict_strategic(text)`: predicts on the input as an adversary would most profitably move it.
- `predict_robust(text)`: assumes the input may already have been moved; leans on the prototypes (`strategic_robust_proto_weight` 0.8, `strategic_robust_head_weight` 0.2).
- `evaluate_strategic_robustness(texts, labels, gaming_levels=[0.0, 0.5, 1.0])`: accuracy when that share of inputs is gamed. `relative_robustness` is `nan` when accuracy without gaming is zero.

With strategic mode off, `predict_strategic` and `predict_robust` are the same as `predict`. Strategic scoring is not part of the portable reference in `docs/deployment.md`.
