# API Documentation

## AdaptiveClassifier

The main class that provides the adaptive classification functionality.

### Constructor

```python
AdaptiveClassifier(
    model_name: str,
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42
)
```

Creates a new adaptive classifier instance.

Parameters:
- `model_name`: Name of the HuggingFace transformer model to use (e.g., "bert-base-uncased")
- `device`: Device to run the model on ("cuda" or "cpu"). If None, automatically detects GPU availability
- `config`: Optional configuration dictionary (see ModelConfig for details)
- `seed`: Random seed for initialization (default: 42)

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
def predict(text: str, k: int = 5) -> List[Tuple[str, float]]
```

Predict labels for a single text input.

Parameters:
- `text`: Input text to classify
- `k`: Number of top predictions to return (default: 5)

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
def save(save_dir: str)
```

Save the classifier state to disk.

Parameters:
- `save_dir`: Directory to save the model state

#### load

```python
@classmethod
def load(cls, save_dir: str, device: Optional[str] = None) -> 'AdaptiveClassifier'
```

Load a saved classifier from disk.

Parameters:
- `save_dir`: Directory containing the saved model state
- `device`: Optional device to load the model onto

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

Prediction Settings:
- `prototype_weight`: Weight for prototype predictions (default: 0.7)
- `neural_weight`: Weight for neural network predictions (default: 0.3)
- `min_confidence`: Minimum confidence threshold (default: 0.1)

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

Hyperparameters such as `prototype_weight` go inside `config`, so grid searches take `{"config": [{...}, {...}]}`. Fitted estimators cannot be pickled (they hold a FAISS index); persist `classifier_` with `save`.
