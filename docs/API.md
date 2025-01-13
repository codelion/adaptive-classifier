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
- `labels`: Optional list of labels to clear. If None, clears all memory

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
