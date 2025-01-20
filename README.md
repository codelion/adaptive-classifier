# Adaptive Classifier

A flexible, adaptive classification system that allows for dynamic addition of new classes and continuous learning from examples. Built on top of transformers from HuggingFace, this library provides an easy-to-use interface for creating and updating text classifiers.

## Features

- 🚀 Works with any transformer classifier model
- 📈 Continuous learning capabilities
- 🎯 Dynamic class addition
- 💾 Safe and efficient state persistence
- 🔄 Prototype-based learning
- 🧠 Neural adaptation layer

## Try Now

| Use Case | Demonstrates | Link |
|----------|----------|-------|
| Basic Example (Cat or Dog)  | Continuous learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Zmvtb3XUFtUImEmYdKpkuqmxKVlRxzt9?usp=sharing) |
| Support Ticket Classification| Realistic examples | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yeVCi_Cdx2jtM7HI0gbU6VlZDJsg_m8u?usp=sharing) |
| Query Classification  | Different configurations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b2q303CLDRQAkC65Rtwcoj09ovR0mGwz?usp=sharing) |
| Multilingual Sentiment Analysis | Ensemble of classifiers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14tfRi_DtL-QgjBMgVRrsLwcov-zqbKBl?usp=sharing) |
| Product Category Classification | Batch processing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VyxVubB8LXXES6qElEYJL241emkV_Wxc?usp=sharing) |

## Installation

```bash
pip install adaptive-classifier
```

## Quick Start

```python
from adaptive_classifier import AdaptiveClassifier

# Initialize with any HuggingFace model
classifier = AdaptiveClassifier("bert-base-uncased")

# Add some examples
texts = [
    "The product works great!",
    "Terrible experience",
    "Neutral about this purchase"
]
labels = ["positive", "negative", "neutral"]

classifier.add_examples(texts, labels)

# Make predictions
predictions = classifier.predict("This is amazing!")
print(predictions)  # [('positive', 0.85), ('neutral', 0.12), ('negative', 0.03)]

# Save the classifier
classifier.save("./my_classifier")

# Load it later
loaded_classifier = AdaptiveClassifier.load("./my_classifier")
```

## Advanced Usage

### Adding New Classes Dynamically

```python
# Add a completely new class
new_texts = [
    "Error code 404 appeared",
    "System crashed after update"
]
new_labels = ["technical"] * 2

classifier.add_examples(new_texts, new_labels)
```

### Continuous Learning

```python
# Add more examples to existing classes
more_examples = [
    "Best purchase ever!",
    "Highly recommend this"
]
more_labels = ["positive"] * 2

classifier.add_examples(more_examples, more_labels)
```

## How It Works

The system combines three key components:

1. **Transformer Embeddings**: Uses state-of-the-art language models for text representation

2. **Prototype Memory**: Maintains class prototypes for quick adaptation to new examples

3. **Adaptive Neural Layer**: Learns refined decision boundaries through continuous training

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- transformers ≥ 4.30.0
- safetensors ≥ 0.3.1
- faiss-cpu ≥ 1.7.4 (or faiss-gpu for GPU support)

## SOTA results on benchmarks with adaptive-classifier

### LLM Router Classifier (Jan 25)

## References

- [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665)
- [Transformer^2: Self-adaptive LLMs](https://arxiv.org/abs/2501.06252)
- [Lamini Classifier Agent Toolkit](https://www.lamini.ai/blog/classifier-agent-toolkit)
- [Protoformer: Embedding Prototypes for Transformers](https://arxiv.org/abs/2206.12710)
- [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{adaptive_classifier,
  title = {Adaptive Classifier: Dynamic Text Classification with Continuous Learning},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/adaptive-classifier}
}
```
