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

# The library is also integrated with Hugging Face. So you can push and load from HF Hub.

# Save to Hub
classifier.push_to_hub("adaptive-classifier/model-name")

# Load from Hub
classifier = AdaptiveClassifier.from_pretrained("adaptive-classifier/model-name")
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

## Benefits of Adaptive Classification in LLM Routing

We evaluate the effectiveness of adaptive classification in optimizing LLM routing decisions. Using the arena-hard-auto-v0.1 dataset with 500 queries, we compared routing performance with and without adaptation while maintaining consistent overall success rates.

### Key Results

| Metric | Without Adaptation | With Adaptation | Impact |
|--------|-------------------|-----------------|---------|
| High Model Routes | 113 (22.6%) | 98 (19.6%) | 0.87x |
| Low Model Routes | 387 (77.4%) | 402 (80.4%) | 1.04x |
| High Model Success Rate | 40.71% | 29.59% | 0.73x |
| Low Model Success Rate | 16.54% | 20.15% | 1.22x |
| Overall Success Rate | 22.00% | 22.00% | 1.00x |
| Cost Savings* | 25.60% | 32.40% | 1.27x |

*Cost savings calculation assumes high-cost model is 2x the cost of low-cost model

### Analysis

The results highlight several key benefits of adaptive classification:

1. **Improved Cost Efficiency**: While maintaining the same overall success rate (22%), the adaptive classifier achieved 32.40% cost savings compared to 25.60% without adaptation - a relative improvement of 1.27x in cost efficiency.

2. **Better Resource Utilization**: The adaptive system routed more queries to the low-cost model (402 vs 387) while reducing high-cost model usage (98 vs 113), demonstrating better resource allocation.

3. **Learning from Experience**: Through adaptation, the system improved the success rate of low-model routes from 16.54% to 20.15% (1.22x increase), showing effective learning from successful cases.

4. **ROI on Adaptation**: The system adapted to 110 new examples during evaluation, leading to a 6.80% improvement in cost savings while maintaining quality - demonstrating significant return on the adaptation investment.

This real-world evaluation demonstrates that adaptive classification can significantly improve cost efficiency in LLM routing without compromising overall performance.

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
