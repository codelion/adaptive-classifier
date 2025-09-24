<div align="center">
  <img src="adaptive-classifier-logo.webp" alt="Adaptive Classifier Logo" width="300"/>
  
  <h1>Adaptive Classifier</h1>
  <p><strong>🚀 Dynamic text classification with continuous learning, strategic defense, and zero-downtime adaptation</strong></p>
  
  [![PyPI - Version](https://img.shields.io/pypi/v/adaptive-classifier)](https://pypi.org/project/adaptive-classifier/)
  [![PyPI - Downloads](https://img.shields.io/pypi/dm/adaptive-classifier)](https://pypi.org/project/adaptive-classifier/)
  [![GitHub Stars](https://img.shields.io/github/stars/codelion/adaptive-classifier)](https://github.com/codelion/adaptive-classifier/stargazers)
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![GitHub Discussions](https://img.shields.io/github/discussions/codelion/adaptive-classifier)](https://github.com/codelion/adaptive-classifier/discussions)

</div>

---

## 🔗 Quick Links

- 📚 **[HuggingFace Organization](https://huggingface.co/adaptive-classifier)** - Pre-trained models and datasets
- 📖 **Articles & Tutorials:**
  - [Adaptive Classifier: Dynamic Text Classification](https://huggingface.co/blog/codelion/adaptive-classifier)
  - [AutoThink: Advanced Reasoning Techniques](https://huggingface.co/blog/codelion/autothink)
  - [Enterprise-Ready Classifiers](https://huggingface.co/blog/codelion/enterprise-ready-classifiers)

---

Adaptive Classifier is a PyTorch-based machine learning library that revolutionizes text classification with **continuous learning**, **dynamic class addition**, and **strategic defense against adversarial inputs**. Built on HuggingFace transformers, it enables zero-downtime model updates and enterprise-grade robustness.

## ✨ Key Features

### 🎯 **Core Capabilities**
- **🚀 Universal Compatibility** - Works with any HuggingFace transformer model
- **📈 Continuous Learning** - Add new examples without catastrophic forgetting
- **🔄 Dynamic Classes** - Add new classes at runtime without retraining
- **⚡ Zero Downtime** - Update models in production without service interruption

### 🛡️ **Advanced Defense**
- **🎮 Strategic Classification** - Game-theoretic defense against adversarial manipulation
- **🔒 Anti-Gaming Protection** - Robust predictions under strategic behavior
- **⚖️ Multiple Prediction Modes** - Regular, strategic, and robust inference options

### 🧠 **Intelligent Architecture** 
- **💾 Prototype Memory** - FAISS-powered efficient similarity search
- **🔬 Adaptive Neural Layer** - Trainable classification head with EWC protection
- **🎯 Hybrid Predictions** - Combines prototype similarity and neural network outputs
- **📊 HuggingFace Integration** - Push/pull models directly from the Hub

---

## 📊 Performance & Benchmarks

### 🛡️ Strategic Classification Defense
Tested on adversarial examples from AI-Secure/adv_glue dataset:

| Metric | Regular Classifier | Strategic Classifier | **Improvement** |
|--------|-------------------|---------------------|----------------|
| Clean Data Accuracy | 80.00% | **82.22%** | **+2.22%** |
| Adversarial Data Accuracy | 60.00% | **82.22%** | **+22.22%** |
| Robustness (vs attack) | -20.00% drop | **0.00% drop** | **Perfect** |

### 🔍 Hallucination Detection
Evaluated on RAGTruth benchmark across multiple task types:

| Task Type | Precision | Recall | **F1 Score** |
|-----------|-----------|--------|-------------|
| QA | 35.50% | 45.11% | 39.74% |
| Summarization | 22.18% | **96.91%** | 36.09% |
| Data-to-Text | **65.00%** | **100.0%** | **78.79%** |
| **Overall** | **40.89%** | **80.68%** | **51.54%** |

### 🚦 LLM Router Optimization
Tested on arena-hard-auto-v0.1 dataset (500 queries):

| Metric | Without Adaptation | With Adaptation | **Improvement** |
|--------|-------------------|----------------|----------------|
| Cost Savings | 25.60% | **32.40%** | **+6.80%** |
| Efficiency Ratio | 1.00x | **1.27x** | **+27%** |
| Resource Utilization | Standard | **Optimized** | **Better** |

> **Key Insight**: Adaptive classification maintains quality while significantly improving cost efficiency and robustness across all tested scenarios.

---

## Try Now

| Use Case | Demonstrates | Link |
|----------|----------|-------|
| Basic Example (Cat or Dog)  | Continuous learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Zmvtb3XUFtUImEmYdKpkuqmxKVlRxzt9?usp=sharing) |
| Support Ticket Classification| Realistic examples | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yeVCi_Cdx2jtM7HI0gbU6VlZDJsg_m8u?usp=sharing) |
| Query Classification  | Different configurations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b2q303CLDRQAkC65Rtwcoj09ovR0mGwz?usp=sharing) |
| Multilingual Sentiment Analysis | Ensemble of classifiers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14tfRi_DtL-QgjBMgVRrsLwcov-zqbKBl?usp=sharing) |
| Product Category Classification | Batch processing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VyxVubB8LXXES6qElEYJL241emkV_Wxc?usp=sharing) |
| Multi-label Classification | Extensibility | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MDL_45QWvGoM2N8NRfUQSy2J7HKmmTsv?usp=sharing) |

## 🚀 Installation

### Quick Install
```bash
pip install adaptive-classifier
```

### 🛠️ Development Setup
```bash
# Clone the repository
git clone https://github.com/codelion/adaptive-classifier.git
cd adaptive-classifier

# Install in development mode
pip install -e .

# Install test dependencies (optional)
pip install pytest pytest-cov pytest-randomly
```

---

## ⚡ Quick Start

### 30-Second Setup
Get started with adaptive classification in under 30 seconds:

```python
from adaptive_classifier import AdaptiveClassifier

# 🎯 Step 1: Initialize with any HuggingFace model
classifier = AdaptiveClassifier("bert-base-uncased")

# 📝 Step 2: Add training examples
texts = ["The product works great!", "Terrible experience", "Neutral about this purchase"]
labels = ["positive", "negative", "neutral"]
classifier.add_examples(texts, labels)

# 🔮 Step 3: Make predictions
predictions = classifier.predict("This is amazing!")
print(predictions)  
# Output: [('positive', 0.85), ('neutral', 0.12), ('negative', 0.03)]
```

### 🏷️ Multi-Label Classification

Classify texts into multiple categories simultaneously with automatic threshold adaptation:

```python
from adaptive_classifier import MultiLabelAdaptiveClassifier

# Initialize multi-label classifier
classifier = MultiLabelAdaptiveClassifier(
    "bert-base-uncased",
    min_predictions=1,    # Ensure at least 1 prediction
    max_predictions=5     # Limit to top 5 predictions
)

# Multi-label training data (each text can have multiple labels)
texts = [
    "AI researchers study climate change using machine learning",
    "Tech startup develops healthcare solutions"
]
labels = [
    ["technology", "science", "climate", "ai"],
    ["technology", "business", "healthcare"]
]

classifier.add_examples(texts, labels)

# Make multi-label predictions
predictions = classifier.predict_multilabel("Medical AI breakthrough announced")
# Output: [('healthcare', 0.72), ('technology', 0.68), ('ai', 0.45)]
```

### 💾 Save & Load Models

```python
# Save locally
classifier.save("./my_classifier")
loaded_classifier = AdaptiveClassifier.load("./my_classifier")

# 🤗 HuggingFace Hub Integration
classifier.push_to_hub("adaptive-classifier/my-model")
hub_classifier = AdaptiveClassifier.from_pretrained("adaptive-classifier/my-model")
```

### 🎮 Strategic Defense (Anti-Gaming)

```python
# Enable strategic classification for adversarial robustness
config = {'enable_strategic_mode': True}
strategic_classifier = AdaptiveClassifier("bert-base-uncased", config=config)

# Robust predictions against manipulation
predictions = strategic_classifier.predict("This product has amazing quality features!")
# Returns predictions that consider potential gaming attempts
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

### Multi-Label Classification with Advanced Configuration

```python
from adaptive_classifier import MultiLabelAdaptiveClassifier

# Configure advanced multi-label settings
classifier = MultiLabelAdaptiveClassifier(
    "bert-base-uncased",
    default_threshold=0.5,      # Base threshold for predictions
    min_predictions=1,          # Minimum labels to return
    max_predictions=10          # Maximum labels to return
)

# Training with diverse multi-label examples
texts = [
    "Scientists develop AI for medical diagnosis and climate research",
    "Tech company launches sustainable energy and healthcare products",
    "Olympic athletes use sports science and nutrition technology"
]
labels = [
    ["science", "ai", "healthcare", "research"],
    ["technology", "business", "environment", "healthcare"],
    ["sports", "science", "health", "technology"]
]

classifier.add_examples(texts, labels)

# Advanced prediction options
predictions = classifier.predict_multilabel(
    "New research on AI applications in environmental science",
    threshold=0.3,     # Custom threshold
    max_labels=5       # Limit results
)

# Get detailed statistics
stats = classifier.get_label_statistics()
print(f"Adaptive threshold: {stats['adaptive_threshold']}")
print(f"Label-specific thresholds: {stats['label_thresholds']}")
```

### Strategic Classification (Anti-Gaming)

```python
# Enable strategic mode to defend against adversarial inputs
config = {
    'enable_strategic_mode': True,
    'cost_function_type': 'linear',
    'cost_coefficients': {
        'sentiment_words': 0.5,    # Cost to change sentiment-bearing words
        'length_change': 0.1,      # Cost to modify text length
        'word_substitution': 0.3   # Cost to substitute words
    },
    'strategic_blend_regular_weight': 0.6,   # Weight for regular predictions
    'strategic_blend_strategic_weight': 0.4  # Weight for strategic predictions
}

classifier = AdaptiveClassifier("bert-base-uncased", config=config)
classifier.add_examples(texts, labels)

# Robust predictions that consider potential manipulation
text = "This product has amazing quality features!"

# Dual prediction (automatic blend of regular + strategic)
predictions = classifier.predict(text)

# Pure strategic prediction (simulates adversarial manipulation)
strategic_preds = classifier.predict_strategic(text)

# Robust prediction (assumes input may already be manipulated)
robust_preds = classifier.predict_robust(text)

print(f"Dual: {predictions}")
print(f"Strategic: {strategic_preds}")
print(f"Robust: {robust_preds}")
```

## 🏷️ Multi-Label Classification

The `MultiLabelAdaptiveClassifier` extends adaptive classification to handle scenarios where each text can belong to multiple categories simultaneously. It automatically handles threshold adaptation for scenarios with many labels.

### Key Features

- **🎯 Automatic Threshold Adaptation**: Dynamically adjusts thresholds based on the number of labels to prevent empty predictions
- **📊 Sigmoid Activation**: Uses proper multi-label architecture with BCE loss instead of softmax
- **⚙️ Configurable Limits**: Set minimum and maximum number of predictions per input
- **📈 Label-Specific Thresholds**: Automatically adjusts thresholds based on label frequency
- **🔄 Incremental Learning**: Add new labels and examples without retraining from scratch

### Usage

```python
from adaptive_classifier import MultiLabelAdaptiveClassifier

# Initialize with configuration
classifier = MultiLabelAdaptiveClassifier(
    "distilbert/distilbert-base-cased",
    default_threshold=0.5,
    min_predictions=1,
    max_predictions=5
)

# Multi-label training data
texts = [
    "Breaking: Scientists discover AI can help predict climate change patterns",
    "Tech giant announces breakthrough in quantum computing for healthcare",
    "Olympic committee adopts new sports technology for athlete performance"
]

labels = [
    ["science", "technology", "climate", "news"],
    ["technology", "healthcare", "quantum", "business"],
    ["sports", "technology", "performance", "news"]
]

# Train the classifier
classifier.add_examples(texts, labels)

# Make predictions
predictions = classifier.predict_multilabel(
    "Revolutionary medical AI system launched by tech startup"
)

# Results: [('technology', 0.85), ('healthcare', 0.72), ('business', 0.45)]
```

### Adaptive Thresholds

The classifier automatically adjusts prediction thresholds based on the number of labels:

| Number of Labels | Threshold | Benefit |
|-----------------|-----------|---------|
| 2-4 labels | 0.5 (default) | Standard precision |
| 5-9 labels | 0.4 (20% lower) | Balanced recall |
| 10-19 labels | 0.3 (40% lower) | Better coverage |
| 20-29 labels | 0.2 (60% lower) | Prevents empty results |
| 30+ labels | 0.1 (80% lower) | Ensures predictions |

This solves the common "No labels met the threshold criteria" issue when dealing with many-label scenarios.

---

## 🏢 Enterprise Use Cases

### 🔍 Hallucination Detection
Detect when LLMs generate information not supported by provided context (51.54% F1, 80.68% recall):

```python
detector = AdaptiveClassifier.from_pretrained("adaptive-classifier/llm-hallucination-detector")
context = "France is in Western Europe. Capital: Paris. Population: ~67 million."
response = "Paris is the capital. Population is 70 million."  # Contains hallucination

prediction = detector.predict(f"Context: {context}\nAnswer: {response}")
# Returns: [('HALLUCINATED', 0.72), ('NOT_HALLUCINATED', 0.28)]
```

### 🚦 Intelligent LLM Routing
Optimize costs by routing queries to appropriate model tiers (32.40% cost savings):

```python
router = AdaptiveClassifier.from_pretrained("adaptive-classifier/llm-router")
query = "Write a function to calculate Fibonacci sequence"

predictions = router.predict(query)
# Returns: [('HIGH', 0.92), ('LOW', 0.08)]
# Route to GPT-4 for complex tasks, GPT-3.5 for simple ones
```

### ⚙️ Configuration Optimization
Automatically predict optimal LLM settings (temperature, top_p) for different query types:

```python
config_optimizer = AdaptiveClassifier.from_pretrained("adaptive-classifier/llm-config-optimizer")
query = "Explain quantum physics concepts"

predictions = config_optimizer.predict(query)
# Returns: [('BALANCED', 0.85), ('CREATIVE', 0.10), ...]
# Automatically suggests temperature range: 0.6-1.0 for balanced responses
```

### 🛡️ Content Moderation
Deploy enterprise-ready classifiers for various moderation tasks:

```python
# Available pre-trained enterprise classifiers:
classifiers = [
    "adaptive-classifier/content-moderation",      # Content safety
    "adaptive-classifier/business-sentiment",      # Business communications
    "adaptive-classifier/pii-detection",           # Privacy protection
    "adaptive-classifier/fraud-detection",         # Financial security
    "adaptive-classifier/email-priority",          # Email routing
    "adaptive-classifier/compliance-classification" # Regulatory compliance
]

# Easy deployment
moderator = AdaptiveClassifier.from_pretrained("adaptive-classifier/content-moderation")
result = moderator.predict("User generated content here...")
```

> **💡 Pro Tip**: All enterprise models support continuous adaptation - add your domain-specific examples to improve performance over time.

---

## Architecture Overview

The Adaptive Classifier combines four key components in a unified architecture:

![Adaptive Classifier Architecture](docs/images/architecture.png)

1. **Transformer Embeddings**: Uses state-of-the-art language models for text representation

2. **Prototype Memory**: Maintains class prototypes for quick adaptation to new examples

3. **Adaptive Neural Layer**: Learns refined decision boundaries through continuous training

4. **Strategic Classification**: Defends against adversarial manipulation using game-theoretic principles. When strategic mode is enabled, the system:
   - Models potential strategic behavior of users trying to game the classifier
   - Uses cost functions to represent the difficulty of manipulating different features
   - Combines regular predictions with strategic-aware predictions for robustness
   - Provides multiple prediction modes: dual (blended), strategic (simulates manipulation), and robust (anti-manipulation)

## Why Adaptive Classification?

Traditional classification approaches face significant limitations when dealing with evolving requirements and adversarial environments:

![Traditional vs Adaptive Classification](docs/images/comparison.png)

The Adaptive Classifier overcomes these limitations through:
- **Dynamic class addition** without full retraining
- **Strategic robustness** against adversarial manipulation
- **Memory-efficient prototypes** with FAISS optimization
- **Zero downtime updates** for production systems
- **Game-theoretic defense** mechanisms

## Continuous Learning Process

The system evolves through distinct phases, each building upon previous knowledge without catastrophic forgetting:

![Continuous Learning Workflow](docs/images/continuous-learning-workflow.png)

The learning process includes:
- **Initial Training**: Bootstrap with basic classes
- **Dynamic Addition**: Seamlessly add new classes as they emerge
- **Continuous Learning**: Refine decision boundaries with EWC protection
- **Strategic Enhancement**: Develop robustness against manipulation
- **Production Deployment**: Full capability with ongoing adaptation

## Order Dependency in Online Learning

When using the adaptive classifier for true online learning (adding examples incrementally), be aware that the order in which examples are added can affect predictions. This is inherent to incremental neural network training.

### The Challenge

```python
# These two scenarios may produce slightly different models:

# Scenario 1
classifier.add_examples(["fish example"], ["aquatic"])
classifier.add_examples(["bird example"], ["aerial"])

# Scenario 2  
classifier.add_examples(["bird example"], ["aerial"])
classifier.add_examples(["fish example"], ["aquatic"])
```

While we've implemented sorted label ID assignment to minimize this effect, the neural network component still learns incrementally, which can lead to order-dependent behavior.

### Solution: Prototype-Only Predictions

For applications requiring strict order independence, you can configure the classifier to rely solely on prototype-based predictions:

```python
# Configure to use only prototypes (order-independent)
config = {
    'prototype_weight': 1.0,  # Use only prototypes
    'neural_weight': 0.0      # Disable neural network contribution
}

classifier = AdaptiveClassifier("bert-base-uncased", config=config)
```

With this configuration:
- Predictions are based solely on similarity to class prototypes (mean embeddings)
- Results are completely order-independent
- Trade-off: May have slightly lower accuracy than the hybrid approach

### Best Practices

1. **For maximum consistency**: Use prototype-only configuration
2. **For maximum accuracy**: Accept some order dependency with the default hybrid approach
3. **For production systems**: Consider batching updates and retraining periodically if strict consistency is required
4. **Model selection matters**: Some models (e.g., `google-bert/bert-large-cased`) may produce poor embeddings for single words. For better results with short inputs, consider:
   - `bert-base-uncased`
   - `sentence-transformers/all-MiniLM-L6-v2`
   - Or any model specifically trained for semantic similarity

---

## 🔗 Related Projects

- **[OpenEvolve](https://github.com/codelion/openevolve)** - Open-source evolutionary coding agent for algorithm discovery
- **[OptiLLM](https://github.com/codelion/optillm)** - Optimizing inference proxy with 20+ techniques for 2-10x accuracy improvements

## 🤝 Community & Contributing

- **🐛 Issues & Bug Reports**: [GitHub Issues](https://github.com/codelion/adaptive-classifier/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/codelion/adaptive-classifier/discussions)
- **📖 Documentation**: [API Reference](docs/API.md)
- **🛠️ Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

## 📚 References

- [Strategic Classification](https://arxiv.org/abs/1506.06980)
- [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665)
- [Transformer^2: Self-adaptive LLMs](https://arxiv.org/abs/2501.06252)
- [Lamini Classifier Agent Toolkit](https://www.lamini.ai/blog/classifier-agent-toolkit)
- [Protoformer: Embedding Prototypes for Transformers](https://arxiv.org/abs/2206.12710)
- [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
- [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://arxiv.org/abs/2401.00396)
- [LettuceDetect: A Hallucination Detection Framework for RAG Applications](https://arxiv.org/abs/2502.17125)

## 📜 Citation

If you use this library in your research, please cite:

```bibtex
@software{adaptive-classifier,
  title = {Adaptive Classifier: Dynamic Text Classification with Continuous Learning},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/adaptive-classifier}
}
```

---

<div align="center">
  <p><strong>Made with ❤️ by <a href="https://huggingface.co/adaptive-classifier">Adaptive Classifier Team</a></strong></p>
  <p>
    <a href="https://github.com/codelion/adaptive-classifier">⭐ Star us on GitHub</a> •
    <a href="https://huggingface.co/adaptive-classifier">🤗 Follow on HuggingFace</a>
  </p>
</div>
