# Adaptive Classifier

A flexible, adaptive classification system that allows for dynamic addition of new classes and continuous learning from examples. Built on top of transformers from HuggingFace, this library provides an easy-to-use interface for creating and updating text classifiers.

[![GitHub Discussions](https://img.shields.io/github/discussions/codelion/adaptive-classifier)](https://github.com/codelion/adaptive-classifier/discussions)

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
| Multi-label Classification | Extensibility | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MDL_45QWvGoM2N8NRfUQSy2J7HKmmTsv?usp=sharing) |

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

## Adaptive Classification with LLMs

### Hallucination Detector

The adaptive classifier can detect hallucinations in language model outputs, especially in Retrieval-Augmented Generation (RAG) scenarios. Despite incorporating external knowledge sources, LLMs often still generate content that isn't supported by the provided context. Our hallucination detector identifies when a model's output contains information that goes beyond what's present in the source material.

The classifier categorizes text into:

- **HALLUCINATED**: Output contains information not supported by or contradictory to the provided context
- **NOT_HALLUCINATED**: Output is faithfully grounded in the provided context

Our hallucination detector has been trained and evaluated on the RAGTruth benchmark, which provides a standardized dataset for assessing hallucination detection across different task types:

#### Performance Across Tasks

| Task Type      | Precision | Recall | F1 Score |
|----------------|-----------|--------|----------|
| QA             | 35.50%    | 45.11% | 39.74%   |
| Summarization  | 22.18%    | 96.91% | 36.09%   |
| Data-to-Text   | 65.00%    | 100.0% | 78.79%   |
| **Overall**    | **40.89%**| **80.68%** | **51.54%** |

The detector shows particularly high recall (80.68% overall), making it effective at catching potential hallucinations, with strong performance on data-to-text generation tasks. The adaptive nature of the classifier means it continues to improve as it processes more examples, making it ideal for production environments where user feedback can be incorporated.

```python
from adaptive_classifier import AdaptiveClassifier

# Load the hallucination detector
detector = AdaptiveClassifier.from_pretrained("adaptive-classifier/llm-hallucination-detector")

# Detect hallucinations in RAG output
context = "France is a country in Western Europe. Its capital is Paris. The population of France is about 67 million people."
query = "What is the capital of France and its population?"
response = "The capital of France is Paris. The population is 70 million."

# Format input as expected by the model
input_text = f"Context: {context}\nQuestion: {query}\nAnswer: {response}"

# Get hallucination prediction
prediction = detector.predict(input_text)
# Returns: [('HALLUCINATED', 0.72), ('NOT_HALLUCINATED', 0.28)]

# Example handling logic
if prediction[0][0] == 'HALLUCINATED' and prediction[0][1] > 0.6:
    print("Warning: Response may contain hallucinations")
    # Implement safeguards: request human review, add disclaimer, etc.
```

This system can be integrated into RAG pipelines as a safety layer, LLM evaluation frameworks, or content moderation systems. The ability to detect hallucinations helps build more trustworthy AI systems, particularly for applications in domains like healthcare, legal, finance, and education where factual accuracy is critical.

The detector can be easily fine-tuned on domain-specific data, making it adaptable to specialized use cases where the definition of hallucination may differ from general contexts.

### LLM Configuration Optimization

The adaptive classifier can also be used to predict optimal configurations for Language Models. Our research shows that model configurations, particularly temperature settings, can significantly impact response quality. Using the adaptive classifier, we can automatically predict the best temperature range for different types of queries:

- **DETERMINISTIC** (T: 0.0-0.1): For queries requiring precise, factual responses
- **FOCUSED** (T: 0.2-0.5): For structured, technical responses with slight flexibility
- **BALANCED** (T: 0.6-1.0): For natural, conversational responses
- **CREATIVE** (T: 1.1-1.5): For more varied and imaginative outputs
- **EXPERIMENTAL** (T: 1.6-2.0): For maximum variability and unconventional responses

Our evaluation on the LLM Arena dataset demonstrates:
- 69.8% success rate in finding optimal configurations
- Consistent response quality (avg. similarity score: 0.64)
- Balanced distribution across temperature classes, with each class finding its appropriate use cases
- BALANCED and CREATIVE temperatures producing the most reliable results (scores: 0.649 and 0.645)

This classifier can be used to automatically optimize LLM configurations based on query characteristics, leading to more consistent and higher-quality responses while reducing the need for manual configuration tuning.

```python
from adaptive_classifier import AdaptiveClassifier

# Load the configuration optimizer
classifier = AdaptiveClassifier.from_pretrained("adaptive-classifier/llm-config-optimizer")

# Get optimal temperature class for a query
predictions = classifier.predict("Your query here")
# Returns: [('BALANCED', 0.85), ('CREATIVE', 0.10), ...]
```

The classifier continuously learns from new examples, adapting its predictions as it processes more queries and observes their performance.

### LLM Router

The adaptive classifier can be used to intelligently route queries between different LLM models based on query complexity and requirements. The classifier learns to categorize queries into:

- **HIGH**: Complex queries requiring advanced reasoning, multi-step problem solving, or deep expertise. Examples include:
  - Code generation and debugging
  - Complex analysis tasks
  - Multi-step mathematical problems
  - Technical explanations
  - Creative writing tasks

- **LOW**: Straightforward queries that can be handled by smaller, faster models. Examples include:
  - Simple factual questions
  - Basic clarifications
  - Formatting tasks
  - Short definitions
  - Basic sentiment analysis

The router can be used to optimize costs and latency while maintaining response quality:

```python
from adaptive_classifier import AdaptiveClassifier

# Load the router classifier
classifier = AdaptiveClassifier.from_pretrained("adaptive-classifier/llm-router")

# Get routing prediction for a query
predictions = classifier.predict("Write a function to calculate the Fibonacci sequence")
# Returns: [('HIGH', 0.92), ('LOW', 0.08)]

# Example routing logic
def route_query(query: str, classifier: AdaptiveClassifier):
    predictions = classifier.predict(query)
    top_class = predictions[0][0]
    
    if top_class == 'HIGH':
        return use_advanced_model(query)  # e.g., GPT-4
    else:
        return use_basic_model(query)     # e.g., GPT-3.5-Turbo
```

We evaluate the effectiveness of adaptive classification in optimizing LLM routing decisions. Using the arena-hard-auto-v0.1 dataset with 500 queries, we compared routing performance with and without adaptation while maintaining consistent overall success rates.

#### Key Results

| Metric | Without Adaptation | With Adaptation | Impact |
|--------|-------------------|-----------------|---------|
| High Model Routes | 113 (22.6%) | 98 (19.6%) | 0.87x |
| Low Model Routes | 387 (77.4%) | 402 (80.4%) | 1.04x |
| High Model Success Rate | 40.71% | 29.59% | 0.73x |
| Low Model Success Rate | 16.54% | 20.15% | 1.22x |
| Overall Success Rate | 22.00% | 22.00% | 1.00x |
| Cost Savings* | 25.60% | 32.40% | 1.27x |

*Cost savings calculation assumes high-cost model is 2x the cost of low-cost model

#### Analysis

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
- [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://arxiv.org/abs/2401.00396)
- [LettuceDetect: A Hallucination Detection Framework for RAG Applications](https://arxiv.org/abs/2502.17125)

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
