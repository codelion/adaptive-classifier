import pytest
import torch
import numpy as np
from adaptive_classifier.memory import PrototypeMemory
from adaptive_classifier.models import Example, ModelConfig

@pytest.fixture
def memory():
    return PrototypeMemory(embedding_dim=768)

@pytest.fixture
def example_embedding():
    return torch.randn(768)

@pytest.fixture
def config():
    return ModelConfig({
        'max_examples_per_class': 5,
        'prototype_update_frequency': 3,
        'similarity_threshold': 0.95
    })

def test_initialization(memory):
    assert memory.embedding_dim == 768
    assert len(memory.examples) == 0
    assert len(memory.prototypes) == 0
    assert memory.index is not None

def test_add_example(memory, example_embedding):
    example = Example("test text", "positive", example_embedding)
    memory.add_example(example, "positive")
    
    assert len(memory.examples["positive"]) == 1
    assert "positive" in memory.prototypes
    assert memory.updates_since_rebuild == 1

def test_prototype_update(memory, example_embedding):
    # Add multiple examples
    examples = [
        Example(f"text_{i}", "positive", example_embedding + i)
        for i in range(3)
    ]
    
    for ex in examples:
        memory.add_example(ex, "positive")
    
    # Check if prototype is mean of embeddings
    expected_prototype = torch.stack([ex.embedding for ex in examples]).mean(dim=0)
    actual_prototype = memory.prototypes["positive"]
    
    assert torch.allclose(expected_prototype, actual_prototype)

def test_nearest_prototypes(memory, example_embedding):
    # Add examples for different classes
    classes = ["positive", "negative", "neutral"]
    
    for cls in classes:
        example = Example(f"text_{cls}", cls, example_embedding + len(cls))
        memory.add_example(example, cls)
    
    # Query
    query = example_embedding + 1  # Should be closest to "positive"
    results = memory.get_nearest_prototypes(query, k=3)
    
    assert len(results) == 3
    assert all(isinstance(label, str) and isinstance(score, float) 
              for label, score in results)

def test_memory_pruning(memory, example_embedding, config):
    memory = PrototypeMemory(embedding_dim=768, config=config)
    
    # Add more examples than max_examples_per_class
    num_examples = config.max_examples_per_class + 3
    
    for i in range(num_examples):
        example = Example(
            f"text_{i}",
            "positive",
            example_embedding + i
        )
        memory.add_example(example, "positive")
    
    # Check if pruning occurred
    assert len(memory.examples["positive"]) == config.max_examples_per_class

def test_index_rebuild(memory, example_embedding, config):
    memory = PrototypeMemory(embedding_dim=768, config=config)
    
    # Add examples until rebuild
    for i in range(config.prototype_update_frequency + 1):
        example = Example(
            f"text_{i}",
            "positive",
            example_embedding + i
        )
        memory.add_example(example, "positive")
    
    # Check if rebuild occurred
    assert memory.updates_since_rebuild == 0
    assert len(memory.label_to_index) == 1
    assert len(memory.index_to_label) == 1

def test_clear_memory(memory, example_embedding):
    # Add some examples
    example = Example("test text", "positive", example_embedding)
    memory.add_example(example, "positive")
    
    # Clear memory
    memory.clear()
    
    # Check if everything is cleared
    assert len(memory.examples) == 0
    assert len(memory.prototypes) == 0
    assert len(memory.label_to_index) == 0
    assert len(memory.index_to_label) == 0
    assert memory.updates_since_rebuild == 0

def test_get_stats(memory, example_embedding):
    # Add examples
    classes = ["positive", "negative"]
    for cls in classes:
        for i in range(3):
            example = Example(
                f"text_{cls}_{i}",
                cls,
                example_embedding + i
            )
            memory.add_example(example, cls)
    
    stats = memory.get_stats()
    
    assert stats['num_classes'] == 2
    assert stats['examples_per_class']['positive'] == 3
    assert stats['examples_per_class']['negative'] == 3
    assert stats['total_examples'] == 6
    assert stats['prototype_dimensions'] == 768

def test_memory_device_handling(memory, example_embedding):
    if torch.cuda.is_available():
        # Test with GPU tensors
        gpu_embedding = example_embedding.cuda()
        example = Example("test text", "positive", gpu_embedding)
        memory.add_example(example, "positive")
        
        # Check if prototype is on CPU for FAISS compatibility
        assert memory.prototypes["positive"].device == torch.device('cpu')

def test_invalid_input_handling(memory):
    with pytest.raises(ValueError):
        # Try to add example with wrong embedding dimension
        wrong_embedding = torch.randn(100)  # Wrong dimension
        example = Example("test text", "positive", wrong_embedding)
        memory.add_example(example, "positive")

def test_prototype_stability(memory, example_embedding):
    # Add same example multiple times
    example = Example("test text", "positive", example_embedding)
    
    for _ in range(5):
        memory.add_example(example, "positive")
    
    # Prototype should be same as embedding
    assert torch.allclose(
        memory.prototypes["positive"],
        example_embedding,
        atol=1e-6
    )

def test_memory_efficiency(memory, example_embedding):
    # Monitor memory usage while adding many examples
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Add many examples
    for i in range(1000):
        example = Example(
            f"text_{i}",
            "positive",
            example_embedding + i
        )
        memory.add_example(example, "positive")
    
    final_memory = process.memory_info().rss
    memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
    
    # Memory growth should be reasonable (adjust threshold as needed)
    assert memory_growth < 1000  # Less than 1GB growth

def test_concurrent_access(memory, example_embedding):
    import threading
    
    def add_examples(label):
        for i in range(100):
            example = Example(
                f"text_{label}_{i}",
                label,
                example_embedding + i
            )
            memory.add_example(example, label)
    
    # Create threads for concurrent access
    threads = []
    labels = ["positive", "negative", "neutral"]
    
    for label in labels:
        thread = threading.Thread(target=add_examples, args=(label,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify results
    stats = memory.get_stats()
    assert stats['num_classes'] == 3
    for label in labels:
        assert label in memory.examples
        assert len(memory.examples[label]) == 100

if __name__ == "__main__":
    pytest.main([__file__])