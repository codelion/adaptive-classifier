"""Tests for Elastic Weight Consolidation (EWC) module."""

import pytest
import torch
import numpy as np
from adaptive_classifier import AdaptiveClassifier
from adaptive_classifier.ewc import EWC
import torch.nn as nn


@pytest.fixture
def simple_model():
    """Create a simple neural network for testing."""
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, num_classes=3):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes)
        
        def forward(self, x):
            return self.fc(x)
    
    return SimpleModel()


@pytest.fixture
def small_dataset():
    """Create a small dataset for testing."""
    # Create embeddings and labels
    embeddings = torch.randn(33, 10)  # 33 samples to test edge case
    labels = torch.tensor([0, 1, 2] * 11)  # 3 classes repeated
    return torch.utils.data.TensorDataset(embeddings, labels)


def test_ewc_single_batch_edge_case(simple_model, small_dataset):
    """Test EWC with dataset size that creates single-sample batch.
    
    This tests the fix for the squeeze() bug that occurred when
    the last batch had only 1 sample.
    """
    device = 'cpu'
    
    # This should not raise an error anymore
    ewc = EWC(
        simple_model,
        small_dataset,
        device=device,
        ewc_lambda=100.0
    )
    
    assert ewc is not None
    assert ewc.fisher_info is not None
    assert ewc.old_params is not None


def test_ewc_various_batch_sizes():
    """Test EWC with various dataset sizes to ensure robustness."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)
        
        def forward(self, x):
            return self.fc(x)
    
    # Test with different dataset sizes that create different batch scenarios
    test_sizes = [1, 31, 32, 33, 64, 65, 100]  # Various edge cases
    
    for size in test_sizes:
        model = SimpleModel()
        embeddings = torch.randn(size, 10)
        labels = torch.randint(0, 3, (size,))
        dataset = torch.utils.data.TensorDataset(embeddings, labels)
        
        # Should not raise any errors
        ewc = EWC(model, dataset, device='cpu', ewc_lambda=100.0)
        
        # Verify EWC was initialized properly
        assert ewc.fisher_info is not None
        assert len(ewc.fisher_info) > 0
        
        # Test EWC loss computation
        loss = ewc.ewc_loss(batch_size=32)
        assert loss is not None
        assert loss.item() >= 0  # Loss should be non-negative


def test_adaptive_classifier_with_many_classes():
    """Test AdaptiveClassifier with many classes (simulates Banking77 scenario)."""
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create classifier
    classifier = AdaptiveClassifier('distilbert-base-uncased', device='cpu')
    
    # Simulate many classes with few examples each
    num_classes = 20
    examples_per_class = 3
    
    texts = []
    labels = []
    
    for class_id in range(num_classes):
        class_name = f"class_{class_id}"
        for example_id in range(examples_per_class):
            texts.append(f"This is example {example_id} for {class_name}")
            labels.append(class_name)
    
    # Add examples in batches (this should trigger EWC when new classes appear)
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # This should not raise any errors
        classifier.add_examples(batch_texts, batch_labels)
    
    # Verify classifier works
    test_text = "This is a test example"
    predictions = classifier.predict(test_text, k=3)
    
    assert predictions is not None
    assert len(predictions) <= 3
    assert all(isinstance(p[0], str) for p in predictions)  # Labels are strings
    assert all(isinstance(p[1], float) for p in predictions)  # Scores are floats


def test_ewc_loss_computation(simple_model, small_dataset):
    """Test that EWC loss is computed correctly."""
    device = 'cpu'
    
    # Initialize EWC
    ewc = EWC(
        simple_model,
        small_dataset,
        device=device,
        ewc_lambda=100.0
    )
    
    # Modify model parameters slightly
    for param in simple_model.parameters():
        param.data += 0.1
    
    # Compute EWC loss
    loss = ewc.ewc_loss()
    
    # Loss should be positive since we changed parameters
    assert loss.item() > 0
    
    # Test with batch size normalization
    loss_normalized = ewc.ewc_loss(batch_size=32)
    assert loss_normalized.item() > 0
    assert loss_normalized.item() != loss.item()  # Should be different due to normalization


def test_progressive_class_addition():
    """Test adding classes progressively (triggers EWC multiple times)."""
    classifier = AdaptiveClassifier('distilbert-base-uncased', device='cpu')
    
    # Phase 1: Add initial classes
    phase1_texts = ["Good product", "Bad service", "Average quality"]
    phase1_labels = ["positive", "negative", "neutral"]
    classifier.add_examples(phase1_texts, phase1_labels)
    
    # Phase 2: Add new classes (should trigger EWC)
    phase2_texts = ["Need help", "Bug report", "Feature request"]
    phase2_labels = ["support", "bug", "feature"]
    classifier.add_examples(phase2_texts, phase2_labels)
    
    # Phase 3: Add more examples to existing classes
    phase3_texts = ["Excellent!", "Terrible!", "It's okay"]
    phase3_labels = ["positive", "negative", "neutral"]
    classifier.add_examples(phase3_texts, phase3_labels)
    
    # Phase 4: Add more new classes (should trigger EWC again)
    phase4_texts = ["Urgent issue", "Question about pricing"]
    phase4_labels = ["urgent", "inquiry"]
    classifier.add_examples(phase4_texts, phase4_labels)
    
    # Verify all classes are learned
    expected_classes = {"positive", "negative", "neutral", "support", 
                       "bug", "feature", "urgent", "inquiry"}
    
    for label in expected_classes:
        assert label in classifier.label_to_id
    
    # Test prediction
    test_text = "This is wonderful!"
    predictions = classifier.predict(test_text, k=3)
    assert predictions is not None
    assert len(predictions) > 0


def test_ewc_with_empty_batch_edge_case():
    """Test EWC handles edge cases gracefully."""
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    model = TinyModel()
    
    # Create a tiny dataset
    embeddings = torch.randn(1, 5)  # Single sample
    labels = torch.tensor([0])
    dataset = torch.utils.data.TensorDataset(embeddings, labels)
    
    # Should handle single sample without errors
    ewc = EWC(model, dataset, device='cpu', ewc_lambda=50.0)
    
    assert ewc is not None
    loss = ewc.ewc_loss()
    assert loss.item() >= 0