import pytest
import torch
import tempfile
import numpy as np
import random
from pathlib import Path
from adaptive_classifier import AdaptiveClassifier

@pytest.fixture
def base_classifier():
    return AdaptiveClassifier("bert-base-uncased")

@pytest.fixture
def sample_data():
    texts = [
        "This is amazing",
        "Terrible experience",
        "Just okay",
        "Love it",
        "Hate it"
    ]
    labels = [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative"
    ]
    return texts, labels

def test_initialization(base_classifier):
    assert base_classifier is not None
    assert hasattr(base_classifier, 'model')
    assert hasattr(base_classifier, 'tokenizer')
    assert hasattr(base_classifier, 'memory')

def test_adding_examples(base_classifier, sample_data):
    texts, labels = sample_data
    base_classifier.add_examples(texts, labels)
    
    # Check if examples were added correctly
    unique_labels = set(labels)
    for label in unique_labels:
        assert label in base_classifier.label_to_id
        assert base_classifier.label_to_id[label] in base_classifier.id_to_label

def test_prediction(base_classifier, sample_data):
    texts, labels = sample_data
    base_classifier.add_examples(texts, labels)
    
    # Test prediction
    predictions = base_classifier.predict("This is fantastic")
    assert len(predictions) > 0
    assert all(isinstance(p[0], str) and isinstance(p[1], float) for p in predictions)
    assert sum(p[1] for p in predictions) > 0  # Scores should be positive

def test_save_load(base_classifier, sample_data):
    """Test saving and loading the classifier with deterministic results."""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    texts, labels = sample_data
    base_classifier.add_examples(texts, labels)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_classifier"
        
        # Ensure model is in eval mode before saving
        base_classifier.model.eval()
        if base_classifier.adaptive_head is not None:
            base_classifier.adaptive_head.eval()
        
        # Save
        base_classifier.save(save_path)
        assert (save_path / "config.json").exists()
        assert (save_path / "tensors.safetensors").exists()
        
        # Load with same device
        loaded_classifier = AdaptiveClassifier.load(save_path, device=base_classifier.device)
        assert loaded_classifier is not None
        assert loaded_classifier.label_to_id == base_classifier.label_to_id
        
        # Ensure loaded model is also in eval mode
        loaded_classifier.model.eval()
        if loaded_classifier.adaptive_head is not None:
            loaded_classifier.adaptive_head.eval()
        
        # Test predictions match
        with torch.no_grad():  # Ensure no gradients affect predictions
            test_text = "This is a test"
            original_preds = base_classifier.predict(test_text)
            loaded_preds = loaded_classifier.predict(test_text)
        
        # Sort predictions by score to handle any order differences
        original_preds = sorted(original_preds, key=lambda x: (-x[1], x[0]))
        loaded_preds = sorted(loaded_preds, key=lambda x: (-x[1], x[0]))
        
        # Use a more reasonable threshold for floating point comparisons
        score_threshold = 5e-2  # Allow small differences due to floating point operations
        
        for (label1, score1), (label2, score2) in zip(original_preds, loaded_preds):
            assert label1 == label2, f"Labels don't match: {label1} vs {label2}"
            assert abs(score1 - score2) < score_threshold, \
                f"Scores differ too much: {score1} vs {score2}"

def test_dynamic_class_addition(base_classifier, sample_data):
    texts, labels = sample_data
    base_classifier.add_examples(texts[:3], labels[:3])
    
    # Add new class
    new_texts = ["Error in system", "Null pointer exception"]
    new_labels = ["technical", "technical"]
    base_classifier.add_examples(new_texts, new_labels)
    
    # Check if new class was added
    assert "technical" in base_classifier.label_to_id
    
    # Test prediction includes new class
    pred = base_classifier.predict("System crash occurred")
    assert any(label == "technical" for label, _ in pred)

def test_empty_input_handling(base_classifier):
    with pytest.raises(ValueError):
        base_classifier.add_examples([], [])
    
    with pytest.raises(ValueError):
        base_classifier.predict("")

def test_mismatched_input_handling(base_classifier):
    with pytest.raises(ValueError):
        base_classifier.add_examples(["text1"], ["label1", "label2"])

def test_device_handling(base_classifier, sample_data):
    texts, labels = sample_data
    base_classifier.add_examples(texts, labels)
    
    # Test CPU predictions
    base_classifier.to("cpu")
    cpu_pred = base_classifier.predict("test")
    
    # Test GPU predictions if available
    if torch.cuda.is_available():
        base_classifier.to("cuda")
        gpu_pred = base_classifier.predict("test")
        
        # Results should be similar
        for (label1, score1), (label2, score2) in zip(cpu_pred, gpu_pred):
            assert label1 == label2
            assert abs(score1 - score2) < 1e-5

def test_batch_prediction(base_classifier, sample_data):
    texts, labels = sample_data
    base_classifier.add_examples(texts, labels)
    
    batch_texts = ["This is good", "This is bad", "This is okay"]
    predictions = base_classifier.predict_batch(batch_texts)
    
    assert len(predictions) == len(batch_texts)
    for pred in predictions:
        assert len(pred) > 0
        assert all(isinstance(p[0], str) and isinstance(p[1], float) for p in pred)

def test_memory_management(base_classifier, sample_data):
    texts, labels = sample_data
    
    # Test memory before adding examples
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    base_classifier.add_examples(texts, labels)
    
    # Test memory after adding examples
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Memory should be managed properly
    if torch.cuda.is_available():
        assert final_memory > initial_memory  # Should use some GPU memory
        
        # Cleanup
        del base_classifier
        torch.cuda.empty_cache()
        
        # Memory should be released
        cleanup_memory = torch.cuda.memory_allocated()
        assert cleanup_memory < final_memory

if __name__ == "__main__":
    pytest.main([__file__])