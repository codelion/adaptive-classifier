import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from adaptive_classifier import AdaptiveClassifier


def test_confidence_consistency_after_save_load():
    """Test that confidence scores remain consistent after save/load"""
    # Create temporary directory for saving
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize classifier
        classifier = AdaptiveClassifier("answerdotai/ModernBERT-base")
        
        # Add training examples
        texts = ["This is a foo example"] * 100 + ["This is a bar example"] * 100
        labels = ["foo"] * 100 + ["bar"] * 100
        
        classifier.add_examples(texts, labels)
        
        # Get prediction before save
        test_text = "This is a foo example"
        predictions_before = classifier.predict(test_text)
        
        # Extract confidence scores
        conf_before = {label: score for label, score in predictions_before}
        
        # Save the model
        save_path = Path(temp_dir) / "test_model"
        classifier.save(str(save_path))
        
        # Load the model
        loaded_classifier = AdaptiveClassifier.load(str(save_path))
        
        # Get prediction after load
        predictions_after = loaded_classifier.predict(test_text)
        conf_after = {label: score for label, score in predictions_after}
        
        # Check that confidence scores are similar (within 1% tolerance)
        assert abs(conf_before["foo"] - conf_after["foo"]) < 0.01, \
            f"Confidence dropped from {conf_before['foo']:.4f} to {conf_after['foo']:.4f}"
        
        # Verify reasonable confidence is maintained (accounting for prototype normalization)
        assert conf_after["foo"] > 0.70, \
            f"Confidence too low after load: {conf_after['foo']:.4f}"


def test_continuous_learning_with_save_load():
    """Test continuous learning scenario with save/load"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initial training
        classifier = AdaptiveClassifier("answerdotai/ModernBERT-base")
        
        # Train with 100 examples
        texts_1 = ["Initial foo example"] * 100 + ["Initial bar example"] * 100
        labels_1 = ["foo"] * 100 + ["bar"] * 100
        classifier.add_examples(texts_1, labels_1)
        
        # Save
        save_path = Path(temp_dir) / "test_model"
        classifier.save(str(save_path))
        
        # Load
        loaded_classifier = AdaptiveClassifier.load(str(save_path))
        
        # Verify training history was preserved
        assert loaded_classifier.training_history["foo"] == 100
        assert loaded_classifier.training_history["bar"] == 100
        
        # Add more examples (continuous learning)
        texts_2 = ["Additional foo example"] * 20 + ["Additional bar example"] * 20
        labels_2 = ["foo"] * 20 + ["bar"] * 20
        loaded_classifier.add_examples(texts_2, labels_2)
        
        # Verify cumulative training history
        assert loaded_classifier.training_history["foo"] == 120
        assert loaded_classifier.training_history["bar"] == 120
        
        # Get predictions - should use established class weights
        predictions = loaded_classifier.predict("This is a foo example")
        conf = {label: score for label, score in predictions}
        
        # Should maintain reasonable confidence for established classes
        # Note: Due to prototype normalization, confidence is lower than neural-only predictions
        assert conf["foo"] > 0.65


def test_backward_compatibility():
    """Test loading models without training_history field"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and save a model
        classifier = AdaptiveClassifier("answerdotai/ModernBERT-base")
        
        texts = ["Foo example"] * 100 + ["Bar example"] * 100
        labels = ["foo"] * 100 + ["bar"] * 100
        classifier.add_examples(texts, labels)
        
        save_path = Path(temp_dir) / "test_model"
        classifier.save(str(save_path))
        
        # Manually remove training_history from config to simulate old model
        import json
        config_path = save_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Remove training_history if present
        config.pop('training_history', None)
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Load the "old" model
        loaded_classifier = AdaptiveClassifier.load(str(save_path))
        
        # Should have estimated training history
        assert loaded_classifier.training_history["foo"] == 100  # 5 saved * 20
        assert loaded_classifier.training_history["bar"] == 100  # 5 saved * 20
        
        # Predictions should work with reasonable confidence
        predictions = loaded_classifier.predict("This is a foo example")
        conf = {label: score for label, score in predictions}
        assert conf["foo"] > 0.65


def test_new_class_detection():
    """Test that new classes with few examples are correctly identified"""
    classifier = AdaptiveClassifier("answerdotai/ModernBERT-base")
    
    # Add established classes
    texts_established = ["Established foo"] * 50 + ["Established bar"] * 50
    labels_established = ["foo"] * 50 + ["bar"] * 50
    classifier.add_examples(texts_established, labels_established)
    
    # Add new class with few examples
    texts_new = ["New baz example"] * 5
    labels_new = ["baz"] * 5
    classifier.add_examples(texts_new, labels_new)
    
    # Verify training history
    assert classifier.training_history["foo"] == 50
    assert classifier.training_history["bar"] == 50
    assert classifier.training_history["baz"] == 5
    
    # Test predictions - new class should use different weights
    predictions = classifier.predict("New baz example")
    
    # The prediction should work (no errors)
    assert len(predictions) > 0
    assert any(label == "baz" for label, _ in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])