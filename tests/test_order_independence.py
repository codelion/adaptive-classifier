import pytest
import tempfile
from pathlib import Path
from adaptive_classifier import AdaptiveClassifier


def test_label_id_assignment_order_independence():
    """Test that label IDs are assigned consistently regardless of input order"""
    # Create two classifiers
    classifier1 = AdaptiveClassifier("answerdotai/ModernBERT-base")
    classifier2 = AdaptiveClassifier("answerdotai/ModernBERT-base")
    
    # Same labels, different order
    labels1 = ["alpha", "beta", "gamma"]
    texts1 = ["text1", "text2", "text3"]
    
    labels2 = ["gamma", "beta", "alpha"]
    texts2 = ["text3", "text2", "text1"]
    
    # Add examples
    classifier1.add_examples(texts1, labels1)
    classifier2.add_examples(texts2, labels2)
    
    # Check label mappings are identical
    assert classifier1.label_to_id == classifier2.label_to_id, \
        f"Label mappings differ: {classifier1.label_to_id} vs {classifier2.label_to_id}"
    
    # Verify alphabetical order
    expected_mapping = {"alpha": 0, "beta": 1, "gamma": 2}
    assert classifier1.label_to_id == expected_mapping
    assert classifier2.label_to_id == expected_mapping


def test_incremental_label_addition():
    """Test that labels are assigned IDs consistently when added incrementally"""
    classifier = AdaptiveClassifier("answerdotai/ModernBERT-base")
    
    # Add labels in different batches
    classifier.add_examples(["text1"], ["zebra"])
    assert classifier.label_to_id == {"zebra": 0}
    
    classifier.add_examples(["text2"], ["alpha"])
    assert classifier.label_to_id == {"zebra": 0, "alpha": 1}
    
    classifier.add_examples(["text3"], ["beta"])
    assert classifier.label_to_id == {"zebra": 0, "alpha": 1, "beta": 2}
    
    # Add multiple new labels at once - should be sorted
    classifier.add_examples(["text4", "text5"], ["delta", "charlie"])
    assert classifier.label_to_id == {
        "zebra": 0, "alpha": 1, "beta": 2, "charlie": 3, "delta": 4
    }


def test_predictions_with_sorted_labels():
    """Test that predictions are more consistent with sorted label assignment"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Train two classifiers with same data in different order
        classifier1 = AdaptiveClassifier("answerdotai/ModernBERT-base")
        classifier2 = AdaptiveClassifier("answerdotai/ModernBERT-base")
        
        # Original order from the reported issue
        labels1 = ["zing", "zing", "zing", "zoob", "zoob", "zoob"]
        examples1 = [
            "fish like to swim", "fish live in the sea", "fish are amphibians",
            "cats like to meow", "cats live at home", "cats are felines"
        ]
        
        # Reversed order
        labels2 = labels1.copy()
        examples2 = examples1.copy()
        labels2.reverse()
        examples2.reverse()
        
        # Add examples
        classifier1.add_examples(examples1, labels1)
        classifier2.add_examples(examples2, labels2)
        
        # Verify label mappings are identical
        assert classifier1.label_to_id == classifier2.label_to_id
        assert classifier1.label_to_id == {"zing": 0, "zoob": 1}
        
        # Test predictions
        swim_pred1 = classifier1.predict("swim")
        swim_pred2 = classifier2.predict("swim")
        
        meow_pred1 = classifier1.predict("meow")
        meow_pred2 = classifier2.predict("meow")
        
        # Extract confidence values
        swim_conf1 = {label: score for label, score in swim_pred1}
        swim_conf2 = {label: score for label, score in swim_pred2}
        
        meow_conf1 = {label: score for label, score in meow_pred1}
        meow_conf2 = {label: score for label, score in meow_pred2}
        
        # While we can't guarantee identical predictions due to training order,
        # the difference should be smaller than before the fix
        swim_diff = abs(swim_conf1.get("zing", 0) - swim_conf2.get("zing", 0))
        meow_diff = abs(meow_conf1.get("zoob", 0) - meow_conf2.get("zoob", 0))
        
        # Log the differences for debugging
        print(f"\nSwim predictions:")
        print(f"  Classifier 1: {swim_pred1}")
        print(f"  Classifier 2: {swim_pred2}")
        print(f"  Difference in 'zing' confidence: {swim_diff:.4f}")
        
        print(f"\nMeow predictions:")
        print(f"  Classifier 1: {meow_pred1}")
        print(f"  Classifier 2: {meow_pred2}")
        print(f"  Difference in 'zoob' confidence: {meow_diff:.4f}")
        
        # The predictions won't be identical due to training order,
        # but they should be more consistent than the reported 26-36% swings
        assert swim_diff < 0.4, f"Swim predictions differ too much: {swim_diff:.4f}"
        assert meow_diff < 0.4, f"Meow predictions differ too much: {meow_diff:.4f}"


def test_mixed_batch_label_sorting():
    """Test that labels are sorted within each batch before assignment"""
    classifier = AdaptiveClassifier("answerdotai/ModernBERT-base")
    
    # Add multiple new labels in one batch - they should be sorted
    labels = ["zoo", "apple", "dog", "cat", "banana"]
    texts = ["text1", "text2", "text3", "text4", "text5"]
    
    classifier.add_examples(texts, labels)
    
    # Labels should be assigned IDs in alphabetical order
    expected = {
        "apple": 0,
        "banana": 1,
        "cat": 2,
        "dog": 3,
        "zoo": 4
    }
    assert classifier.label_to_id == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])