import pytest
import tempfile
from pathlib import Path
from adaptive_classifier import AdaptiveClassifier


def test_single_example_confidence_consistency():
    """Test confidence consistency with single example per class"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create classifier
        classifier = AdaptiveClassifier("google-bert/bert-large-cased")
        
        # Add single example per class
        examples = {
            "foo": ["fish"],
            "bar": ["cat"]
        }
        
        for label, examples_list in examples.items():
            classifier.add_examples(examples_list, [label] * len(examples_list))
        
        # Get predictions before save
        fish_before = classifier.predict("fish")
        cat_before = classifier.predict("cat")
        
        # Extract confidence values
        fish_conf_before = fish_before[0][1]
        cat_conf_before = cat_before[0][1]
        
        # Save
        save_path = Path(temp_dir) / "test_model"
        classifier.save(str(save_path))
        
        # Load
        loaded_classifier = AdaptiveClassifier.load(str(save_path))
        
        # Get predictions after load
        fish_after = loaded_classifier.predict("fish")
        cat_after = loaded_classifier.predict("cat")
        
        # Extract confidence values
        fish_conf_after = fish_after[0][1]
        cat_conf_after = cat_after[0][1]
        
        # Check confidence consistency
        assert abs(fish_conf_before - fish_conf_after) < 0.01, \
            f"Fish confidence changed: {fish_conf_before:.4f} -> {fish_conf_after:.4f}"
        
        assert abs(cat_conf_before - cat_conf_after) < 0.01, \
            f"Cat confidence changed: {cat_conf_before:.4f} -> {cat_conf_after:.4f}"
        
        # Verify training history
        assert loaded_classifier.training_history["foo"] == 1
        assert loaded_classifier.training_history["bar"] == 1


def test_exact_reported_case():
    """Test the exact case reported by the user"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Exactly as reported
        classifier = AdaptiveClassifier("google-bert/bert-large-cased")
        
        examples = {
            "foo": ["fish"],
            "bar": ["cat"]
        }
        
        for label, examples in examples.items():
            classifier.add_examples(examples, [label] * len(examples))
        
        # Before save
        result_fish_before = classifier.predict("fish")
        result_cat_before = classifier.predict("cat")
        
        # Save
        save_path = Path(temp_dir) / "foobar"
        classifier.save(str(save_path))
        
        # Load
        loaded_classifier = AdaptiveClassifier.load(str(save_path))
        
        # After load
        result_fish_after = loaded_classifier.predict("fish")
        result_cat_after = loaded_classifier.predict("cat")
        
        # The reported issue shows:
        # fish: 0.9997 -> 0.8997 (0.1 drop)
        # cat: 0.9999 -> 0.8998 (0.1 drop)
        
        # Check if confidence drops by ~0.1
        fish_drop = result_fish_before[0][1] - result_fish_after[0][1]
        cat_drop = result_cat_before[0][1] - result_cat_after[0][1]
        
        # We should not see a 0.1 drop
        assert abs(fish_drop) < 0.01, f"Fish confidence dropped by {fish_drop:.4f}"
        assert abs(cat_drop) < 0.01, f"Cat confidence dropped by {cat_drop:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])