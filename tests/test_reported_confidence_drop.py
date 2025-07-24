import pytest
import tempfile
from pathlib import Path
from adaptive_classifier import AdaptiveClassifier


def test_reported_confidence_values():
    """Test for the exact confidence drop reported:
    fish: 0.9997 -> 0.8997
    cat: 0.9999 -> 0.8998
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use the exact setup from the report
        classifier = AdaptiveClassifier("google-bert/bert-large-cased")
        
        examples = {
            "foo": ["fish"],
            "bar": ["cat"]
        }
        
        for label, examples in examples.items():
            classifier.add_examples(examples, [label] * len(examples))
        
        # Get predictions before save
        result_fish_before = classifier.predict("fish")
        result_cat_before = classifier.predict("cat")
        
        # Save
        save_path = Path(temp_dir) / "foobar"
        classifier.save(str(save_path))
        
        # Load
        loaded_classifier = AdaptiveClassifier.load(str(save_path))
        
        # Get predictions after load
        result_fish_after = loaded_classifier.predict("fish")
        result_cat_after = loaded_classifier.predict("cat")
        
        # Extract confidence values
        fish_conf_before = result_fish_before[0][1]
        cat_conf_before = result_cat_before[0][1]
        fish_conf_after = result_fish_after[0][1]
        cat_conf_after = result_cat_after[0][1]
        
        print(f"\nFish confidence: {fish_conf_before:.4f} -> {fish_conf_after:.4f}")
        print(f"Cat confidence: {cat_conf_before:.4f} -> {cat_conf_after:.4f}")
        
        # If we see the reported behavior (0.9997 -> 0.8997), it means:
        # - Before save: getting pure neural predictions
        # - After save: getting blended predictions
        
        # The fix should ensure consistency
        assert abs(fish_conf_before - fish_conf_after) < 0.01, \
            f"Fish confidence dropped from {fish_conf_before:.4f} to {fish_conf_after:.4f}"
        
        assert abs(cat_conf_before - cat_conf_after) < 0.01, \
            f"Cat confidence dropped from {cat_conf_before:.4f} to {cat_conf_after:.4f}"
        
        # Both before and after should be around 0.8997 (blended prediction)
        # not 0.9997 (pure neural prediction)
        assert 0.85 < fish_conf_before < 0.95, \
            f"Before save confidence should be blended, got {fish_conf_before:.4f}"
        
        assert 0.85 < fish_conf_after < 0.95, \
            f"After load confidence should be blended, got {fish_conf_after:.4f}"


if __name__ == "__main__":
    test_reported_confidence_values()