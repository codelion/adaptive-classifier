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
        
        # The reported bug was a pure-neural score (0.9997) before saving versus a
        # blended one (0.8997) afterwards. Both sides must now be the same blend:
        # the right label, confident but not the saturated head-only value. (The
        # exact blended value moved in 0.3.0, when prototype scores were sharpened
        # and the head's weight started ramping up with the number of examples, so
        # this no longer pins a 0.85-0.95 window.)
        assert result_fish_before[0][0] == "foo" and result_fish_after[0][0] == "foo"
        assert result_cat_before[0][0] == "bar" and result_cat_after[0][0] == "bar"
        assert 0.5 < fish_conf_before < 0.9995, \
            f"Before save confidence should be a blend, got {fish_conf_before:.4f}"

        assert 0.5 < fish_conf_after < 0.9995, \
            f"After load confidence should be a blend, got {fish_conf_after:.4f}"

if __name__ == "__main__":
    test_reported_confidence_values()