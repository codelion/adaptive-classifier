"""Test ONNX Runtime integration - Phase 2: Export and reload."""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from adaptive_classifier import AdaptiveClassifier


def _check_optimum_installed():
    """Helper to check if optimum is installed."""
    try:
        import optimum.onnxruntime
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _check_optimum_installed(),
    reason="optimum[onnxruntime] not installed"
)
def test_export_onnx_basic():
    """Test basic ONNX export functionality."""
    model_name = "prajjwal1/bert-tiny"

    # Initialize with PyTorch
    classifier = AdaptiveClassifier(model_name, use_onnx=False, device="cpu")

    # Add some examples
    texts = ["positive example", "negative example"]
    labels = ["positive", "negative"]
    classifier.add_examples(texts, labels)

    # Export to ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "onnx_model"
        result_path = classifier.export_onnx(onnx_path, quantize=False)

        # Check that ONNX files exist
        assert result_path.exists()
        assert (result_path / "model.onnx").exists()
        print(f"✓ ONNX model exported to {result_path}")


@pytest.mark.skipif(
    not _check_optimum_installed(),
    reason="optimum[onnxruntime] not installed"
)
def test_save_with_onnx():
    """Test saving classifier with ONNX export integrated."""
    model_name = "prajjwal1/bert-tiny"

    # Initialize and train classifier
    classifier = AdaptiveClassifier(model_name, use_onnx=False, device="cpu")
    texts = ["positive text", "negative text", "neutral text"]
    labels = ["positive", "negative", "neutral"]
    classifier.add_examples(texts, labels)

    # Save with ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "classifier_with_onnx"
        classifier._save_pretrained(save_path, include_onnx=True, quantize_onnx=False)

        # Verify all files exist
        assert (save_path / "config.json").exists()
        assert (save_path / "examples.json").exists()
        assert (save_path / "model.safetensors").exists()
        assert (save_path / "onnx").exists()
        assert (save_path / "onnx" / "model.onnx").exists()
        print("✓ Classifier saved with ONNX")


@pytest.mark.skipif(
    not _check_optimum_installed(),
    reason="optimum[onnxruntime] not installed"
)
def test_load_onnx_model():
    """Test loading a saved ONNX model."""
    model_name = "prajjwal1/bert-tiny"

    # Train and save classifier with ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "classifier_onnx"

        # Create and save
        classifier_orig = AdaptiveClassifier(model_name, use_onnx=False, device="cpu")
        texts = ["happy", "sad", "angry"]
        labels = ["positive", "negative", "negative"]
        classifier_orig.add_examples(texts, labels)
        classifier_orig._save_pretrained(save_path, include_onnx=True)

        # Load with ONNX
        classifier_loaded = AdaptiveClassifier._from_pretrained(
            str(save_path),
            use_onnx=True
        )

        # Verify ONNX is being used
        assert classifier_loaded.use_onnx is True
        print("✓ ONNX model loaded successfully")

        # Test that it works
        predictions = classifier_loaded.predict("very happy")
        assert len(predictions) > 0
        print(f"✓ Predictions work: {predictions[:2]}")


@pytest.mark.skipif(
    not _check_optimum_installed(),
    reason="optimum[onnxruntime] not installed"
)
def test_onnx_prediction_consistency():
    """Test that predictions are consistent after export and reload."""
    model_name = "prajjwal1/bert-tiny"
    test_text = "This is a test for consistency"

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "classifier_consistency"

        # Create and train classifier
        classifier_pytorch = AdaptiveClassifier(model_name, use_onnx=False, device="cpu")
        texts = ["good", "bad", "okay"]
        labels = ["positive", "negative", "neutral"]
        classifier_pytorch.add_examples(texts, labels)

        # Get prediction with PyTorch
        pred_pytorch = classifier_pytorch.predict(test_text, k=3)

        # Save with ONNX
        classifier_pytorch._save_pretrained(save_path, include_onnx=True)

        # Load ONNX version
        classifier_onnx = AdaptiveClassifier._from_pretrained(
            str(save_path),
            use_onnx=True
        )

        # Get prediction with ONNX
        pred_onnx = classifier_onnx.predict(test_text, k=3)

        # Compare predictions (should be very similar)
        print(f"PyTorch predictions: {pred_pytorch}")
        print(f"ONNX predictions: {pred_onnx}")

        # Check that top prediction matches
        assert pred_pytorch[0][0] == pred_onnx[0][0], \
            "Top prediction differs between PyTorch and ONNX"

        # Check that scores are similar (within 5%)
        for (label_pt, score_pt), (label_ox, score_ox) in zip(pred_pytorch, pred_onnx):
            assert label_pt == label_ox, f"Label mismatch: {label_pt} vs {label_ox}"
            score_diff = abs(score_pt - score_ox)
            assert score_diff < 0.05, \
                f"Score difference too large for {label_pt}: {score_diff}"

        print("✓ Predictions are consistent between PyTorch and ONNX")


@pytest.mark.skipif(
    not _check_optimum_installed(),
    reason="optimum[onnxruntime] not installed"
)
def test_auto_detection_loads_onnx():
    """Test that auto-detection loads ONNX when available on CPU."""
    model_name = "prajjwal1/bert-tiny"

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "classifier_auto"

        # Create and save with ONNX
        classifier_orig = AdaptiveClassifier(model_name, use_onnx=False, device="cpu")
        texts = ["example one", "example two"]
        labels = ["class1", "class2"]
        classifier_orig.add_examples(texts, labels)
        classifier_orig._save_pretrained(save_path, include_onnx=True)

        # Load with auto-detection on CPU
        classifier_auto = AdaptiveClassifier._from_pretrained(
            str(save_path),
            use_onnx="auto",
            device="cpu"
        )

        # Should automatically use ONNX on CPU
        assert classifier_auto.use_onnx is True
        print("✓ Auto-detection correctly loads ONNX on CPU")


def test_fallback_when_onnx_not_available():
    """Test that loading works even when ONNX not in save directory."""
    model_name = "prajjwal1/bert-tiny"

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "classifier_no_onnx"

        # Create and save WITHOUT ONNX
        classifier_orig = AdaptiveClassifier(model_name, use_onnx=False, device="cpu")
        texts = ["text one", "text two"]
        labels = ["A", "B"]
        classifier_orig.add_examples(texts, labels)
        classifier_orig._save_pretrained(save_path, include_onnx=False)

        # Try to load with ONNX requested
        classifier_loaded = AdaptiveClassifier._from_pretrained(
            str(save_path),
            use_onnx=True  # Request ONNX even though it's not available
        )

        # Should fall back to PyTorch
        assert classifier_loaded.use_onnx is False
        print("✓ Correctly falls back to PyTorch when ONNX not available")

        # Should still work
        predictions = classifier_loaded.predict("test")
        assert len(predictions) > 0


if __name__ == "__main__":
    print("Testing ONNX Phase 2 implementation...")
    print(f"Optimum installed: {_check_optimum_installed()}")

    if not _check_optimum_installed():
        print("⊗ Skipping tests - optimum[onnxruntime] not installed")
        exit(0)

    print("\n1. Testing basic ONNX export...")
    test_export_onnx_basic()

    print("\n2. Testing save with ONNX...")
    test_save_with_onnx()

    print("\n3. Testing load ONNX model...")
    test_load_onnx_model()

    print("\n4. Testing prediction consistency...")
    test_onnx_prediction_consistency()

    print("\n5. Testing auto-detection...")
    test_auto_detection_loads_onnx()

    print("\n6. Testing fallback when ONNX not available...")
    test_fallback_when_onnx_not_available()

    print("\n✓ All Phase 2 tests passed!")
