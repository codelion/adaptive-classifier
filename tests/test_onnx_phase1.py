"""Test ONNX Runtime integration - Phase 1: Basic initialization and embeddings."""

import pytest
import torch
import numpy as np
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
def test_onnx_initialization():
    """Test that ONNX model initializes correctly."""
    # Use a small model for testing
    model_name = "prajjwal1/bert-tiny"

    # Initialize with ONNX explicitly enabled
    classifier = AdaptiveClassifier(model_name, use_onnx=True, device="cpu")

    # Verify ONNX is being used
    assert classifier.use_onnx is True
    assert hasattr(classifier.model, "model")  # ORTModel has this attribute


def test_auto_detection_cpu():
    """Test that auto-detection uses ONNX on CPU."""
    model_name = "prajjwal1/bert-tiny"

    # Initialize with auto-detection on CPU
    classifier = AdaptiveClassifier(model_name, device="cpu", use_onnx="auto")

    # Should use ONNX on CPU if available
    # If optimum not installed, should fall back to PyTorch
    if _check_optimum_installed():
        assert classifier.use_onnx is True
    else:
        assert classifier.use_onnx is False


def test_auto_detection_gpu():
    """Test that auto-detection uses PyTorch on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model_name = "prajjwal1/bert-tiny"

    # Initialize with auto-detection on GPU
    classifier = AdaptiveClassifier(model_name, device="cuda", use_onnx="auto")

    # Should use PyTorch on GPU
    assert classifier.use_onnx is False


@pytest.mark.skipif(
    not _check_optimum_installed(),
    reason="optimum[onnxruntime] not installed"
)
def test_embedding_consistency():
    """Test that ONNX and PyTorch produce similar embeddings."""
    model_name = "prajjwal1/bert-tiny"
    test_text = "This is a test sentence for embedding comparison."

    # Initialize PyTorch model
    classifier_pytorch = AdaptiveClassifier(model_name, use_onnx=False, device="cpu")

    # Initialize ONNX model
    classifier_onnx = AdaptiveClassifier(model_name, use_onnx=True, device="cpu")

    # Get embeddings from both
    embedding_pytorch = classifier_pytorch._get_embeddings([test_text])[0]
    embedding_onnx = classifier_onnx._get_embeddings([test_text])[0]

    # Convert to numpy for comparison
    emb_pytorch_np = embedding_pytorch.cpu().numpy()
    emb_onnx_np = embedding_onnx.cpu().numpy()

    # Check shapes match
    assert emb_pytorch_np.shape == emb_onnx_np.shape

    # Check embeddings are similar (cosine similarity > 0.99)
    cosine_sim = np.dot(emb_pytorch_np, emb_onnx_np) / (
        np.linalg.norm(emb_pytorch_np) * np.linalg.norm(emb_onnx_np)
    )

    print(f"Cosine similarity between PyTorch and ONNX embeddings: {cosine_sim:.6f}")
    assert cosine_sim > 0.99, f"Embeddings differ too much: cosine_sim={cosine_sim}"


@pytest.mark.skipif(
    not _check_optimum_installed(),
    reason="optimum[onnxruntime] not installed"
)
def test_onnx_with_training():
    """Test that ONNX model works with adaptive classifier training."""
    model_name = "prajjwal1/bert-tiny"

    # Initialize with ONNX
    classifier = AdaptiveClassifier(model_name, use_onnx=True, device="cpu")

    # Add some examples
    texts = [
        "This is a positive example",
        "This is a negative example",
        "Another positive case",
        "Another negative case"
    ]
    labels = ["positive", "negative", "positive", "negative"]

    # This should work without errors
    classifier.add_examples(texts, labels)

    # Test prediction
    predictions = classifier.predict("This seems positive")

    # Verify we got predictions
    assert len(predictions) > 0
    assert all(isinstance(label, str) and isinstance(score, float)
               for label, score in predictions)


def test_explicit_disable_onnx():
    """Test that ONNX can be explicitly disabled."""
    model_name = "prajjwal1/bert-tiny"

    # Explicitly disable ONNX
    classifier = AdaptiveClassifier(model_name, use_onnx=False, device="cpu")

    # Should not use ONNX
    assert classifier.use_onnx is False


def test_fallback_on_import_error():
    """Test that classifier falls back to PyTorch if optimum not installed."""
    model_name = "prajjwal1/bert-tiny"

    # Even if we request ONNX, should gracefully fall back if not available
    classifier = AdaptiveClassifier(model_name, use_onnx=True, device="cpu")

    # Should either use ONNX or have fallen back to PyTorch
    assert classifier.use_onnx in [True, False]

    # Should be functional regardless
    embedding = classifier._get_embeddings(["test"])[0]
    assert embedding is not None
    assert embedding.shape[0] > 0


if __name__ == "__main__":
    # Run tests
    print("Testing ONNX Phase 1 implementation...")
    print(f"Optimum installed: {_check_optimum_installed()}")

    print("\n1. Testing ONNX initialization...")
    if _check_optimum_installed():
        test_onnx_initialization()
        print("✓ ONNX initialization works")
    else:
        print("⊗ Skipped (optimum not installed)")

    print("\n2. Testing auto-detection on CPU...")
    test_auto_detection_cpu()
    print("✓ Auto-detection on CPU works")

    print("\n3. Testing explicit disable...")
    test_explicit_disable_onnx()
    print("✓ Explicit disable works")

    print("\n4. Testing fallback...")
    test_fallback_on_import_error()
    print("✓ Fallback mechanism works")

    if _check_optimum_installed():
        print("\n5. Testing embedding consistency...")
        test_embedding_consistency()
        print("✓ Embedding consistency verified")

        print("\n6. Testing ONNX with training...")
        test_onnx_with_training()
        print("✓ ONNX works with training")

    print("\n✓ All Phase 1 tests passed!")
