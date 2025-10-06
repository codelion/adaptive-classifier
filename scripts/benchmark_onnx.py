"""Benchmark script comparing PyTorch vs ONNX vs Quantized ONNX performance."""

import time
import argparse
import tempfile
from pathlib import Path
import numpy as np
from adaptive_classifier import AdaptiveClassifier


def check_optimum_installed():
    """Check if optimum is installed."""
    try:
        import optimum.onnxruntime
        return True
    except ImportError:
        return False


def benchmark_inference(classifier, texts, num_runs=100):
    """Benchmark inference speed."""
    # Warmup
    for _ in range(5):
        classifier.predict(texts[0])

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        for text in texts:
            classifier.predict(text)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_query = (total_time / (num_runs * len(texts))) * 1000  # ms

    return avg_time_per_query, total_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX vs PyTorch performance")
    parser.add_argument("--model", type=str, default="prajjwal1/bert-tiny",
                        help="HuggingFace model name to benchmark")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of benchmark runs")
    parser.add_argument("--skip-quantized", action="store_true",
                        help="Skip quantized ONNX benchmarking")
    args = parser.parse_args()

    if not check_optimum_installed():
        print("⚠️  optimum[onnxruntime] not installed. Skipping ONNX benchmarks.")
        print("Install with: pip install optimum[onnxruntime]")
        return

    print("=" * 70)
    print("ONNX Runtime Benchmark for Adaptive Classifier")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Runs per test: {args.runs}")
    print()

    # Prepare test data
    test_texts = [
        "This is a positive example",
        "This seems negative to me",
        "A neutral statement here",
        "Another test case for benchmarking performance",
        "The quick brown fox jumps over the lazy dog"
    ]

    print("Preparing classifiers...")
    print()

    # Train a baseline classifier
    classifier_base = AdaptiveClassifier(args.model, use_onnx=False, device="cpu")
    training_texts = [
        "great product", "terrible experience", "okay item",
        "loved it", "hated it", "it's fine",
        "amazing quality", "poor service", "average performance"
    ]
    training_labels = [
        "positive", "negative", "neutral",
        "positive", "negative", "neutral",
        "positive", "negative", "neutral"
    ]
    classifier_base.add_examples(training_texts, training_labels)

    # Save and create ONNX versions
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "classifier"

        # Save with ONNX versions
        print("Exporting ONNX models...")
        classifier_base._save_pretrained(
            save_path,
            include_onnx=True,
            quantize_onnx=not args.skip_quantized
        )

        # Load PyTorch version
        print("Loading PyTorch model...")
        classifier_pytorch = AdaptiveClassifier._from_pretrained(
            str(save_path),
            use_onnx=False
        )

        # Load ONNX version
        print("Loading ONNX model...")
        classifier_onnx = AdaptiveClassifier._from_pretrained(
            str(save_path),
            use_onnx=True
        )

        print()
        print("Starting benchmarks...")
        print("-" * 70)

        # Benchmark PyTorch
        print("\n1. PyTorch Baseline")
        print("   Running benchmark...")
        pytorch_avg, pytorch_total = benchmark_inference(
            classifier_pytorch, test_texts, args.runs
        )
        print(f"   ✓ Average time per query: {pytorch_avg:.2f}ms")
        print(f"   ✓ Total time: {pytorch_total:.2f}s")

        # Benchmark ONNX
        print("\n2. ONNX Runtime")
        print("   Running benchmark...")
        onnx_avg, onnx_total = benchmark_inference(
            classifier_onnx, test_texts, args.runs
        )
        print(f"   ✓ Average time per query: {onnx_avg:.2f}ms")
        print(f"   ✓ Total time: {onnx_total:.2f}s")
        speedup = pytorch_avg / onnx_avg
        print(f"   ✓ Speedup: {speedup:.2f}x faster than PyTorch")

        # Test prediction accuracy
        print("\n3. Accuracy Verification")
        test_text = "This is amazing!"
        pred_pytorch = classifier_pytorch.predict(test_text)
        pred_onnx = classifier_onnx.predict(test_text)

        print(f"   PyTorch top prediction: {pred_pytorch[0]}")
        print(f"   ONNX top prediction: {pred_onnx[0]}")

        if pred_pytorch[0][0] == pred_onnx[0][0]:
            print("   ✓ Predictions match!")
        else:
            print("   ⚠️  Predictions differ slightly")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"PyTorch:     {pytorch_avg:.2f}ms/query  (baseline)")
    print(f"ONNX:        {onnx_avg:.2f}ms/query  ({speedup:.2f}x faster)")
    print()

    if speedup > 2.0:
        print("🚀 ONNX provides significant speedup! (>2x)")
    elif speedup > 1.2:
        print("⚡ ONNX provides moderate speedup")
    else:
        print("ℹ️  ONNX provides marginal speedup")

    print()
    print("=" * 70)
    print("\nRecommendation:")
    if speedup > 1.5:
        print("✓ Use ONNX for CPU inference for better performance!")
        print("  classifier = AdaptiveClassifier(model_name, use_onnx=True)")
    else:
        print("ℹ️  ONNX speedup is modest for this model.")
        print("  Consider using smaller models (distilbert, MiniLM) for better gains.")


if __name__ == "__main__":
    main()
