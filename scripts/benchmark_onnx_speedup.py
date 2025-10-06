#!/usr/bin/env python3
"""Benchmark ONNX vs PyTorch performance for adaptive classifier."""

import time
import logging
import datasets
from adaptive_classifier import AdaptiveClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_model(model_id: str, test_texts: list, use_onnx: bool, num_runs: int = 3):
    """Benchmark a model configuration."""
    mode = "ONNX (Quantized)" if use_onnx else "PyTorch"
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {mode}")
    logger.info(f"{'='*60}")

    # Load model
    logger.info(f"Loading model from {model_id}...")
    start = time.time()
    classifier = AdaptiveClassifier.load(model_id, use_onnx=use_onnx)
    load_time = time.time() - start
    logger.info(f"Model loaded in {load_time:.2f}s")

    # Warm-up run (not timed)
    logger.info("Warming up...")
    _ = classifier.predict_batch(test_texts[:5])

    # Benchmark runs
    times = []
    for run in range(num_runs):
        logger.info(f"Run {run + 1}/{num_runs}...")
        start = time.time()
        predictions = classifier.predict_batch(test_texts)
        elapsed = time.time() - start
        times.append(elapsed)
        logger.info(f"  Completed in {elapsed:.3f}s ({len(test_texts)/elapsed:.1f} samples/sec)")

    avg_time = sum(times) / len(times)
    throughput = len(test_texts) / avg_time

    logger.info(f"\nResults for {mode}:")
    logger.info(f"  Average time: {avg_time:.3f}s")
    logger.info(f"  Throughput: {throughput:.1f} samples/sec")
    logger.info(f"  Per-sample latency: {avg_time*1000/len(test_texts):.1f}ms")

    return {
        'mode': mode,
        'load_time': load_time,
        'avg_time': avg_time,
        'throughput': throughput,
        'times': times
    }

def main():
    # Configuration
    model_id = "adaptive-classifier/llm-router"
    num_samples = 100
    num_runs = 3

    logger.info(f"Benchmark Configuration:")
    logger.info(f"  Model: {model_id}")
    logger.info(f"  Samples: {num_samples}")
    logger.info(f"  Runs per config: {num_runs}")

    # Load test data
    logger.info(f"\nLoading test dataset...")
    dataset = datasets.load_dataset("routellm/gpt4_dataset", split="validation")
    test_data = dataset.select(range(min(num_samples, len(dataset))))
    test_texts = [item['prompt'] for item in test_data]
    logger.info(f"Loaded {len(test_texts)} test samples")

    # Benchmark PyTorch version
    pytorch_results = benchmark_model(model_id, test_texts, use_onnx=False, num_runs=num_runs)

    # Benchmark ONNX version
    onnx_results = benchmark_model(model_id, test_texts, use_onnx=True, num_runs=num_runs)

    # Compare results
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPARISON SUMMARY")
    logger.info(f"{'='*60}")

    speedup = pytorch_results['avg_time'] / onnx_results['avg_time']
    throughput_increase = onnx_results['throughput'] / pytorch_results['throughput']
    latency_reduction = (1 - onnx_results['avg_time'] / pytorch_results['avg_time']) * 100

    logger.info(f"\nPyTorch (Baseline):")
    logger.info(f"  Average time: {pytorch_results['avg_time']:.3f}s")
    logger.info(f"  Throughput: {pytorch_results['throughput']:.1f} samples/sec")

    logger.info(f"\nONNX Quantized:")
    logger.info(f"  Average time: {onnx_results['avg_time']:.3f}s")
    logger.info(f"  Throughput: {onnx_results['throughput']:.1f} samples/sec")

    logger.info(f"\nSpeedup:")
    logger.info(f"  🚀 {speedup:.2f}x faster")
    logger.info(f"  📈 {throughput_increase:.2f}x throughput increase")
    logger.info(f"  ⏱️  {latency_reduction:.1f}% latency reduction")

    logger.info(f"\nModel Size Comparison:")
    logger.info(f"  PyTorch: Uses full precision weights")
    logger.info(f"  ONNX Quantized: 65.6 MB (4x smaller than unquantized)")

    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARK COMPLETE")
    logger.info(f"{'='*60}")

    return {
        'pytorch': pytorch_results,
        'onnx': onnx_results,
        'speedup': speedup,
        'throughput_increase': throughput_increase,
        'latency_reduction': latency_reduction
    }

if __name__ == "__main__":
    results = main()
