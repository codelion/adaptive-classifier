#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hallucination detector for language models based on the adaptive-classifier library.
Inspired by the LettuceDetect paper.
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import logging
import json
import time
from sklearn.metrics import precision_recall_fscore_support

from adaptive_classifier import AdaptiveClassifier

# Set up logging - will be reconfigured in main based on command-line args
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a hallucination detector using adaptive-classifier"
    )
    parser.add_argument(
        "--train-percentage",
        type=float,
        default=20.0,
        help="Percentage of dataset to use for training (default: 20%%)",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Fixed number of examples to use for training (overrides --train-percentage if specified)",
    )
    parser.add_argument(
        "--use-all-data",
        action="store_true",
        help="Use all available data for training and testing (overrides other training parameters)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Hugging Face model name to use as the base model",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./hallucination-detector",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        help="Name to use when pushing to HuggingFace Hub under adaptive-classifier/",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=0,
        help="Minimum number of examples required per task (default: 0, use all available data)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable more detailed logging"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    return parser.parse_args()

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_and_split_dataset(args):
    """Load RAGTruth dataset and split according to specified parameters."""
    try:
        dataset = load_dataset("flowaicom/RAGTruth_test")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.info("Trying alternative dataset ID...")
        try:
            dataset = load_dataset("RAGTruth/test")
        except Exception as e:
            logger.error(f"Error loading alternative dataset: {e}")
            raise ValueError("Failed to load RAGTruth dataset. Please check the dataset ID or your internet connection.")
    
    logger.info(f"Dataset loaded with structure: {dataset}")
    
    # Check the dataset structure
    splits = {}
    
    # If it's a DatasetDict (contains multiple datasets by task)
    if hasattr(dataset, 'keys') and callable(dataset.keys):
        task_types = list(dataset.keys())
        logger.info(f"Found task types in DatasetDict: {task_types}")
        
        for task_type in task_types:
            # Get dataset for this task type
            task_dataset = dataset[task_type]
            
            # Determine training size based on arguments
            dataset_size = len(task_dataset)
            
            if args.use_all_data:
                # Use all data for both training and testing
                logger.info(f"Using all {dataset_size} examples for both training and testing for task {task_type}")
                splits[task_type] = {
                    "train": task_dataset,
                    "test": task_dataset
                }
                continue
            
            if args.train_count is not None:
                # Fixed number of examples
                train_size = min(args.train_count, dataset_size)
                logger.info(f"Using fixed count of {train_size} examples for training for task {task_type}")
            else:
                # Percentage-based split
                train_size = int(dataset_size * (args.train_percentage / 100))
                # Ensure minimum 1 example for training if percentage > 0
                if args.train_percentage > 0 and train_size == 0:
                    train_size = 1
                    logger.warning(f"Training percentage {args.train_percentage}% resulted in 0 examples for {task_type}. Using 1 example instead.")
                logger.info(f"Using {train_size} examples ({args.train_percentage}%) for training for task {task_type}")
            
            # If train size equals dataset size, we need to use same data for testing
            if train_size >= dataset_size:
                logger.warning(f"Training size {train_size} >= dataset size {dataset_size} for task {task_type}. Using same data for testing.")
                splits[task_type] = {
                    "train": task_dataset,
                    "test": task_dataset
                }
                continue
            
            # Shuffle data
            shuffled_data = task_dataset.shuffle(seed=args.seed)
            
            # Split data
            train_data = shuffled_data.select(range(train_size))
            test_data = shuffled_data.select(range(train_size, len(task_dataset)))
            
            # Add to splits
            splits[task_type] = {
                "train": train_data,
                "test": test_data
            }
            
            # Verify the split worked correctly
            logger.info(f"Task {task_type}: Calculated {train_size} training examples, actually got {len(train_data)}")
            logger.info(f"Task {task_type}: {len(train_data)} training examples, {len(test_data)} test examples")
    
    # If it's a single dataset, try to group by task_type
    else:
        main_data = dataset
        # Filter and group by task type
        task_types = set()
        for example in main_data:
            if 'task_type' in example and example['task_type']:
                task_types.add(example['task_type'])
        
        logger.info(f"Found task types in dataset: {task_types}")
        
        # Group data by task type
        task_data = {}
        for task_type in task_types:
            task_data[task_type] = [ex for ex in main_data if ex.get('task_type') == task_type]
            logger.info(f"Task {task_type}: {len(task_data[task_type])} examples")
        
        # Split each task type's data
        for task_type, examples in task_data.items():
            # Determine training size based on arguments
            dataset_size = len(examples)
            
            if args.use_all_data:
                # Use all data for both training and testing
                logger.info(f"Using all {dataset_size} examples for both training and testing for task {task_type}")
                splits[task_type] = {
                    "train": examples,
                    "test": examples
                }
                continue
            
            if args.train_count is not None:
                # Fixed number of examples
                train_size = min(args.train_count, dataset_size)
                logger.info(f"Using fixed count of {train_size} examples for training for task {task_type}")
            else:
                # Percentage-based split
                train_size = int(dataset_size * (args.train_percentage / 100))
                # Ensure minimum 1 example for training if percentage > 0
                if args.train_percentage > 0 and train_size == 0:
                    train_size = 1
                    logger.warning(f"Training percentage {args.train_percentage}% resulted in 0 examples for {task_type}. Using 1 example instead.")
                logger.info(f"Using {train_size} examples ({args.train_percentage}%) for training for task {task_type}")
            
            # If train size equals dataset size, we need to use same data for testing
            if train_size >= dataset_size:
                logger.warning(f"Training size {train_size} >= dataset size {dataset_size} for task {task_type}. Using same data for testing.")
                splits[task_type] = {
                    "train": examples,
                    "test": examples
                }
                continue
            
            # Shuffle data - convert to list if needed
            shuffled_examples = examples.copy() if hasattr(examples, 'copy') else list(examples)
            np.random.seed(args.seed)
            np.random.shuffle(shuffled_examples)
            
            # Ensure the examples are indexed properly
            if hasattr(shuffled_examples, 'select') and callable(getattr(shuffled_examples, 'select')):
                # If this is a Dataset object that supports select
                train_data = shuffled_examples.select(range(train_size))
                test_data = shuffled_examples.select(range(train_size, len(shuffled_examples)))
            else:
                # Standard list slicing
                train_data = shuffled_examples[:train_size]
                test_data = shuffled_examples[train_size:]
            
            splits[task_type] = {
                "train": train_data,
                "test": test_data
            }
            
            logger.info(f"Task {task_type}: {len(train_data)} training examples, {len(test_data)} test examples")
    
    if not splits:
        logger.warning("No task types found in the dataset.")
    
    return splits

def format_input_for_hallucination_detection(example, task_type):
    """Format the input for hallucination detection based on task type."""
    # First, convert example to dict if it's not already
    if not isinstance(example, dict):
        example = dict(example)
    
    # Use the schema fields we know about
    source_info = example.get("source_info", "")
    prompt = example.get("prompt", "")
    response = example.get("response", "")
    
    # Format based on task type
    if task_type == "qa":
        return f"Context: {source_info}\nQuestion: {prompt}\nAnswer: {response}"
    
    elif task_type == "data2text":
        return f"Context: {source_info}\nGenerated Text: {response}"
    
    elif task_type == "summarization":
        return f"Article: {source_info}\nSummary: {response}"
    
    else:
        # Default format for unknown task types
        return f"Source: {source_info}\nPrompt: {prompt}\nResponse: {response}"

def prepare_examples(examples, task_type):
    """Prepare examples based on task type."""
    texts = []
    labels = []
    
    # Handle different dataset types
    if hasattr(examples, '__iter__') and not isinstance(examples, (dict, str)):
        # If examples is a Dataset object or list-like
        iterator = examples
    else:
        # If it's a single example or unknown type
        iterator = [examples]
    
    for example in iterator:
        # Convert example to dict if needed
        if hasattr(example, 'to_dict'):
            # This is a Dataset row
            example = example[0] if isinstance(example, tuple) else example
            try:
                example = example.to_dict()
            except:
                # Fall back to dictionary-style access
                example_dict = {}
                for key in example.keys():
                    example_dict[key] = example[key]
                example = example_dict
        elif not isinstance(example, dict):
            # Try to convert to dict
            try:
                example = dict(example)
            except:
                logger.warning(f"Could not convert example to dictionary: {type(example)}")
                continue
            
        # Format input
        text = format_input_for_hallucination_detection(example, task_type)
        
        # Check if there are hallucination annotations based on the score field
        # IMPORTANT: According to the dataset description:
        # Score 0 = Response is NOT faithful, hallucination detected
        # Score 1 = Response is faithful, NO hallucination detected
        score = example.get("score", None)
        # Convert to numeric if it's not
        if not isinstance(score, (int, float)) and score is not None:
            try:
                score = float(score) if score else None
            except (ValueError, TypeError):
                score = None
        
        # Determine hallucination based on score
        # If score is 0, the response contains hallucinations
        # If score is 1, the response is faithful (no hallucinations)
        if score is not None:
            has_hallucination = score == 0  # Score 0 means hallucinated
        else:
            # Default behavior if score is not provided
            logger.warning(f"No score found for example in task {task_type}, assuming no hallucination")
            has_hallucination = False
        
        texts.append(text)
        labels.append("HALLUCINATED" if has_hallucination else "NOT_HALLUCINATED")
    
    # Log distribution before returning
    hallucinated_count = sum(1 for label in labels if label == "HALLUCINATED")
    non_hallucinated_count = len(labels) - hallucinated_count
    logger.info(f"Prepared {len(texts)} examples: {hallucinated_count} HALLUCINATED, {non_hallucinated_count} NOT_HALLUCINATED")
    
    return texts, labels

def evaluate_classifier(classifier, test_texts, test_labels):
    """Evaluate the classifier at the example level."""
    all_predictions = []
    
    start_time = time.time()
    for text in tqdm(test_texts, desc="Evaluating"):
        try:
            predictions = classifier.predict(text)
            # Get the top prediction
            top_pred = predictions[0][0] if predictions else "NOT_HALLUCINATED"
        except Exception as e:
            logger.warning(f"Error during prediction: {e}. Using default prediction.")
            top_pred = "NOT_HALLUCINATED"
            
        all_predictions.append(top_pred)
    end_time = time.time()
    
    # Calculate metrics
    y_true = [1 if label == "HALLUCINATED" else 0 for label in test_labels]
    y_pred = [1 if pred == "HALLUCINATED" else 0 for pred in all_predictions]
    
    # Handle edge case where all predictions are the same
    if len(set(y_pred)) <= 1 or len(set(y_true)) <= 1:
        logger.warning("All predictions or ground truth are the same class. Metrics might be unreliable.")
        # Handle case with all predictions the same
        if len(set(y_pred)) <= 1:
            if y_pred[0] == 1:  # All hallucinated
                precision = sum(y_true) / len(y_true) if len(y_true) > 0 else 0
                recall = 1.0 if sum(y_true) > 0 else 0
            else:  # All not hallucinated
                precision = 0
                recall = 0 if sum(y_true) > 0 else 1.0
            
            # Calculate F1 from precision and recall
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            # Calculate using sklearn, which will set warning
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
    else:
        # Normal case
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
    
    # Calculate throughput
    throughput = len(test_texts) / (end_time - start_time) if (end_time - start_time) > 0 else 0
    
    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "throughput": throughput,
        "predictions": all_predictions,
        "true_labels": test_labels
    }

def train_and_evaluate(args):
    """Train and evaluate the hallucination detector."""
    logger.info(f"Loading dataset with training parameters:")
    if args.use_all_data:
        logger.info("  - Using all data for both training and testing")
    elif args.train_count is not None:
        logger.info(f"  - Using fixed count of {args.train_count} examples for training")
    else:
        logger.info(f"  - Using {args.train_percentage}% of data for training")
    
    splits = load_and_split_dataset(args)
    
    # Check if we have any data
    if not splits:
        logger.error("No data found in dataset. Please check the dataset structure.")
        return {}
    
    # Initialize classifier
    logger.info(f"Initializing adaptive classifier with {args.model_name}")
    
    config= {'max_length': 1000}
    classifier = AdaptiveClassifier(args.model_name, config=config)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training stats
    task_stats = {}
    
    # Process each task
    for task_type in splits.keys():
        logger.info(f"Processing task: {task_type}")
        
        # Prepare examples
        train_texts, train_labels = prepare_examples(splits[task_type]["train"], task_type)
        test_texts, test_labels = prepare_examples(splits[task_type]["test"], task_type)
        
        # Skip if we have no data or don't meet minimum example requirement
        if len(train_texts) == 0 or len(test_texts) == 0:
            logger.warning(f"No data for task {task_type}. Skipping.")
            continue
        elif len(train_texts) < args.min_examples or len(test_texts) < args.min_examples:
            logger.warning(f"Not enough data for task {task_type}. Minimum required: {args.min_examples}. Found: {len(train_texts)} train, {len(test_texts)} test. Skipping.")
            continue
        
        # Log class distribution
        train_positive = sum(1 for label in train_labels if label == "HALLUCINATED")
        train_negative = len(train_labels) - train_positive
        test_positive = sum(1 for label in test_labels if label == "HALLUCINATED")
        test_negative = len(test_labels) - test_positive
        
        logger.info(f"Train distribution for {task_type}: {train_positive} HALLUCINATED, {train_negative} NOT_HALLUCINATED")
        logger.info(f"Test distribution for {task_type}: {test_positive} HALLUCINATED, {test_negative} NOT_HALLUCINATED")
        
        logger.info(f"Training on {len(train_texts)} examples for task {task_type}")
        
        # Add examples in batches
        batch_size = args.batch_size
        for i in range(0, len(train_texts), batch_size):
            end_idx = min(i + batch_size, len(train_texts))
            batch_texts = train_texts[i:end_idx]
            batch_labels = train_labels[i:end_idx]
            classifier.add_examples(batch_texts, batch_labels)
            
            # Log progress for larger datasets
            if (i // batch_size) % 10 == 0 and i > 0:
                logger.info(f"Added {i + len(batch_texts)} examples so far...")
        
        # Evaluate on test set
        logger.info(f"Evaluating on {len(test_texts)} examples for task {task_type}")
        metrics = evaluate_classifier(classifier, test_texts, test_labels)
        
        # Confusion matrix for better understanding
        true_positives = sum(1 for true, pred in zip(metrics["true_labels"], metrics["predictions"]) 
                           if true == "HALLUCINATED" and pred == "HALLUCINATED")
        false_positives = sum(1 for true, pred in zip(metrics["true_labels"], metrics["predictions"]) 
                            if true == "NOT_HALLUCINATED" and pred == "HALLUCINATED")
        true_negatives = sum(1 for true, pred in zip(metrics["true_labels"], metrics["predictions"]) 
                           if true == "NOT_HALLUCINATED" and pred == "NOT_HALLUCINATED")
        false_negatives = sum(1 for true, pred in zip(metrics["true_labels"], metrics["predictions"]) 
                            if true == "HALLUCINATED" and pred == "NOT_HALLUCINATED")
        
        confusion_matrix = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        }
        
        # Store results
        task_stats[task_type] = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "throughput": metrics["throughput"],
            "num_train_examples": len(train_texts),
            "num_test_examples": len(test_texts),
            "confusion_matrix": confusion_matrix
        }
        
        logger.info(f"Task: {task_type} - Precision: {metrics['precision']:.2f}%, "
                    f"Recall: {metrics['recall']:.2f}%, F1: {metrics['f1']:.2f}%")
        logger.info(f"Throughput: {metrics['throughput']:.2f} examples/second")
        logger.info(f"Confusion Matrix: TP={true_positives}, FP={false_positives}, "
                    f"TN={true_negatives}, FN={false_negatives}")
        
        # Remove large objects before JSON serialization
        if "predictions" in metrics:
            del metrics["predictions"]
        if "true_labels" in metrics:
            del metrics["true_labels"]
    
    # Only proceed with overall stats if we have task stats
    if not task_stats:
        logger.warning("No task statistics available. Skipping overall metrics calculation.")
        return {"metadata": {
            "train_percentage": args.train_percentage,
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": "No data processed successfully"
        }}
    
    # Calculate overall metrics (weighted by dataset size)
    total_test = sum(stats["num_test_examples"] for stats in task_stats.values())
    
    if total_test > 0:
        overall_precision = sum(stats["precision"] * stats["num_test_examples"] / total_test for stats in task_stats.values())
        overall_recall = sum(stats["recall"] * stats["num_test_examples"] / total_test for stats in task_stats.values())
        overall_f1 = sum(stats["f1"] * stats["num_test_examples"] / total_test for stats in task_stats.values())
        overall_throughput = sum(stats["throughput"] * stats["num_test_examples"] / total_test for stats in task_stats.values())
        
        # Sum confusion matrices
        overall_tp = sum(stats["confusion_matrix"]["true_positives"] for stats in task_stats.values())
        overall_fp = sum(stats["confusion_matrix"]["false_positives"] for stats in task_stats.values())
        overall_tn = sum(stats["confusion_matrix"]["true_negatives"] for stats in task_stats.values())
        overall_fn = sum(stats["confusion_matrix"]["false_negatives"] for stats in task_stats.values())
        
        task_stats["overall"] = {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "throughput": overall_throughput,
            "confusion_matrix": {
                "true_positives": overall_tp,
                "false_positives": overall_fp,
                "true_negatives": overall_tn,
                "false_negatives": overall_fn
            }
        }
        
        logger.info(f"Overall - Precision: {overall_precision:.2f}%, "
                    f"Recall: {overall_recall:.2f}%, F1: {overall_f1:.2f}%")
        logger.info(f"Overall Throughput: {overall_throughput:.2f} examples/second")
        logger.info(f"Overall Confusion Matrix: TP={overall_tp}, FP={overall_fp}, "
                    f"TN={overall_tn}, FN={overall_fn}")
    
    # Add training metadata
    task_stats["metadata"] = {
        "train_percentage": args.train_percentage,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save the model
    logger.info(f"Saving model to {save_dir}")
    classifier.save(save_dir)
    
    # Save metrics
    metrics_file = save_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(task_stats, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        repo_id = f"adaptive-classifier/{args.push_to_hub}"
        logger.info(f"Pushing model to HuggingFace Hub: {repo_id}")
        try:
            classifier.push_to_hub(repo_id)
            logger.info(f"Successfully pushed to {repo_id}")
        except Exception as e:
            logger.error(f"Error pushing to HuggingFace Hub: {e}")
            logger.info("Trying alternative push method...")
            
            try:
                from huggingface_hub import upload_folder
                upload_folder(folder_path=str(save_dir), repo_id=repo_id, repo_type="model")
                logger.info(f"Successfully pushed to {repo_id} using upload_folder")
            except Exception as e:
                logger.error(f"Error using alternative push method: {e}")
    
    return task_stats

def print_comparison_table(stats, args):
    """Print a comparison table similar to the one in the paper."""
    print("\nRESULTS COMPARISON WITH PAPER")
    print("=" * 80)
    print(f"{'TASK':<15} {'METHOD':<25} {'PRECISION':<10} {'RECALL':<10} {'F1':<10}")
    print("-" * 80)
    
    # Paper baseline results - standard benchmarks from LettuceDetect paper
    paper_baselines = {
        "qa": [
            ("Luna (paper)", 37.8, 80.0, 51.3),
            ("LettuceDetect-large", 65.93, 75.0, 70.18)
        ],
        "data2txt": [
            ("Luna (paper)", 64.9, 91.2, 75.9),
            ("LettuceDetect-large", 90.45, 86.7, 88.54)
        ],
        "summarization": [
            ("Luna (paper)", 40.0, 76.5, 52.5),
            ("LettuceDetect-large", 64.0, 55.88, 59.69)
        ]
    }
    
    # Overall paper results
    overall_paper_results = [
        ("Luna (paper)", 52.7, 86.1, 65.4),
        ("LettuceDetect-large", 80.44, 78.05, 79.22)
    ]
    
    # Print results for each task
    tasks = [task for task in stats.keys() if task not in ["overall", "metadata"]]
    for task in tasks:
        # Print our results
        print(f"{task:<15} {'Our Model':<25} {stats[task]['precision']:<10.2f} {stats[task]['recall']:<10.2f} {stats[task]['f1']:<10.2f}")
        
        # Print paper baselines if available for this task
        if task.lower() in paper_baselines:
            for method, precision, recall, f1 in paper_baselines[task.lower()]:
                print(f"{'':<15} {method:<25} {precision:<10.1f} {recall:<10.1f} {f1:<10.1f}")
        else:
            print(f"{'':<15} {'(No paper baseline)':<25} {'-':<10} {'-':<10} {'-':<10}")
    
    # Print overall results
    print("-" * 80)
    if "overall" in stats:
        print(f"{'Overall':<15} {'Our Model':<25} {stats['overall']['precision']:<10.2f} {stats['overall']['recall']:<10.2f} {stats['overall']['f1']:<10.2f}")
    else:
        print(f"{'Overall':<15} {'Our Model':<25} {'-':<10} {'-':<10} {'-':<10}")
        
    # Print paper overall results
    for method, precision, recall, f1 in overall_paper_results:
        print(f"{'':<15} {method:<25} {precision:<10.1f} {recall:<10.1f} {f1:<10.1f}")
    
    print("=" * 80)
    print(f"Training with {args.train_percentage}% of data using model: {args.model_name}")
    if "overall" in stats:
        print(f"Throughput: {stats['overall']['throughput']:.2f} examples/second")

def main():
    args = parse_args()
    
    # Configure logging based on arguments
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Enable more verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    set_seeds(args.seed)
    
    try:
        stats = train_and_evaluate(args)
        
        if not stats or not any(key not in ["metadata"] for key in stats.keys()):
            logger.warning("No statistics available. Cannot generate report.")
            return
            
        # Print final comparison table
        print("\nResults Summary:")
        print("=" * 80)
        print(f"{'Task':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Throughput':<12}")
        print("-" * 80)
        
        for task_type, metrics in stats.items():
            if task_type not in ["overall", "metadata"]:
                print(f"{task_type:<15} {metrics['precision']:<12.2f} {metrics['recall']:<12.2f} "
                      f"{metrics['f1']:<12.2f} {metrics['throughput']:<12.2f}")
        
        print("-" * 80)
        if "overall" in stats:
            print(f"{'overall':<15} {stats['overall']['precision']:<12.2f} {stats['overall']['recall']:<12.2f} "
                  f"{stats['overall']['f1']:<12.2f} {stats['overall']['throughput']:<12.2f}")
        print("=" * 80)
        
        # Compare with paper results
        print_comparison_table(stats, args)
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()
