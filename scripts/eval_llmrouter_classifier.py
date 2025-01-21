import argparse
import argparse
import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple

import datasets
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from huggingface_hub import HfFolder

from adaptive_classifier import AdaptiveClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_args() -> argparse.Namespace:
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark Adaptive Classifier')
    parser.add_argument(
        '--model', 
        type=str, 
        default='distilbert/distilbert-base-cased',
        help='Base transformer model to use'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--max-samples', 
        type=int, 
        default=1200,
        help='Maximum number of samples to use (for testing)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='benchmark_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push the trained model to HuggingFace Hub'
    )
    parser.add_argument(
        '--hub-repo',
        type=str,
        help='HuggingFace Hub repository ID (e.g. "username/model-name") for pushing the model'
    )
    parser.add_argument(
        '--hub-token',
        type=str,
        help='HuggingFace Hub token. If not provided, will look for the token in the environment'
    )
    return parser.parse_args()

def load_dataset(max_samples: int = None) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Load and preprocess the dataset."""
    logger.info("Loading routellm/gpt4_dataset...")
    
    # Load dataset
    dataset = datasets.load_dataset("routellm/gpt4_dataset")
    
    def preprocess_function(example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert scores to binary labels."""
        score = example['mixtral_score']
        # Scores 4-5 -> LOW, 1-3 -> HIGH
        label = 'LOW' if score >= 4 else 'HIGH'
        return {
            'text': example['prompt'],
            'label': label
        }
    
    # Process train and validation sets
    train_dataset = dataset['train'].map(preprocess_function)
    val_dataset = dataset['validation'].map(preprocess_function)
    
    # Limit samples if specified
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(max_samples, len(val_dataset))))
    
    logger.info(f"Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def train_classifier(
    model_name: str,
    train_dataset: datasets.Dataset,
    batch_size: int
) -> AdaptiveClassifier:
    """Train the adaptive classifier with improved balancing and configuration."""
    logger.info(f"Initializing classifier with model: {model_name}")
    
    # Count class distribution
    labels = train_dataset['label']
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info(f"Original class distribution: {label_counts}")
    
    # Calculate class weights
    total_samples = sum(label_counts.values())
    class_weights = {
        label: total_samples / (len(label_counts) * count)
        for label, count in label_counts.items()
    }
    
    logger.info(f"Class weights: {class_weights}")
    
    # Initialize classifier with optimized config
    classifier = AdaptiveClassifier(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        config={
            'batch_size': batch_size,
            'max_examples_per_class': 500,  # Reduced to prevent overfitting
            'prototype_update_frequency': 50,  # More frequent updates
            'learning_rate': 0.0005,  # Slightly higher learning rate
            'similarity_threshold': 0.7,  # Increased threshold
            'prototype_weight': 0.8,  # Give more weight to prototypes
            'neural_weight': 0.2,  # Less weight to neural network
        }
    )
    
    # Create balanced batches
    texts = train_dataset['text']
    
    # Group examples by label
    examples_by_label = {label: [] for label in label_counts.keys()}
    for text, label in zip(texts, labels):
        examples_by_label[label].append(text)
    
    # Find minority class size
    min_class_size = min(len(examples) for examples in examples_by_label.values())
    
    # Create balanced training data
    balanced_texts = []
    balanced_labels = []
    
    # Sample equally from each class
    for label, examples in examples_by_label.items():
        # Oversample minority class, undersample majority class
        if len(examples) < min_class_size * 2:
            # Oversample smaller classes
            sampled_examples = random.choices(examples, k=min_class_size * 2)
        else:
            # Undersample larger classes
            sampled_examples = random.sample(examples, min_class_size * 2)
        
        balanced_texts.extend(sampled_examples)
        balanced_labels.extend([label] * len(sampled_examples))
    
    # Shuffle consistently
    combined = list(zip(balanced_texts, balanced_labels))
    random.Random(42).shuffle(combined)
    balanced_texts, balanced_labels = zip(*combined)
    
    # Process in batches
    total_batches = (len(balanced_texts) + batch_size - 1) // batch_size
    logger.info(f"Total batches: {total_batches}")
    
    for i in tqdm(range(0, len(balanced_texts), batch_size), total=total_batches):
        try:
            batch_texts = balanced_texts[i:i + batch_size]
            batch_labels = balanced_labels[i:i + batch_size]
            
            # Debug information
            if i % (batch_size * 10) == 0:
                logger.debug(f"Batch {i//batch_size + 1}/{total_batches}")
                label_counts = {label: batch_labels.count(label) 
                              for label in set(batch_labels)}
                logger.debug(f"Batch class distribution: {label_counts}")
            
            classifier.add_examples(batch_texts, batch_labels)
            
        except Exception as e:
            logger.error(f"Error in batch {i//batch_size + 1}")
            logger.error(str(e))
            raise
    
    # Log final statistics
    memory_stats = classifier.get_memory_stats()
    logger.info(f"Final memory stats: {memory_stats}")
    
    return classifier

def evaluate_classifier(
    classifier: AdaptiveClassifier,
    val_dataset: datasets.Dataset,
    batch_size: int
) -> Dict[str, Any]:
    """Evaluate the classifier."""
    logger.info("Starting evaluation...")
    
    predictions = []
    true_labels = val_dataset['label']
    texts = val_dataset['text']
    
    # Process in batches
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches):
        batch_texts = texts[i:i + batch_size]
        batch_predictions = classifier.predict_batch(batch_texts, k=1)
        predictions.extend([pred[0][0] for pred in batch_predictions])
    
    # Calculate metrics
    report = classification_report(true_labels, predictions, output_dict=True)
    conf_matrix = confusion_matrix(true_labels, predictions).tolist()
    
    # Get memory stats
    memory_stats = classifier.get_memory_stats()
    example_stats = classifier.get_example_statistics()
    
    results = {
        'metrics': report,
        'confusion_matrix': conf_matrix,
        'memory_stats': memory_stats,
        'example_stats': example_stats
    }
    
    return results

def push_to_hub(
    classifier: AdaptiveClassifier,
    repo_id: str,
    token: str = None,
    metrics: Dict[str, Any] = None
) -> str:
    """Push the classifier to HuggingFace Hub.
    
    Args:
        classifier: Trained classifier to push
        repo_id: HuggingFace Hub repository ID
        token: HuggingFace Hub token
        metrics: Optional evaluation metrics to add to model card
        
    Returns:
        URL of the model on the Hub
    """
    logger.info(f"Pushing model to HuggingFace Hub: {repo_id}")
    
    # Set token if provided
    if token:
        HfFolder.save_token(token)
    
    try:
        # Push to hub with evaluation results in model card
        url = classifier.push_to_hub(
            repo_id,
            commit_message="Upload from benchmark script",
        )
        logger.info(f"Successfully pushed model to Hub: {url}")
        return url
    except Exception as e:
        logger.error(f"Error pushing to Hub: {str(e)}")
        raise

def save_results(
    classifier: AdaptiveClassifier,
    results: Dict[str, Any],
    args: argparse.Namespace
):
    """Save evaluation results and optionally push to Hub."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"benchmark_results_{timestamp}.json"
    filepath = os.path.join(args.output_dir, filename)
    
    # Add run configuration to results
    results['config'] = {
        'model': args.model,
        'batch_size': args.batch_size,
        'max_samples': args.max_samples,
        'timestamp': timestamp
    }

    # Save classifier locally
    classifier.save(args.output_dir)
    
    # Save results locally
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")
    
    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        if not args.hub_repo:
            raise ValueError("--hub-repo must be specified when using --push-to-hub")
        
        hub_url = push_to_hub(
            classifier,
            args.hub_repo,
            args.hub_token,
            metrics=results['metrics']
        )
        results['hub_url'] = hub_url
        
        # Update saved results with hub URL
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary to console
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"Model: {args.model}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print("\nPer-class metrics:")
    for label in ['HIGH', 'LOW']:
        metrics = results['metrics'][label]
        print(f"\n{label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")
    print("\nConfusion Matrix:")
    print("            Predicted")
    print("             HIGH  LOW")
    print(f"Actual HIGH  {results['confusion_matrix'][0][0]:4d}  {results['confusion_matrix'][0][1]:4d}")
    print(f"      LOW   {results['confusion_matrix'][1][0]:4d}  {results['confusion_matrix'][1][1]:4d}")
    
    if args.push_to_hub:
        print(f"\nModel pushed to HuggingFace Hub: {results['hub_url']}")

def main():
    """Main execution function."""
    args = setup_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Load dataset
    train_dataset, val_dataset = load_dataset(args.max_samples)
    
    # Train classifier
    classifier = train_classifier(args.model, train_dataset, args.batch_size)
    
    # Evaluate
    results = evaluate_classifier(classifier, val_dataset, args.batch_size)
    
    # Save and display results
    save_results(classifier, results, args)

if __name__ == "__main__":
    main()
