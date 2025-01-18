import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

import datasets
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

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
        default='bert-base-uncased',
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
        default=None,
        help='Maximum number of samples to use (for testing)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='benchmark_results',
        help='Directory to save results'
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
    """Train the adaptive classifier."""
    logger.info(f"Initializing classifier with model: {model_name}")
    
    # Initialize classifier
    classifier = AdaptiveClassifier(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        config={
            'batch_size': batch_size,
            'max_examples_per_class': 5000,  # Increased for this task
            'prototype_update_frequency': 100,
            'learning_rate': 0.0001  # Lower learning rate for stability
        }
    )
    
    # Prepare batches
    texts = train_dataset['text']
    labels = train_dataset['label']
    
    # Process in batches
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    logger.info("Training classifier...")
    start_time = time.time()
    
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches):
        try:
            batch_texts = texts[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            # Debug information before adding examples
            logger.debug(f"Processing batch {i//batch_size + 1}/{total_batches}")
            logger.debug(f"Batch size: {len(batch_texts)}")
            logger.debug(f"Sample text: {batch_texts[0][:100]}...")
            logger.debug(f"Labels in batch: {set(batch_labels)}")
            
            # Try to add examples and catch any errors
            try:
                classifier.add_examples(batch_texts, batch_labels)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    # Print detailed debugging information
                    logger.error("Size mismatch error occurred!")
                    logger.error(f"Error message: {str(e)}")
                    logger.error("\nDebug Information:")
                    logger.error(f"Batch index: {i}")
                    logger.error(f"Number of texts: {len(batch_texts)}")
                    logger.error(f"Number of labels: {len(batch_labels)}")
                    logger.error(f"Unique labels in batch: {set(batch_labels)}")
                    logger.error("\nFirst few examples:")
                    for j in range(min(3, len(batch_texts))):
                        logger.error(f"Example {j}:")
                        logger.error(f"Text: {batch_texts[j][:100]}...")
                        logger.error(f"Label: {batch_labels[j]}")
                    logger.error("\nClassifier state:")
                    logger.error(f"Label to ID mapping: {classifier.label_to_id}")
                    logger.error(f"Number of classes: {len(classifier.label_to_id)}")
                    raise  # Re-raise the error after logging details
                else:
                    raise  # Re-raise other RuntimeErrors
                    
            # Log progress periodically
            if (i // batch_size) % 31 == 0:
                logger.info(f"Processed {i + len(batch_texts)} examples")
                memory_stats = classifier.get_memory_stats()
                logger.info(f"Current memory stats: {memory_stats}")
                
        except Exception as e:
            logger.error(f"Error processing batch starting at index {i}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("\nBatch details:")
            logger.error(f"Batch size: {len(batch_texts)}")
            logger.error(f"First text in batch: {batch_texts[0][:100]}...")
            logger.error(f"Labels in batch: {set(batch_labels)}")
            raise  # Re-raise the exception after logging details
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
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

def save_results(results: Dict[str, Any], args: argparse.Namespace):
    """Save evaluation results."""
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
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")
    
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
    save_results(results, args)

if __name__ == "__main__":
    main()
