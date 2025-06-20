#!/usr/bin/env python3
"""
Evaluation script for Strategic Classifier on AI-Secure/adv_glue dataset.

This script evaluates the strategic classification feature of the adaptive classifier
using the adversarial GLUE dataset (adv_sst2 subset) to test robustness against
adversarial examples.

Usage:
    python eval_strategic_classifier_adv_glue.py [--model MODEL_NAME] [--output OUTPUT_FILE]
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import random

# Add src to path to import adaptive_classifier
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from adaptive_classifier import AdaptiveClassifier

def convert_to_serializable(obj):
    """Convert numpy/torch types to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_adv_glue_dataset() -> Tuple[List[str], List[str]]:
    """Load the AI-Secure/adv_glue dataset (adv_sst2 subset, validation split).
    
    Returns:
        Tuple of (texts, labels) where labels are converted to string format
    """
    logger.info("Loading AI-Secure/adv_glue dataset (adv_sst2 subset)...")
    
    try:
        # Load the dataset
        dataset = load_dataset("AI-Secure/adv_glue", "adv_sst2", split="validation")
        
        # Extract texts and labels
        texts = dataset['sentence']
        labels = dataset['label']
        
        # Convert labels to string format (0 -> "negative", 1 -> "positive")
        label_map = {0: "negative", 1: "positive"}
        labels = [label_map[label] for label in labels]
        
        logger.info(f"Loaded {len(texts)} examples")
        logger.info(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return texts, labels
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def split_dataset(
    texts: List[str], 
    labels: List[str], 
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split dataset into train and test sets.
    
    Args:
        texts: List of text samples
        labels: List of corresponding labels
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_texts, test_texts, train_labels, test_labels)
    """
    logger.info(f"Splitting dataset with test_size={test_size}")
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels  # Maintain label distribution
    )
    
    logger.info(f"Train set: {len(train_texts)} examples")
    logger.info(f"Test set: {len(test_texts)} examples")
    
    return train_texts, test_texts, train_labels, test_labels

def create_strategic_config(model_name: str, cost_strategy: str = "balanced") -> Dict[str, Any]:
    """Create configuration for strategic classification with balanced cost functions.
    
    Args:
        model_name: Name of the HuggingFace model to get embedding dimension from
        cost_strategy: Cost function strategy ('balanced', 'sparse_low', 'uniform_low', 'minimal')
    
    Returns:
        Configuration dictionary with strategic settings
    """
    # Get embedding dimension from the model configuration
    try:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(model_name)
        embedding_dim = model_config.hidden_size
        
        logger.info(f"Model {model_name} embedding dimension: {embedding_dim}")
    except Exception as e:
        logger.error(f"Failed to get embedding dimension for model {model_name}: {e}")
        raise RuntimeError(f"Could not determine embedding dimension for model {model_name}. "
                         f"Please ensure the model exists and is accessible.")
    
    # Create more balanced cost coefficients
    if cost_strategy == "balanced":
        # 50% of dimensions manipulable at moderate cost - more realistic
        manipulable_dims = int(embedding_dim * 0.5)  # 50% of dimensions
        cost_coefficients = [0.0] * embedding_dim
        
        import random
        random.seed(42)
        manipulable_indices = random.sample(range(embedding_dim), manipulable_dims)
        
        for idx in manipulable_indices:
            cost_coefficients[idx] = 0.3  # Lower cost for more realistic manipulation
        
        logger.info(f"Balanced cost function: {manipulable_dims} manipulable dimensions with cost 0.3")
        
    elif cost_strategy == "sparse_low":
        # Sparse but with lower costs
        manipulable_dims = int(embedding_dim * 0.2)  # 20% of dimensions
        cost_coefficients = [0.0] * embedding_dim
        
        import random
        random.seed(42)
        manipulable_indices = random.sample(range(embedding_dim), manipulable_dims)
        
        for idx in manipulable_indices:
            cost_coefficients[idx] = 0.4  # Moderate cost
        
        logger.info(f"Sparse low cost function: {manipulable_dims} manipulable dimensions with cost 0.4")
        
    elif cost_strategy == "uniform_low":
        # Uniform low cost - all dimensions can be modified cheaply
        cost_coefficients = [0.15] * embedding_dim  # Very low uniform cost
        
        logger.info(f"Uniform low cost function: 0.15 across all {embedding_dim} dimensions")
        
    elif cost_strategy == "minimal":
        # Minimal strategic influence for debugging
        cost_coefficients = [0.05] * embedding_dim  # Minimal cost
        
        logger.info(f"Minimal cost function: 0.05 across all {embedding_dim} dimensions")
        
    else:
        # Legacy support for old strategies with lower costs
        if cost_strategy == "sparse_high":
            manipulable_dims = int(embedding_dim * 0.3)  # Increased to 30%
            cost_coefficients = [0.0] * embedding_dim
            
            import random
            random.seed(42)
            manipulable_indices = random.sample(range(embedding_dim), manipulable_dims)
            
            for idx in manipulable_indices:
                cost_coefficients[idx] = 0.4  # Reduced from 0.8 to 0.4
            
            logger.info(f"Sparse high (adjusted) cost function: {manipulable_dims} manipulable dimensions with cost 0.4")
        else:
            raise ValueError(f"Unknown cost strategy: {cost_strategy}")
    
    return {
        'enable_strategic_mode': True,
        'cost_function_type': 'linear',
        'cost_coefficients': cost_coefficients,
        'strategic_lambda': 0.05,  # Much lower strategic weight
        'strategic_training_frequency': 10,  # Less frequent strategic training
        # More conservative blending - favor regular component
        'strategic_blend_regular_weight': 0.7,  # Increased regular weight
        'strategic_blend_strategic_weight': 0.3,  # Decreased strategic weight
        # Robust prediction weights
        'strategic_robust_proto_weight': 0.8,
        'strategic_robust_head_weight': 0.2,
        # Strategic prediction weights
        'strategic_prediction_proto_weight': 0.5,
        'strategic_prediction_head_weight': 0.5
    }

def train_classifier(
    model_name: str,
    train_texts: List[str],
    train_labels: List[str],
    config: Dict[str, Any] = None
) -> AdaptiveClassifier:
    """Train an adaptive classifier on the training data.
    
    Args:
        model_name: Name of the transformer model to use
        train_texts: Training texts
        train_labels: Training labels
        config: Optional configuration for the classifier
        
    Returns:
        Trained AdaptiveClassifier instance
    """
    logger.info(f"Training classifier with model: {model_name}")
    logger.info(f"Strategic mode: {config.get('enable_strategic_mode', False) if config else False}")
    
    # Initialize classifier
    classifier = AdaptiveClassifier(model_name, config=config)
    
    # Debug: Check if strategic mode is properly enabled
    if config and config.get('enable_strategic_mode'):
        logger.info(f"Strategic mode enabled: {classifier.strategic_mode}")
        logger.info(f"Strategic cost function: {classifier.strategic_cost_function is not None}")
        logger.info(f"Strategic optimizer: {classifier.strategic_optimizer is not None}")
        
        # If strategic mode failed to initialize, this is a critical error
        if not classifier.strategic_mode:
            error_msg = (
                f"CRITICAL ERROR: Strategic mode failed to initialize!\n"
                f"Config cost_coefficients: {config.get('cost_coefficients')}\n"
                f"Config cost_function_type: {config.get('cost_function_type')}\n"
                f"Strategic mode is required for this evaluation. "
                f"Please fix the strategic classifier implementation."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    # Add training examples in batches for better performance
    batch_size = 50
    for i in range(0, len(train_texts), batch_size):
        batch_texts = train_texts[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]
        
        classifier.add_examples(batch_texts, batch_labels)
        
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i + len(batch_texts)} / {len(train_texts)} training examples")
    
    logger.info("Training completed")
    return classifier

def evaluate_classifier(
    classifier: AdaptiveClassifier,
    test_texts: List[str],
    test_labels: List[str],
    mode: str = "regular"
) -> Dict[str, Any]:
    """Evaluate a classifier on test data.
    
    Args:
        classifier: Trained classifier
        test_texts: Test texts
        test_labels: Test labels
        mode: Evaluation mode ("regular", "strategic", "robust", "dual")
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating classifier in {mode} mode...")
    
    predictions = []
    prediction_probs = []
    prediction_times = []
    
    # Get predictions for each test example
    for i, text in enumerate(test_texts):
        start_time = time.time()
        
        # Get prediction based on mode
        if mode == "regular":
            if classifier.strategic_mode:
                # Use internal regular prediction method
                pred_results = classifier._predict_regular(text, k=2)
            else:
                pred_results = classifier.predict(text, k=2)
        elif mode == "strategic":
            pred_results = classifier.predict_strategic(text, k=2)
        elif mode == "robust":
            pred_results = classifier.predict_robust(text, k=2)
        elif mode == "dual":
            pred_results = classifier.predict(text, k=2)  # Uses dual system in strategic mode
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")
        
        end_time = time.time()
        
        # Extract top prediction
        if pred_results:
            top_pred, top_prob = pred_results[0]
            predictions.append(top_pred)
            prediction_probs.append(top_prob)
        else:
            # Fallback if no prediction
            predictions.append("negative")  # Default prediction
            prediction_probs.append(0.5)
        
        prediction_times.append(end_time - start_time)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Evaluated {i + 1} / {len(test_texts)} examples")
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='weighted'
    )
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        test_labels, predictions, average=None, labels=["negative", "positive"]
    )
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions, labels=["negative", "positive"])
    
    # Confidence statistics
    avg_confidence = np.mean(prediction_probs)
    std_confidence = np.std(prediction_probs)
    avg_prediction_time = np.mean(prediction_times)
    
    results = {
        'mode': mode,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class_metrics': {
            'negative': {
                'precision': float(precision_per_class[0]),
                'recall': float(recall_per_class[0]),
                'f1_score': float(f1_per_class[0]),
                'support': int(support_per_class[0])
            },
            'positive': {
                'precision': float(precision_per_class[1]),
                'recall': float(recall_per_class[1]),
                'f1_score': float(f1_per_class[1]),
                'support': int(support_per_class[1])
            }
        },
        'confusion_matrix': cm.tolist(),
        'avg_confidence': float(avg_confidence),
        'std_confidence': float(std_confidence),
        'avg_prediction_time': float(avg_prediction_time),
        'total_predictions': len(predictions)
    }
    
    logger.info(f"{mode.capitalize()} mode results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1-score: {f1:.4f}")
    logger.info(f"  Avg confidence: {avg_confidence:.4f}")
    logger.info(f"  Avg prediction time: {avg_prediction_time:.4f}s")
    
    return results

def generate_manipulated_data(
    strategic_classifier: AdaptiveClassifier,
    test_texts: List[str],
    manipulation_level: float = 1.0
) -> List[torch.Tensor]:
    """Generate strategically manipulated versions of test data.
    
    Args:
        strategic_classifier: Classifier with strategic capabilities
        test_texts: Original test texts
        manipulation_level: Level of manipulation (0.0 = no manipulation, 1.0 = full manipulation)
        
    Returns:
        List of manipulated embeddings
    """
    if not strategic_classifier.strategic_mode:
        logger.warning("Strategic mode not enabled - returning original embeddings")
        return strategic_classifier._get_embeddings(test_texts)
    
    logger.info(f"Generating manipulated data with manipulation level: {manipulation_level}")
    
    manipulated_embeddings = []
    original_embeddings = strategic_classifier._get_embeddings(test_texts)
    
    # Create classifier function for strategic optimization
    def classifier_func(x):
        with torch.no_grad():
            if strategic_classifier.adaptive_head is not None:
                strategic_classifier.adaptive_head.eval()
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                logits = strategic_classifier.adaptive_head(x.to(strategic_classifier.device))
                return F.softmax(logits, dim=-1)
            else:
                num_classes = len(strategic_classifier.label_to_id) if strategic_classifier.label_to_id else 1
                return torch.ones(1, num_classes) / num_classes
    
    for i, original_embedding in enumerate(original_embeddings):
        if torch.rand(1).item() < manipulation_level:
            # Apply strategic manipulation
            try:
                manipulated_embedding = strategic_classifier.strategic_cost_function.compute_best_response(
                    original_embedding, classifier_func
                )
                manipulated_embeddings.append(manipulated_embedding)
            except Exception as e:
                logger.warning(f"Strategic manipulation failed for example {i}: {e}")
                manipulated_embeddings.append(original_embedding)
        else:
            # No manipulation
            manipulated_embeddings.append(original_embedding)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1} / {len(test_texts)} manipulated examples")
    
    return manipulated_embeddings

def evaluate_classifier_on_embeddings(
    classifier: AdaptiveClassifier,
    embeddings: List[torch.Tensor],
    test_labels: List[str],
    mode: str = "regular"
) -> Dict[str, Any]:
    """Evaluate a classifier on pre-computed embeddings.
    
    Args:
        classifier: Trained classifier
        embeddings: List of embeddings to evaluate on
        test_labels: True labels
        mode: Evaluation mode (for strategic classifiers)
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating classifier on manipulated data in {mode} mode...")
    
    predictions = []
    prediction_probs = []
    prediction_times = []
    
    # Get predictions for each embedding
    for i, embedding in enumerate(embeddings):
        start_time = time.time()
        
        # Use the embedding directly with prediction logic
        try:
            if mode == "regular" or not classifier.strategic_mode:
                # Get predictions from embedding (regular mode)
                pred_results = classifier._predict_from_embedding(embedding, k=2)
            elif mode == "strategic":
                pred_results = classifier._predict_from_embedding(embedding, k=2, strategic=True)
            elif mode == "robust":
                pred_results = classifier._predict_from_embedding(embedding, k=2, robust=True)
            elif mode == "dual":
                pred_results = classifier._predict_from_embedding(embedding, k=2)
            else:
                raise ValueError(f"Unknown evaluation mode: {mode}")
        except Exception as e:
            logger.warning(f"Prediction failed for embedding {i}: {e}")
            # Fallback to regular prediction
            pred_results = classifier._predict_from_embedding(embedding, k=2)
        
        end_time = time.time()
        
        # Extract top prediction
        if pred_results:
            top_pred, top_prob = pred_results[0]
            predictions.append(top_pred)
            prediction_probs.append(top_prob)
        else:
            # Fallback if no prediction
            predictions.append("negative")  # Default prediction
            prediction_probs.append(0.5)
        
        prediction_times.append(end_time - start_time)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1} / {len(embeddings)} examples")
    
    # Calculate metrics (same as before)
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='weighted'
    )
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        test_labels, predictions, average=None, labels=["negative", "positive"]
    )
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions, labels=["negative", "positive"])
    
    # Confidence statistics
    avg_confidence = np.mean(prediction_probs)
    std_confidence = np.std(prediction_probs)
    avg_prediction_time = np.mean(prediction_times)
    
    results = {
        'mode': mode,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class_metrics': {
            'negative': {
                'precision': float(precision_per_class[0]),
                'recall': float(recall_per_class[0]),
                'f1_score': float(f1_per_class[0]),
                'support': int(support_per_class[0])
            },
            'positive': {
                'precision': float(precision_per_class[1]),
                'recall': float(recall_per_class[1]),
                'f1_score': float(f1_per_class[1]),
                'support': int(support_per_class[1])
            }
        },
        'confusion_matrix': cm.tolist(),
        'avg_confidence': float(avg_confidence),
        'std_confidence': float(std_confidence),
        'avg_prediction_time': float(avg_prediction_time),
        'total_predictions': len(predictions)
    }
    
    logger.info(f"{mode.capitalize()} mode results on manipulated data:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1-score: {f1:.4f}")
    logger.info(f"  Avg confidence: {avg_confidence:.4f}")
    logger.info(f"  Avg prediction time: {avg_prediction_time:.4f}s")
    
    return results

def evaluate_strategic_robustness(
    strategic_classifier: AdaptiveClassifier,
    test_texts: List[str],
    test_labels: List[str]
) -> Dict[str, Any]:
    """Evaluate strategic robustness at different gaming levels.
    
    Args:
        strategic_classifier: Classifier with strategic capabilities
        test_texts: Test texts  
        test_labels: Test labels
        
    Returns:
        Dictionary of robustness metrics
    """
    if not strategic_classifier.strategic_mode:
        logger.error("Strategic mode not enabled - cannot evaluate robustness")
        raise RuntimeError("Strategic mode not enabled - cannot evaluate robustness")
    
    logger.info("Evaluating strategic robustness...")
    
    # Test different gaming levels
    gaming_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    robustness_results = {}
    
    for level in gaming_levels:
        logger.info(f"Testing gaming level: {level}")
        
        # Simulate strategic behavior by evaluating with different prediction modes
        if level == 0.0:
            # No strategic behavior - use regular predictions
            results = evaluate_classifier(strategic_classifier, test_texts, test_labels, mode="regular")
        elif level == 1.0:
            # Full strategic behavior - use strategic predictions
            results = evaluate_classifier(strategic_classifier, test_texts, test_labels, mode="strategic")
        else:
            # Mixed behavior - use dual predictions (which blend regular and strategic)
            results = evaluate_classifier(strategic_classifier, test_texts, test_labels, mode="dual")
        
        robustness_results[f'gaming_level_{level}'] = {
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'avg_confidence': results['avg_confidence']
        }
    
    # Calculate robustness metrics
    baseline_accuracy = robustness_results['gaming_level_0.0']['accuracy']
    strategic_accuracy = robustness_results['gaming_level_1.0']['accuracy']
    
    robustness_score = baseline_accuracy - strategic_accuracy
    relative_robustness = strategic_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0.0
    
    robustness_results['summary'] = {
        'baseline_accuracy': baseline_accuracy,
        'strategic_accuracy': strategic_accuracy,
        'robustness_score': robustness_score,
        'relative_robustness': relative_robustness
    }
    
    logger.info(f"Robustness Score: {robustness_score:.4f}")
    logger.info(f"Relative Robustness: {relative_robustness:.4f}")
    
    return robustness_results

def evaluate_comparison_on_manipulated_data(
    regular_classifier: AdaptiveClassifier,
    strategic_classifier: AdaptiveClassifier,
    test_texts: List[str],
    test_labels: List[str]
) -> Dict[str, Any]:
    """Perform comparison by evaluating both classifiers on manipulated data.
    
    Args:
        regular_classifier: Regular classifier (no strategic training)
        strategic_classifier: Strategic classifier
        test_texts: Original test texts
        test_labels: Test labels
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("="*60)
    logger.info("EVALUATION ON MANIPULATED DATA")
    logger.info("="*60)
    
    # Generate strategically manipulated data
    manipulated_embeddings = generate_manipulated_data(
        strategic_classifier, test_texts, manipulation_level=1.0
    )
    
    # Evaluate regular classifier on manipulated data
    logger.info("Evaluating regular classifier on manipulated data...")
    regular_on_manipulated = evaluate_classifier_on_embeddings(
        regular_classifier, manipulated_embeddings, test_labels, mode="regular"
    )
    
    # Evaluate strategic classifier on manipulated data
    logger.info("Evaluating strategic classifier on manipulated data...")
    strategic_on_manipulated = evaluate_classifier_on_embeddings(
        strategic_classifier, manipulated_embeddings, test_labels, mode="dual"
    )
    
    # Calculate improvement
    accuracy_improvement = strategic_on_manipulated['accuracy'] - regular_on_manipulated['accuracy']
    f1_improvement = strategic_on_manipulated['f1_score'] - regular_on_manipulated['f1_score']
    
    return {
        'regular_on_manipulated': regular_on_manipulated,
        'strategic_on_manipulated': strategic_on_manipulated,
        'comparison': {
            'accuracy_improvement': accuracy_improvement,
            'f1_improvement': f1_improvement,
            'relative_accuracy_improvement': accuracy_improvement / regular_on_manipulated['accuracy'] if regular_on_manipulated['accuracy'] > 0 else 0.0
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Strategic Classifier on AI-Secure/adv_glue dataset")
    parser.add_argument(
        "--model", 
        type=str, 
        default="answerdotai/ModernBERT-base",
        help="HuggingFace model name to use (default: answerdotai/ModernBERT-base). "
             "The script automatically adapts to any model's embedding dimension."
    )
    parser.add_argument(
        "--cost-strategy",
        type=str,
        default="balanced",
        choices=["balanced", "sparse_low", "uniform_low", "minimal", "sparse_high"],
        help="Strategic cost function strategy. Options: "
             "'balanced' (50%% manipulable dims, cost 0.3), "
             "'sparse_low' (20%% manipulable dims, cost 0.4), "
             "'uniform_low' (all dims, cost 0.15), "
             "'minimal' (all dims, cost 0.05 - for debugging), "
             "'sparse_high' (legacy - adjusted to be less restrictive)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="strategic_classifier_evaluation_results.json",
        help="Output file for results (default: strategic_classifier_evaluation_results.json)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of data to use for testing (default: 0.3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Start evaluation
    logger.info("Starting Strategic Classifier Evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Cost strategy: {args.cost_strategy}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Random seed: {args.seed}")
    
    start_time = time.time()
    
    try:
        # 1. Load dataset
        texts, labels = load_adv_glue_dataset()
        
        # 2. Split dataset
        train_texts, test_texts, train_labels, test_labels = split_dataset(
            texts, labels, test_size=args.test_size, random_state=args.seed
        )
        
        # 3. Train regular classifier
        logger.info("="*60)
        logger.info("TRAINING REGULAR CLASSIFIER")
        logger.info("="*60)
        
        regular_classifier = train_classifier(
            args.model, train_texts, train_labels, config=None
        )
        
        # 4. Train strategic classifier
        logger.info("="*60)
        logger.info("TRAINING STRATEGIC CLASSIFIER")
        logger.info("="*60)
        
        strategic_config = create_strategic_config(args.model, args.cost_strategy)
        strategic_classifier = train_classifier(
            args.model, train_texts, train_labels, config=strategic_config
        )
        
        # Strategic mode must be enabled for this evaluation
        if not strategic_classifier.strategic_mode:
            raise RuntimeError("Strategic mode failed to initialize. Cannot proceed with strategic evaluation.")
        
        # 5. Evaluate both classifiers
        logger.info("="*60)
        logger.info("EVALUATION PHASE")
        logger.info("="*60)
        
        # Evaluate regular classifier
        regular_results = evaluate_classifier(
            regular_classifier, test_texts, test_labels, mode="regular"
        )
        
        # Evaluate strategic classifier in different modes
        strategic_dual_results = evaluate_classifier(
            strategic_classifier, test_texts, test_labels, mode="dual"
        )
        
        strategic_only_results = evaluate_classifier(
            strategic_classifier, test_texts, test_labels, mode="strategic"
        )
        
        robust_results = evaluate_classifier(
            strategic_classifier, test_texts, test_labels, mode="robust"
        )
        
        # 6. Perform comparison on manipulated data
        comparison_results = evaluate_comparison_on_manipulated_data(
            regular_classifier, strategic_classifier, test_texts, test_labels
        )
        
        # 7. Evaluate strategic robustness (original method)
        robustness_results = evaluate_strategic_robustness(
            strategic_classifier, test_texts, test_labels
        )
        
        # 8. Compile final results
        end_time = time.time()
        total_time = end_time - start_time
        
        final_results = {
            'metadata': {
                'model_name': args.model,
                'dataset': 'AI-Secure/adv_glue (adv_sst2)',
                'evaluation_date': datetime.now().isoformat(),
                'total_examples': len(texts),
                'train_examples': len(train_texts),
                'test_examples': len(test_texts),
                'test_size': args.test_size,
                'random_seed': args.seed,
                'total_evaluation_time': total_time
            },
            'dataset_info': {
                'train_label_distribution': dict(zip(*np.unique(train_labels, return_counts=True))),
                'test_label_distribution': dict(zip(*np.unique(test_labels, return_counts=True)))
            },
            'regular_classifier': regular_results,
            'strategic_classifier': {
                'dual_mode': strategic_dual_results,
                'strategic_only_mode': strategic_only_results,
                'robust_mode': robust_results
            },
            'strategic_robustness': robustness_results,
            'fair_comparison_on_manipulated_data': fair_comparison_results,
            'comparison': {
                'accuracy_improvement': strategic_dual_results['accuracy'] - regular_results['accuracy'],
                'f1_improvement': strategic_dual_results['f1_score'] - regular_results['f1_score'],
                'relative_accuracy_improvement': (strategic_dual_results['accuracy'] - regular_results['accuracy']) / regular_results['accuracy'] if regular_results['accuracy'] > 0 else 0.0
            },
            'config': {
                'strategic_config': strategic_config,
                'cost_strategy': args.cost_strategy
            }
        }
        
        # 9. Save results
        output_path = Path(args.output)
        # Convert all numpy/torch types to JSON serializable types
        serializable_results = convert_to_serializable(final_results)
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, sort_keys=True)
        
        # 10. Print summary
        logger.info("="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Regular Classifier Accuracy: {regular_results['accuracy']:.4f}")
        logger.info(f"Strategic Classifier (Dual) Accuracy: {strategic_dual_results['accuracy']:.4f}")
        logger.info(f"Strategic Classifier (Strategic-only) Accuracy: {strategic_only_results['accuracy']:.4f}")
        logger.info(f"Strategic Classifier (Robust) Accuracy: {robust_results['accuracy']:.4f}")
        
        logger.info("")  # Empty line for spacing
        logger.info(f"Accuracy Improvement: {final_results['comparison']['accuracy_improvement']:.4f}")
        logger.info(f"F1-score Improvement: {final_results['comparison']['f1_improvement']:.4f}")
        logger.info(f"Relative Accuracy Improvement: {final_results['comparison']['relative_accuracy_improvement']:.4f}")
        
        if robustness_results and 'summary' in robustness_results:
            logger.info("")  # Empty line for spacing
            logger.info(f"Strategic Robustness Score: {robustness_results['summary']['robustness_score']:.4f}")
            logger.info(f"Relative Robustness: {robustness_results['summary']['relative_robustness']:.4f}")
        
        # Fair comparison on manipulated data
        if comparison_results and 'fair_comparison' in comparison_results:
            logger.info("")  # Empty line for spacing
            logger.info("COMPARISON ON MANIPULATED DATA:")
            logger.info(f"Regular Classifier on Manipulated Data: {comparison_results['regular_on_manipulated']['accuracy']:.4f}")
            logger.info(f"Strategic Classifier on Manipulated Data: {comparison_results['strategic_on_manipulated']['accuracy']:.4f}")
            logger.info(f"Accuracy Improvement: {comparison_results['fair_comparison']['accuracy_improvement']:.4f}")
            logger.info(f"F1-score Improvement: {comparison_results['fair_comparison']['f1_improvement']:.4f}")
        
        logger.info("")  # Empty line for spacing
        logger.info(f"Total Evaluation Time: {total_time:.2f} seconds")
        logger.info(f"Results saved to: {output_path}")
        
        logger.info("="*60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
