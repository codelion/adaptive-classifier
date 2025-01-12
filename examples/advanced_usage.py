import torch
from adaptive_classifier import AdaptiveClassifier
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str]):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def demonstrate_batch_processing():
    """Example of processing large datasets efficiently"""
    logger.info("Demonstrating batch processing...")
    
    # Initialize classifier
    classifier = AdaptiveClassifier("bert-base-uncased")
    
    # Create a larger dataset
    texts = []
    labels = []
    
    # Simulate customer feedback dataset
    feedback_data = [
        ("The product is amazing!", "positive"),
        ("Worst experience ever", "negative"),
        ("It works as expected", "neutral"),
        # Add more examples...
    ]
    
    for text, label in feedback_data:
        texts.extend([text] * 10)  # Replicate each example 10 times for demo
        labels.extend([label] * 10)
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Process in batches
    start_time = time.time()
    for batch_idx, (batch_texts, batch_labels) in enumerate(dataloader):
        classifier.add_examples(batch_texts, batch_labels)
        if batch_idx % 10 == 0:
            logger.info(f"Processed batch {batch_idx}")
    
    logger.info(f"Processing time: {time.time() - start_time:.2f} seconds")
    
    return classifier

def demonstrate_continuous_learning():
    """Example of continuous learning with performance monitoring"""
    logger.info("Demonstrating continuous learning...")
    
    classifier = AdaptiveClassifier("bert-base-uncased")
    
    # Initial training
    initial_texts = [
        "Great product, highly recommend",
        "Terrible service, avoid",
        "Average experience, nothing special"
    ]
    initial_labels = ["positive", "negative", "neutral"]
    
    classifier.add_examples(initial_texts, initial_labels)
    
    # Function to evaluate performance
    def evaluate_performance(test_texts: List[str], test_labels: List[str]) -> float:
        correct = 0
        total = len(test_texts)
        
        for text, true_label in zip(test_texts, test_labels):
            predictions = classifier.predict(text)
            predicted_label = predictions[0][0]  # Get top prediction
            if predicted_label == true_label:
                correct += 1
        
        return correct / total
    
    # Initial evaluation
    test_texts = [
        "This is fantastic",
        "Don't buy this",
        "It's okay I guess"
    ]
    test_labels = ["positive", "negative", "neutral"]
    
    initial_accuracy = evaluate_performance(test_texts, test_labels)
    logger.info(f"Initial accuracy: {initial_accuracy:.2f}")
    
    # Simulate continuous learning
    for i in range(3):
        # New batch of examples
        new_texts = [
            f"Really enjoyed using it {i}",
            f"Disappointed with quality {i}",
            f"Standard product {i}"
        ]
        new_labels = ["positive", "negative", "neutral"]
        
        classifier.add_examples(new_texts, new_labels)
        
        # Evaluate after update
        accuracy = evaluate_performance(test_texts, test_labels)
        logger.info(f"Accuracy after update {i+1}: {accuracy:.2f}")
    
    return classifier

def demonstrate_persistence():
    # 1. Create and train initial classifier
    print("Phase 1: Creating and training initial classifier")
    classifier = AdaptiveClassifier("bert-base-uncased")
    
    # Add some initial examples
    initial_texts = [
        "This product is amazing!",
        "Terrible experience",
        "Neutral about this"
    ]
    initial_labels = ["positive", "negative", "neutral"]
    
    classifier.add_examples(initial_texts, initial_labels)
    
    # Save the state
    print("\nSaving classifier ...")
    classifier.save("./demo_classifier")
    
    # 2. Load the classifier in a new session
    print("\nPhase 2: Loading classifier from saved state")
    loaded_classifier = AdaptiveClassifier.load("./demo_classifier")
    
    # Verify it works with existing classes
    test_text = "This is fantastic!"
    predictions = loaded_classifier.predict(test_text)
    print(f"\nPredictions using loaded classifier:")
    print(f"Text: {test_text}")
    for label, score in predictions:
        print(f"{label}: {score:.4f}")
    
    # 3. Add new examples to loaded classifier
    print("\nPhase 3: Adding new examples to loaded classifier")
    new_texts = [
        "Technical error occurred",
        "System crashed"
    ]
    new_labels = ["technical"] * 2
    
    loaded_classifier.add_examples(new_texts, new_labels)
    
    # Save updated state
    print("\nSaving updated classifier ...")
    loaded_classifier.save("./demo_classifier")
    
    # Show final class distribution
    print("\nFinal class distribution:")
    for label, examples in loaded_classifier.memory.examples.items():
        print(f"{label}: {len(examples)} examples")

def demonstrate_multi_language():
    """Example of handling multiple languages"""
    logger.info("Demonstrating multi-language support...")
    
    # Use a multilingual model
    classifier = AdaptiveClassifier("bert-base-multilingual-uncased")
    
    # Add examples in different languages
    texts = [
        # English
        "This is great",
        "This is terrible",
        # Spanish
        "Esto es excelente",
        "Esto es terrible",
        # French
        "C'est excellent",
        "C'est terrible"
    ]
    
    labels = ["positive", "negative"] * 3
    
    classifier.add_examples(texts, labels)
    
    # Test in different languages
    test_texts = [
        "This is wonderful",  # English
        "Esto es maravilloso",  # Spanish
        "C'est merveilleux"  # French
    ]
    
    for text in test_texts:
        predictions = classifier.predict(text)
        logger.info(f"\nText: {text}")
        logger.info("Predictions:")
        for label, score in predictions:
            logger.info(f"{label}: {score:.4f}")
    
    return classifier

def main():
    demonstrate_batch_processing()
    demonstrate_continuous_learning()
    demonstrate_persistence()
    demonstrate_multi_language()

if __name__ == "__main__":
    main()