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
    classifier = AdaptiveClassifier("distilbert/distilbert-base-cased")
    
    # Create a larger dataset
    texts = []
    labels = []
    
    # Simulate customer feedback dataset
    feedback_data = [
        # Positive feedback
        ("The product is amazing!", "positive"),
        ("Exceeded all my expectations, truly worth every penny", "positive"),
        ("Customer service was incredibly helpful and responsive", "positive"),
        ("Best purchase I've made this year", "positive"),
        ("The quality is outstanding", "positive"),
        ("Shipping was super fast and packaging was perfect", "positive"),
        ("Really impressed with the durability", "positive"),
        ("Great value for money", "positive"),
        ("The features are exactly what I needed", "positive"),
        ("Easy to use and very intuitive", "positive"),
        ("Fantastic product, will definitely buy again", "positive"),
        ("Love how lightweight and portable it is", "positive"),
        ("The installation process was seamless", "positive"),
        ("Brilliant design and functionality", "positive"),
        ("Top-notch quality and performance", "positive"),

        # Negative feedback
        ("Worst experience ever", "negative"),
        ("Product broke after just one week", "negative"),
        ("Customer support never responded to my emails", "negative"),
        ("Completely disappointed with the quality", "negative"),
        ("Not worth the money at all", "negative"),
        ("Arrived damaged and return process was horrible", "negative"),
        ("The instructions were impossible to follow", "negative"),
        ("Poor build quality, feels cheap", "negative"),
        ("Missing essential features that were advertised", "negative"),
        ("Terrible battery life", "negative"),
        ("Keeps malfunctioning randomly", "negative"),
        ("The worst customer service I've ever experienced", "negative"),
        ("Save your money and avoid this product", "negative"),
        ("Doesn't work as advertised", "negative"),
        ("Had to return it immediately", "negative"),

        # Neutral feedback
        ("It works as expected", "neutral"),
        ("Average product, nothing special", "neutral"),
        ("Does the job, but could be better", "neutral"),
        ("Reasonable price for what you get", "neutral"),
        ("Some good features, some bad ones", "neutral"),
        ("Pretty standard quality", "neutral"),
        ("Not bad, not great", "neutral"),
        ("Meets basic requirements", "neutral"),
        ("Similar to other products in this category", "neutral"),
        ("Acceptable performance for the price", "neutral"),
        ("Middle-of-the-road quality", "neutral"),
        ("Functions adequately", "neutral"),
        ("Basic functionality works fine", "neutral"),
        ("Got what I paid for", "neutral"),
        ("Standard delivery time and service", "neutral"),

        # Technical feedback
        ("Getting error code 404 when trying to sync", "technical"),
        ("App crashes after latest update", "technical"),
        ("Can't connect to WiFi despite correct password", "technical"),
        ("Battery drains even when device is off", "technical"),
        ("Screen freezes during startup", "technical"),
        ("Bluetooth pairing fails consistently", "technical"),
        ("System shows unrecognized device error", "technical"),
        ("Software keeps reverting to previous version", "technical"),
        ("Memory full error after minimal usage", "technical"),
        ("Device overheats during normal operation", "technical"),
        ("USB port not recognizing connections", "technical"),
        ("Network connectivity drops randomly", "technical"),
        ("Authentication failed error on login", "technical"),
        ("Sync process stuck at 99%", "technical"),
        ("Database connection timeout error", "technical")
    ]
    
    # Number of times to replicate each example
    num_replications = 10  # This will create 10x more data
    
    for text, label in feedback_data:
        # Add multiple copies of each example
        texts.extend([text] * num_replications)
        labels.extend([label] * num_replications)
    
    logger.info(f"Total examples: {len(texts)}")
    logger.info(f"Examples per class: {sum(1 for l in labels if l == 'positive')}/{sum(1 for l in labels if l == 'negative')}/"
                f"{sum(1 for l in labels if l == 'neutral')}/{sum(1 for l in labels if l == 'technical')}")
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, labels)
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate expected number of batches
    expected_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
    logger.info(f"Expected number of batches: {expected_batches}")
    
    # Process in batches
    start_time = time.time()
    for batch_idx, (batch_texts, batch_labels) in enumerate(dataloader):
        classifier.add_examples(batch_texts, batch_labels)
        if batch_idx % 5 == 0:  # Log every 5 batches
            logger.info(f"Processed batch {batch_idx + 1}/{expected_batches}")
            
        # Optional: print batch sizes to verify
        if batch_idx in [0, expected_batches // 2, expected_batches - 1]:  # Print first, middle, and last batch
            logger.info(f"Batch {batch_idx + 1} size: {len(batch_texts)}")
    
    processing_time = time.time() - start_time
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info(f"Average time per batch: {processing_time/expected_batches:.2f} seconds")
    
    return classifier

def demonstrate_continuous_learning():
    """Example of continuous learning with performance monitoring"""
    logger.info("Demonstrating continuous learning...")
    
    classifier = AdaptiveClassifier("distilbert/distilbert-base-cased")
    
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
    classifier = AdaptiveClassifier("distilbert/distilbert-base-cased")
    
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
    classifier = AdaptiveClassifier("distilbert/distilbert-base-multilingual-cased")
    
    texts = [
        # English - Positive
        "This is great",
        "I love this product",
        "Amazing experience",
        "Excellent service",
        "Best purchase ever",
        "Highly recommended",
        "Really impressive quality",
        "Fantastic results",
        
        # English - Negative
        "This is terrible",
        "Worst experience ever",
        "Don't waste your money",
        "Very disappointed",
        "Poor quality product",
        "Absolutely horrible",
        "Complete waste of time",
        "Not worth buying",
        
        # Spanish - Positive
        "Esto es excelente",
        "Me encanta este producto",
        "Una experiencia maravillosa",
        "Servicio excepcional",
        "La mejor compra",
        "Muy recomendable",
        "Calidad impresionante",
        "Resultados fantásticos",
        
        # Spanish - Negative
        "Esto es terrible",
        "La peor experiencia",
        "No malgastes tu dinero",
        "Muy decepcionado",
        "Producto de mala calidad",
        "Absolutamente horrible",
        "Pérdida total de tiempo",
        "No vale la pena comprarlo",
    ]

    labels = ["positive"] * 8 + ["negative"] * 8 \
        + ["positive"] * 8 + ["negative"] * 8 
    
    classifier.add_examples(texts, labels)
    
    # Test in different languages
    test_texts = [
        # English
        "This is wonderful",        # Positive
        "This is terrible",         # Negative
        
        # Spanish
        "Esto es maravilloso",      # Positive
        "Esto es terrible",         # Negative
    ]

    # Print test results
    print("\nTesting predictions in multiple languages:")
    for text in test_texts:
        predictions = classifier.predict(text)
        print(f"\nText: {text}")
        print("Predictions:")
        for label, score in predictions:
            print(f"{label}: {score:.4f}")
    
    return classifier

def main():
    demonstrate_batch_processing()
    demonstrate_continuous_learning()
    demonstrate_persistence()
    demonstrate_multi_language()

if __name__ == "__main__":
    main()