import torch
import numpy as np
import random
from adaptive_classifier import AdaptiveClassifier

def main():

    # Initialize classifier
    classifier = AdaptiveClassifier("distilbert/distilbert-base-cased")
    
    # Initial training data with atleast 5 examples per class
    texts = [
        # Positive examples
        "The product works great!",
        "Amazing service, very satisfied",
        "This exceeded my expectations",
        "Best purchase I've made this year",
        "Really impressed with the quality",
        "Fantastic product, will buy again",
        "Highly recommend this to everyone",
        
        # Negative examples
        "Terrible experience, don't buy",
        "Worst product ever",
        "Complete waste of money",
        "Poor quality and bad service",
        "Would not recommend to anyone",
        "Disappointed with the purchase",
        "Product broke after first use",
        
        # Neutral examples
        "Product arrived on time",
        "Does what it says",
        "Average product, nothing special",
        "Meets basic requirements",
        "Fair price for what you get",
        "Standard quality product",
        "Works as expected"
    ]
    
    labels = [
        # Positive labels
        "positive", "positive", "positive", "positive", 
        "positive", "positive", "positive",
        
        # Negative labels
        "negative", "negative", "negative", "negative",
        "negative", "negative", "negative",
        
        # Neutral labels
        "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral"
    ]
    
    # Add examples
    print("Adding initial examples...")
    classifier.add_examples(texts, labels)
    
    # Test predictions
    test_texts = [
        "This is a fantastic product!",
        "Disappointed with this bad product",
        "Average product, as expected"
    ]
    
    print("\nTesting predictions:")
    classifier.model.eval()
    
    with torch.no_grad():
        for text in test_texts:
            predictions = classifier.predict(text)
            print(f"\nText: {text}")
            print("Predictions:")
            for label, score in predictions:
                print(f"{label}: {score:.4f}")
    
    # Save the classifier
    print("\nSaving classifier...")
    classifier.save("./demo_classifier")
    
    # Load the classifier
    print("\nLoading classifier...")
    loaded_classifier = AdaptiveClassifier.load("./demo_classifier")
    
    # Add new technical class with more examples
    print("\nAdding new technical class...")
    technical_texts = [
        "Error code 404 appeared",
        "System crashed after update",
        "Cannot connect to database",
        "Memory allocation failed",
        "Null pointer exception detected",
        "API endpoint not responding",
        "Stack overflow in main thread"
    ]
    technical_labels = ["technical"] * len(technical_texts)
    
    loaded_classifier.add_examples(technical_texts, technical_labels)
    
    # Test new predictions
    print("\nTesting technical classification:")
    technical_test = "API giving null pointer exception"
    
    loaded_classifier.model.eval()
    
    with torch.no_grad():
        predictions = loaded_classifier.predict(technical_test)
        print(f"\nText: {technical_test}")
        print("Predictions:")
        for label, score in predictions:
            print(f"{label}: {score:.4f}")

if __name__ == "__main__":
    main()