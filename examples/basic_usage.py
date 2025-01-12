from adaptive_classifier import AdaptiveClassifier

def main():
    # Initialize classifier
    classifier = AdaptiveClassifier("bert-base-uncased")
    
    # Initial training data
    texts = [
        "The product works great!",
        "Amazing service, very satisfied",
        "This exceeded my expectations",
        "Terrible experience, don't buy",
        "Worst product ever",
        "Product arrived on time",
        "Does what it says"
    ]
    
    labels = [
        "positive", "positive", "positive",
        "negative", "negative",
        "neutral", "neutral"
    ]
    
    # Add examples
    print("Adding initial examples...")
    classifier.add_examples(texts, labels)
    
    # Test predictions
    test_texts = [
        "This is fantastic!",
        "I hate this product",
        "It's okay, nothing special"
    ]
    
    print("\nTesting predictions:")
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
    
    # Add new class
    print("\nAdding new technical class...")
    technical_texts = [
        "Error code 404 appeared",
        "System crashed after update"
    ]
    technical_labels = ["technical"] * 2
    
    loaded_classifier.add_examples(technical_texts, technical_labels)
    
    # Test new predictions
    print("\nTesting technical classification:")
    technical_test = "Getting null pointer exception"
    predictions = loaded_classifier.predict(technical_test)
    print(f"\nText: {technical_test}")
    print("Predictions:")
    for label, score in predictions:
        print(f"{label}: {score:.4f}")

if __name__ == "__main__":
    main()