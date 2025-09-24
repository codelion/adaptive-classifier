#!/usr/bin/env python3
"""
Multi-Label Adaptive Classifier Example

This example demonstrates how to use the MultiLabelAdaptiveClassifier
for text classification tasks where each text can belong to multiple categories.

Key features demonstrated:
1. Training with multi-label data
2. Making multi-label predictions
3. Adaptive threshold handling for many labels
4. Label-specific threshold customization
5. Saving and loading multi-label models
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptive_classifier import MultiLabelAdaptiveClassifier
import torch


def create_sample_data():
    """Create sample multi-label training data."""

    # Sample texts with multiple labels each
    training_data = [
        {
            "text": "Scientists discover new species of butterfly in Amazon rainforest with unique wing patterns",
            "labels": ["science", "nature", "discovery", "biology"]
        },
        {
            "text": "Tech startup raises $50M in Series A funding to develop AI-powered healthcare solutions",
            "labels": ["technology", "business", "healthcare", "funding"]
        },
        {
            "text": "Climate change impacts ocean temperature causing coral bleaching in Great Barrier Reef",
            "labels": ["environment", "climate", "nature", "science"]
        },
        {
            "text": "NBA playoffs feature exciting games with record-breaking performances by star players",
            "labels": ["sports", "entertainment", "basketball"]
        },
        {
            "text": "New renewable energy technology could reduce costs by 40% according to MIT research",
            "labels": ["technology", "science", "environment", "energy"]
        },
        {
            "text": "Archaeological team uncovers 2000-year-old Roman artifacts in excavation site",
            "labels": ["history", "science", "discovery", "archaeology"]
        },
        {
            "text": "Stock market reaches new highs as investors show confidence in economic recovery",
            "labels": ["business", "finance", "economy"]
        },
        {
            "text": "Machine learning breakthrough helps doctors diagnose rare diseases more accurately",
            "labels": ["technology", "healthcare", "science", "ai"]
        },
        {
            "text": "Wildlife conservation efforts show success in protecting endangered tiger populations",
            "labels": ["nature", "environment", "conservation", "wildlife"]
        },
        {
            "text": "Olympic athletes prepare for upcoming games with intensive training programs",
            "labels": ["sports", "olympics", "training", "fitness"]
        },
        {
            "text": "Quantum computing research makes progress toward solving complex optimization problems",
            "labels": ["technology", "science", "computing", "research"]
        },
        {
            "text": "Sustainable agriculture practices help farmers reduce environmental impact while increasing yield",
            "labels": ["environment", "agriculture", "sustainability", "farming"]
        },
        {
            "text": "Music festival features artists from diverse genres attracting thousands of fans",
            "labels": ["entertainment", "music", "culture", "events"]
        },
        {
            "text": "Space agency announces plans for Mars mission with new rocket technology",
            "labels": ["science", "space", "technology", "exploration"]
        },
        {
            "text": "Educational technology helps students learn programming through interactive online courses",
            "labels": ["education", "technology", "programming", "learning"]
        }
    ]

    # Extract texts and labels
    texts = [item["text"] for item in training_data]
    labels = [item["labels"] for item in training_data]

    return texts, labels


def demonstrate_basic_usage():
    """Demonstrate basic multi-label classification."""

    print("=" * 60)
    print("MULTI-LABEL ADAPTIVE CLASSIFIER - BASIC USAGE")
    print("=" * 60)

    # Create classifier
    classifier = MultiLabelAdaptiveClassifier(
        model_name="distilbert/distilbert-base-cased",
        default_threshold=0.5,
        min_predictions=1,  # Ensure at least 1 prediction
        max_predictions=5   # Limit to top 5 predictions
    )

    # Load training data
    texts, labels = create_sample_data()

    print(f"Training with {len(texts)} examples")
    print(f"Example text: {texts[0][:60]}...")
    print(f"Example labels: {labels[0]}")

    # Train the classifier
    classifier.add_examples(texts, labels)

    # Get statistics
    stats = classifier.get_label_statistics()
    print(f"\nTraining completed:")
    print(f"- Total labels: {stats['num_classes']}")
    print(f"- Total examples: {stats['total_examples']}")
    print(f"- Adaptive threshold: {stats['adaptive_threshold']:.3f}")

    return classifier


def demonstrate_predictions(classifier):
    """Demonstrate making predictions."""

    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)

    # Test texts
    test_texts = [
        "Researchers develop new AI algorithm for medical diagnosis",
        "Football team wins championship in exciting final match",
        "Solar panel efficiency increases with new manufacturing technique",
        "Ancient civilization discovered through satellite imagery analysis"
    ]

    for text in test_texts:
        print(f"\nText: {text}")

        # Make multi-label prediction
        predictions = classifier.predict_multilabel(text)

        print("Predictions:")
        if predictions:
            for label, confidence in predictions:
                print(f"  {label}: {confidence:.4f}")
        else:
            print("  No predictions above threshold")

    return test_texts


def demonstrate_threshold_adjustment(classifier):
    """Demonstrate threshold adjustment for different scenarios."""

    print("\n" + "=" * 60)
    print("THRESHOLD ADJUSTMENT")
    print("=" * 60)

    test_text = "AI researchers publish breakthrough study on climate modeling using machine learning"

    print(f"Test text: {test_text}")

    # Try different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"\n{'Threshold':<10} {'Predictions':<12} {'Labels'}")
    print("-" * 50)

    for threshold in thresholds:
        predictions = classifier.predict_multilabel(test_text, threshold=threshold)
        labels_str = ", ".join([label for label, _ in predictions[:3]])

        print(f"{threshold:<10.1f} {len(predictions):<12} {labels_str}")


def demonstrate_saving_loading(classifier):
    """Demonstrate saving and loading the model."""

    print("\n" + "=" * 60)
    print("SAVING AND LOADING")
    print("=" * 60)

    # Save the model
    save_path = "./multilabel_classifier"
    print(f"Saving classifier to {save_path}")
    classifier.save(save_path)

    # Load the model
    print("Loading classifier...")
    loaded_classifier = MultiLabelAdaptiveClassifier.load(save_path)

    # Verify it works
    test_text = "New medical technology helps treat cancer patients"

    print(f"\nTesting loaded classifier:")
    print(f"Text: {test_text}")

    predictions = loaded_classifier.predict_multilabel(test_text)
    print("Predictions:")
    for label, confidence in predictions:
        print(f"  {label}: {confidence:.4f}")

    return loaded_classifier


def demonstrate_incremental_learning(classifier):
    """Demonstrate adding new labels incrementally."""

    print("\n" + "=" * 60)
    print("INCREMENTAL LEARNING - ADDING NEW LABELS")
    print("=" * 60)

    # Add new examples with new labels
    new_texts = [
        "Chef creates innovative fusion cuisine combining Asian and European flavors",
        "Food delivery service expands to new cities with sustainable packaging",
        "Restaurant industry adapts to new dining trends post-pandemic",
        "Cooking show features celebrity chefs competing in culinary challenges"
    ]

    new_labels = [
        ["food", "cuisine", "cooking", "culture"],
        ["business", "food", "sustainability"],
        ["business", "food", "trends"],
        ["entertainment", "food", "cooking", "tv"]
    ]

    print("Adding new examples with 'food' and 'cooking' labels...")
    classifier.add_examples(new_texts, new_labels)

    # Test with food-related text
    food_text = "Nutritionist recommends healthy meal planning for busy professionals"

    print(f"\nTesting with food-related text:")
    print(f"Text: {food_text}")

    predictions = classifier.predict_multilabel(food_text)
    print("Predictions:")
    for label, confidence in predictions:
        print(f"  {label}: {confidence:.4f}")

    # Show updated statistics
    stats = classifier.get_label_statistics()
    print(f"\nUpdated statistics:")
    print(f"- Total labels: {stats['num_classes']}")
    print(f"- Total examples: {stats['total_examples']}")


def main():
    """Main function to run all demonstrations."""

    print("Multi-Label Adaptive Classifier Example")
    print("Fixing the 'No labels met the threshold criteria' issue\n")

    try:
        # Basic usage
        classifier = demonstrate_basic_usage()

        # Making predictions
        demonstrate_predictions(classifier)

        # Threshold adjustment
        demonstrate_threshold_adjustment(classifier)

        # Saving and loading
        loaded_classifier = demonstrate_saving_loading(classifier)

        # Incremental learning
        demonstrate_incremental_learning(loaded_classifier)

        print("\n" + "=" * 60)
        print("EXAMPLE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        # Final statistics
        final_stats = loaded_classifier.get_label_statistics()
        print(f"\nFinal Model Statistics:")
        print(f"- Labels: {final_stats['num_classes']}")
        print(f"- Examples: {final_stats['total_examples']}")
        print(f"- Default threshold: {final_stats['default_threshold']}")
        print(f"- Adaptive threshold: {final_stats['adaptive_threshold']:.3f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()