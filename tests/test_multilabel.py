import pytest
import torch
import tempfile
import os
from pathlib import Path
from adaptive_classifier import MultiLabelAdaptiveClassifier, MultiLabelAdaptiveHead


@pytest.fixture
def sample_multilabel_data():
    """Sample multi-label training data."""
    texts = [
        "Scientists study climate change effects on polar ice caps",
        "Tech company develops AI for medical diagnosis",
        "Athletes train for upcoming Olympic games",
        "Researchers discover new species in Amazon rainforest",
        "Startup raises funding for sustainable energy project"
    ]

    labels = [
        ["science", "climate", "environment"],
        ["technology", "healthcare", "ai"],
        ["sports", "fitness", "olympics"],
        ["science", "nature", "discovery"],
        ["business", "technology", "environment"]
    ]

    return texts, labels


@pytest.fixture
def multilabel_classifier():
    """Create a MultiLabelAdaptiveClassifier instance."""
    return MultiLabelAdaptiveClassifier(
        "distilbert/distilbert-base-cased",
        default_threshold=0.5,
        min_predictions=1,
        max_predictions=5
    )


def test_multilabel_classifier_initialization(multilabel_classifier):
    """Test MultiLabelAdaptiveClassifier initialization."""
    assert multilabel_classifier.default_threshold == 0.5
    assert multilabel_classifier.min_predictions == 1
    assert multilabel_classifier.max_predictions == 5
    assert multilabel_classifier.adaptive_head is None


def test_multilabel_head_initialization():
    """Test MultiLabelAdaptiveHead initialization."""
    head = MultiLabelAdaptiveHead(768, 5)
    assert head.num_classes == 5
    assert isinstance(head.model, torch.nn.Sequential)

    # Test forward pass
    input_tensor = torch.randn(1, 768)
    output = head(input_tensor)
    assert output.shape == (1, 5)
    assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output


def test_multilabel_head_update_classes():
    """Test updating number of classes in MultiLabelAdaptiveHead."""
    head = MultiLabelAdaptiveHead(768, 3)
    original_weight = head.model[-1].weight.data.clone()
    original_bias = head.model[-1].bias.data.clone()

    # Update to more classes
    head.update_num_classes(5)
    assert head.num_classes == 5

    # Check that original weights are preserved
    assert torch.equal(head.model[-1].weight[:3], original_weight)
    assert torch.equal(head.model[-1].bias[:3], original_bias)


def test_adaptive_threshold_calculation(multilabel_classifier):
    """Test adaptive threshold calculation for different numbers of labels."""
    # Test threshold scaling with number of labels
    assert multilabel_classifier._get_adaptive_threshold(2) == 0.5
    assert multilabel_classifier._get_adaptive_threshold(5) == 0.4
    assert multilabel_classifier._get_adaptive_threshold(10) == 0.3
    assert multilabel_classifier._get_adaptive_threshold(20) == 0.2
    assert multilabel_classifier._get_adaptive_threshold(30) == 0.1


def test_multilabel_training(multilabel_classifier, sample_multilabel_data):
    """Test training with multi-label data."""
    texts, labels = sample_multilabel_data

    # Train classifier
    multilabel_classifier.add_examples(texts, labels)

    # Check that labels were added correctly
    expected_labels = set()
    for label_list in labels:
        expected_labels.update(label_list)

    assert len(multilabel_classifier.label_to_id) == len(expected_labels)
    assert set(multilabel_classifier.label_to_id.keys()) == expected_labels

    # Check that adaptive head was initialized
    assert multilabel_classifier.adaptive_head is not None
    assert isinstance(multilabel_classifier.adaptive_head, MultiLabelAdaptiveHead)


def test_multilabel_prediction(multilabel_classifier, sample_multilabel_data):
    """Test multi-label prediction."""
    texts, labels = sample_multilabel_data

    # Train classifier
    multilabel_classifier.add_examples(texts, labels)

    # Make prediction
    test_text = "AI researchers study climate change using machine learning"
    predictions = multilabel_classifier.predict_multilabel(test_text)

    # Check predictions format
    assert isinstance(predictions, list)
    for label, confidence in predictions:
        assert isinstance(label, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    # Check that we get at least min_predictions
    assert len(predictions) >= multilabel_classifier.min_predictions


def test_threshold_filtering(multilabel_classifier, sample_multilabel_data):
    """Test that threshold filtering works correctly."""
    texts, labels = sample_multilabel_data
    multilabel_classifier.add_examples(texts, labels)

    test_text = "Scientific research on environmental issues"

    # Test with different thresholds
    high_threshold_preds = multilabel_classifier.predict_multilabel(test_text, threshold=0.9)
    low_threshold_preds = multilabel_classifier.predict_multilabel(test_text, threshold=0.1)

    # Lower threshold should give more predictions (or at least not fewer)
    assert len(low_threshold_preds) >= len(high_threshold_preds)

    # With min_predictions=1, we should always get at least 1 prediction
    assert len(high_threshold_preds) >= 1
    assert len(low_threshold_preds) >= 1


def test_many_labels_scenario(multilabel_classifier):
    """Test the specific scenario that caused 'No labels met the threshold criteria'."""
    # Create many labels
    num_labels = 25
    texts = []
    labels = []

    for i in range(num_labels):
        for j in range(3):  # 3 examples per label
            texts.append(f"This is example {j} about topic {i}")
            labels.append([f"label_{i:02d}"])

    # Train with many labels
    multilabel_classifier.add_examples(texts, labels)

    # Test prediction
    test_text = "This is a general text about various topics"
    predictions = multilabel_classifier.predict_multilabel(test_text)

    # Should not return empty result
    assert len(predictions) > 0
    assert not isinstance(predictions, str)  # Should not be error message

    # Should respect adaptive threshold
    adaptive_threshold = multilabel_classifier._get_adaptive_threshold(num_labels)
    assert adaptive_threshold < 0.5  # Should be lower for many labels


def test_label_specific_thresholds(multilabel_classifier, sample_multilabel_data):
    """Test label-specific threshold updates."""
    texts, labels = sample_multilabel_data

    # Add more examples for some labels to make them common
    additional_texts = ["More science content"] * 10
    additional_labels = [["science"]] * 10

    all_texts = texts + additional_texts
    all_labels = labels + additional_labels

    multilabel_classifier.add_examples(all_texts, all_labels)

    # Check that thresholds were updated
    assert len(multilabel_classifier.label_thresholds) > 0

    # Science should have a different threshold due to higher frequency
    if "science" in multilabel_classifier.label_thresholds:
        science_threshold = multilabel_classifier.label_thresholds["science"]
        # Should be adjusted based on frequency
        assert isinstance(science_threshold, float)


def test_save_load_multilabel(multilabel_classifier, sample_multilabel_data):
    """Test saving and loading multi-label classifier."""
    texts, labels = sample_multilabel_data
    multilabel_classifier.add_examples(texts, labels)

    test_text = "Test prediction text"
    original_predictions = multilabel_classifier.predict_multilabel(test_text, max_labels=5)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "multilabel_classifier"

        # Save classifier
        multilabel_classifier.save(str(save_path))

        # Load classifier with same configuration
        loaded_classifier = MultiLabelAdaptiveClassifier.load(
            str(save_path),
            device=multilabel_classifier.device
        )
        # Set same max_predictions for fair comparison
        loaded_classifier.max_predictions = multilabel_classifier.max_predictions

        # Test that predictions are similar
        loaded_predictions = loaded_classifier.predict_multilabel(test_text, max_labels=5)

        assert len(loaded_predictions) <= 5  # Should respect max_labels
        assert len(original_predictions) <= 5
        # Check that we get predictions from both
        assert len(loaded_predictions) > 0
        assert len(original_predictions) > 0


def test_statistics_reporting(multilabel_classifier, sample_multilabel_data):
    """Test that statistics are reported correctly."""
    texts, labels = sample_multilabel_data
    multilabel_classifier.add_examples(texts, labels)

    stats = multilabel_classifier.get_label_statistics()

    # Check required fields
    assert 'label_thresholds' in stats
    assert 'adaptive_threshold' in stats
    assert 'default_threshold' in stats
    assert 'min_predictions' in stats
    assert 'max_predictions' in stats
    assert 'num_classes' in stats
    assert 'total_examples' in stats

    # Check values
    assert stats['default_threshold'] == multilabel_classifier.default_threshold
    assert stats['min_predictions'] == multilabel_classifier.min_predictions
    assert stats['max_predictions'] == multilabel_classifier.max_predictions