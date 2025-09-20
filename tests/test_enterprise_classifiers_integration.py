#!/usr/bin/env python3
"""
Integration tests for enterprise classifiers hosted on Hugging Face Hub.
These tests verify that all published enterprise classifiers maintain their expected
performance and consistency after code changes.
"""

import pytest
import time
import sys
from pathlib import Path

# Add the src directory to the path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adaptive_classifier import AdaptiveClassifier

# Expected metrics for all enterprise classifiers
# Based on the retrained models with v0.0.17
CLASSIFIER_METRICS = {
    "business-sentiment": {
        "min_accuracy": 0.95,
        "expected": 0.988,
        "classes": 4,
        "class_names": ["mixed", "negative", "neutral", "positive"]
    },
    "compliance-classification": {
        "min_accuracy": 0.60,
        "expected": 0.653,
        "classes": 5,
        "class_names": ["gdpr", "hipaa", "other", "pci", "sox"]
    },
    "content-moderation": {
        "min_accuracy": 0.95,
        "expected": 1.000,
        "classes": 3,
        "class_names": ["appropriate", "inappropriate", "spam"]
    },
    "customer-intent": {
        "min_accuracy": 0.80,
        "expected": 0.852,
        "classes": 4,
        "class_names": ["complaint", "information", "purchase", "support"]
    },
    "document-quality": {
        "min_accuracy": 0.95,
        "expected": 1.000,
        "classes": 2,
        "class_names": ["high_quality", "poor_quality"]
    },
    "document-type": {
        "min_accuracy": 0.95,
        "expected": 0.980,
        "classes": 5,
        "class_names": ["contract", "email", "invoice", "memo", "report"]
    },
    "email-priority": {
        "min_accuracy": 0.80,
        "expected": 0.839,
        "classes": 3,
        "class_names": ["low", "normal", "urgent"]
    },
    "email-security": {
        "min_accuracy": 0.90,
        "expected": 0.938,
        "classes": 4,
        "class_names": ["legitimate", "phishing", "spam", "suspicious"]
    },
    "escalation-detection": {
        "min_accuracy": 0.95,
        "expected": 0.976,
        "classes": 2,
        "class_names": ["normal", "urgent"]
    },
    "expense-category": {
        "min_accuracy": 0.80,
        "expected": 0.842,
        "classes": 5,
        "class_names": ["equipment", "meals", "office_supplies", "other", "travel"]
    },
    "fraud-detection": {
        "min_accuracy": 0.90,
        "expected": 0.927,
        "classes": 2,
        "class_names": ["fraudulent", "legitimate"]
    },
    "language-detection": {
        "min_accuracy": 0.95,
        "expected": 1.000,
        "classes": 4,
        "class_names": ["english", "french", "other", "spanish"]
    },
    "pii-detection": {
        "min_accuracy": 0.95,
        "expected": 1.000,
        "classes": 2,
        "class_names": ["contains_pii", "no_pii"]
    },
    "product-category": {
        "min_accuracy": 0.80,
        "expected": 0.852,
        "classes": 4,
        "class_names": ["books", "clothing", "electronics", "home_garden"]
    },
    "risk-assessment": {
        "min_accuracy": 0.70,
        "expected": 0.756,
        "classes": 2,
        "class_names": ["high_risk", "low_risk"]
    },
    "support-ticket": {
        "min_accuracy": 0.80,
        "expected": 0.829,
        "classes": 4,
        "class_names": ["account", "billing", "general_inquiry", "technical"]
    },
    "vendor-classification": {
        "min_accuracy": 0.90,
        "expected": 0.927,
        "classes": 2,
        "class_names": ["new_vendor", "trusted_partner"]
    }
}

# Domain-specific test sentences for each classifier
TEST_SENTENCES = {
    "business-sentiment": [
        "Our quarterly earnings exceeded all expectations and we're seeing tremendous growth",
        "The company is facing significant financial challenges this quarter",
        "The meeting was informative and covered standard business procedures",
        "The results were good but could have been better in some areas"
    ],
    "compliance-classification": [
        "We need to ensure all patient health information is properly encrypted and access-controlled",
        "The financial data must comply with Sarbanes-Oxley reporting requirements",
        "Credit card processing systems must meet PCI DSS standards",
        "Personal data collection requires explicit GDPR consent mechanisms",
        "This document doesn't fall under any specific compliance framework"
    ],
    "content-moderation": [
        "This is a helpful educational article about machine learning",
        "This content contains inappropriate language and offensive material",
        "Click here to win $1000000 - limited time offer!!!"
    ],
    "customer-intent": [
        "I'm very disappointed with this product and want to file a complaint",
        "Can you tell me more about your premium subscription features?",
        "I'd like to buy three units of product SKU-12345",
        "My account is locked and I can't log in, please help"
    ],
    "document-quality": [
        "This is a well-structured document with clear headings, proper grammar, and comprehensive content",
        "tis doc has many erors and is vry hard too read"
    ],
    "document-type": [
        "This agreement sets forth the terms and conditions between the parties",
        "Subject: Quarterly Team Meeting - Please confirm your attendance",
        "Invoice #2024-001 - Amount Due: $1,250.00 - Payment Terms: Net 30",
        "MEMO: To All Staff - New Office Hours Effective Monday",
        "Annual Sales Report - Q4 2024 Performance Analysis"
    ],
    "email-priority": [
        "FYI - Monthly newsletter with company updates",
        "Please review the attached document when you have time",
        "URGENT: Server down - need immediate assistance"
    ],
    "email-security": [
        "Your monthly bank statement is ready for review",
        "Verify your account immediately by clicking this suspicious link",
        "You've won a lottery you never entered! Claim your prize now!",
        "This email has unusual sending patterns but isn't clearly malicious"
    ],
    "escalation-detection": [
        "Everything is working fine, no issues to report",
        "CRITICAL: Production system failure affecting all users"
    ],
    "expense-category": [
        "Dell laptop computer for development work - $1,200",
        "Business lunch with client at Italian restaurant - $85",
        "Office paper, pens, and printer supplies - $45",
        "Flight and hotel for conference in Chicago - $850",
        "Miscellaneous office expenses and supplies - $120"
    ],
    "fraud-detection": [
        "Multiple small transactions of $9.99 within minutes from different countries",
        "Regular monthly subscription payment of $9.99 to Netflix"
    ],
    "language-detection": [
        "This is a sentence written in English language",
        "Bonjour, comment allez-vous aujourd'hui?",
        "Hola, ¿cómo estás hoy?",
        "这是一个中文句子"  # This should be classified as "other"
    ],
    "pii-detection": [
        "My name is John Smith and my SSN is 123-45-6789",
        "The weather is nice today and I'm going for a walk"
    ],
    "product-category": [
        "The latest thriller novel by bestselling author",
        "Cotton blend t-shirt available in multiple colors",
        "4K Ultra HD Smart TV with streaming capabilities",
        "Outdoor garden furniture set with weather-resistant coating"
    ],
    "risk-assessment": [
        "Multiple failed login attempts from suspicious IP addresses",
        "Regular user accessing normal business applications"
    ],
    "support-ticket": [
        "I forgot my password and can't access my account",
        "My credit card was charged twice for the same purchase",
        "What are your business hours and contact information?",
        "The software keeps crashing when I try to export data"
    ],
    "vendor-classification": [
        "ABC Corp - first time supplier, no previous transaction history",
        "Microsoft - long-term technology partner with established relationship"
    ]
}


@pytest.mark.integration
class TestEnterpriseClassifiers:
    """Integration tests for enterprise classifiers."""

    @pytest.mark.parametrize("classifier_name", list(CLASSIFIER_METRICS.keys()))
    def test_model_loading(self, classifier_name):
        """Test that each enterprise classifier can be loaded from HuggingFace Hub."""
        repo_name = f"adaptive-classifier/{classifier_name}"

        try:
            classifier = AdaptiveClassifier.load(repo_name)
            assert classifier is not None
            assert hasattr(classifier, 'predict')
            assert hasattr(classifier, 'label_to_id')
            assert hasattr(classifier, 'id_to_label')
        except Exception as e:
            pytest.fail(f"Failed to load {repo_name}: {str(e)}")

    @pytest.mark.parametrize("classifier_name", list(CLASSIFIER_METRICS.keys()))
    def test_prediction_functionality(self, classifier_name):
        """Test that each classifier can make predictions."""
        repo_name = f"adaptive-classifier/{classifier_name}"
        classifier = AdaptiveClassifier.load(repo_name)

        # Test with domain-specific sentences
        test_sentences = TEST_SENTENCES[classifier_name]

        for sentence in test_sentences:
            predictions = classifier.predict(sentence, k=3)

            # Verify prediction format
            assert isinstance(predictions, list)
            assert len(predictions) > 0

            for label, confidence in predictions:
                assert isinstance(label, str)
                assert isinstance(confidence, float)
                assert 0.0 <= confidence <= 1.0

            # Verify predicted labels are valid
            expected_classes = CLASSIFIER_METRICS[classifier_name]["class_names"]
            for label, _ in predictions:
                assert label in expected_classes, f"Unexpected label '{label}' for {classifier_name}"

    @pytest.mark.parametrize("classifier_name", list(CLASSIFIER_METRICS.keys()))
    def test_k_parameter_consistency(self, classifier_name):
        """Test that k=1 and k=2 produce consistent top predictions (regression test for k parameter bug)."""
        repo_name = f"adaptive-classifier/{classifier_name}"
        classifier = AdaptiveClassifier.load(repo_name)

        test_sentences = TEST_SENTENCES[classifier_name]

        for sentence in test_sentences:
            # Get predictions with k=1 and k=2
            pred_k1 = classifier.predict(sentence, k=1)
            pred_k2 = classifier.predict(sentence, k=2)

            # Both should return results
            assert len(pred_k1) >= 1
            assert len(pred_k2) >= 1

            # Top prediction should be the same
            top_label_k1 = pred_k1[0][0]
            top_label_k2 = pred_k2[0][0]

            assert top_label_k1 == top_label_k2, (
                f"k=1 and k=2 give different top predictions for {classifier_name}: "
                f"k=1={top_label_k1}, k=2={top_label_k2}, sentence='{sentence[:50]}...'"
            )

            # Confidence scores should be very close (within 1%)
            top_conf_k1 = pred_k1[0][1]
            top_conf_k2 = pred_k2[0][1]

            conf_diff = abs(top_conf_k1 - top_conf_k2)
            assert conf_diff < 0.01, (
                f"k=1 and k=2 confidence scores differ significantly for {classifier_name}: "
                f"k=1={top_conf_k1:.3f}, k=2={top_conf_k2:.3f}, diff={conf_diff:.3f}"
            )

    @pytest.mark.parametrize("classifier_name", list(CLASSIFIER_METRICS.keys()))
    def test_prediction_stability(self, classifier_name):
        """Test that repeated predictions are consistent."""
        repo_name = f"adaptive-classifier/{classifier_name}"
        classifier = AdaptiveClassifier.load(repo_name)

        # Use first test sentence
        test_sentence = TEST_SENTENCES[classifier_name][0]

        # Make multiple predictions
        predictions = []
        for _ in range(3):
            pred = classifier.predict(test_sentence, k=2)
            predictions.append(pred)

        # All predictions should have the same top result
        first_top = predictions[0][0]
        for i, pred in enumerate(predictions[1:], 1):
            current_top = pred[0]
            assert first_top[0] == current_top[0], (
                f"Prediction {i+1} differs from first for {classifier_name}: "
                f"first={first_top[0]}, current={current_top[0]}"
            )

    @pytest.mark.parametrize("classifier_name", list(CLASSIFIER_METRICS.keys()))
    def test_inference_performance(self, classifier_name):
        """Test that inference completes within reasonable time."""
        repo_name = f"adaptive-classifier/{classifier_name}"
        classifier = AdaptiveClassifier.load(repo_name)

        test_sentence = TEST_SENTENCES[classifier_name][0]

        # Time a single prediction
        start_time = time.time()
        predictions = classifier.predict(test_sentence, k=3)
        end_time = time.time()

        inference_time = end_time - start_time

        # Should complete within 2 seconds (generous for CI environments)
        assert inference_time < 2.0, (
            f"Inference too slow for {classifier_name}: {inference_time:.2f}s"
        )

        # Should return valid predictions
        assert len(predictions) > 0

    @pytest.mark.parametrize("classifier_name", list(CLASSIFIER_METRICS.keys()))
    def test_class_coverage(self, classifier_name):
        """Test that the classifier knows about all expected classes."""
        repo_name = f"adaptive-classifier/{classifier_name}"
        classifier = AdaptiveClassifier.load(repo_name)

        expected_classes = set(CLASSIFIER_METRICS[classifier_name]["class_names"])
        expected_count = CLASSIFIER_METRICS[classifier_name]["classes"]

        # Check label mappings
        actual_classes = set(classifier.label_to_id.keys())

        assert len(actual_classes) == expected_count, (
            f"Wrong number of classes for {classifier_name}: "
            f"expected {expected_count}, got {len(actual_classes)}"
        )

        assert actual_classes == expected_classes, (
            f"Class mismatch for {classifier_name}: "
            f"expected {expected_classes}, got {actual_classes}"
        )

    def test_all_classifiers_loadable(self):
        """Test that all enterprise classifiers can be loaded successfully."""
        successful_loads = 0
        failed_loads = []

        for classifier_name in CLASSIFIER_METRICS.keys():
            repo_name = f"adaptive-classifier/{classifier_name}"
            try:
                classifier = AdaptiveClassifier.load(repo_name)
                assert classifier is not None
                successful_loads += 1
            except Exception as e:
                failed_loads.append((classifier_name, str(e)))

        total_classifiers = len(CLASSIFIER_METRICS)

        # Report results
        print(f"\nClassifier Loading Summary:")
        print(f"Successfully loaded: {successful_loads}/{total_classifiers}")

        if failed_loads:
            print(f"Failed to load:")
            for name, error in failed_loads:
                print(f"  - {name}: {error}")

        # All classifiers should load successfully
        assert successful_loads == total_classifiers, (
            f"Failed to load {len(failed_loads)} classifiers: {[name for name, _ in failed_loads]}"
        )

    def test_integration_health_check(self):
        """Overall health check for the enterprise classifier ecosystem."""
        print(f"\n{'='*60}")
        print("ENTERPRISE CLASSIFIER INTEGRATION HEALTH CHECK")
        print(f"{'='*60}")

        results = {
            "total_classifiers": len(CLASSIFIER_METRICS),
            "high_accuracy": 0,    # >95%
            "good_accuracy": 0,    # 80-95%
            "acceptable_accuracy": 0,  # 60-80%
            "low_accuracy": 0,     # <60%
        }

        for classifier_name, metrics in CLASSIFIER_METRICS.items():
            expected_acc = metrics["expected"]

            if expected_acc >= 0.95:
                results["high_accuracy"] += 1
            elif expected_acc >= 0.80:
                results["good_accuracy"] += 1
            elif expected_acc >= 0.60:
                results["acceptable_accuracy"] += 1
            else:
                results["low_accuracy"] += 1

        print(f"Total classifiers: {results['total_classifiers']}")
        print(f"High accuracy (≥95%): {results['high_accuracy']}")
        print(f"Good accuracy (80-95%): {results['good_accuracy']}")
        print(f"Acceptable accuracy (60-80%): {results['acceptable_accuracy']}")
        print(f"Low accuracy (<60%): {results['low_accuracy']}")

        # Health assertions
        assert results["total_classifiers"] == 17, "Should have exactly 17 enterprise classifiers"
        assert results["high_accuracy"] >= 6, "Should have at least 6 high-accuracy classifiers"
        assert results["low_accuracy"] == 0, "Should have no low-accuracy classifiers"

        print(f"✅ Enterprise classifier ecosystem is healthy!")
        print(f"{'='*60}")


# Optional: Run specific classifier tests
if __name__ == "__main__":
    # Run tests for a specific classifier
    import sys
    if len(sys.argv) > 1:
        classifier_name = sys.argv[1]
        if classifier_name in CLASSIFIER_METRICS:
            pytest.main([f"-v", f"-k", f"test_{classifier_name}", __file__])
        else:
            print(f"Unknown classifier: {classifier_name}")
            print(f"Available classifiers: {list(CLASSIFIER_METRICS.keys())}")
    else:
        pytest.main(["-v", __file__])