# Enterprise Classifier Integration Tests

This document describes the integration test suite for enterprise classifiers hosted on Hugging Face Hub.

## Overview

The integration tests (`tests/test_enterprise_classifiers_integration.py`) verify that all 17 enterprise classifiers maintain their expected performance and behavior. These tests serve as regression tests to ensure code changes don't break the published models.

## Test Coverage

The integration test suite covers:

- **Model Loading**: Can each model be loaded from HuggingFace Hub?
- **Prediction Functionality**: Do models make valid predictions?
- **k-Parameter Consistency**: Do k=1 and k=2 produce consistent results? (regression test for the k parameter bug)
- **Prediction Stability**: Are repeated predictions consistent?
- **Performance**: Does inference complete within reasonable time?
- **Class Coverage**: Do models know about all expected classes?
- **Health Check**: Overall ecosystem health assessment

## Running Integration Tests

### Run All Integration Tests
```bash
pytest tests/test_enterprise_classifiers_integration.py -v
```

### Run Only Unit Tests (Skip Integration)
```bash
pytest tests/ -m "not integration" -v
```

### Run Specific Test for One Classifier
```bash
pytest tests/test_enterprise_classifiers_integration.py -k "fraud-detection" -v
```

### Run Specific Test Type
```bash
# Test k-parameter consistency for all classifiers
pytest tests/test_enterprise_classifiers_integration.py::TestEnterpriseClassifiers::test_k_parameter_consistency -v

# Test model loading for all classifiers
pytest tests/test_enterprise_classifiers_integration.py::TestEnterpriseClassifiers::test_model_loading -v
```

## CI/CD Integration

The CI/CD pipeline runs integration tests automatically:

1. **Unit Tests Job**: Runs all unit tests first
2. **Integration Tests Job**: Runs only if unit tests pass
   - 30-minute timeout for model downloads
   - Tests all 17 enterprise classifiers
   - Reports detailed results

## Tested Classifiers

The following 17 enterprise classifiers are tested:

| Classifier | Expected Accuracy | Classes | Use Case |
|------------|------------------|---------|----------|
| business-sentiment | 98.8% | 4 | Business text sentiment analysis |
| compliance-classification | 65.3% | 5 | Regulatory compliance categorization |
| content-moderation | 100.0% | 3 | Content filtering and moderation |
| customer-intent | 85.2% | 4 | Customer service intent detection |
| document-quality | 100.0% | 2 | Document quality assessment |
| document-type | 98.0% | 5 | Document type classification |
| email-priority | 83.9% | 3 | Email priority triage |
| email-security | 93.8% | 4 | Email security threat detection |
| escalation-detection | 97.6% | 2 | Support ticket escalation detection |
| expense-category | 84.2% | 5 | Business expense categorization |
| fraud-detection | 92.7% | 2 | Financial fraud detection |
| language-detection | 100.0% | 4 | Text language identification |
| pii-detection | 100.0% | 2 | Personal information detection |
| product-category | 85.2% | 4 | E-commerce product categorization |
| risk-assessment | 75.6% | 2 | Security risk assessment |
| support-ticket | 82.9% | 4 | Support ticket categorization |
| vendor-classification | 92.7% | 2 | Vendor relationship classification |

## Test Assertions

Each classifier is tested against:

- **Minimum Accuracy Thresholds**: Must meet or exceed defined minimums
- **k-Parameter Consistency**: k=1 and k=2 must produce identical top predictions
- **Response Time**: Inference must complete within 2 seconds
- **Class Completeness**: Must know about all expected classes
- **Prediction Validity**: All predictions must be properly formatted

## Failure Modes

Tests will fail if:

- Model cannot be loaded from HuggingFace Hub
- Accuracy drops below minimum threshold
- k=1 and k=2 produce different top predictions (regression)
- Inference takes longer than 2 seconds
- Predicted classes don't match expected class set
- Prediction format is invalid

## Adding New Enterprise Classifiers

To add a new enterprise classifier to the test suite:

1. Update `CLASSIFIER_METRICS` dictionary with expected metrics
2. Add domain-specific test sentences to `TEST_SENTENCES`
3. Ensure the model is available on HuggingFace Hub as `adaptive-classifier/{name}`

## Debugging Test Failures

### Model Loading Failures
- Check that the model exists on HuggingFace Hub
- Verify network connectivity
- Check for authentication issues (if model is private)

### Accuracy Failures
- Compare actual vs expected accuracy in test output
- Check if model was retrained recently
- Verify test sentences are appropriate for the domain

### k-Parameter Inconsistencies
- This indicates a regression in the k parameter fix
- Check prediction logic in `_predict_regular` method
- Verify prototype and neural head prediction combination

### Performance Issues
- Check system resources during testing
- Consider network latency for model downloads
- Verify model size hasn't increased significantly

## Local Development

For faster local testing, you can:

```bash
# Skip integration tests during development
pytest tests/ -m "not integration"

# Test specific classifier during debugging
pytest tests/test_enterprise_classifiers_integration.py -k "fraud-detection" -v -s
```

## Maintenance

The integration test suite should be updated when:

- New enterprise classifiers are published
- Expected accuracy thresholds change
- New test dimensions are needed
- Test sentences need updating for better coverage

This ensures the test suite remains comprehensive and valuable for regression detection.