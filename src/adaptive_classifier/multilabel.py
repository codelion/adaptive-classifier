import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set, Union
import logging
from collections import defaultdict

from .classifier import AdaptiveClassifier
from .models import AdaptiveHead

logger = logging.getLogger(__name__)


class MultiLabelAdaptiveHead(nn.Module):
    """Multi-label version of adaptive head using sigmoid activation."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [input_dim // 2]

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim

        # Final layer with sigmoid for multi-label
        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, x):
        logits = self.model(x)
        # Apply sigmoid for multi-label prediction
        return torch.sigmoid(logits)

    def update_num_classes(self, new_num_classes: int):
        """Update the number of output classes while preserving existing weights."""
        if new_num_classes <= self.num_classes:
            return

        # Get the final layer
        final_layer = self.model[-1]

        # Create new final layer
        new_final_layer = nn.Linear(final_layer.in_features, new_num_classes)

        # Copy existing weights
        with torch.no_grad():
            new_final_layer.weight[:self.num_classes] = final_layer.weight
            new_final_layer.bias[:self.num_classes] = final_layer.bias

            # Initialize new class weights with small random values
            nn.init.xavier_uniform_(new_final_layer.weight[self.num_classes:])
            nn.init.zeros_(new_final_layer.bias[self.num_classes:])

        # Replace the final layer
        self.model[-1] = new_final_layer
        self.num_classes = new_num_classes


class MultiLabelAdaptiveClassifier(AdaptiveClassifier):
    """
    Multi-label extension of AdaptiveClassifier that can predict multiple labels per input.

    Handles the "No labels met the threshold criteria" issue by implementing:
    1. Adaptive thresholds based on number of labels
    2. Minimum predictions per sample
    3. Label-specific threshold adjustments
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        default_threshold: float = 0.5,
        min_predictions: int = 1,
        max_predictions: Optional[int] = None
    ):
        super().__init__(model_name, device, config, seed)

        # Multi-label specific configuration
        self.default_threshold = default_threshold
        self.min_predictions = min_predictions
        self.max_predictions = max_predictions
        self.label_thresholds = {}  # Per-label thresholds

        # Override adaptive head with multi-label version
        self.adaptive_head = None

    def _initialize_adaptive_head(self):
        """Initialize multi-label adaptive head."""
        num_classes = len(self.label_to_id)
        hidden_dims = [self.embedding_dim, self.embedding_dim // 2]

        self.adaptive_head = MultiLabelAdaptiveHead(
            self.embedding_dim,
            num_classes,
            hidden_dims=hidden_dims
        ).to(self.device)

    def _get_adaptive_threshold(self, num_labels: int) -> float:
        """
        Calculate adaptive threshold based on number of labels.

        With more labels, individual prediction scores tend to be lower,
        so we need a lower threshold to avoid "No labels met the threshold criteria".
        """
        if num_labels <= 2:
            return self.default_threshold
        elif num_labels <= 5:
            return self.default_threshold * 0.8
        elif num_labels <= 10:
            return self.default_threshold * 0.6
        elif num_labels <= 20:
            return self.default_threshold * 0.4
        else:
            # For many labels (20+), use very low threshold
            return self.default_threshold * 0.2

    def predict_multilabel(
        self,
        text: str,
        threshold: Optional[float] = None,
        max_labels: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Predict multiple labels for input text.

        Args:
            text: Input text to classify
            threshold: Confidence threshold for predictions (adaptive if None)
            max_labels: Maximum number of labels to return

        Returns:
            List of (label, confidence) tuples for labels above threshold
        """
        if not text:
            raise ValueError("Empty input text")

        num_labels = len(self.label_to_id)
        if num_labels == 0:
            return []

        # Use adaptive threshold if not specified
        if threshold is None:
            threshold = self._get_adaptive_threshold(num_labels)

        max_labels = max_labels or self.max_predictions

        with torch.no_grad():
            # Get embedding
            embedding = self._get_embeddings([text])[0]

            # Get predictions from neural head
            if self.adaptive_head is not None:
                self.adaptive_head.eval()
                input_embedding = embedding.unsqueeze(0).to(self.device)
                probabilities = self.adaptive_head(input_embedding).squeeze(0)

                # Convert to label predictions
                predictions = []
                for i, prob in enumerate(probabilities):
                    if i < len(self.id_to_label):
                        label = self.id_to_label[i]
                        # Use label-specific threshold if available
                        label_threshold = self.label_thresholds.get(label, threshold)
                        if prob.item() >= label_threshold:
                            predictions.append((label, prob.item()))

                # Sort by confidence
                predictions.sort(key=lambda x: x[1], reverse=True)

                # Apply max_labels limit
                if max_labels and len(predictions) > max_labels:
                    predictions = predictions[:max_labels]

            else:
                # Fallback to prototype-based prediction
                proto_predictions = self.memory.get_nearest_prototypes(
                    embedding,
                    k=min(num_labels, max_labels) if max_labels else num_labels
                )

                # Filter by threshold
                predictions = [
                    (label, score) for label, score in proto_predictions
                    if score >= threshold
                ]

        # Ensure minimum predictions if required
        if len(predictions) < self.min_predictions and self.adaptive_head is not None:
            # Add top predictions even if below threshold
            with torch.no_grad():
                input_embedding = embedding.unsqueeze(0).to(self.device)
                probabilities = self.adaptive_head(input_embedding).squeeze(0)

                # Get top predictions
                values, indices = torch.topk(
                    probabilities,
                    min(self.min_predictions, len(self.id_to_label))
                )

                additional_predictions = []
                for val, idx in zip(values, indices):
                    if idx.item() < len(self.id_to_label):
                        label = self.id_to_label[idx.item()]
                        score = val.item()

                        # Only add if not already included
                        if not any(pred[0] == label for pred in predictions):
                            additional_predictions.append((label, score))

                # Add additional predictions to meet minimum
                predictions.extend(additional_predictions[:self.min_predictions - len(predictions)])
                predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions

    def predict(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Override base predict to use multi-label prediction.
        Falls back to single-label prediction if needed.
        """
        # Use multi-label prediction but limit to k results
        multilabel_preds = self.predict_multilabel(text, max_labels=k)

        if multilabel_preds:
            return multilabel_preds[:k]
        else:
            # Fallback to base prediction if no multi-label predictions
            return super().predict(text, k)

    def add_examples(self, texts: List[str], labels: List[List[str]]):
        """
        Add multi-label training examples.

        Args:
            texts: List of input texts
            labels: List of label lists (each text can have multiple labels)
        """
        if not texts or not labels:
            raise ValueError("Empty input lists")
        if len(texts) != len(labels):
            raise ValueError("Mismatched text and label lists")

        # Flatten labels for single-label training approach
        # We'll train one example per text-label pair
        flattened_texts = []
        flattened_labels = []

        for text, text_labels in zip(texts, labels):
            if not text_labels:  # Skip texts with no labels
                continue

            # For multi-label, we create multiple training examples
            # Each example represents the text with one of its labels
            for label in text_labels:
                flattened_texts.append(text)
                flattened_labels.append(label)

        if flattened_texts:
            # Use parent class method with flattened examples
            super().add_examples(flattened_texts, flattened_labels)

        # Update label-specific thresholds based on training data
        self._update_label_thresholds()

    def _update_label_thresholds(self):
        """Update per-label thresholds based on training data distribution."""
        if not self.memory.examples:
            return

        # Calculate label frequencies
        label_counts = defaultdict(int)
        total_examples = 0

        for label, examples in self.memory.examples.items():
            label_counts[label] = len(examples)
            total_examples += len(examples)

        # Adjust thresholds based on label frequency
        # Rare labels get lower thresholds, common labels get higher thresholds
        for label, count in label_counts.items():
            frequency = count / total_examples

            if frequency < 0.05:  # Very rare labels (< 5%)
                self.label_thresholds[label] = self.default_threshold * 0.3
            elif frequency < 0.1:  # Rare labels (< 10%)
                self.label_thresholds[label] = self.default_threshold * 0.5
            elif frequency > 0.3:  # Very common labels (> 30%)
                self.label_thresholds[label] = self.default_threshold * 1.2
            else:  # Normal frequency labels
                self.label_thresholds[label] = self.default_threshold

        logger.debug(f"Updated label thresholds: {self.label_thresholds}")

    def _train_adaptive_head(self, epochs: int = 10):
        """Train multi-label adaptive head with BCE loss."""
        if not self.memory.examples:
            return

        # Prepare multi-label training data
        all_embeddings = []
        all_labels = []

        # Create label matrix for multi-label training
        num_classes = len(self.label_to_id)

        # Collect unique texts and their labels
        text_to_labels = defaultdict(set)
        for label, examples in self.memory.examples.items():
            for example in examples:
                text_to_labels[example.text].add(label)

        # Create training data with proper multi-label targets
        for text, labels in text_to_labels.items():
            # Get embedding for this text (take first occurrence)
            embedding = None
            for label in labels:
                for example in self.memory.examples[label]:
                    if example.text == text:
                        embedding = example.embedding
                        break
                if embedding is not None:
                    break

            if embedding is not None:
                all_embeddings.append(embedding)

                # Create multi-hot encoded label vector
                label_vector = torch.zeros(num_classes)
                for label in labels:
                    if label in self.label_to_id:
                        label_vector[self.label_to_id[label]] = 1.0

                all_labels.append(label_vector)

        if not all_embeddings:
            return

        all_embeddings = torch.stack(all_embeddings)
        all_labels = torch.stack(all_labels)

        # Normalize embeddings
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(all_embeddings, all_labels)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(32, len(all_embeddings)),
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )

        # Training setup
        self.adaptive_head.train()
        criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label
        optimizer = torch.optim.AdamW(
            self.adaptive_head.parameters(),
            lr=0.001,
            weight_decay=0.01
        )

        best_loss = float('inf')
        patience_counter = 0
        patience = 3

        for epoch in range(epochs):
            total_loss = 0
            for batch_embeddings, batch_labels in loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.adaptive_head(batch_embeddings)

                loss = criterion(outputs, batch_labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.adaptive_head.parameters(),
                    max_norm=1.0
                )
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.debug(f"Early stopping at epoch {epoch + 1}")
                    break

        self.train_steps += 1

    def get_label_statistics(self) -> Dict[str, Any]:
        """Get statistics about label distribution and thresholds."""
        stats = super().get_example_statistics()

        # Add multi-label specific stats
        stats['label_thresholds'] = dict(self.label_thresholds)
        stats['adaptive_threshold'] = self._get_adaptive_threshold(len(self.label_to_id))
        stats['default_threshold'] = self.default_threshold
        stats['min_predictions'] = self.min_predictions
        stats['max_predictions'] = self.max_predictions

        return stats