import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set, Union
import logging
from collections import defaultdict

from .classifier import AdaptiveClassifier, _synchronized
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

    def remove_classes(self, keep_indices: list):
        """Keep only the given output rows, in the given order."""
        if not keep_indices:
            raise ValueError("Cannot remove every class from the head")
        layer = self.model[-1]
        new_layer = nn.Linear(layer.in_features, len(keep_indices))
        with torch.no_grad():
            index = torch.tensor(keep_indices, dtype=torch.long, device=layer.weight.device)
            new_layer.weight.copy_(layer.weight[index])
            new_layer.bias.copy_(layer.bias[index])
        self.model[-1] = new_layer.to(layer.weight.device)
        self.num_classes = len(keep_indices)


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
        max_predictions: Optional[int] = None,
        use_onnx: Optional[Union[bool, str]] = "auto",
        trust_remote_code: bool = False
    ):
        if not 0.0 <= default_threshold <= 1.0:
            raise ValueError(f"default_threshold must be between 0 and 1; got {default_threshold!r}")
        if min_predictions < 0:
            raise ValueError(f"min_predictions must be >= 0; got {min_predictions!r}")
        if max_predictions is not None and max_predictions < 1:
            raise ValueError(f"max_predictions must be >= 1 or None; got {max_predictions!r}")
        if (config or {}).get('enable_strategic_mode'):
            raise ValueError(
                "Strategic mode is not supported by MultiLabelAdaptiveClassifier; "
                "use AdaptiveClassifier for it")
        super().__init__(model_name, device, config, seed, use_onnx, trust_remote_code)

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

    def _extra_state(self) -> Dict[str, Any]:
        """Settings that `save` stores beside the base classifier's."""
        return {
            'multilabel': {
                'default_threshold': self.default_threshold,
                'min_predictions': self.min_predictions,
                'max_predictions': self.max_predictions,
                'label_thresholds': dict(self.label_thresholds),
            }
        }

    def _restore_extra_state(self, state: Dict[str, Any]):
        saved = (state or {}).get('multilabel') or {}
        self.default_threshold = saved.get('default_threshold', self.default_threshold)
        self.min_predictions = saved.get('min_predictions', self.min_predictions)
        self.max_predictions = saved.get('max_predictions', self.max_predictions)
        self.label_thresholds = dict(saved.get('label_thresholds', {}))

    def _drop_classes(self, drop):
        super()._drop_classes(drop)
        for label in drop:
            self.label_thresholds.pop(label, None)

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

    @_synchronized
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
        if max_labels is not None and (isinstance(max_labels, bool) or not isinstance(max_labels, int) or max_labels < 0):
            raise ValueError(f"max_labels must be a non-negative integer or None; got {max_labels!r}")
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1; got {threshold!r}")

        num_labels = len(self.label_to_id)
        if num_labels == 0:
            return []

        # An explicit threshold applies to every label; otherwise each label uses the
        # threshold learned for it (falling back to the adaptive one).
        explicit_threshold = threshold is not None
        if threshold is None:
            threshold = self._get_adaptive_threshold(num_labels)

        if max_labels is None:
            max_labels = self.max_predictions

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
                        label_threshold = threshold if explicit_threshold else self.label_thresholds.get(label, threshold)
                        if prob.item() >= label_threshold:
                            predictions.append((label, prob.item()))

                # Sort by confidence
                predictions.sort(key=lambda x: x[1], reverse=True)

                # Apply max_labels limit
                if max_labels is not None and len(predictions) > max_labels:
                    predictions = predictions[:max_labels]

            else:
                # Fallback to prototype-based prediction
                proto_predictions = self.memory.get_nearest_prototypes(
                    embedding,
                    k=num_labels if max_labels is None else min(num_labels, max_labels)
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

        # max_labels is a hard cap: it also bounds the min_predictions top-up.
        if max_labels is not None:
            predictions = predictions[:max_labels]
        return predictions

    @_synchronized
    def predict(
        self,
        text: str,
        k: int = 5,
        abstain_below: Optional[float] = None,
        abstain_ood: bool = False,
    ) -> List[Tuple[str, float]]:
        """The labels `predict_multilabel` picks, at most `k` of them.

        Falls back to the single-label prediction when no label clears its
        threshold and `min_predictions` is 0. An empty list means the classifier
        abstained: the best score is below `abstain_below`, or the text is out of
        distribution (`abstain_ood`).
        """
        self._validate_k(k)
        if abstain_ood and self.is_ood(text):
            return []
        multilabel_preds = self.predict_multilabel(text, max_labels=k)

        if multilabel_preds:
            predictions = multilabel_preds[:k]
        else:
            # Fallback to base prediction if no multi-label predictions
            predictions = super().predict(text, k)
        if abstain_below is not None and (not predictions or predictions[0][1] < abstain_below):
            return []
        return predictions

    @_synchronized
    def predict_batch(self, texts: List[str], k: int = 5, batch_size: int = 32) -> List[List[Tuple[str, float]]]:
        """`predict` for each text. (The single-label batch path softmaxes the scores, which is wrong for sigmoid outputs.)"""
        if not texts:
            raise ValueError("Empty input batch")
        self._validate_k(k)
        return [self.predict(text, k) for text in texts]

    @_synchronized
    def add_examples(self, texts: List[str], labels: List[List[str]]):
        """
        Add multi-label training examples.

        Args:
            texts: List of input texts
            labels: One list of labels per text (each text can have several).
                A text with an empty label list is skipped.
        """
        if not texts or not labels:
            raise ValueError("Empty input lists")
        if len(texts) != len(labels):
            raise ValueError("Mismatched text and label lists")
        for text_labels in labels:
            # A bare string would otherwise be split into one class per character.
            if not isinstance(text_labels, (list, tuple, set, frozenset)):
                raise ValueError(
                    "labels must be a list of label lists, one per text; got "
                    f"{type(text_labels).__name__} {text_labels!r} (wrap a single label as ['label'])"
                )

        # One training example per (text, label) pair; a label repeated within a
        # text counts once.
        flattened_texts = []
        flattened_labels = []
        for text, text_labels in zip(texts, labels):
            for label in dict.fromkeys(text_labels):
                flattened_texts.append(text)
                flattened_labels.append(label)

        if flattened_texts:
            # The parent validates that texts and labels are non-empty strings.
            super().add_examples(flattened_texts, flattened_labels)
        elif not all(isinstance(t, str) for t in texts):
            raise ValueError("texts must all be strings")

        # Update label-specific thresholds based on training data
        self._update_label_thresholds()

    @_synchronized
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

    def _train_new_classes(self, old_head, new_classes):
        """Retrain the head on every stored example when classes are added.

        The single-label routine behind this hook uses a softmax loss, which
        does not apply to independent per-label outputs.
        """
        self._train_adaptive_head()

    def _train_adaptive_head(self, epochs: int = 10):
        """Train the multi-label head with binary cross-entropy.

        Like the single-label head, it runs at least `head_steps` optimiser
        steps with a cosine-decayed learning rate and no early stopping; a few
        epochs over a small memory is too little for the head to learn anything.
        """
        if not self.memory.examples:
            return

        # A text's targets are every label it was stored under.
        text_to_labels = defaultdict(set)
        text_to_embedding = {}
        for label, examples in self.memory.examples.items():
            for example in examples:
                text_to_labels[example.text].add(label)
                text_to_embedding.setdefault(example.text, example.embedding)

        num_classes = len(self.label_to_id)
        texts = sorted(text_to_labels)
        targets = torch.zeros(len(texts), num_classes)
        for row, text in enumerate(texts):
            for label in text_to_labels[text]:
                if label in self.label_to_id:
                    targets[row, self.label_to_id[label]] = 1.0
        embeddings = F.normalize(torch.stack([text_to_embedding[t] for t in texts]), p=2, dim=1)
        embeddings, targets = embeddings.to(self.device), targets.to(self.device)

        batch_size = min(32, len(embeddings))
        batches_per_epoch = math.ceil(len(embeddings) / batch_size)
        total_steps = max(epochs * batches_per_epoch,
                          int(getattr(self.config, 'head_steps', 0) or 0))

        self.adaptive_head.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(
            self.adaptive_head.parameters(),
            lr=getattr(self.config, 'head_learning_rate', 0.001),
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        generator = torch.Generator().manual_seed(42)

        step = 0
        while step < total_steps:
            order = torch.randperm(len(embeddings), generator=generator)
            for start in range(0, len(order), batch_size):
                index = order[start:start + batch_size]
                optimizer.zero_grad()
                loss = criterion(self.adaptive_head(embeddings[index]), targets[index])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adaptive_head.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                step += 1
                if step >= total_steps:
                    break

        self.adaptive_head.eval()
        self.train_steps += 1

    def merge_classifiers(self, other):
        # Not synchronized: the base class never holds two classifiers' locks at once.
        merged = super().merge_classifiers(other)
        self._update_label_thresholds()
        return merged

    @_synchronized
    def remove_examples(self, texts, label=None, retrain=True):
        removed = super().remove_examples(texts, label=label, retrain=retrain)
        self._update_label_thresholds()
        return removed

    @_synchronized
    def clear_memory(self, labels=None):
        super().clear_memory(labels)
        if labels is None:
            self.label_thresholds = {}

    def calibrate(self, *args, **kwargs):
        raise NotImplementedError(
            "Confidence calibration assumes one label per text and is not available "
            "for MultiLabelAdaptiveClassifier")

    calibration_report = calibrate
    predict_set = calibrate

    @_synchronized
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