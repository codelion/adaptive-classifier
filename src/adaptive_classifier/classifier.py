import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Tuple, Any, Set
import logging
import copy
from pathlib import Path
from safetensors.torch import save_file, load_file
import json
from sklearn.cluster import KMeans

from .models import Example, AdaptiveHead, ModelConfig
from .memory import PrototypeMemory
from .ewc import EWC


logger = logging.getLogger(__name__)

class AdaptiveClassifier:
    """A flexible classifier that can adapt to new classes and examples."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42  # Add seed parameter
    ):
        """Initialize the adaptive classifier.
        
        Args:
            model_name: Name of the HuggingFace transformer model
            device: Device to run the model on (default: auto-detect)
            config: Optional configuration dictionary
        """
        # Set seed for initialization
        torch.manual_seed(seed)
        self.config = ModelConfig(config)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize transformer model and tokenizer
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize memory system
        self.embedding_dim = self.model.config.hidden_size
        self.memory = PrototypeMemory(
            self.embedding_dim,
            config=self.config
        )
        
        # Initialize adaptive head
        self.adaptive_head = None
        
        # Label mappings
        self.label_to_id = {}
        self.id_to_label = {}
        
        # Statistics
        self.train_steps = 0
    
    def add_examples(self, texts: List[str], labels: List[str]):
        """Add new examples with special handling for new classes."""
        if not texts or not labels:
            raise ValueError("Empty input lists")
        if len(texts) != len(labels):
            raise ValueError("Mismatched text and label lists")
        
        # Check for new classes
        new_classes = set(labels) - set(self.label_to_id.keys())
        is_adding_new_classes = len(new_classes) > 0
        
        # Update label mappings
        for label in new_classes:
            idx = len(self.label_to_id)
            self.label_to_id[label] = idx
            self.id_to_label[idx] = label
        
        # Get embeddings for all texts
        embeddings = self._get_embeddings(texts)
        
        # Add examples to memory
        for text, embedding, label in zip(texts, embeddings, labels):
            example = Example(text, label, embedding)
            self.memory.add_example(example, label)
        
        # Special handling for new classes
        if is_adding_new_classes:
            # Store old head for EWC
            old_head = copy.deepcopy(self.adaptive_head) if self.adaptive_head is not None else None
            
            # Initialize new head with more output classes
            self._initialize_adaptive_head()
            
            # Train with focus on new classes
            self._train_new_classes(old_head, new_classes)
        else:
            # Regular training for existing classes
            self._train_adaptive_head()
    
    def _train_new_classes(self, old_head: Optional[nn.Module], new_classes: Set[str]):
        """Train the model with focus on new classes while preserving old class knowledge."""
        if not self.memory.examples:
            return
        
        # Prepare training data with balanced sampling
        all_embeddings = []
        all_labels = []
        examples_per_class = {}
        
        # Count examples per class
        for label in self.memory.examples:
            examples_per_class[label] = len(self.memory.examples[label])
        
        # Calculate sampling weights to balance old and new classes
        min_examples = min(examples_per_class.values())
        sampling_weights = {}
        
        for label, count in examples_per_class.items():
            if label in new_classes:
                # Oversample new classes
                sampling_weights[label] = 2.0
            else:
                # Sample old classes proportionally
                sampling_weights[label] = min_examples / count
        
        # Sample examples with weights
        for label, examples in self.memory.examples.items():
            weight = sampling_weights[label]
            num_samples = max(min_examples, int(len(examples) * weight))
            
            # Randomly sample with replacement if needed
            indices = np.random.choice(
                len(examples),
                size=num_samples,
                replace=num_samples > len(examples)
            )
            
            for idx in indices:
                example = examples[idx]
                all_embeddings.append(example.embedding)
                all_labels.append(self.label_to_id[label])
        
        all_embeddings = torch.stack(all_embeddings)
        all_labels = torch.tensor(all_labels)
        
        # Create dataset and initialize EWC with lower penalty for new classes
        dataset = torch.utils.data.TensorDataset(all_embeddings, all_labels)
        
        if old_head is not None:
            ewc = EWC(
                old_head,
                dataset,
                device=self.device,
                ewc_lambda=10.0  # Lower EWC penalty to allow better learning of new classes
            )
        
        # Training setup
        self.adaptive_head.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.adaptive_head.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # Create data loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )
        
        # Training loop
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(15):  # More epochs for new classes
            total_loss = 0
            for batch_embeddings, batch_labels in loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.adaptive_head(batch_embeddings)
                
                # Compute task loss
                task_loss = criterion(outputs, batch_labels)
                
                # Add EWC loss if applicable
                if old_head is not None:
                    ewc_loss = ewc.ewc_loss(batch_size=len(batch_embeddings))
                    loss = task_loss + ewc_loss
                else:
                    loss = task_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.adaptive_head.parameters(),
                    max_norm=1.0
                )
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.debug(f"Early stopping at epoch {epoch + 1}")
                    break
        
        self.train_steps += 1
    
    def predict(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Predict with adjusted weights for new classes."""
        if not text:
            raise ValueError("Empty input text")
        
        # Ensure deterministic behavior
        with torch.no_grad():
            # Get embedding
            embedding = self._get_embeddings([text])[0]
            
            # Get prototype predictions
            proto_preds = self.memory.get_nearest_prototypes(embedding, k=k)
            
            # Get neural predictions if available
            if self.adaptive_head is not None:
                self.adaptive_head.eval()  # Ensure eval mode
                logits = self.adaptive_head(embedding.to(self.device))
                probs = F.softmax(logits, dim=0)
                
                values, indices = torch.topk(probs, min(k, len(self.id_to_label)))
                head_preds = [
                    (self.id_to_label[idx.item()], prob.item())
                    for prob, idx in zip(values, indices)
                ]
            else:
                head_preds = []
        
        # Combine predictions with adjusted weights
        combined_scores = {}
        
        # Use neural predictions more for recent classes
        for label, score in proto_preds:
            if label in self.memory.examples and len(self.memory.examples[label]) < 10:
                # For newer classes (fewer examples), trust neural predictions more
                weight = 0.3  # Lower prototype weight for new classes
            else:
                weight = 0.7  # Higher prototype weight for established classes
            combined_scores[label] = score * weight
        
        for label, score in head_preds:
            if label in self.memory.examples and len(self.memory.examples[label]) < 10:
                weight = 0.7  # Higher neural weight for new classes
            else:
                weight = 0.3  # Lower neural weight for established classes
            combined_scores[label] = combined_scores.get(label, 0) + score * weight
        
        # Normalize scores
        predictions = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        total = sum(score for _, score in predictions)
        if total > 0:
            predictions = [(label, score/total) for label, score in predictions]
        
        return predictions[:k]
    
    def save(self, save_dir: str):
        """Save classifier state with representative examples."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Select representative examples for each class
        saved_examples = {}
        for label, examples in self.memory.examples.items():
            saved_examples[label] = [
                ex.to_dict() for ex in 
                self.select_representative_examples(examples, k=5)
            ]
        
        # Save configuration and metadata
        config = {
            'model_name': self.model.config._name_or_path,
            'embedding_dim': self.embedding_dim,
            'label_to_id': self.label_to_id,
            'id_to_label': {str(k): v for k, v in self.id_to_label.items()},
            'train_steps': self.train_steps,
            'config': self.config.to_dict(),
            'examples': saved_examples
        }
        
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f)
        
        # Save model tensors
        tensor_dict = {}
        
        # Save prototypes
        for label, proto in self.memory.prototypes.items():
            tensor_dict[f'prototype_{label}'] = proto
        
        # Save adaptive head if it exists
        if self.adaptive_head is not None:
            for name, param in self.adaptive_head.state_dict().items():
                tensor_dict[f'adaptive_head_{name}'] = param
        
        # Save tensors
        save_file(tensor_dict, save_dir / 'tensors.safetensors')
    
    @classmethod
    def load(cls, save_dir: str, device: Optional[str] = None) -> 'AdaptiveClassifier':
        """Load classifier with saved examples."""
        save_dir = Path(save_dir)
        
        # Load configuration
        with open(save_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Initialize classifier
        classifier = cls(
            config['model_name'],
            device=device,
            config=config.get('config', None)
        )
        
        # Restore label mappings
        classifier.label_to_id = config['label_to_id']
        classifier.id_to_label = {
            int(k): v for k, v in config['id_to_label'].items()
        }
        classifier.train_steps = config['train_steps']
        
        # Load tensors
        tensors = load_file(save_dir / 'tensors.safetensors')
        
        # Restore saved examples
        saved_examples = config['examples']
        for label, examples_data in saved_examples.items():
            classifier.memory.examples[label] = [
                Example.from_dict(ex_data) for ex_data in examples_data
            ]
        
        # Restore prototypes
        for label in classifier.label_to_id.keys():
            prototype_key = f'prototype_{label}'
            if prototype_key in tensors:
                prototype = tensors[prototype_key]
                classifier.memory.prototypes[label] = prototype
        
        # Rebuild memory system
        classifier.memory._restore_from_save()
        
        # Restore adaptive head if it exists
        adaptive_head_params = {
            k.replace('adaptive_head_', ''): v 
            for k, v in tensors.items() 
            if k.startswith('adaptive_head_')
        }
        
        if adaptive_head_params:
            classifier._initialize_adaptive_head()
            classifier.adaptive_head.load_state_dict(adaptive_head_params)
        
        return classifier
    
    def to(self, device: str) -> 'AdaptiveClassifier':
        """Move the model to specified device.
        
        Args:
            device: Device to move to ("cuda" or "cpu")
            
        Returns:
            Self for chaining
        """
        self.device = device
        self.model = self.model.to(device)
        if self.adaptive_head is not None:
            self.adaptive_head = self.adaptive_head.to(device)
        return self
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        return self.memory.get_stats()
    
    def _initialize_adaptive_head(self):
        """Initialize or reinitialize the adaptive head with improved configuration."""
        num_classes = len(self.label_to_id)
        hidden_dims = [self.embedding_dim, self.embedding_dim // 2]
        
        self.adaptive_head = AdaptiveHead(
            self.embedding_dim,
            num_classes,
            hidden_dims=hidden_dims
        ).to(self.device)

    def _get_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Get embeddings for input texts with improved caching."""
        # Sort texts for consistent tokenization
        sorted_indices = list(range(len(texts)))
        sorted_indices.sort(key=lambda i: texts[i])
        sorted_texts = [texts[i] for i in sorted_indices]
        
        # Temporarily set model to eval mode
        was_training = self.model.training
        self.model.eval()
        
        # Get embeddings
        with torch.no_grad():
            inputs = self.tokenizer(
                sorted_texts,
                max_length=self.config.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Restore original training mode
        if was_training:
            self.model.train()
        
        # Restore original order
        original_order = [0] * len(texts)
        for i, idx in enumerate(sorted_indices):
            original_order[idx] = embeddings[i].cpu()
        
        return original_order

    def get_example_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored examples and model state."""
        stats = {
            'total_examples': sum(len(exs) for exs in self.memory.examples.values()),
            'examples_per_class': {
                label: len(exs) for label, exs in self.memory.examples.items()
            },
            'num_classes': len(self.label_to_id),
            'train_steps': self.train_steps,
            'memory_usage': {
                'prototypes': sum(p.nelement() * p.element_size() 
                                for p in self.memory.prototypes.values()),
                'examples': sum(sum(ex.embedding.nelement() * ex.embedding.element_size() 
                                  for ex in exs) 
                              for exs in self.memory.examples.values())
            }
        }
        
        if self.adaptive_head is not None:
            stats['model_params'] = sum(p.nelement() for p in 
                                      self.adaptive_head.parameters())
        
        return stats

    def predict_batch(
        self,
        texts: List[str],
        k: int = 5,
        batch_size: int = 32
    ) -> List[List[Tuple[str, float]]]:
        """Predict labels for a batch of texts with improved batching."""
        if not texts:
            raise ValueError("Empty input batch")
        
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Get embeddings for batch
            batch_embeddings = self._get_embeddings(batch_texts)
            
            # Get predictions for each embedding
            batch_predictions = []
            for embedding in batch_embeddings:
                # Get prototype predictions
                proto_preds = self.memory.get_nearest_prototypes(
                    embedding,
                    k=k
                )
                
                # Get neural predictions if available
                if self.adaptive_head is not None:
                    self.adaptive_head.eval()
                    with torch.no_grad():
                        logits = self.adaptive_head(embedding.to(self.device))
                        probs = F.softmax(logits, dim=0)
                        
                        values, indices = torch.topk(
                            probs,
                            min(k, len(self.id_to_label))
                        )
                        head_preds = [
                            (self.id_to_label[idx.item()], prob.item())
                            for prob, idx in zip(values, indices)
                        ]
                else:
                    head_preds = []
                
                # Combine predictions
                combined_scores = {}
                proto_weight = 0.7  # More weight to prototypes
                head_weight = 0.3   # Less weight to neural network
                
                for label, score in proto_preds:
                    combined_scores[label] = score * proto_weight
                    
                for label, score in head_preds:
                    combined_scores[label] = (
                        combined_scores.get(label, 0) + score * head_weight
                    )
                
                # Sort and normalize predictions
                predictions = sorted(
                    combined_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Normalize scores
                total = sum(score for _, score in predictions)
                if total > 0:
                    predictions = [(label, score/total) 
                                 for label, score in predictions]
                
                batch_predictions.append(predictions[:k])
            
            all_predictions.extend(batch_predictions)
        
        return all_predictions

    def clear_memory(self, labels: Optional[List[str]] = None):
        """Clear memory for specified labels or all if none specified."""
        if labels is None:
            self.memory.clear()
        else:
            for label in labels:
                if label in self.memory.examples:
                    del self.memory.examples[label]
                if label in self.memory.prototypes:
                    del self.memory.prototypes[label]
            self.memory._rebuild_index()

    def merge_classifiers(self, other: 'AdaptiveClassifier') -> 'AdaptiveClassifier':
        """Merge another classifier into this one."""
        # Verify compatibility
        if self.embedding_dim != other.embedding_dim:
            raise ValueError("Classifiers have different embedding dimensions")
            
        # Merge label mappings
        next_idx = max(self.id_to_label.keys()) + 1
        for label in other.label_to_id:
            if label not in self.label_to_id:
                self.label_to_id[label] = next_idx
                self.id_to_label[next_idx] = label
                next_idx += 1
        
        # Merge examples and update prototypes
        for label, examples in other.memory.examples.items():
            for example in examples:
                self.memory.add_example(example, label)
        
        # Retrain adaptive head
        if self.adaptive_head is not None:
            self._initialize_adaptive_head()
            self._train_adaptive_head()
        
        return self
    
    def _train_adaptive_head(self, epochs: int = 10):
        """Train the adaptive head with improved stability."""
        if not self.memory.examples:
            return
            
        # Prepare training data
        all_embeddings = []
        all_labels = []
        
        # Sort examples for deterministic order
        for label in sorted(self.memory.examples.keys()):
            examples = sorted(self.memory.examples[label], key=lambda x: x.text)
            for example in examples:
                all_embeddings.append(example.embedding)
                all_labels.append(self.label_to_id[label])
        
        all_embeddings = torch.stack(all_embeddings)
        all_labels = torch.tensor(all_labels)
        
        # Normalize embeddings for stable training
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        
        # Create deterministic data loader
        dataset = torch.utils.data.TensorDataset(all_embeddings, all_labels)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(32, len(all_embeddings)),  # Smaller batch size for stability
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )
        
        # Training setup
        self.adaptive_head.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(  # Switch to AdamW for better stability
            self.adaptive_head.parameters(),
            lr=0.001,  # Lower learning rate
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler for stability
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
        )
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 3  # Early stopping patience
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_embeddings, batch_labels in loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.adaptive_head(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.adaptive_head.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.debug(f"Early stopping at epoch {epoch + 1}")
                    break
        
        self.train_steps += 1
    
    def _update_adaptive_head(self):
        """Update adaptive head for new classes."""
        num_classes = len(self.label_to_id)
        
        if self.adaptive_head is None:
            self._initialize_adaptive_head()
        elif num_classes > self.adaptive_head.model[-1].out_features:
            self.adaptive_head.update_num_classes(num_classes)

    def select_representative_examples(self, examples: List[Example], k: int = 5) -> List[Example]:
        """Select k most representative examples using k-means clustering.
        
        Args:
            examples: List of examples to select from
            k: Number of examples to select
            
        Returns:
            List of selected examples
        """
        if len(examples) <= k:
            return examples
            
        # Stack embeddings
        embeddings = torch.stack([ex.embedding for ex in examples])
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Use k-means to find centroids
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        kmeans.fit(embeddings.numpy())
        
        # Find examples closest to centroids
        selected_indices = []
        centroids = torch.tensor(kmeans.cluster_centers_)
        
        for centroid in centroids:
            # Compute distances to centroid
            distances = torch.norm(embeddings - centroid, dim=1)
            # Get index of closest example
            closest_idx = torch.argmin(distances).item()
            selected_indices.append(closest_idx)
        
        return [examples[idx] for idx in selected_indices]
