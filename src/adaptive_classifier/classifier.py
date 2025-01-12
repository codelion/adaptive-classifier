import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
from safetensors.torch import save_file, load_file
import json
import time

from .models import Example, AdaptiveHead, ModelConfig
from .memory import PrototypeMemory

logger = logging.getLogger(__name__)

class AdaptiveClassifier:
    """A flexible classifier that can adapt to new classes and examples."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the adaptive classifier.
        
        Args:
            model_name: Name of the HuggingFace transformer model
            device: Device to run the model on (default: auto-detect)
            config: Optional configuration dictionary
        """
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
        """Add new examples to the classifier.
        
        Args:
            texts: List of text examples
            labels: List of corresponding labels
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not texts or not labels:
            raise ValueError("Empty input lists")
        if len(texts) != len(labels):
            raise ValueError("Mismatched text and label lists")
        
        # Update label mappings
        unique_labels = set(labels)
        for label in unique_labels:
            if label not in self.label_to_id:
                idx = len(self.label_to_id)
                self.label_to_id[label] = idx
                self.id_to_label[idx] = label
        
        # Get embeddings for all texts
        embeddings = self._get_embeddings(texts)
        
        # Add examples to memory
        for text, embedding, label in zip(texts, embeddings, labels):
            example = Example(text, label, embedding)
            self.memory.add_example(example, label)
        
        # Initialize or update adaptive head
        self._update_adaptive_head()
        
        # Train adaptive head
        self._train_adaptive_head()
    
    def predict(
        self,
        text: str,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Predict top-k labels for input text.
        
        Args:
            text: Input text
            k: Number of predictions to return
            
        Returns:
            List of (label, score) tuples
        """
        if not text:
            raise ValueError("Empty input text")
        
        # Get embedding
        embedding = self._get_embeddings([text])[0]
        
        # Get prototype predictions
        proto_preds = self.memory.get_nearest_prototypes(embedding, k=k)
        
        # Get neural predictions if available
        if self.adaptive_head is not None:
            self.adaptive_head.eval()
            with torch.no_grad():
                logits = self.adaptive_head(embedding.to(self.device))
                probs = F.softmax(logits, dim=0)
                
                # Convert to predictions
                values, indices = torch.topk(probs, min(k, len(self.id_to_label)))
                head_preds = [
                    (self.id_to_label[idx.item()], prob.item())
                    for prob, idx in zip(values, indices)
                ]
        else:
            head_preds = []
        
        # Combine predictions
        combined_scores = {}
        
        # Weight for combining predictions
        proto_weight = 0.7  # Adjust as needed
        head_weight = 0.3   # Adjust as needed
        
        for label, score in proto_preds:
            combined_scores[label] = score * proto_weight
            
        for label, score in head_preds:
            combined_scores[label] = combined_scores.get(label, 0) + score * head_weight
        
        # Sort and return top-k
        predictions = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return predictions[:k]
    
    def predict_batch(
        self,
        texts: List[str],
        k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """Predict labels for a batch of texts.
        
        Args:
            texts: List of input texts
            k: Number of predictions per text
            
        Returns:
            List of prediction lists
        """
        if not texts:
            raise ValueError("Empty input batch")
        
        # Get embeddings for batch
        embeddings = self._get_embeddings(texts)
        
        # Get predictions for each embedding
        predictions = []
        for embedding in embeddings:
            # Get prototype predictions
            proto_preds = self.memory.get_nearest_prototypes(embedding, k=k)
            
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
            proto_weight = 0.7
            head_weight = 0.3
            
            for label, score in proto_preds:
                combined_scores[label] = score * proto_weight
                
            for label, score in head_preds:
                combined_scores[label] = (
                    combined_scores.get(label, 0) + score * head_weight
                )
            
            # Sort and add to results
            batch_preds = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:k]
            
            predictions.append(batch_preds)
        
        return predictions
    
    def save(self, save_dir: str):
        """Save the classifier state.
        
        Args:
            save_dir: Directory to save the state
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration and metadata
        config = {
            'model_name': self.model.config._name_or_path,
            'embedding_dim': self.embedding_dim,
            'label_to_id': self.label_to_id,
            'id_to_label': {str(k): v for k, v in self.id_to_label.items()},
            'train_steps': self.train_steps,
            'config': self.config.to_dict()
        }
        
        # Add example metadata
        examples_metadata = {}
        for label, examples in self.memory.examples.items():
            examples_metadata[label] = [
                {
                    'text': ex.text,
                    'label': ex.label
                }
                for ex in examples
            ]
        config['examples_metadata'] = examples_metadata
        
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f)
        
        # Save all tensors
        tensor_dict = {}
        
        # Save example embeddings
        for label, examples in self.memory.examples.items():
            for idx, ex in enumerate(examples):
                tensor_dict[f'example_{label}_{idx}'] = ex.embedding
        
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
    def load(
        cls,
        save_dir: str,
        device: Optional[str] = None
    ) -> 'AdaptiveClassifier':
        """Load a classifier from saved state.
        
        Args:
            save_dir: Directory containing the saved state
            device: Device to load the model on
            
        Returns:
            Loaded classifier instance
        """
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
        
        # Rebuild examples from metadata
        examples_metadata = config['examples_metadata']
        for label, examples_meta in examples_metadata.items():
            for idx, ex_meta in enumerate(examples_meta):
                embedding = tensors[f'example_{label}_{idx}']
                example = Example(
                    text=ex_meta['text'],
                    label=ex_meta['label'],
                    embedding=embedding
                )
                classifier.memory.examples[label].append(example)
        
        # Rebuild prototypes
        for label in classifier.label_to_id.keys():
            prototype_key = f'prototype_{label}'
            if prototype_key in tensors:
                prototype = tensors[prototype_key]
                classifier.memory.prototypes[label] = prototype
                classifier.memory.index.add(prototype.unsqueeze(0).numpy())
        
        # Rebuild adaptive head if it exists
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
    
    def _get_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Get embeddings for input texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding tensors
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return [emb.cpu() for emb in embeddings]
    
    def _initialize_adaptive_head(self):
        """Initialize or reinitialize the adaptive head."""
        num_classes = len(self.label_to_id)
        self.adaptive_head = AdaptiveHead(
            self.embedding_dim,
            num_classes
        ).to(self.device)
    
    def _train_adaptive_head(self, epochs: int = 5):
        """Train the adaptive head on current examples.
        
        Args:
            epochs: Number of training epochs
        """
        if not self.memory.examples:
            return
            
        # Prepare training data
        all_embeddings = []
        all_labels = []
        
        for label, examples in self.memory.examples.items():
            label_id = self.label_to_id[label]
            for example in examples:
                all_embeddings.append(example.embedding)
                all_labels.append(label_id)
        
        all_embeddings = torch.stack(all_embeddings)
        all_labels = torch.tensor(all_labels)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(all_embeddings, all_labels)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Training loop
        optimizer = torch.optim.Adam(
            self.adaptive_head.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        self.adaptive_head.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_embeddings, batch_labels in loader:
                optimizer.zero_grad()
                
                batch_embeddings = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.adaptive_head(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.train_steps += 1
    
    def _update_adaptive_head(self):
        """Update adaptive head for new classes."""
        num_classes = len(self.label_to_id)
        
        if self.adaptive_head is None:
            self._initialize_adaptive_head()
        elif num_classes > self.adaptive_head.model[-1].out_features:
            self.adaptive_head.update_num_classes(num_classes)
