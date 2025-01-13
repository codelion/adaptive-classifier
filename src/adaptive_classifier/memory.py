import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import faiss
import logging
from .models import Example, ModelConfig

logger = logging.getLogger(__name__)

class PrototypeMemory:
    """Memory system that maintains prototypes for each class."""
    
    def __init__(
        self,
        embedding_dim: int,
        config: Optional[ModelConfig] = None
    ):
        """Initialize the prototype memory system.
        
        Args:
            embedding_dim: Dimension of the embeddings
            config: Optional model configuration
        """
        self.embedding_dim = embedding_dim
        self.config = config or ModelConfig()
        
        # Initialize storage
        self.examples = defaultdict(list)  # label -> List[Example]
        self.prototypes = {}  # label -> tensor
        
        # Initialize FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.label_to_index = {}  # label -> index in FAISS
        self.index_to_label = {}  # index in FAISS -> label
        
        # Statistics
        self.updates_since_rebuild = 0
    
    def add_example(self, example: Example, label: str):
        """Add a new example to memory.
        
        Args:
            example: Example to add
            label: Class label
        """
        # Check if we need to prune examples
        if (len(self.examples[label]) >= 
            self.config.max_examples_per_class):
            self._prune_examples(label)
            
        # Add new example
        self.examples[label].append(example)
        
        # Update prototype
        self._update_prototype(label)
        
        # Check if we need to rebuild index
        self.updates_since_rebuild += 1
        if (self.updates_since_rebuild >= 
            self.config.prototype_update_frequency):
            self._rebuild_index()
    
    def get_nearest_prototypes(
            self,
            query_embedding: torch.Tensor,
            k: int = 5,
            min_similarity: Optional[float] = None
        ) -> List[Tuple[str, float]]:
            """Find the nearest prototype neighbors for a query.
            
            Args:
                query_embedding: Query embedding tensor
                k: Number of neighbors to return
                min_similarity: Optional minimum similarity threshold
                
            Returns:
                List of (label, similarity) tuples
            """
            # Handle empty index case
            if self.index.ntotal == 0:
                return []
                
            # Ensure the query is in the right format and normalized
            query_np = torch.nn.functional.normalize(
                query_embedding, 
                p=2, 
                dim=0
            ).unsqueeze(0).numpy()
            
            # Search the index with valid k
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_np, k)
            
            # Convert to labels and scores
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0:  # Valid index
                    label = self.index_to_label[int(idx)]
                    # Convert distance to similarity score with sharper scaling
                    similarity = float(torch.exp(-torch.tensor(dist) / 5.0))
                    
                    # Apply minimum similarity threshold if specified
                    if min_similarity is None or similarity >= min_similarity:
                        results.append((label, similarity))
            
            # Normalize similarities using softmax if we have results
            if results:
                similarities = torch.tensor([score for _, score in results])
                normalized_similarities = torch.nn.functional.softmax(
                    similarities / 0.1,  # Sharp temperature
                    dim=0
                )
                results = [
                    (label, float(score)) 
                    for (label, _), score in zip(results, normalized_similarities)
                ]
            
            return results
    
    def _update_prototype(self, label: str):
        """Update the prototype for a given label.
        
        Args:
            label: Class label to update
        """
        examples = self.examples[label]
        if not examples:
            return
            
        # Compute mean of embeddings
        embeddings = torch.stack([ex.embedding for ex in examples])
        prototype = torch.mean(embeddings, dim=0)
        
        # Update prototype
        self.prototypes[label] = prototype
        
        # Update index if label exists
        if label in self.label_to_index:
            idx = self.label_to_index[label]
            self.index.remove_ids(torch.tensor([idx]))
            self.index.add(prototype.unsqueeze(0).numpy())
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from scratch."""
        # Clear existing index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.label_to_index.clear()
        self.index_to_label.clear()
        
        # Add all prototypes in sorted order to ensure consistent indices
        sorted_labels = sorted(self.prototypes.keys())
        for i, label in enumerate(sorted_labels):
            prototype = self.prototypes[label]
            self.index.add(prototype.unsqueeze(0).numpy())
            self.label_to_index[label] = i
            self.index_to_label[i] = label
            
        self.updates_since_rebuild = 0

    def _restore_from_save(self):
        """Restore index and mappings after loading from save."""
        # Clear existing index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.label_to_index.clear()
        self.index_to_label.clear()
        
        # Add prototypes in sorted order for consistency
        sorted_labels = sorted(self.prototypes.keys())
        for i, label in enumerate(sorted_labels):
            prototype = self.prototypes[label]
            self.index.add(prototype.unsqueeze(0).numpy())
            self.label_to_index[label] = i
            self.index_to_label[i] = label
            
        self.updates_since_rebuild = 0
    
    def _prune_examples(self, label: str):
        """Prune examples for a given label to maintain memory bounds.
        
        Args:
            label: Label to prune examples for
        """
        examples = self.examples[label]
        if not examples:
            return
            
        # Compute distances to prototype
        prototype = self.prototypes[label]
        distances = []
        
        for ex in examples:
            dist = torch.norm(ex.embedding - prototype).item()
            distances.append(dist)
            
        # Sort by distance and keep closest examples
        sorted_indices = np.argsort(distances)
        keep_indices = sorted_indices[:self.config.max_examples_per_class]
        
        # Update examples
        self.examples[label] = [examples[i] for i in keep_indices]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        return {
            'num_classes': len(self.prototypes),
            'examples_per_class': {
                label: len(examples)
                for label, examples in self.examples.items()
            },
            'total_examples': sum(
                len(examples) for examples in self.examples.values()
            ),
            'prototype_dimensions': self.embedding_dim,
            'updates_since_rebuild': self.updates_since_rebuild
        }
    
    def clear(self):
        """Clear all memory."""
        self.examples.clear()
        self.prototypes.clear()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.label_to_index.clear()
        self.index_to_label.clear()
        self.updates_since_rebuild = 0