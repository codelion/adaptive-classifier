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
        self.strategic_prototypes = {}  # label -> strategic prototype tensor
        # Class spread recorded when a classifier was saved. A reloaded memory
        # keeps only a few representative examples, which understate the spread.
        self.saved_radii = {}  # label -> float
        
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
            
        Raises:
            ValueError: If example embedding dimension doesn't match memory dimension
        """
        # Validate embedding dimension
        if example.embedding is None:
            raise ValueError("Example must have an embedding")
        if example.embedding.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Example embedding dimension {example.embedding.size(-1)} "
                f"does not match memory dimension {self.embedding_dim}"
            )
            
        # Add new example
        self.examples[label].append(example)
        
        # Check if we need to prune examples after adding
        if len(self.examples[label]) > self.config.max_examples_per_class:
            self._prune_examples(label)
        
        # Update prototype
        self._update_prototype(label)

        # Only increment if we haven't just rebuilt
        if not getattr(self, 'just_rebuilt', False):
            self.updates_since_rebuild += 1
            # print(f"updates_since_rebuild: {self.updates_since_rebuild}")
        
        # If we've hit the frequency, rebuild
        if self.updates_since_rebuild >= self.config.prototype_update_frequency:
            # print("Index rebuild")
            self._rebuild_index()
            self.just_rebuilt = True
        else:
            self.just_rebuilt = False
            
        # print(f"updates_since_rebuild: {self.updates_since_rebuild}")
    
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
            # Ensure index is up to date
            if self.updates_since_rebuild >= self.config.prototype_update_frequency:
                self._rebuild_index()
                
            # Handle empty index case
            if self.index.ntotal == 0:
                return []
                
            # Ensure the query is in the right format
            query_np = query_embedding.unsqueeze(0).numpy()
                
            # Search the index with valid k
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_np, k)
            
            temperature = getattr(self.config, 'prototype_temperature', None)
            valid = [(int(i), float(d)) for i, d in zip(indices[0], distances[0]) if i >= 0]
            if not valid:
                return []

            if temperature:
                # Softmax over negative squared distance: the nearest class gets
                # most of the mass when it is clearly nearer than the others.
                scores = torch.softmax(
                    torch.tensor([-d / temperature for _, d in valid]), dim=0
                )
            else:
                # Legacy scoring (before 0.3.0): exp(-distance), then softmax.
                scores = torch.softmax(
                    torch.tensor([float(np.exp(-d)) for _, d in valid]), dim=0
                )

            results = [
                (self.index_to_label[i], float(score))
                for (i, _), score in zip(valid, scores)
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
            
        # Explicitly reset the counter
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
        """Prune examples for a given label to maintain memory bounds."""
        examples = self.examples[label]
        if not examples:
            return
            
        # Compute distances to mean embedding (more stable than current prototype)
        embeddings = torch.stack([ex.embedding for ex in examples])
        mean_embedding = torch.mean(embeddings, dim=0)
        
        distances = []
        for ex in examples:
            dist = torch.norm(ex.embedding - mean_embedding).item()
            distances.append(dist)
            
        # Sort by distance and keep closest examples
        sorted_indices = np.argsort(distances)
        keep_indices = sorted_indices[:self.config.max_examples_per_class]
        
        # Update examples - ensure we don't exceed max size
        self.examples[label] = [examples[i] for i in keep_indices]
        assert len(self.examples[label]) <= self.config.max_examples_per_class
    
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
    
    def remove_class(self, label: str):
        """Drop a class and everything stored for it."""
        self.examples.pop(label, None)
        self.prototypes.pop(label, None)
        self.strategic_prototypes.pop(label, None)
        self.saved_radii.pop(label, None)
        self._rebuild_index()

    def remove_examples(self, label: str, texts) -> int:
        """Remove examples of `label` whose text is in `texts`.

        The prototype is recomputed from what remains; a class left with no
        examples is removed. Returns the number of examples removed.
        """
        wanted = set(texts)
        before = self.examples.get(label, [])
        kept = [ex for ex in before if ex.text not in wanted]
        removed = len(before) - len(kept)
        if removed == 0:
            return 0
        if not kept:
            self.remove_class(label)
            return removed
        self.examples[label] = kept
        self.prototypes[label] = torch.mean(torch.stack([ex.embedding for ex in kept]), dim=0)
        self.strategic_prototypes.pop(label, None)
        # The saved spread described the old membership.
        self.saved_radii.pop(label, None)
        self._rebuild_index()
        return removed

    def class_radius(self, label: str) -> float:
        """Distance from the prototype to the farthest known example."""
        radius = self.saved_radii.get(label, 0.0)
        examples = self.examples.get(label)
        if examples and label in self.prototypes:
            embeddings = torch.stack([ex.embedding for ex in examples])
            distances = torch.norm(embeddings - self.prototypes[label], dim=1)
            radius = max(radius, float(distances.max()))
        return radius

    def ood_score(self, query_embedding: torch.Tensor):
        """Return (score, label): the smallest distance-to-radius ratio over classes.

        1.0 means the query is exactly as far from the prototype as the most
        remote training example. Returns (inf, None) when the memory is empty.
        """
        best_score, best_label = float('inf'), None
        for label, prototype in self.prototypes.items():
            radius = max(self.class_radius(label), self.config.ood_min_radius)
            score = float(torch.norm(query_embedding - prototype)) / radius
            if score < best_score:
                best_score, best_label = score, label
        return best_score, best_label

    def clear(self):
        """Clear all memory."""
        self.examples.clear()
        self.prototypes.clear()
        self.saved_radii.clear()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.label_to_index.clear()
        self.index_to_label.clear()
        self.updates_since_rebuild = 0
    
    def compute_strategic_prototypes(self, cost_function, classifier_func):
        """Compute strategic prototypes for all classes.
        
        Args:
            cost_function: Strategic cost function
            classifier_func: Current classifier function
        """
        for label, examples in self.examples.items():
            if examples:
                strategic_embeddings = []
                
                for example in examples:
                    # Compute where this example would strategically move
                    strategic_embedding = cost_function.compute_best_response(
                        example.embedding, classifier_func
                    )
                    strategic_embeddings.append(strategic_embedding)
                
                # Compute mean of strategic embeddings
                if strategic_embeddings:
                    strategic_prototype = torch.stack(strategic_embeddings).mean(dim=0)
                    self.strategic_prototypes[label] = strategic_prototype
    
    def get_strategic_prototypes(self, query_embedding: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """Get nearest strategic prototypes.
        
        Args:
            query_embedding: Query embedding tensor
            k: Number of neighbors to return
            
        Returns:
            List of (label, similarity) tuples
        """
        if not self.strategic_prototypes:
            return self.get_nearest_prototypes(query_embedding, k)
        
        # Compute similarities to strategic prototypes
        similarities = []
        for label, prototype in self.strategic_prototypes.items():
            # Compute cosine similarity
            sim = F.cosine_similarity(
                query_embedding.unsqueeze(0), 
                prototype.unsqueeze(0)
            ).item()
            similarities.append((label, sim))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]