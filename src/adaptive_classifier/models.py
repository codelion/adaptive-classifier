import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class Example:
    """Represents a single training example."""
    text: str
    label: str
    embedding: Optional[torch.Tensor] = None

class AdaptiveHead(nn.Module):
    """Neural network head that adapts to new classes."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[list] = None
    ):
        """Initialize the adaptive head.
        
        Args:
            input_dim: Dimension of input embeddings
            num_classes: Number of classes to predict
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
            
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def update_num_classes(self, num_classes: int):
        """Update the output layer for new number of classes.
        
        Args:
            num_classes: New number of classes
        """
        current_weight = self.model[-1].weight
        current_bias = self.model[-1].bias
        
        if num_classes > current_weight.size(0):
            # Create new layer with more classes
            new_layer = nn.Linear(
                current_weight.size(1),
                num_classes
            )
            
            # Copy existing weights
            with torch.no_grad():
                new_layer.weight[:current_weight.size(0)] = current_weight
                new_layer.bias[:current_weight.size(0)] = current_bias
                
            # Replace last layer
            self.model[-1] = new_layer

class ModelConfig:
    """Configuration for the adaptive classifier."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Model settings
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 2e-5)
        self.warmup_steps = self.config.get('warmup_steps', 0)
        
        # Memory settings
        self.max_examples_per_class = self.config.get('max_examples_per_class', 1000)
        self.prototype_update_frequency = self.config.get('prototype_update_frequency', 100)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.95)
        
        # Optimization settings
        self.device_map = self.config.get('device_map', 'auto')
        self.quantization = self.config.get('quantization', None)
        self.gradient_checkpointing = self.config.get('gradient_checkpointing', False)
        
    def update(self, **kwargs):
        """Update configuration parameters.
        
        Args:
            **kwargs: Key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary of configuration parameters
        """
        return {
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'max_examples_per_class': self.max_examples_per_class,
            'prototype_update_frequency': self.prototype_update_frequency,
            'similarity_threshold': self.similarity_threshold,
            'device_map': self.device_map,
            'quantization': self.quantization,
            'gradient_checkpointing': self.gradient_checkpointing
        }