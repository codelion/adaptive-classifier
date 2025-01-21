from dataclasses import dataclass
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

@dataclass
class Example:
    """Represents a single training example."""
    text: str
    label: str
    embedding: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert example to dictionary for saving."""
        return {
            'text': self.text,
            'label': self.label,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Example':
        """Create example from dictionary."""
        embedding = torch.tensor(data['embedding']) if data['embedding'] is not None else None
        return cls(text=data['text'], label=data['label'], embedding=embedding)

class AdaptiveHead(nn.Module):
    """Neural network head with stable initialization and deterministic behavior."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[list] = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim]  # Simpler architecture with one hidden layer
            
        layers = []
        prev_dim = input_dim
        
        # Initialize layers with specific initialization strategy
        for dim in hidden_dims:
            linear = nn.Linear(prev_dim, dim)
            # Use Kaiming initialization with fixed seed
            torch.manual_seed(42)
            nn.init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            
            layers.extend([
                linear,
                nn.ReLU(),
                nn.Dropout(0.1)  # Replace BatchNorm with small dropout
            ])
            prev_dim = dim
            
        # Output layer with specific initialization
        output_layer = nn.Linear(prev_dim, num_classes)
        torch.manual_seed(42)
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass ensuring consistent output shape."""
        # Always preserve batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension [D] -> [1, D]
            
        output = self.model(x)  # Should be [B, C] where B is batch size, C is num_classes
        
        # Never squeeze output - always return [B, C]
        return output
    
    def update_num_classes(self, num_classes: int):
        """Update output layer with stable weight initialization."""
        current_weight = self.model[-1].weight
        current_bias = self.model[-1].bias
        
        if num_classes > current_weight.size(0):
            new_layer = nn.Linear(current_weight.size(1), num_classes)
            torch.manual_seed(42)
            nn.init.xavier_uniform_(new_layer.weight)
            nn.init.zeros_(new_layer.bias)
            
            # Copy existing weights
            with torch.no_grad():
                new_layer.weight[:current_weight.size(0)] = current_weight
                new_layer.bias[:current_weight.size(0)] = current_bias
                
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
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.warmup_steps = self.config.get('warmup_steps', 0)
        
        # Memory settings
        self.max_examples_per_class = self.config.get('max_examples_per_class', 1000)
        self.prototype_update_frequency = self.config.get('prototype_update_frequency', 100)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.6)
        
        # EWC settings
        self.ewc_lambda = self.config.get('ewc_lambda', 100.0)
        self.num_representative_examples = self.config.get('num_representative_examples', 5)
        
        # Training settings
        self.epochs = self.config.get('epochs', 10)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 3)
        self.min_examples_per_class = self.config.get('min_examples_per_class', 3)
        
        # Prediction settings
        self.prototype_weight = self.config.get('prototype_weight', 0.7)
        self.neural_weight = self.config.get('neural_weight', 0.3)
        self.min_confidence = self.config.get('min_confidence', 0.1)
        
        # Device settings
        self.device_map = self.config.get('device_map', 'auto')
        self.quantization = self.config.get('quantization', None)
        self.gradient_checkpointing = self.config.get('gradient_checkpointing', False)
        
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'max_examples_per_class': self.max_examples_per_class,
            'prototype_update_frequency': self.prototype_update_frequency,
            'similarity_threshold': self.similarity_threshold,
            'ewc_lambda': self.ewc_lambda,
            'num_representative_examples': self.num_representative_examples,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'min_examples_per_class': self.min_examples_per_class,
            'prototype_weight': self.prototype_weight,
            'neural_weight': self.neural_weight,
            'min_confidence': self.min_confidence,
            'device_map': self.device_map,
            'quantization': self.quantization,
            'gradient_checkpointing': self.gradient_checkpointing
        }