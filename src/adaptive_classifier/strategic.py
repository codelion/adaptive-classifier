import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class StrategicCostFunction(ABC):
    """Abstract base class for strategic cost functions."""
    
    @abstractmethod
    def compute_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the cost of moving from x to y.
        
        Args:
            x: Original input tensor
            y: Modified input tensor
            
        Returns:
            Cost tensor
        """
        pass
    
    @abstractmethod
    def compute_best_response(self, x: torch.Tensor, f: callable) -> torch.Tensor:
        """Compute the best response for input x given classifier f.
        
        Args:
            x: Original input tensor
            f: Classifier function
            
        Returns:
            Best response tensor
        """
        pass


class SeparableCostFunction(StrategicCostFunction):
    """Separable cost function of the form c(x,y) = max{0, c2(y) - c1(x)}."""
    
    def __init__(
        self,
        c1_coefficients: Union[Dict[str, float], torch.Tensor],
        c2_coefficients: Union[Dict[str, float], torch.Tensor],
        feature_names: Optional[List[str]] = None
    ):
        """Initialize separable cost function.
        
        Args:
            c1_coefficients: Coefficients for c1 function (original state value)
            c2_coefficients: Coefficients for c2 function (target state value)
            feature_names: Optional list of feature names for dict-based coefficients
        """
        if isinstance(c1_coefficients, dict) and isinstance(c2_coefficients, dict):
            if feature_names is None:
                raise ValueError("feature_names required when using dict coefficients")
            self.c1 = torch.tensor([c1_coefficients.get(name, 0.0) for name in feature_names])
            self.c2 = torch.tensor([c2_coefficients.get(name, 0.0) for name in feature_names])
            self.feature_names = feature_names
        else:
            self.c1 = c1_coefficients if isinstance(c1_coefficients, torch.Tensor) else torch.tensor(c1_coefficients)
            self.c2 = c2_coefficients if isinstance(c2_coefficients, torch.Tensor) else torch.tensor(c2_coefficients)
            self.feature_names = feature_names
    
    def compute_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute separable cost c(x,y) = max{0, c2(y) - c1(x)}."""
        c1_x = torch.dot(self.c1, x)
        c2_y = torch.dot(self.c2, y)
        return torch.relu(c2_y - c1_x)
    
    def compute_best_response(self, x: torch.Tensor, f: callable) -> torch.Tensor:
        """Compute best response for separable cost function.
        
        This implements Algorithm 1 from the strategic classification paper.
        """
        # For separable costs, the best response has a threshold structure
        # The agent will move to the cheapest point that gets accepted
        
        # Generate candidate points based on thresholds
        candidates = self._generate_candidates(x)
        
        best_utility = float('-inf')
        best_response = x.clone()
        
        for candidate in candidates:
            # Compute utility: classifier output - cost
            with torch.no_grad():
                f_candidate = f(candidate.unsqueeze(0)).squeeze()
                if len(f_candidate.shape) > 0:  # Multi-class case
                    f_candidate = torch.max(f_candidate)  # Use max probability as utility
                
            cost = self.compute_cost(x, candidate)
            utility = f_candidate - cost
            
            if utility > best_utility:
                best_utility = utility
                best_response = candidate
        
        return best_response
    
    def _generate_candidates(self, x: torch.Tensor, num_candidates: int = 50) -> List[torch.Tensor]:
        """Generate candidate points for optimization."""
        candidates = [x]  # Always include the original point
        
        # Generate candidates by varying each feature
        for i in range(len(x)):
            for delta in torch.linspace(-2.0, 2.0, 10):  # Reasonable range for feature changes
                if delta == 0:
                    continue
                candidate = x.clone()
                candidate[i] += delta
                candidates.append(candidate)
        
        # Generate some random candidates
        for _ in range(num_candidates - len(candidates)):
            noise = torch.randn_like(x) * 0.5  # Small random perturbations
            candidate = x + noise
            candidates.append(candidate)
        
        return candidates[:num_candidates]


class LinearCostFunction(SeparableCostFunction):
    """Linear cost function c(x,y) = <alpha, y-x>_+."""
    
    def __init__(
        self,
        alpha: Union[Dict[str, float], torch.Tensor],
        feature_names: Optional[List[str]] = None
    ):
        """Initialize linear cost function.
        
        Args:
            alpha: Cost coefficients for each feature
            feature_names: Optional list of feature names for dict-based coefficients
        """
        if isinstance(alpha, dict):
            if feature_names is None:
                raise ValueError("feature_names required when using dict coefficients")
            alpha_tensor = torch.tensor([alpha.get(name, 0.0) for name in feature_names])
        else:
            alpha_tensor = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)
        
        # For linear costs: c(x,y) = <alpha, y-x>_+ = max{0, <alpha,y> - <alpha,x>}
        super().__init__(alpha_tensor, alpha_tensor, feature_names)
        self.alpha = alpha_tensor
    
    def compute_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute linear cost c(x,y) = <alpha, y-x>_+."""
        diff = y - x
        cost = torch.dot(self.alpha, diff)
        return torch.relu(cost)


class CostFunctionFactory:
    """Factory for creating cost functions from configuration."""
    
    @staticmethod
    def create_cost_function(
        cost_type: str,
        cost_coefficients: Dict[str, float],
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> StrategicCostFunction:
        """Create a cost function from configuration.
        
        Args:
            cost_type: Type of cost function ('linear', 'separable')
            cost_coefficients: Dictionary of feature costs
            feature_names: List of feature names
            **kwargs: Additional arguments for specific cost functions
            
        Returns:
            Configured cost function
        """
        if cost_type == "linear":
            return LinearCostFunction(cost_coefficients, feature_names)
        elif cost_type == "separable":
            # For separable, use same coefficients for both c1 and c2 by default
            c2_coefficients = kwargs.get('c2_coefficients', cost_coefficients)
            return SeparableCostFunction(cost_coefficients, c2_coefficients, feature_names)
        else:
            raise ValueError(f"Unknown cost function type: {cost_type}")


class StrategicOptimizer:
    """Optimizer for strategic training using the paper's algorithms."""
    
    def __init__(self, cost_function: StrategicCostFunction):
        """Initialize strategic optimizer.
        
        Args:
            cost_function: Cost function to use for strategic optimization
        """
        self.cost_function = cost_function
    
    def strategic_loss(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        strategic_lambda: float = 0.1
    ) -> torch.Tensor:
        """Compute strategic loss for training.
        
        Args:
            model: Neural network model
            embeddings: Input embeddings
            labels: True labels
            strategic_lambda: Weight for strategic loss component
            
        Returns:
            Combined loss tensor
        """
        # Regular classification loss
        outputs = model(embeddings)
        regular_loss = F.cross_entropy(outputs, labels)
        
        # Strategic loss: optimize for worst-case strategic responses
        strategic_loss = 0.0
        
        for i, (embedding, label) in enumerate(zip(embeddings, labels)):
            # Compute best response for this embedding
            def classifier_func(x):
                return torch.softmax(model(x), dim=-1)
            
            best_response = self.cost_function.compute_best_response(embedding, classifier_func)
            
            # Loss should be high if the best response gets wrong classification
            strategic_output = model(best_response.unsqueeze(0))
            strategic_prediction = torch.argmax(strategic_output, dim=-1)
            
            # Add penalty if strategic response changes the prediction
            if strategic_prediction != label:
                strategic_loss += F.cross_entropy(strategic_output, label.unsqueeze(0))
        
        strategic_loss = strategic_loss / len(embeddings) if len(embeddings) > 0 else 0.0
        
        return regular_loss + strategic_lambda * strategic_loss
    
    def compute_strategic_prototypes(
        self,
        examples: List,
        classifier_func: callable
    ) -> torch.Tensor:
        """Compute strategic prototypes - where agents would move to.
        
        Args:
            examples: List of examples for a class
            classifier_func: Current classifier function
            
        Returns:
            Strategic prototype tensor
        """
        strategic_embeddings = []
        
        for example in examples:
            # Compute where this example would strategically move
            strategic_embedding = self.cost_function.compute_best_response(
                example.embedding, classifier_func
            )
            strategic_embeddings.append(strategic_embedding)
        
        if strategic_embeddings:
            return torch.stack(strategic_embeddings).mean(dim=0)
        else:
            return torch.zeros_like(examples[0].embedding)


class StrategicEvaluator:
    """Evaluator for strategic robustness."""
    
    def __init__(self, cost_function: StrategicCostFunction):
        """Initialize strategic evaluator.
        
        Args:
            cost_function: Cost function for strategic behavior
        """
        self.cost_function = cost_function
    
    def evaluate_robustness(
        self,
        classifier,
        test_embeddings: torch.Tensor,
        test_labels: torch.Tensor,
        gaming_levels: List[float] = [0.0, 0.5, 1.0]
    ) -> Dict[str, float]:
        """Evaluate classifier robustness under strategic behavior.
        
        Args:
            classifier: Trained classifier
            test_embeddings: Test embeddings
            test_labels: Test labels
            gaming_levels: List of gaming intensity levels
            
        Returns:
            Dictionary of robustness metrics
        """
        results = {}
        
        for level in gaming_levels:
            # Simulate strategic behavior at this level
            strategic_embeddings = self._simulate_strategic_behavior(
                test_embeddings, classifier, level
            )
            
            # Evaluate accuracy on strategic inputs
            with torch.no_grad():
                outputs = classifier(strategic_embeddings)
                predictions = torch.argmax(outputs, dim=-1)
                accuracy = (predictions == test_labels).float().mean().item()
            
            results[f'accuracy_gaming_{level}'] = accuracy
        
        # Compute robustness metrics
        results['robustness_score'] = results['accuracy_gaming_0.0'] - results['accuracy_gaming_1.0']
        results['relative_robustness'] = results['accuracy_gaming_1.0'] / results['accuracy_gaming_0.0']
        
        return results
    
    def _simulate_strategic_behavior(
        self,
        embeddings: torch.Tensor,
        classifier,
        gaming_level: float
    ) -> torch.Tensor:
        """Simulate strategic behavior at given gaming level.
        
        Args:
            embeddings: Original embeddings
            classifier: Classifier to game against
            gaming_level: Intensity of gaming (0.0 = no gaming, 1.0 = full gaming)
            
        Returns:
            Modified embeddings after strategic behavior
        """
        strategic_embeddings = []
        
        def classifier_func(x):
            with torch.no_grad():
                return torch.softmax(classifier(x), dim=-1)
        
        for embedding in embeddings:
            if torch.rand(1).item() < gaming_level:
                # Apply strategic behavior
                strategic_embedding = self.cost_function.compute_best_response(
                    embedding, classifier_func
                )
            else:
                # No strategic behavior
                strategic_embedding = embedding
            
            strategic_embeddings.append(strategic_embedding)
        
        return torch.stack(strategic_embeddings)
