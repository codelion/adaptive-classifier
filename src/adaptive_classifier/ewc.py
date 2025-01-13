import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import copy

class EWC:
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""
    
    def __init__(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        device: str = 'cpu',
        ewc_lambda: float = 100.0
    ):
        """Initialize EWC.
        
        Args:
            model: Neural network model
            dataset: Dataset to compute Fisher information
            device: Device to use
            ewc_lambda: Importance of old tasks
        """
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        
        # Store old parameters
        self.old_params = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        
        # Compute Fisher information
        self.fisher_info = self._compute_fisher(dataset)
    
    def _compute_fisher(
        self,
        dataset: torch.utils.data.Dataset
    ) -> Dict[str, torch.Tensor]:
        """Compute Fisher information matrix.
        
        Args:
            dataset: Dataset to compute Fisher information
            
        Returns:
            Dictionary of parameter names to Fisher information
        """
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
        self.model.eval()
        
        # Create dataloader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True
        )
        
        # Compute Fisher information
        for batch_embeddings, batch_labels in loader:
            self.model.zero_grad()
            
            batch_embeddings = batch_embeddings.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Get model outputs
            outputs = self.model(batch_embeddings)
            
            # Compute loss
            probs = F.softmax(outputs, dim=1)
            log_probs = F.log_softmax(outputs, dim=1)
            
            # Sample from output distribution
            sampled_labels = torch.multinomial(probs, 1).squeeze()
            
            # Compute loss with sampled labels
            loss = F.nll_loss(log_probs, sampled_labels)
            
            # Compute gradients
            loss.backward()
            
            # Accumulate Fisher information
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(loader)
        
        return fisher
    
    def ewc_loss(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Compute EWC loss.
        
        Args:
            batch_size: Batch size for normalization
            
        Returns:
            EWC loss tensor
        """
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                # Compute squared distance
                _loss = (self.fisher_info[n] * (p - self.old_params[n]) ** 2).sum()
                loss += _loss
        
        # Normalize by batch size if provided
        if batch_size is not None:
            loss = loss / batch_size
            
        return self.ewc_lambda * loss