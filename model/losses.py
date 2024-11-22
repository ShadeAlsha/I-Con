import torch
import torch.nn as nn
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class KernelComputations:
    """Separate class for kernel-related computations."""
    
    @staticmethod
    def compute_kernel_cross_entropy(target: torch.Tensor, 
                                   learned: torch.Tensor, 
                                   eps: float = 1e-7, 
                                   log: bool = True) -> torch.Tensor:
        """Compute cross entropy between target and learned kernel matrices."""
        if target.shape != learned.shape:
            raise ValueError(f"Shape mismatch: target {target.shape} != learned {learned.shape}")
            
        target_flat = target.flatten()
        learned_flat = learned.flatten()
        non_zero_mask = target_flat > eps

        target_filtered = target_flat[non_zero_mask]
        learned_filtered = learned_flat[non_zero_mask]
        
        if torch.isnan(learned_filtered).any():
            raise ValueError("NaN values detected in learned kernel")
            
        log_q = learned_filtered if log else torch.log(learned_filtered.clamp(min=eps))
        cross_entropy_loss = -torch.sum(target_filtered * log_q)
        
        return cross_entropy_loss / target.shape[0]

class KernelLoss:
    """Enhanced loss function factory with input validation."""
    
    @staticmethod
    def get_loss_fn(loss_type: str):
        loss_functions = {
            'kl': lambda x, y: F.kl_div(y.clamp(min=1e-10).log(), x, reduction='batchmean'),
            'ce': KernelComputations.compute_kernel_cross_entropy,
            'l2': lambda x, y: F.mse_loss(x, y),
            'tv': lambda x, y: 0.5 * torch.abs(x - y).mean(),
            'hellinger': lambda x, y, log: (torch.sqrt(x.clamp(min=1e-10)) - 
                                     torch.sqrt(y.clamp(min=1e-10))).pow(2).mean(),
            'orthogonality': lambda x, y: -(x * y).mean(),
            'jsd': lambda x, y: 0.5 * (
                F.kl_div(y.clamp(min=1e-10).log(), x, reduction='batchmean') +
                F.kl_div(x.clamp(min=1e-10).log(), y, reduction='batchmean')
            ),
            'none': lambda x, y: torch.tensor(0.0, device=x.device)
        }
        if loss_type not in loss_functions:
            raise ValueError(f"Unsupported loss_type: {loss_type}. "
                           f"Available types: {list(loss_functions.keys())}")
        return loss_functions[loss_type]
