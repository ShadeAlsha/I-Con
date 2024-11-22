# kernels/clustering.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional, Tuple
from .base import Kernel, KernelConfig

class ClusteringKernel(Kernel):
    """Creates a kernel based on cluster assignment probabilities."""
    
    def __init__(self, config: Optional[KernelConfig] = None):
        super().__init__(config)
    
    def forward(self, cluster_probs: Union[torch.Tensor, List[torch.Tensor]],
                labels: Optional[torch.Tensor] = None,
                idx: Optional[torch.Tensor] = None,
                return_log: bool = False) -> torch.Tensor:
        """Compute clustering-based similarity matrix.
        
        Args:
            cluster_probs: Cluster assignment probabilities or list of [probs1, probs2]
            labels: Optional labels (not used)
            idx: Optional indices (not used)
        
        Returns:
            Kernel matrix based on cluster assignments
        """
        # Handle input format
        if isinstance(cluster_probs, list) and len(cluster_probs) == 2:
            probs1, probs2 = cluster_probs
        else:
            probs1 = probs2 = cluster_probs
            
        probs1 = probs1.to(device=self.config.device)
        probs2 = probs2.to(device=self.config.device)
        
        # Compute cluster sizes
        cluster_sizes = probs1.sum(dim=0)
        
        # Compute normalized kernel
        kernel = (probs1 / (cluster_sizes + self.config.eps)) @ probs2.t()
        
        # Apply masking if configured
        if self.config.mask_diagonal:
            kernel.fill_diagonal_(0)
        
        # Already normalized, so no further normalization needed
        
        
        # Symmetrize if configured
        if self.config.symmetric:
            kernel = (kernel + kernel.t()) / 2
            
        
        if return_log:
            return kernel.log()
        return kernel
