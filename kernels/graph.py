# kernels/graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from .base import Kernel
from .distance import DistanceKernel

class KNNKernel(Kernel):
    """Computes k-nearest neighbor graph as a kernel."""
    
    def __init__(self, k: int, metric = 'euclidean', mapper=None, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance_kernel = DistanceKernel(metric=metric, mapper=mapper)
    
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None,
                idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute KNN graph from features."""
        # Compute pairwise distances
        distances = self.distance_kernel(features)
        
        # Handle self-connections
        if self.mask_diagonal:
            distances.fill_diagonal_(float('inf'))
        
        # Get KNN indices
        _, nn_idx = torch.topk(distances, k=min(self.k, distances.size(1)), 
                              dim=1, largest=False)
        
        # Create adjacency matrix
        n = distances.size(0)
        adj_matrix = torch.zeros_like(distances)
        row_idx = torch.arange(n, device=distances.device).view(-1, 1).expand(-1, self.k)
        adj_matrix[row_idx.flatten(), nn_idx.flatten()] = 1.0
        
        # Normalize if configured
        if self.normalize:
            deg = adj_matrix.sum(dim=1, keepdim=True)
            adj_matrix = adj_matrix / (deg + self.eps)
            
        return adj_matrix

class LabelKernel(Kernel):
    """Creates a kernel based on label similarity."""
    
    def __init__(self, sparse: bool = False, **kwargs):
        super().__init__( **kwargs)
        self.sparse = sparse
    
    def forward(self, features: Optional[torch.Tensor] = None,
                labels: torch.Tensor = None,
                idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute label-based similarity matrix."""
        if labels is None:
            raise ValueError("Labels must be provided for LabelKernel")
            
        n = labels.size(0)
        labels = labels.view(-1, 1)
        
        # Create similarity matrix
        if self.sparse:
            # Efficient implementation for sparse matrices
            matches = (labels == labels.t())
            indices = torch.nonzero(matches).t()
            values = torch.ones(indices.size(1), device=labels.device)
            adj_matrix = torch.sparse_coo_tensor(indices, values, (n, n))
        else:
            # Dense implementation
            adj_matrix = (labels == labels.t()).float()
        
        # Remove self-loops if configured
        if self.mask_diagonal:
            if self.sparse:
                adj_matrix = adj_matrix.to_dense()
                adj_matrix.fill_diagonal_(0)
                adj_matrix = adj_matrix.to_sparse()
            else:
                adj_matrix.fill_diagonal_(0)
        
        # Normalize if configured
        if self.normalize:
            if self.sparse:
                adj_matrix = adj_matrix.to_dense()
            row_sums = adj_matrix.sum(dim=1, keepdim=True)
            adj_matrix = adj_matrix / (row_sums + self.eps)
            if self.sparse and not self.symmetric:
                adj_matrix = adj_matrix.to_sparse()
                
        return adj_matrix

class AugmentationKernel(Kernel):
    """Kernel for defining relationships between augmented views of data."""
    
    def __init__(
        self,
        block: Optional[torch.Tensor] = None,
        block_size: Optional[int] = None,
        **kwargs):
        """Initialize augmentation kernel with block pattern."""
        super().__init__(**kwargs)
        
        if block is not None:
            self.block = block
            self.block_size = block.shape[0]
        elif block_size is not None:
            self.block = torch.ones(block_size, block_size)
            self.block_size = block_size
            if block_size < 2:
                self.mask_diagonal = False
        else:
            raise ValueError("Either block or block_size must be provided")
            
    def forward(
        self,
        features: Union[torch.Tensor, List[torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        return_log: bool = False
    ) -> torch.Tensor:
        """Compute augmentation kernel matrix."""
        if isinstance(features, list):
            batch_size = features[0].shape[0]
            device = features[0].device
        else:
            batch_size = features.shape[0]
            device = features.device
            
        kernel_matrix = torch.zeros(batch_size, batch_size, device=device)
        
        # If block is provided, create block diagonal matrix
        if self.block is not None:
            block = self.block.to(device=device)
            num_blocks = batch_size // self.block_size
            if num_blocks > 0:  # Only create if we have complete blocks
                kernel_matrix[:num_blocks*self.block_size, :num_blocks*self.block_size] = \
                    torch.block_diag(*[block] * num_blocks)
                
        # If indices provided, fill with ones based on indices
        if idx is not None:
            idx_mat = idx.unsqueeze(0)
            kernel_matrix.masked_fill_((idx_mat == idx_mat.t()), 1.0)
            
        if self.mask_diagonal:
            kernel_matrix = kernel_matrix.fill_diagonal_(0)
            
        if self.normalize:
            if return_log:
                kernel_matrix = torch.log(kernel_matrix.clamp(min=1e-8))
                kernel_matrix = kernel_matrix - torch.logsumexp(kernel_matrix, dim=1, keepdim=True)
            else:
                kernel_matrix = F.normalize(kernel_matrix, p=1, dim=1)
            
        return kernel_matrix