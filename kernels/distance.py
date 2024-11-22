# kernels/distance.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional, Tuple
from .base import Kernel, KernelConfig

class DistanceKernel(Kernel):
    """Enhanced distance computation kernel with optimizations."""
    
    VALID_METRICS = {'euclidean', 'cosine', 'dot', 'manhattan', 'minkowski'}
    
    def __init__(self, metric: str = 'euclidean', config: Optional[KernelConfig] = None):
        super().__init__(config)
        if metric not in self.VALID_METRICS:
            raise ValueError(f"Metric must be one of {self.VALID_METRICS}")
        self.metric = metric
            
    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]], 
                labels: Optional[torch.Tensor] = None,
                idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute pairwise distances with optimizations."""
        # Process input
        if isinstance(features, list):
            x1, x2 = features
        else:
            x1 = x2 = features
            
        #x1 = x1.to(device=self.config.device, dtype=self.config.dtype)
        #x2 = x2.to(device=self.config.device, dtype=self.config.dtype)
            
        # Compute distances based on metric
        if self.metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.metric == 'cosine':
            return self._cosine_distance(x1, x2)
        elif self.metric == 'dot':
            return self._dot_product(x1, x2)
        elif self.metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        else:  # minkowski
            return self._minkowski_distance(x1, x2)

    def _euclidean_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Optimized Euclidean distance computation."""
        x1_norm = (x1**2).sum(1).view(-1, 1)
        x2_norm = (x2**2).sum(1).view(1, -1)
        dist = x1_norm + x2_norm - 2.0 * torch.mm(x1, x2.t())
        return torch.clamp(dist, 0.0, float('inf'))

    def _cosine_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Optimized cosine distance computation."""
        x1_normalized = F.normalize(x1, p=2, dim=1)
        x2_normalized = F.normalize(x2, p=2, dim=1)
        return 1 - torch.mm(x1_normalized, x2_normalized.t())

    @staticmethod
    def _dot_product(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute dot product distance."""
        return -torch.mm(x1, x2.t())

    def _manhattan_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute Manhattan (L1) distance."""
        return torch.cdist(x1, x2, p=1)

    def _minkowski_distance(self, x1: torch.Tensor, x2: torch.Tensor, p: float = 3) -> torch.Tensor:
        """Compute Minkowski distance."""
        return torch.cdist(x1, x2, p=p)

class GaussianKernel(Kernel):
    """Enhanced Gaussian/RBF kernel with adaptive bandwidth selection."""
    
    def __init__(
        self,
        sigma: Optional[float] = None,
        perplexity: float = 30.0,
        metric: str = 'euclidean',
        config: Optional[KernelConfig] = None
    ):
        super().__init__(config)
        self.sigma = sigma
        self.perplexity = perplexity
        self.distance_kernel = DistanceKernel(metric=metric, config=config)
        self._init_bandwidth_estimator()

    def _init_bandwidth_estimator(self):
        """Initialize the bandwidth estimator."""
        self.bandwidth_estimator = AdaptiveBandwidthEstimator(
            perplexity=self.perplexity,
            tolerance=1e-5,
            max_iter=50,
            device=self.config.device
        )

    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]], 
                labels: Optional[torch.Tensor] = None,
                idx: Optional[torch.Tensor] = None,
                return_log: bool = False) -> torch.Tensor:
        """Compute Gaussian kernel with automatic bandwidth selection."""
        distances = self.distance_kernel(features)
        
        if self.config.mask_diagonal:
            distances = fill_diagonal(distances,float('inf'))
            
        if self.sigma is None:
            sigma = self.bandwidth_estimator(distances)
        else:
            sigma = torch.full((distances.size(0),), self.sigma, device=distances.device)
        
        logits = -distances / (2 * sigma.view(-1, 1) ** 2)
        
        if return_log:
            affinities = F.log_softmax(logits, dim=1)
        else:
            affinities = F.softmax(logits, dim=1)
                
        return affinities

class AdaptiveBandwidthEstimator:
    """Improved bandwidth estimation using binary search with early stopping."""
    
    def __init__(self, perplexity: float, tolerance: float = 1e-5,
                 max_iter: int = 50, device: str = 'cuda'):
        self.perplexity = perplexity
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.device = device
        self.target_entropy = torch.log(torch.tensor(perplexity, device=device))

    @torch.no_grad()
    def __call__(self, distances: torch.Tensor) -> torch.Tensor:
        """Estimate optimal bandwidths for each point."""
        n = distances.size(0)
        sigma = torch.ones(n, device=self.device)
        
        # Initialize bounds
        lower = torch.full_like(sigma, 1e-10)
        upper = torch.full_like(sigma, 1e10)
        
        # Binary search with early stopping
        for _ in range(self.max_iter):
            # Compute probabilities and entropy
            P = self._compute_gaussian_probs(distances, sigma)
            entropy = -torch.sum(P * torch.log(P + 1e-7), dim=1)
            
            # Check convergence
            error = torch.abs(entropy - self.target_entropy)
            if torch.all(error < self.tolerance):
                break
                
            # Update bounds
            mask_low = entropy < self.target_entropy
            mask_high = entropy > self.target_entropy
            
            lower[mask_low] = sigma[mask_low]
            upper[mask_high] = sigma[mask_high]
            sigma = torch.sqrt(lower * upper)
            
        return sigma

    def _compute_gaussian_probs(self, distances: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian probabilities efficiently."""
        logits = -distances / (2 * sigma.view(-1, 1) ** 2)
        logits_max = torch.max(logits, dim=1, keepdim=True)[0]
        logits = logits - logits_max
        
        exp_logits = torch.exp(logits)
        exp_sum = torch.sum(exp_logits, dim=1, keepdim=True)
        
        return exp_logits / exp_sum

class CauchyKernel(Kernel):
    """Implements heavy-tailed Cauchy kernel, suitable for low-dimensional embeddings."""
    
    def __init__(
        self,
        gamma: float = 1.0,
        metric: str = 'euclidean',
        config: Optional[KernelConfig] = None
    ):
        super().__init__(config)
        self.gamma = gamma
        self.distance_kernel = DistanceKernel(metric=metric, config=config)
    
    def forward(
        self,
        features: Union[torch.Tensor, List[torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        return_log: bool = False
    ) -> torch.Tensor:
        distances = self.distance_kernel(features)
        
        if self.config.mask_diagonal:
            distances = fill_diagonal(distances,float('inf'))
            
        affinities = 1 / (1 + (distances / self.gamma) ** 2)
        
        if self.config.normalize:
            if return_log:
                #affinities = torch.log(affinities) - torch.logsumexp(torch.log(affinities), dim=1, keepdim=True)
                affinities = torch.log(affinities) - torch.sum(affinities, dim=1, keepdim=True)
            else:
                affinities = F.normalize(affinities, p=1, dim=1)
                
        if self.config.symmetric and not return_log:
            affinities = (affinities + affinities.t()) / 2
            
        return affinities

def fill_diagonal(tensor, value):
    result = tensor.clone()  # Clone the tensor to avoid modifying the original
    idx = torch.arange(tensor.size(0))
    result[idx, idx] = value
    return result