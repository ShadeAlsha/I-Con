# kernels/base.py
import torch
import torch.nn as nn
from typing import Union, List, Optional, Dict, Tuple
from dataclasses import dataclass

@dataclass
class KernelConfig:
    """Configuration for kernel computation."""
    metric: str = 'euclidean'
    mask_diagonal: bool = True
    symmetric: bool = False
    normalize: bool = True
    device: str = 'cpu'
    eps: float = 0.0
    dtype: torch.dtype = torch.float32

class Kernel(nn.Module):
    """Base class for all kernel implementations."""
    
    def __init__(self, config: Optional[KernelConfig] = None):
        super().__init__()
        self.config = config or KernelConfig()
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
        
    def __add__(self, other: Union['Kernel', float, int]) -> 'Kernel':
        if isinstance(other, Kernel):
            return CompositeKernel([self, other], operation='add')
        elif isinstance(other, (int, float)):
            return ConstantAddedKernel(self, other)
        raise ValueError(f"Unsupported addition with type {type(other)}")
    
    def __radd__(self, other: Union['Kernel', float]) -> 'Kernel':
        return self.__add__(other)
    
    def __sub__(self, other: Union['Kernel', float, int]) -> 'Kernel':
        if isinstance(other, Kernel):
            return CompositeKernel([self, other], operation='sub')
        elif isinstance(other, (int, float)):
            return ConstantAddedKernel(self, -other)
        raise ValueError(f"Unsupported subtraction with type {type(other)}")
    
    def __mul__(self, scalar: float) -> 'Kernel':
        return ScaledKernel(self, scalar)
    
    def __rmul__(self, scalar: float) -> 'Kernel':
        return self.__mul__(scalar)
    
    def normalize(self) -> 'Kernel':
        return NormalizedKernel(self)
    
    def binarize(self) -> 'Kernel':
        return BinarizedKernel(self)
    
    def leak(self, alpha: float) -> 'Kernel':
        return LeakKernel(self, alpha)
    
    def low_rank(self, q: int = 10) -> 'Kernel':
        return LowRankKernel(self, q)
    
    def spectral(self, q: int = 10) -> 'Kernel':
        return SpectralKernel(self, q)
    
class CompositeKernel(Kernel):
    """Combines multiple kernels with various operations."""
    
    VALID_OPERATIONS = {'add', 'sub', 'max', 'compose', 'mul'}
    
    def __init__(self, kernels: List[Kernel], operation: str = 'add', 
                 config: Optional[KernelConfig] = None):
        super().__init__(config)
        if operation not in self.VALID_OPERATIONS:
            raise ValueError(f"Operation must be one of {self.VALID_OPERATIONS}")
        self.kernels = nn.ModuleList(kernels)
        self.operation = operation
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.operation == 'add':
            return sum(kernel(*args, **kwargs) for kernel in self.kernels)
        
        elif self.operation == 'sub':
            result = self.kernels[0](*args, **kwargs)
            for kernel in self.kernels[1:]:
                result = result - kernel(*args, **kwargs)
            return result
        
        elif self.operation == 'max':
            kernel_outputs = [kernel(*args, **kwargs) for kernel in self.kernels]
            return torch.max(torch.stack(kernel_outputs), dim=0)[0]
        
        elif self.operation == 'compose':
            result = self.kernels[0](*args, **kwargs)
            for kernel in self.kernels[1:]:
                result = kernel(result)
            return result
            
        elif self.operation == 'mul':
            result = self.kernels[0](*args, **kwargs)
            for kernel in self.kernels[1:]:
                result = result * kernel(*args, **kwargs)
            return result

class ScaledKernel(Kernel):
    """Scales a kernel by a constant factor."""
    
    def __init__(self, kernel: Kernel, scalar: float, config: Optional[KernelConfig] = None):
        super().__init__(config)
        self.kernel = kernel
        self.scalar = scalar
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.scalar * self.kernel(*args, **kwargs)

class ConstantAddedKernel(Kernel):
    """Adds a constant to a kernel."""
    
    def __init__(self, kernel: Kernel, constant: float, config: Optional[KernelConfig] = None):
        super().__init__(config)
        self.kernel = kernel
        self.constant = constant
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.kernel(*args, **kwargs) + self.constant

class NormalizedKernel(Kernel):
    """Row-normalizes a kernel matrix."""
    
    def __init__(self, kernel: Kernel, config: Optional[KernelConfig] = None):
        super().__init__(config)
        self.kernel = kernel
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        kernel_output = self.kernel(*args, **kwargs)
        if isinstance(kernel_output, torch.Tensor):
            row_sums = kernel_output.sum(dim=1, keepdim=True)
            return kernel_output / (row_sums + self.config.eps)
        else:
            raise ValueError("Kernel output must be a tensor for normalization")
        

class BinarizedKernel(Kernel):
    """Binarizes a kernel matrix."""
    
    def __init__(self, kernel: Kernel, threshold: float = 0.0,
                 config: Optional[KernelConfig] = None):
        super().__init__(config)
        self.kernel = kernel
        self.threshold = threshold
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        kernel_output = self.kernel(*args, **kwargs)
        binary_output = (kernel_output > self.threshold).float()
        
        if self.config.normalize:
            row_sums = binary_output.sum(dim=1, keepdim=True)
            binary_output = binary_output / (row_sums + self.config.eps)
            
        return binary_output

class LeakKernel(Kernel):
    """Adds a leakage term to the kernel matrix for regularization.
    
    The leakage term adds a uniform component to the kernel, which can help prevent
    over-focusing on strong connections and maintain some level of global connectivity.
    
    Args:
        kernel: Base kernel to add leakage to
        alpha: Leakage coefficient between 0 and 1
            - alpha=0: Original kernel (no leakage)
            - alpha=1: Uniform kernel (complete leakage)
    """
    
    def __init__(self, kernel: Kernel, alpha: float, config: Optional[KernelConfig] = None):
        super().__init__(config)
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.kernel = kernel
        self.alpha = alpha
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Compute base kernel
        kernel_output = self.kernel(*args, **kwargs)
        
        # Create uniform component
        n = kernel_output.size(1)
        uniform_kernel = torch.ones_like(kernel_output) / n
        
        # Combine with leakage
        leaked_kernel = (1 - self.alpha) * kernel_output + self.alpha * uniform_kernel
        return leaked_kernel

class LowRankKernel(Kernel):
    """Creates a low-rank approximation of a kernel using SVD."""
    
    def __init__(self, kernel: Kernel, q: int = 10, config: Optional[KernelConfig] = None):
        super().__init__(config)
        self.kernel = kernel
        self.q = q
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        kernel_output = self.kernel(*args, **kwargs)
        
        # Handle small matrices
        q = min(self.q, min(kernel_output.size()))
        
        try:
            # Try regular SVD first
            U, S, Vt = torch.linalg.svd(kernel_output, full_matrices=False)
        except RuntimeError:
            # Fallback to more stable but slower eigendecomposition for symmetric matrices
            if self.config.symmetric:
                eigenvalues, eigenvectors = torch.linalg.eigh(kernel_output)
                # Sort in descending order
                idx = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Reconstruct using top components
                U = eigenvectors
                S = torch.sqrt(torch.abs(eigenvalues))  # Handle small negative eigenvalues
                Vt = eigenvectors.t()
            else:
                raise
        
        # Reconstruct using top components
        low_rank = U[:, :q] @ torch.diag(S[:q]) @ Vt[:q, :]
        
        if self.config.normalize:
            row_sums = low_rank.sum(dim=1, keepdim=True)
            low_rank = low_rank / (row_sums + self.config.eps)
            
        return low_rank

class SpectralKernel(Kernel):
    """Creates a spectral approximation using top eigenvectors."""
    
    def __init__(self, kernel: Kernel, q: int = 10, config: Optional[KernelConfig] = None):
        super().__init__(config)
        self.kernel = kernel
        self.q = q
    
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        kernel_output = self.kernel(*args, **kwargs)
        
        # Handle small matrices
        q = min(self.q, kernel_output.size(0))
        
        # Compute eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(kernel_output)
        except RuntimeError:
            # Fallback to more stable but less accurate method
            eigenvalues, eigenvectors = torch.linalg.eig(kernel_output)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
            
        # Sort eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top q components
        top_eigenvalues = eigenvalues[:q]
        top_eigenvectors = eigenvectors[:, :q]
        
        # Handle negative eigenvalues
        top_eigenvalues = torch.abs(top_eigenvalues)
        
        # Reconstruct using top components
        reconstructed = top_eigenvectors @ torch.diag(top_eigenvalues) @ top_eigenvectors.t()
        
        if self.config.normalize:
            row_sums = reconstructed.sum(dim=1, keepdim=True)
            reconstructed = reconstructed / (row_sums + self.config.eps)
            
        return reconstructed
