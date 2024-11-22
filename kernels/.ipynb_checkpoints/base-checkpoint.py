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
    symmetric: bool = True
    normalize: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps: float = 1e-10
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