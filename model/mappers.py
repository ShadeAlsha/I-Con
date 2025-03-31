from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple

####################### Parametric Mappers #######################
class MLPMapper(nn.Module):
    """Parametric MLP-based mapper."""
    def __init__(
        self, 
        input_dim: int = 28*28, 
        hidden_dims: Tuple[int, ...] = (512, 512, 2000),
        output_dim: int = 2,
        probabilities: bool = False,
        mode: str = 'softmax',
        normalize: bool = False
    ):
        if hidden_dims is None:
            hidden_dims = ()
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        self.probabilities = probabilities
        self.mode = mode
        self._output_dim = output_dim
        self.normalize = normalize
        
    @property
    def output_dim(self):
        return self._output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        
        if self.probabilities:
            if self.mode == 'softmax':
                logits = F.softmax(logits, dim=1)
            elif self.mode == 'cauchy':
                logits = 1 / (1 + logits**2)
                logits = F.normalize(logits, p=1, dim=1)
                
        if self.normalize:
            logits = F.normalize(logits, dim=1)
            
        return logits

class SimpleCNN(nn.Module):
    """
    A deeper CNN architecture with skip connections. 
    We define three "blocks":
      - Block1: 1->16->16 channels + skip, then pool
      - Block2: 16->16->16 channels + skip, then pool
      - Block3: 16->32->32 channels + skip (no pool at end)
    Finally, we flatten and use 2 fully connected layers.
    """
    def __init__(self, output_dim=2, softmax = False, normalize_feats=False, unit_sphere=False):
        super().__init__()
        
        # ----- Block 1 (1 -> 16 -> 16) -----
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        # ----- Block 2 (16 -> 16 -> 16) -----
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        # ----- Block 3 (16 -> 32 -> 32) -----
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        # ----- Fully connected layers -----
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.output_dim = output_dim
        self.softmax = softmax
        self.normalize_feats = normalize_feats
        self.unit_sphere = unit_sphere

    def forward(self, x):
        # x shape: [batch, 1, 28, 28]

        # ----- Block 1 -----
        # conv1 + relu
        x = x.view(-1, 1, 28, 28)
        out1 = F.relu(self.conv1(x))  
        # conv2 (skip from out1)
        out2 = self.conv2(out1)        
        out2 = F.relu(out2 + out1)     # skip connection
        out2 = F.max_pool2d(out2, 2)   # [batch, 16, 14, 14]

        # ----- Block 2 -----
        out3 = F.relu(self.conv3(out2))
        out4 = self.conv4(out3)
        out4 = F.relu(out4 + out3)     # skip connection
        out4 = F.max_pool2d(out4, 2)   # [batch, 16, 7, 7]

        # ----- Block 3 -----
        out5 = F.relu(self.conv5(out4)) 
        out6 = self.conv6(out5)
        out6 = F.relu(out6 + out5)     # skip connection
        # shape is now [batch, 32, 7, 7]

        # Flatten
        out6 = out6.view(-1, 32 * 7 * 7)

        # Fully connected layers
        out6 = F.relu(self.fc1(out6))
        out6 = self.fc2(out6)
        if self.softmax:
            out6 = F.softmax(out6)
        if self.normalize_feats:
            # Normalize final output (Feature-wise standardization: Mean 0, Std 1)
            mean = out6.mean(dim=0, keepdim=True)
            std = out6.std(dim=0, keepdim=True) + 1e-5  
            out6 = (out6 - mean) 
            if std.min().item() > 1:
                out6 = out6 / std
        if self.unit_sphere:
            out6 = F.normalize(out6, p=2, dim=1)
        return out6

class ResNet(nn.Module):
    def __init__(self, model_type="resnet50"):
        """
        Initialize a ResNet model.
        
        Args:
            model_type (str): Type of ResNet model to use. Options: "resnet18", "resnet34", "resnet50".
        """
        super(ResNet, self).__init__()
        
        # Choose the ResNet model based on the specified type
        if model_type == "resnet18":
            self.encoder = models.resnet18(weights=None)  # Do not load pretrained weights
            self.output_dim = 512
        elif model_type == "resnet34":
            self.encoder = models.resnet34(weights=None)
            self.output_dim = 512
        elif model_type == "resnet50":
            self.encoder = models.resnet50(weights=None)
            self.output_dim = 2048
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'resnet18', 'resnet34', or 'resnet50'.")
        
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()
    
    def forward(self, x):
        return self.encoder(x)
####################### Non-Parametric Mappers #######################

class NonParametricMapper(ABC):
    """Abstract base class for non-parametric mappers."""
    
    @abstractmethod
    def update(self, x: torch.Tensor) -> None:
        """Update internal representations with new data."""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension of the mapper."""
        pass   
class CodebookMapper(nn.Module, NonParametricMapper):
    """Non-parametric mapper using a learned codebook."""
    
    def __init__(
        self, 
        codebook_size: int,
        feature_dim: int,
        tau: float = 0.07,
        use_straight_through: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.feature_dim = feature_dim
        self.tau = tau
        self.use_straight_through = use_straight_through
        self._output_dim = feature_dim
        
        # Initialize codebook
        init_bound = 1.0 / codebook_size
        codebook = torch.rand(codebook_size, feature_dim, device=device) * 2 * init_bound - init_bound
        self.register_buffer('codebook', F.normalize(codebook, dim=1))
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input
        x_normalized = F.normalize(x, dim=1)
        
        # Compute similarities
        similarities = torch.matmul(x_normalized, self.codebook.t()) / self.tau
        
        # Soft assignments
        soft_assign = F.softmax(similarities, dim=1)
        
        # Quantize
        quantized = torch.matmul(soft_assign, self.codebook)
        
        if self.training and self.use_straight_through:
            # Straight-through gradient estimation
            quantized = x_normalized + (quantized - x_normalized).detach()
            
        return quantized
    
    def update(self, x: torch.Tensor) -> None:
        """Update codebook using EMA."""
        if self.training:
            with torch.no_grad():
                x_normalized = F.normalize(x, dim=1)
                
                # Find nearest neighbors
                similarities = torch.matmul(x_normalized, self.codebook.t())
                indices = similarities.argmax(dim=1)
                
                # Update codebook
                for idx in range(self.codebook_size):
                    mask = (indices == idx)
                    if mask.any():
                        self.codebook[idx] = F.normalize(
                            0.9 * self.codebook[idx] + 0.1 * x_normalized[mask].mean(dim=0),
                            dim=0
                        )

class MemoryBankMapper(nn.Module, NonParametricMapper):
    """Non-parametric mapper using a memory bank."""
    
    def __init__(
        self,
        feature_dim: int,
        memory_size: int = 1000,
        momentum: float = 0.1,
        sigma: float = 1.0,
        device: str = 'cuda'
    ):
        super().__init__()
        self.memory_size = memory_size
        self.momentum = momentum
        self.sigma = sigma
        self._output_dim = memory_size
        
        # Initialize memory bank
        self.register_buffer('memory', torch.randn(memory_size, feature_dim))
        self.memory = F.normalize(self.memory, dim=1)
        self.register_buffer('pointer', torch.zeros(1, dtype=torch.long))
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = F.normalize(x, dim=1)
        
        # Compute kernel distances to memory bank
        similarities = torch.matmul(x_normalized, self.memory.t())
        kernel_weights = torch.exp(similarities / self.sigma)
        kernel_weights = F.normalize(kernel_weights, p=1, dim=1)
        
        return kernel_weights
    
    def update(self, x: torch.Tensor) -> None:
        """Update memory bank."""
        if self.training:
            with torch.no_grad():
                x_normalized = F.normalize(x, dim=1)
                batch_size = x.shape[0]
                
                # Update memory bank
                n_slots = min(batch_size, self.memory_size - self.pointer[0])
                if n_slots > 0:
                    self.memory[self.pointer[0]:self.pointer[0] + n_slots] = \
                        self.momentum * self.memory[self.pointer[0]:self.pointer[0] + n_slots] + \
                        (1 - self.momentum) * x_normalized[:n_slots]
                    
                    self.pointer[0] = (self.pointer[0] + n_slots) % self.memory_size
                    
                    # Handle remaining samples if any
                    if n_slots < batch_size:
                        remaining = batch_size - n_slots
                        self.memory[:remaining] = \
                            self.momentum * self.memory[:remaining] + \
                            (1 - self.momentum) * x_normalized[n_slots:]
                        self.pointer[0] = remaining

class KNNMapper(nn.Module, NonParametricMapper):
    """Non-parametric mapper using k-nearest neighbors."""
    
    def __init__(
        self,
        feature_dim: int,
        k: int = 5,
        sigma: float = 1.0,
        device: str = 'cuda'
    ):
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.feature_dim = feature_dim
        self._output_dim = k
        
        # Initialize reference points
        self.register_buffer('reference_points', None)
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reference_points is None:
            return torch.zeros(x.size(0), self.k, device=x.device)
            
        # Compute distances to all reference points
        x_normalized = F.normalize(x, dim=1)
        distances = torch.cdist(x_normalized, self.reference_points)
        
        # Get k nearest neighbors
        _, indices = distances.topk(self.k, dim=1, largest=False)
        
        # Compute kernel weights
        weights = torch.exp(-distances.gather(1, indices) / self.sigma)
        weights = F.normalize(weights, p=1, dim=1)
        
        return weights
    
    def update(self, x: torch.Tensor) -> None:
        """Update reference points."""
        if self.training and self.reference_points is None:
            self.reference_points = F.normalize(x.detach().clone(), dim=1)
            
class OneHotEncoder(nn.Module):
    def __init__(self, num_classes):
        super(OneHotEncoder, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return F.one_hot(x, num_classes=self.num_classes).float()
