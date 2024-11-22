# model/kernel_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass

@dataclass
class KernelModelConfig:
    """Configuration class for KernelModel."""
    mapper: nn.Module
    target_kernel: nn.Module
    learned_kernel: nn.Module
    mapper2: Optional[nn.Module] = None
    num_classes: Optional[int] = None
    embeddings_map: Optional[nn.Module] = None
    embeddings_map2: Optional[nn.Module] = None
    lr: float = 1e-3
    accuracy_mode: Optional[str] = None
    use_ema: bool = False
    ema_momentum: float = 0.999
    loss_type: str = 'kl'
    decay_factor: float = 0.9
    linear_probe: bool = False

class KernelLoss:
    """Loss function factory for kernel-based losses."""
    
    @staticmethod
    def get_loss_fn(loss_type: str):
        loss_functions = {
            'kl': lambda x, y: F.kl_div(y.log(), x, reduction='batchmean'),
            'ce': lambda x, y: F.cross_entropy(x, y),
            'l2': lambda x, y: F.mse_loss(x, y),
            'tv': lambda x, y: 0.5 * torch.abs(x - y).mean(),
            'hellinger': lambda x, y: (torch.sqrt(x) - torch.sqrt(y)).pow(2).mean(),
            'orthogonality': lambda x, y: -(x * y).mean(),
            'jsd': lambda x, y: 0.5 * (F.kl_div(y.log(), x, reduction='batchmean') + 
                                     F.kl_div(x.log(), y, reduction='batchmean')),
            'none': lambda x, y: torch.tensor(0.0, device=x.device)
        }
        if loss_type not in loss_functions:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        return loss_functions[loss_type]

class KernelModel(pl.LightningModule):
    def __init__(self, config: KernelModelConfig):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        self._setup_model()
        self._setup_metrics()
        self.validation_outputs = []
        
    def _setup_model(self):
        """Initialize model components."""
        self.mapper = self.config.mapper
        self.target_kernel = self.config.target_kernel
        self.learned_kernel = self.config.learned_kernel
        self.embeddings_map = self.config.embeddings_map or nn.Identity()
        self.embeddings_map2 = self.config.embeddings_map2 or self.embeddings_map
        
        # Setup mapper2 (EMA or alternate)
        self.mapper2 = self._initialize_second_mapper()
        
        # Setup linear probe if needed
        self.linear_classifier = (nn.Linear(self.mapper.output_dim, self.config.num_classes) 
                                if self.config.linear_probe else nn.Identity())
        
        # Initialize loss function
        self.loss_fn = KernelLoss.get_loss_fn(self.config.loss_type)
        
    def _initialize_second_mapper(self) -> nn.Module:
        """Initialize the second mapper based on configuration."""
        if self.config.use_ema:
            return self._create_ema_mapper()
        return self.config.mapper2 or self.mapper

    def _setup_metrics(self):
        """Initialize accuracy metrics."""
        if self.config.accuracy_mode == 'regular':
            self.train_acc = self.val_acc = Accuracy(task="multiclass", 
                                                   num_classes=self.config.num_classes)
        elif self.config.accuracy_mode == 'unsupervised':
            self.train_acc = self.val_acc = UnsupervisedAccuracy(self.config.num_classes)
        else:
            self.train_acc = self.val_acc = None

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both mappers."""
        return self.mapper(x1), self.mapper2(x2)

    def _compute_loss(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        """Compute all losses for a batch."""
        features, labels, idx = self._process_batch(batch)
        embeddings1, embeddings2 = self(features[0], features[1])
        proj1, proj2 = self.embeddings_map(embeddings1), self.embeddings_map2(embeddings2)
        
        learned_kernel = self.learned_kernel([proj1, proj2], labels, idx)
        target_kernel = self.target_kernel([features[0], features[1]], labels, idx)
        
        losses = {
            'kernel_loss': self.loss_fn(target_kernel, learned_kernel),
            'linear_probe_loss': self._compute_linear_probe_loss(embeddings1, labels)
        }
        
        return {
            'losses': losses,
            'metrics': {
                'embeddings': (proj1, proj2),
                'kernels': (learned_kernel, target_kernel),
                'logits': self.linear_classifier(embeddings1.detach()),
                'labels': labels
            }
        }

    def _process_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Process input batch to handle different formats."""
        features, labels, idx = batch
        if len(features.shape) == 3 and features.shape[1] >= 2:
            features = self._split_features(features)
            labels = labels.repeat(features[0].size(0) // labels.size(0))
        else:
            features = [features, features]
        return features, labels, idx

    def _compute_linear_probe_loss(self, embeddings: torch.Tensor, 
                                 labels: torch.Tensor) -> torch.Tensor:
        """Compute linear probe loss if enabled."""
        if self.config.linear_probe:
            logits = self.linear_classifier(embeddings)
            return F.cross_entropy(logits, labels)
        return torch.tensor(0.0, device=embeddings.device)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        param_groups = [
            {'params': self.mapper.parameters()},
            {'params': self.embeddings_map.parameters()}
        ]
        if self.config.linear_probe:
            param_groups.append({'params': self.linear_classifier.parameters(), 'lr': 5e-3})

        optimizer = torch.optim.AdamW(param_groups, lr=self.config.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return [optimizer], [scheduler]

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Execute training step."""
        results = self._compute_loss(batch)
        loss = sum(results['losses'].values())
        self._log_metrics('train', results, loss)
        if self.config.use_ema:
            self._update_ema()
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """Execute validation step."""
        results = self._compute_loss(batch)
        loss = sum(results['losses'].values())
        self._log_metrics('val', results, loss)
        self.validation_outputs.append(results['metrics'])

    def _log_metrics(self, phase: str, results: Dict, loss: torch.Tensor) -> None:
        """Log metrics for current step."""
        self.log(f'{phase}_loss', loss)
        for loss_name, loss_value in results['losses'].items():
            self.log(f'{phase}_{loss_name}', loss_value)
            
        if getattr(self, f'{phase}_acc'):
            logits = results['metrics']['logits']
            labels = results['metrics']['labels']
            getattr(self, f'{phase}_acc')(logits.argmax(dim=-1), labels)
            self.log(f'{phase}_accuracy', getattr(self, f'{phase}_acc'))
