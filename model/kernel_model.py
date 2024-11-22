import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Dict, Tuple, List, Union, Any
from .losses import KernelLoss
from .ema import EMAUpdater
from .model_config import KernelModelConfig
from .metrics import UnsupervisedAccuracy
from torchmetrics import Accuracy

class KernelModel(pl.LightningModule):
    def __init__(self, config: KernelModelConfig):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        self._setup_model()
        self._setup_metrics()
        self._setup_monitoring()
        
    def _setup_model(self):
        """Initialize and validate model components."""
        self.mapper = self.config.mapper
        self.target_kernel = self.config.target_kernel
        self.learned_kernel = self.config.learned_kernel
        self.embeddings_map = self.config.embeddings_map or nn.Identity()
        self.embeddings_map2 = self.config.embeddings_map2 or self.embeddings_map
        
        if self.config.use_ema:
            self.ema = EMAUpdater(self.mapper, self.config.ema_momentum)
            self.mapper2 = self.mapper
        else:
            self.mapper2 = self.config.mapper2 or self.mapper
            
        self.linear_classifier = (
            nn.Linear(self.mapper.output_dim, self.config.num_classes)
            if self.config.linear_probe else nn.Identity()
        )
        print(self.linear_classifier.weight.dtype)
        
        self.loss_fn = KernelLoss.get_loss_fn(self.config.loss_type)
        
    def _setup_metrics(self):
        """Initialize performance metrics."""
        if self.config.accuracy_mode == 'regular' and self.config.num_classes:
            self.train_acc = self.val_acc = Accuracy(
                task="multiclass",
                num_classes=self.config.num_classes,
            )
        elif self.config.accuracy_mode == 'unsupervised' and self.config.num_classes:
            self.train_acc = self.val_acc = UnsupervisedAccuracy(
                n_classes=self.config.num_classes
            )
        else:
            self.train_acc = self.val_acc = None
        # Add gradient norm monitoring
        self.grad_norm = torch.tensor(0.0)
        
    def _setup_monitoring(self):
        """Initialize training monitoring."""
        self.automatic_optimization = not self.config.use_mixed_precision
        self.validation_outputs = []
        self.best_val_loss = float('inf')
        
    #@torch.cuda.amp.autocast()
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with mixed precision support."""
        return self.mapper(x1), self.mapper2(x2)
    
    def _validate_batch(self, batch: Tuple[torch.Tensor, ...]) -> None:
        """Validate input batch structure and dimensions."""
        if not (isinstance(batch, tuple) or  isinstance(batch, list))or len(batch) != 3:
            raise ValueError("Batch must be a tuple of (features, labels, idx)")
        
        features, labels, idx = batch
        if not all(isinstance(t, torch.Tensor) for t in (features, labels, idx)):
            raise TypeError("All batch elements must be torch.Tensor")
            
        if len(features.shape) < 2:
            raise ValueError("Features must have at least 2 dimensions")
            
    def _compute_kernels(self, 
                        embeddings1: torch.Tensor,
                        embeddings2: torch.Tensor,
                        features: List[torch.Tensor],
                        labels: torch.Tensor,
                        idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute target and learned kernels with error handling."""
        try:
            proj1 = self.embeddings_map(embeddings1)
            proj2 = self.embeddings_map2(embeddings2)
            
            learned_kernel = self.learned_kernel([proj1, proj2], labels, idx, return_log=self.config.log_kernel_loss)
            target_kernel = self.target_kernel([features[0], features[1]], labels, idx)
            
            if torch.isnan(learned_kernel).any() or torch.isnan(target_kernel).any():
                raise ValueError("NaN values detected in kernel computation")
                
            return learned_kernel, target_kernel
        except RuntimeError as e:
            self.log("kernel_computation_error", True)
            raise
            
    def _compute_loss(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
        """Compute model losses with comprehensive error handling."""
        self._validate_batch(batch)
        features, labels, idx = self._process_batch(batch)
        
        # Compute embeddings and kernels
        embeddings1, embeddings2 = self(features[0], features[1])
        learned_kernel, target_kernel = self._compute_kernels(
            embeddings1, embeddings2, features, labels, idx
        )
        
        # Compute losses
        losses = {
            'kernel_loss': self.loss_fn(target_kernel, learned_kernel, log=self.config.log_kernel_loss),
            'linear_probe_loss': self._compute_linear_probe_loss(embeddings1, labels)
        }
        # Validate losses
        if any(torch.isnan(loss) for loss in losses.values()):
            raise ValueError("NaN loss detected")
            
        return {
            'losses': losses,
            'metrics': {
                'embeddings': (embeddings1, embeddings2),
                'kernels': (torch.exp(learned_kernel) if self.config.log_kernel_loss else learned_kernel,
                            target_kernel),
                'logits': self.linear_classifier(embeddings1.detach()),
                'labels': labels
            }
        }
        
        
    def _compute_linear_probe_loss(self, embeddings: torch.Tensor, 
                                 labels: torch.Tensor) -> torch.Tensor:
        """Compute linear probe loss if enabled."""
        if self.config.linear_probe:
            logits = self.linear_classifier(embeddings)
            return F.cross_entropy(logits, labels)
        return torch.tensor(0.0, device=embeddings.device)
    
    def configure_optimizers(self):
        """Configure optimizers with learning rate scheduling."""
        param_groups = [
            {'params': self.mapper.parameters(), 'lr': self.config.lr},
            {'params': self.embeddings_map.parameters(), 'lr': self.config.lr}
        ]
        
        if self.config.linear_probe:
            param_groups.append({
                'params': self.linear_classifier.parameters(),
                'lr': self.config.lr * 5
            })
            
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=self.config.lr * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
        
    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Execute training step with gradient clipping and EMA updates."""
        results = self._compute_loss(batch)
        loss = sum(results['losses'].values())
        
        # Manual optimization for mixed precision
        if self.config.use_mixed_precision:
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(loss)
            
            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                self.grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.config.gradient_clip_val
                )
                
            optimizer.step()
            
        # Update EMA if enabled
        if self.config.use_ema:
            self.ema.update()
            
        self._log_metrics('train', results, loss)
        return loss
        
    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict:
        """Execute validation step with EMA if enabled."""
        if self.config.use_ema:
            with self.ema.average_parameters():
                results = self._compute_loss(batch)
        else:
            results = self._compute_loss(batch)
            
        loss = sum(results['losses'].values())
        self._log_metrics('val', results, loss)
        
        # Track best model
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            
        metrics = results['metrics']
        return {
            'embeddings': metrics['embeddings'][0].detach().cpu(),
            'logits': metrics['logits'].detach().cpu(),
            'labels': metrics['labels'].detach().cpu(),
            'learned_kernel': results['metrics']['kernels'][0].detach().cpu(),
            'target_kernel': results['metrics']['kernels'][1].detach().cpu()
        }
        
    def _log_metrics(self, phase: str, results: Dict, loss: torch.Tensor) -> None:
        """Enhanced metric logging with gradient and LR monitoring."""
        # Log losses
        self.log(f'{phase}_loss', loss, prog_bar=True)
        for loss_name, loss_value in results['losses'].items():
            self.log(f'{phase}_{loss_name}', loss_value)
            
        # Log accuracy if available
        if accuracy_metric := getattr(self, f'{phase}_acc'):
            logits = results['metrics']['logits']
            labels = results['metrics']['labels']
            if isinstance(accuracy_metric, UnsupervisedAccuracy):
                # Only update stats, compute later in epoch end
                accuracy_metric.update(logits.argmax(dim=-1), labels)
            else:
                # For supervised accuracy, compute as before
                accuracy_metric(logits.argmax(dim=-1), labels)
                self.log(f'{phase}_accuracy', accuracy_metric, prog_bar=True)
            
        # Log gradient norm during training
        if phase == 'train':
            self.log('grad_norm', self.grad_norm)
            
        # Log learning rates
        if self.trainer is not None:
            for i, optimizer in enumerate(self.trainer.optimizers):
                for j, param_group in enumerate(optimizer.param_groups):
                    self.log(f'lr_group_{j}', param_group['lr'])
                    
    def _process_batch(self, 
                      batch: Tuple[torch.Tensor, ...]) -> Tuple[List[torch.Tensor], 
                                                              torch.Tensor, 
                                                              torch.Tensor]:
        """Process input batch with improved error handling and validation."""
        features, labels, idx = batch
        
        # Handle multiple views in features
        if len(features.shape) >= 3 and features.shape[1] == 2:
            features = [features[:, 0], features[:, 1]]
        else:
            features = [features, features]
            
        # Validate shapes
        if not all(f.size(0) == labels.size(0) for f in features):
            raise ValueError("Batch size mismatch between features and labels")
            
        return features, labels, idx
    def training_epoch_end(self, outputs: List[Dict]) -> None:
        """Compute unsupervised accuracy at epoch end if needed."""
        if isinstance(self.train_acc, UnsupervisedAccuracy):
            accuracy = self.train_acc.compute()
            self.log('train_accuracy', accuracy, prog_bar=True)
            self.train_acc.reset()    
    def validation_epoch_end(self, validation_step_outputs: List[Dict]) -> None:
        """Process validation outputs with memory efficiency."""
        outputs = {}
        
        for key in validation_step_outputs[0].keys():
            tensors = [batch[key] for batch in validation_step_outputs]
            outputs[key] = torch.cat(tensors, dim=0)
                
        self.aggregated_val_outputs = (
            outputs['embeddings'],
            outputs['logits'],
            outputs['labels'],
            outputs['learned_kernel'],
            outputs['target_kernel']
        )
        # Compute unsupervised accuracy at epoch end if needed
        if isinstance(self.val_acc, UnsupervisedAccuracy):
            accuracy = self.val_acc.compute()
            self.log('val_accuracy', accuracy, prog_bar=True)
            self.val_acc.reset()
        
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save additional model state in checkpoints."""
        checkpoint['best_val_loss'] = self.best_val_loss
        if self.config.use_ema:
            checkpoint['ema_state'] = self.ema.shadow
            
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load additional model state from checkpoints."""
        self.best_val_loss = checkpoint['best_val_loss']
        if self.config.use_ema and 'ema_state' in checkpoint:
            self.ema.shadow = checkpoint['ema_state']
            
    def get_progress_bar_dict(self) -> Dict[str, Union[int, float, str]]:
        """Customize progress bar metrics."""
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        
        # Add custom metrics to progress bar
        if hasattr(self, 'grad_norm'):
            items['grad'] = f'{self.grad_norm:.3f}'
        if hasattr(self, 'best_val_loss'):
            items['best_val'] = f'{self.best_val_loss:.3f}'
            
        return items