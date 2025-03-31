import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Dict, Tuple, List, Union, Any
from .losses import KernelLoss
from .ema import EMAUpdater
from .model_config import KernelModelConfig
from .metrics import UnsupervisedAccuracy, Accuracy


class KernelModel(pl.LightningModule):
    def __init__(self, config: KernelModelConfig):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        self._setup_model()
        self._setup_metrics()
        self._setup_monitoring()

    def _setup_model(self):
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

        self.loss_fn = KernelLoss.get_loss_fn(self.config.loss_type)

    def _setup_metrics(self):
        if self.config.accuracy_mode == 'regular' and self.config.num_classes:
            self.train_acc = Accuracy(num_classes=self.config.num_classes, ignore_index=-1)
            self.val_acc = Accuracy(num_classes=self.config.num_classes, ignore_index=-1)
        elif self.config.accuracy_mode == 'unsupervised' and self.config.num_classes:
            self.train_acc = UnsupervisedAccuracy(n_classes=self.config.num_classes)
            self.val_acc = UnsupervisedAccuracy(n_classes=self.config.num_classes)
        else:
            self.train_acc = self.val_acc = None

        self.grad_norm = torch.tensor(0.0)

    def _setup_monitoring(self):
        self.automatic_optimization = not self.config.use_mixed_precision
        self.validation_outputs = []

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mapper(x1), self.mapper2(x2)

    def _validate_batch(self, batch: Tuple[torch.Tensor, ...]) -> None:
        if not (isinstance(batch, tuple) or isinstance(batch, list)) or len(batch) != 3:
            raise ValueError("Batch must be a tuple of (features, labels, idx)")

    def _compute_kernels(self,
                         embeddings1: torch.Tensor,
                         embeddings2: torch.Tensor,
                         features: List[torch.Tensor],
                         labels: torch.Tensor,
                         idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self._validate_batch(batch)
        features, labels, idx = self._process_batch(batch)
        embeddings1, embeddings2 = self(features[0], features[1])

        learned_kernel, target_kernel = self._compute_kernels(
            embeddings1, embeddings2, features, labels, idx
        )

        losses = {
            'kernel_loss': self.loss_fn(target_kernel, learned_kernel, log=self.config.log_kernel_loss),
            'linear_probe_loss': self._compute_linear_probe_loss(embeddings1.detach(), labels)
        }

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

    def _compute_linear_probe_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.config.linear_probe:
            logits = self.linear_classifier(embeddings)
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            if not torch.isnan(loss):
                return loss
        return torch.tensor(0.0, device=embeddings.device)

    def configure_optimizers(self):
        param_groups = [
            {'params': self.mapper.parameters(), 'lr': self.config.lr},
            {'params': self.embeddings_map.parameters(), 'lr': self.config.lr},
        ]

        if self.config.linear_probe:
            param_groups.append({
                'params': self.linear_classifier.parameters(),
                'lr': self.config.lr * 5
            })

        if hasattr(self.learned_kernel, 'learnable_gamma') and self.learned_kernel.learnable_gamma:
            param_groups.append({
                'params': [self.learned_kernel.gamma],
                'lr': 0.001 * self.config.lr
            })

        if self.config.optimizer == 'adam':
            optimizer = torch.optim.SGD(param_groups, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(param_groups, weight_decay=self.config.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {self.config.optimizer} not supported")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        results = self._compute_loss(batch)
        loss = sum(results['losses'].values())

        if self.config.use_mixed_precision:
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(loss)
            if self.config.gradient_clip_val > 0:
                self.grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.config.gradient_clip_val
                )
            optimizer.step()

        if self.config.use_ema:
            self.ema.update()

        self._log_metrics('train', results, loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict:
        if self.config.use_ema:
            with self.ema.average_parameters():
                results = self._compute_loss(batch)
        else:
            results = self._compute_loss(batch)

        loss = sum(results['losses'].values())
        self._log_metrics('val', results, loss)

        metrics = results['metrics']
        return {
            'embeddings': metrics['embeddings'][0].detach().cpu(),
            'logits': metrics['logits'].detach().cpu(),
            'labels': metrics['labels'].detach().cpu(),
            'learned_kernel': results['metrics']['kernels'][0].clip(1e-10).detach().cpu(),
            'target_kernel': results['metrics']['kernels'][1].clip(1e-10).detach().cpu(),
        }

    def on_train_epoch_end(self) -> None:
        if isinstance(self.train_acc, UnsupervisedAccuracy):
            accuracy = self.train_acc.compute()
            self.log('train_accuracy', accuracy, prog_bar=True)
            self.train_acc.reset()


    def on_validation_epoch_end(self) -> None:
        if not self.validation_outputs:
            return

        outputs = {}
        keys = self.validation_outputs[0].keys()
        for key in keys:
            tensors = [batch[key] for batch in self.validation_outputs]
            outputs[key] = torch.cat(tensors, dim=0)

        self.aggregated_val_outputs = (
            outputs['embeddings'],
            outputs['logits'],
            outputs['labels'],
            outputs['learned_kernel'],
            outputs['target_kernel']
        )

        if isinstance(self.val_acc, UnsupervisedAccuracy):
            accuracy = self.val_acc.compute()
            self.log('val_accuracy', accuracy, prog_bar=True)
            self.val_acc.reset()

        # Clear stored outputs for next epoch
        self.validation_outputs.clear()


    def _log_metrics(self, phase: str, results: Dict, loss: torch.Tensor) -> None:
        self.log(f'{phase}_loss', loss, prog_bar=True)
        for loss_name, loss_value in results['losses'].items():
            self.log(f'{phase}_{loss_name}', loss_value)

        if accuracy_metric := getattr(self, f'{phase}_acc'):
            logits = results['metrics']['logits']
            labels = results['metrics']['labels']
            if isinstance(accuracy_metric, UnsupervisedAccuracy):
                accuracy_metric.update(logits.argmax(dim=-1), labels)
            else:
                accuracy_metric(logits.argmax(dim=-1), labels)
                self.log(f'{phase}_accuracy', accuracy_metric, prog_bar=True)

        if phase == 'train':
            self.log('grad_norm', self.grad_norm)

        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):
            opts = [opts]

        for i, opt in enumerate(opts):
            for j, group in enumerate(opt.param_groups):
                self.log(f'lr_group_{i}_{j}', group['lr'])



    def _process_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        features, labels, idx = batch
        if len(features.shape) >= 3:
            if features.shape[1] == 2:
                features = [features[:, 0], features[:, 1]]
            elif features.shape[1] == 1:
                features = [features[:, 0], features[:, 0]]
            else:
                features = [features, features]
        elif len(features.shape) == 3:
            if features.shape[1] == 1:
                features = [features[:, 0], features[:, 0]]
        elif len(features.shape) == 2:
            features = [features, features]
        return features, labels, idx

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.config.use_ema:
            checkpoint['ema_state'] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.config.use_ema and 'ema_state' in checkpoint:
            self.ema.shadow = checkpoint['ema_state']

    def get_progress_bar_dict(self) -> Dict[str, Union[int, float, str]]:
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        if hasattr(self, 'grad_norm'):
            items['grad'] = f'{self.grad_norm:.3f}'
        return items
