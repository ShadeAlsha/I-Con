from model import *
from kernels import *
from dataloaders import *
from visualization import *
from dataclasses import dataclass
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint

@dataclass
class TrainingConfig:
    batch_size: int = 256
    num_views: int = 2
    num_workers: int = 0
    max_epochs: int = 150
    accelerator: str = 'gpu'
    devices: int = 4
    precision: int = 32
    data_root: str = "/datadrive/pytorch-data"
    checkpoint_dir: str = "cifar100-checkpoints"
    save_top_k: int = 3
    monitor: str = "train_kernel_loss"
    save_last: bool = True
    every_n_epochs: int = 1
    projection_dim: int = 128
    sigma = 0.7
    leak = 0

def setup_checkpointing(config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename='epoch_{epoch:03d}-kernel_loss_{train_kernel_loss:.4f}',
        save_top_k=config.save_top_k,
        monitor='train_kernel_loss',
        mode='min',
        save_last=config.save_last,
        every_n_epochs=config.every_n_epochs
    )
    
    return checkpoint_callback

def main():
    training_config = TrainingConfig()
    
    train_loader = get_cifar10_dataloaders(
        root=training_config.data_root, 
        batch_size=training_config.batch_size, 
        num_views=training_config.num_views
    )
    
    kernel_config = KernelConfig(mask_diagonal=False)
    
    mapper = ResNet50()
    embeddings_map = MLPMapper(
        input_dim=mapper.output_dim,
        hidden_dims=(mapper.output_dim,),
        output_dim=training_config.projection_dim
    )
    
    model_config = KernelModelConfig(
        mapper=mapper,
        target_kernel=AugmentationKernel(block_size=1, config=kernel_config).leak(training_config.leak),
        learned_kernel=GaussianKernel(sigma=training_config.sigma, config=kernel_config),
        embeddings_map=embeddings_map,
        lr=1e-3,
        loss_type='ce',
        log_kernel_loss=True,
        accuracy_mode='regular',
        num_classes=10,
        linear_probe=True
    )
    
    
    model = KernelModel(model_config)
    
    checkpoint_callback = setup_checkpointing(training_config)
    callbacks = [
        checkpoint_callback,
    ]
    
    trainer = pl.Trainer(
        strategy="ddp",
        max_epochs=training_config.max_epochs,
        accelerator=training_config.accelerator,
        devices=training_config.devices,
        precision=training_config.precision,
        callbacks=callbacks
    )
    
    trainer.fit(model, train_loader)
    print(f"\nTraining completed!")
    print(f"Best checkpoint path: {checkpoint_callback.best_model_path}")
    print(f"Best kernel loss: {checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    main()