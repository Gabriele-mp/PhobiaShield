"""
Training Script for PhobiaShield

Usage:
    python scripts/train.py                                    # Default config
    python scripts/train.py model=baseline training=fast_test  # Override config
    python scripts/train.py training.epochs=100 training.lr=0.01  # Override params
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
import numpy as np

from data.dataset import PhobiaDataset, get_transforms
from models.phobia_net import create_model
from models.loss import PhobiaLoss


class Trainer:
    """Training manager for PhobiaNet."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Set seed
        self._set_seed(config.seed)
        
        # Initialize W&B
        if config.wandb.enabled:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.name,
                config=OmegaConf.to_container(config, resolve=True),
                tags=config.wandb.tags,
                notes=config.wandb.notes,
                mode=config.wandb.mode,
            )
        
        # Create dataloaders
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Create model
        self.model = create_model(OmegaConf.to_container(config.model, resolve=True))
        self.model = self.model.to(self.device)
        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Create loss function
        self.criterion = PhobiaLoss(
            lambda_coord=config.model.loss.lambda_coord,
            lambda_obj=config.model.loss.lambda_obj,
            lambda_noobj=config.model.loss.lambda_noobj,
            lambda_class=config.model.loss.lambda_class,
            grid_size=config.model.architecture.grid_size,
            num_boxes=config.model.architecture.num_boxes_per_cell,
            num_classes=config.model.output.num_classes,
        )
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler() if config.training.scheduler.enabled else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _create_dataloaders(self):
        """Create train and validation dataloaders."""
        cfg = self.config
        
        # Transforms
        train_transform = get_transforms(cfg, mode="train")
        val_transform = get_transforms(cfg, mode="val")
        
        # Datasets
        train_dataset = PhobiaDataset(
            root_dir=cfg.paths.raw_data,
            annotations_file=cfg.data.paths.train_annotations,
            image_size=tuple(cfg.data.image.size),
            transform=train_transform,
            grid_size=cfg.model.architecture.grid_size,
            num_boxes=cfg.model.architecture.num_boxes_per_cell,
            num_classes=cfg.model.output.num_classes,
        )
        
        val_dataset = PhobiaDataset(
            root_dir=cfg.paths.raw_data,
            annotations_file=cfg.data.paths.val_annotations,
            image_size=tuple(cfg.data.image.size),
            transform=val_transform,
            grid_size=cfg.model.architecture.grid_size,
            num_boxes=cfg.model.architecture.num_boxes_per_cell,
            num_classes=cfg.model.output.num_classes,
        )
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
            drop_last=cfg.data.dataloader.drop_last,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
        )
        
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def _create_optimizer(self):
        """Create optimizer."""
        cfg = self.config.training.optimizer
        
        if cfg.type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.lr,
                betas=tuple(cfg.betas),
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.type == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=cfg.nesterov,
            )
        elif cfg.type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.lr,
                betas=tuple(cfg.betas),
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        cfg = self.config.training.scheduler
        
        if cfg.type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.step_size,
                gamma=cfg.gamma,
            )
        elif cfg.type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.T_max,
                eta_min=cfg.eta_min,
            )
        elif cfg.type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=cfg.mode,
                factor=cfg.factor,
                patience=cfg.patience,
                threshold=cfg.threshold,
            )
        else:
            raise ValueError(f"Unknown scheduler: {cfg.type}")
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_coord_loss = 0
        total_conf_obj_loss = 0
        total_conf_noobj_loss = 0
        total_class_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss, loss_dict = self.criterion(outputs, targets)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip.enabled:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip.max_norm,
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_coord_loss += loss_dict["coord_loss"]
            total_conf_obj_loss += loss_dict["conf_loss_obj"]
            total_conf_noobj_loss += loss_dict["conf_loss_noobj"]
            total_class_loss += loss_dict["class_loss"]
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "coord": f"{loss_dict['coord_loss']:.4f}",
                "class": f"{loss_dict['class_loss']:.4f}",
            })
            
            # Log to W&B
            if self.config.wandb.enabled and batch_idx % self.config.logging.log_every_n_steps == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_coord_loss": loss_dict["coord_loss"],
                    "train/batch_conf_obj_loss": loss_dict["conf_loss_obj"],
                    "train/batch_conf_noobj_loss": loss_dict["conf_loss_noobj"],
                    "train/batch_class_loss": loss_dict["class_loss"],
                })
        
        # Average losses
        num_batches = len(self.train_loader)
        metrics = {
            "train/loss": total_loss / num_batches,
            "train/coord_loss": total_coord_loss / num_batches,
            "train/conf_obj_loss": total_conf_obj_loss / num_batches,
            "train/conf_noobj_loss": total_conf_noobj_loss / num_batches,
            "train/class_loss": total_class_loss / num_batches,
        }
        
        return metrics
    
    def validate(self) -> dict:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        total_coord_loss = 0
        total_conf_obj_loss = 0
        total_conf_noobj_loss = 0
        total_class_loss = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, targets)
                
                # Accumulate losses
                total_loss += loss.item()
                total_coord_loss += loss_dict["coord_loss"]
                total_conf_obj_loss += loss_dict["conf_loss_obj"]
                total_conf_noobj_loss += loss_dict["conf_loss_noobj"]
                total_class_loss += loss_dict["class_loss"]
        
        # Average losses
        num_batches = len(self.val_loader)
        metrics = {
            "val/loss": total_loss / num_batches,
            "val/coord_loss": total_coord_loss / num_batches,
            "val/conf_obj_loss": total_conf_obj_loss / num_batches,
            "val/conf_noobj_loss": total_conf_noobj_loss / num_batches,
            "val/class_loss": total_class_loss / num_batches,
        }
        
        return metrics
    
    def save_checkpoint(self, filename: str = "checkpoint.pth"):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.paths.checkpoints)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_dir / filename)
        print(f"Checkpoint saved: {checkpoint_dir / filename}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if (epoch + 1) % self.config.training.validation.check_every_n_epochs == 0:
                val_metrics = self.validate()
            else:
                val_metrics = {}
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics["epoch"] = epoch
            all_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            
            # Log to W&B
            if self.config.wandb.enabled:
                wandb.log(all_metrics)
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")
            print(f"  Train Loss: {train_metrics['train/loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val/loss']:.4f}")
            print(f"  LR: {all_metrics['lr']:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.checkpoint.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
            
            # Save best model
            if val_metrics and val_metrics["val/loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val/loss"]
                self.save_checkpoint("best_model.pth")
                print("  ✓ Best model saved!")
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics["val/loss"])
                else:
                    self.scheduler.step()
        
        print("\n" + "="*50)
        print("Training completed!")
        print("="*50)
        
        # Save final model
        self.save_checkpoint("final_model.pth")
        
        if self.config.wandb.enabled:
            wandb.finish()


@hydra.main(config_path="../cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    print(OmegaConf.to_yaml(cfg))
    
    # Create trainer
    trainer = Trainer(cfg)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
