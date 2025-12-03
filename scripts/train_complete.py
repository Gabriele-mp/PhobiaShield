"""
PhobiaShield Training Script - Complete
The Architect Module

Usage:
    python scripts/train_complete.py
    python scripts/train_complete.py training=fast_test
    python scripts/train_complete.py model=tiny_yolo training.epochs=50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import wandb
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.phobia_net import PhobiaNet
from src.models.loss import PhobiaLoss
from src.data.phobia_dataset import PhobiaDataset


class Trainer:
    """Training manager for PhobiaShield."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"PhobiaShield Training - The Architect")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*60}\n")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        
        # Setup
        self._setup_model()
        self._setup_data()
        self._setup_training()
        self._setup_wandb()
    
    def _setup_model(self):
        """Initialize model and loss."""
        print("🔧 Setting up model...")
        
        # Load model config
        model_cfg = OmegaConf.load(self.cfg.model.config_path)
        
        # Initialize model
        self.model = PhobiaNet(model_cfg).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Model: {self.cfg.model.name}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Size: {total_params * 4 / 1e6:.2f} MB\n")
        
        # Initialize loss
        self.loss_fn = PhobiaLoss(
            lambda_coord=self.cfg.loss.lambda_coord,
            lambda_obj=self.cfg.loss.lambda_obj,
            lambda_noobj=self.cfg.loss.lambda_noobj,
            lambda_class=self.cfg.loss.lambda_class,
            grid_size=model_cfg.architecture.grid_size,
            num_boxes=model_cfg.architecture.num_boxes_per_cell,
            num_classes=model_cfg.output.num_classes
        )
        print(f"✓ Loss function initialized")
        print(f"  Lambdas: coord={self.cfg.loss.lambda_coord}, "
              f"obj={self.cfg.loss.lambda_obj}, "
              f"noobj={self.cfg.loss.lambda_noobj}, "
              f"class={self.cfg.loss.lambda_class}\n")
    
    def _setup_data(self):
        """Setup dataloaders."""
        print("📦 Setting up data...")
        
        data_root = Path(self.cfg.data.root)
        
        # Train dataset
        train_dataset = PhobiaDataset(
            img_dir=data_root / 'train' / 'images',
            label_dir=data_root / 'train' / 'labels',
            img_size=self.cfg.data.img_size,
            grid_size=self.cfg.data.grid_size,
            num_boxes=self.cfg.data.num_boxes,
            num_classes=self.cfg.data.num_classes,
            augment=True
        )
        
        # Val dataset
        val_dataset = PhobiaDataset(
            img_dir=data_root / 'val' / 'images',
            label_dir=data_root / 'val' / 'labels',
            img_size=self.cfg.data.img_size,
            grid_size=self.cfg.data.grid_size,
            num_boxes=self.cfg.data.num_boxes,
            num_classes=self.cfg.data.num_classes,
            augment=False
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"✓ Train set: {len(train_dataset)} images")
        print(f"✓ Val set:   {len(val_dataset)} images")
        print(f"✓ Batch size: {self.cfg.training.batch_size}\n")
    
    def _setup_training(self):
        """Setup optimizer and scheduler."""
        print("⚙️  Setting up training...")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        print(f"✓ Optimizer: Adam")
        print(f"  LR: {self.cfg.training.learning_rate}")
        print(f"  Weight decay: {self.cfg.training.weight_decay}")
        print(f"✓ Scheduler: ReduceLROnPlateau\n")
    
    def _setup_wandb(self):
        """Initialize Weights & Biases."""
        if self.cfg.logging.use_wandb:
            wandb.init(
                project=self.cfg.logging.wandb_project,
                name=self.cfg.logging.run_name,
                config=OmegaConf.to_container(self.cfg, resolve=True)
            )
            wandb.watch(self.model, log='all', log_freq=100)
            print("✓ W&B initialized\n")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_coord = 0
        total_obj = 0
        total_noobj = 0
        total_class = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.training.epochs}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss, metrics = self.loss_fn(outputs, targets)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.cfg.training.grad_clip
                )
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += metrics['total_loss']
            total_coord += metrics['coord_loss']
            total_obj += metrics['conf_loss_obj']
            total_noobj += metrics['conf_loss_noobj']
            total_class += metrics['class_loss']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.2f}",
                'coord': f"{metrics['coord_loss']:.2f}",
                'noobj': f"{metrics['conf_loss_noobj']:.2f}"
            })
            
            # Log to W&B
            if self.cfg.logging.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': metrics['total_loss'],
                    'train/coord_loss': metrics['coord_loss'],
                    'train/conf_loss_obj': metrics['conf_loss_obj'],
                    'train/conf_loss_noobj': metrics['conf_loss_noobj'],
                    'train/class_loss': metrics['class_loss'],
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        # Epoch metrics
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'coord': total_coord / n_batches,
            'obj': total_obj / n_batches,
            'noobj': total_noobj / n_batches,
            'class': total_class / n_batches
        }
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate model."""
        self.model.eval()
        
        total_loss = 0
        total_coord = 0
        total_obj = 0
        total_noobj = 0
        total_class = 0
        
        pbar = tqdm(self.val_loader, desc=f"Validation")
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss, metrics = self.loss_fn(outputs, targets)
            
            # Accumulate
            total_loss += metrics['total_loss']
            total_coord += metrics['coord_loss']
            total_obj += metrics['conf_loss_obj']
            total_noobj += metrics['conf_loss_noobj']
            total_class += metrics['class_loss']
            
            pbar.set_postfix({'val_loss': f"{metrics['total_loss']:.2f}"})
        
        # Metrics
        n_batches = len(self.val_loader)
        val_metrics = {
            'loss': total_loss / n_batches,
            'coord': total_coord / n_batches,
            'obj': total_obj / n_batches,
            'noobj': total_noobj / n_batches,
            'class': total_class / n_batches
        }
        
        # Log to W&B
        if self.cfg.logging.use_wandb:
            wandb.log({
                'val/loss': val_metrics['loss'],
                'val/coord_loss': val_metrics['coord'],
                'val/conf_loss_obj': val_metrics['obj'],
                'val/conf_loss_noobj': val_metrics['noobj'],
                'val/class_loss': val_metrics['class'],
                'epoch': epoch
            })
        
        return val_metrics
    
    def train(self):
        """Main training loop."""
        print(f"🚀 Starting training for {self.cfg.training.epochs} epochs\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.cfg.training.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Scheduler step
            self.scheduler.step(val_metrics['loss'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.2f}")
            print(f"  Val Loss:   {val_metrics['loss']:.2f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}\n")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ Best model saved (val_loss: {best_val_loss:.2f})\n")
            
            # Save regular checkpoint
            if (epoch + 1) % self.cfg.training.save_every == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("✅ Training complete!")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.cfg.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': OmegaConf.to_container(self.cfg, resolve=True)
        }
        
        if is_best:
            path = checkpoint_dir / 'best_model.pth'
        else:
            path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    
    # Print config
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60 + "\n")
    
    # Create trainer
    trainer = Trainer(cfg)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()