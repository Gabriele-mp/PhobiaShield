"""
PhobiaShield - Training Script
The Architect Module

CORRECTED VERSION with:
- Proper lambda weights (obj=5.0, noobj=0.05)
- Balanced class weights
- NO Focal Loss by default (causes issues)
- NO GIoU by default (unstable)
- ResNet + CBAM enabled
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys
from omegaconf import OmegaConf
import argparse
import math

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from src.models.phobia_net import PhobiaNet
from src.models.loss import PhobiaLoss
from src.data.phobia_dataset import PhobiaDataset


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup + cosine annealing.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 0.0,
        cosine_t0: int = 10,
        cosine_t_mult: int = 2
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.cosine_t0 = cosine_t0
        self.cosine_t_mult = cosine_t_mult
        
        self.current_epoch = 0
        
        # Initialize Cosine scheduler
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_t0,
            T_mult=cosine_t_mult,
            eta_min=min_lr
        )
    
    def step(self, epoch: int = None):
        """Update learning rate."""
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (self.current_epoch / self.warmup_epochs)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine annealing phase
            self.cosine_scheduler.step(self.current_epoch - self.warmup_epochs)
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def train_phobiashield(
    data_root='data/phobiashield_final',
    model_config='cfg/model/tiny_yolo_5class.yaml',
    epochs=60,
    batch_size=8,
    lr=0.00005,
    device='cuda',
    resume=None,
    early_stop_patience=10,
    use_focal=False,  # Default: OFF (causes low confidence)
    use_giou=False,   # Default: OFF (unstable)
    warmup_epochs=5,
    cosine_t0=10,
    use_residual=True,
    use_attention=True
):
    """
    Main training function with CORRECTED settings.
    """
    
    print("="*60)
    print("PhobiaShield Training - CORRECTED VERSION")
    print("The Architect")
    print("="*60)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")
    
    # Load config
    print("🔧 Loading configuration...")
    try:
        model_cfg = OmegaConf.load(model_config)
        num_classes = model_cfg.output.num_classes
        print(f"✓ Model config loaded: {num_classes} classes")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        print("Using default: 5 classes")
        num_classes = 5
    
    # Initialize model with optimizations
    print("\n🔧 Initializing model...")
    print(f"   ResNet blocks: {'ON' if use_residual else 'OFF'}")
    print(f"   CBAM attention: {'ON' if use_attention else 'OFF'}")
    
    try:
        model = PhobiaNet(
            model_cfg,
            use_residual=use_residual,
            use_attention=use_attention
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model initialized: {total_params:,} parameters ({total_params*4/1e6:.2f} MB)")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return
    
    # Initialize loss with CORRECTED lambda weights
    print("\n🔧 Initializing loss function...")
    print(f"   Focal Loss: {'ON' if use_focal else 'OFF'}")
    print(f"   GIoU Loss: {'ON' if use_giou else 'OFF'}")
    
    # CORRECTED: More balanced class weights
    class_weights = [2.0, 5.0, 3.0, 1.0, 10.0]  # Clown, Shark, Spider, Blood, Needle
    print(f"   Class weights: {class_weights}")
    print(f"   Lambda obj: 10.0 (CORRECTED from 5.0)")
    print(f"   Lambda noobj: 0.01 (CORRECTED from 0.05)")
    
    loss_fn = PhobiaLoss(
        lambda_coord=5.0,
        lambda_obj=5.0,      # CORRECTED!
        lambda_noobj=0.05,   # CORRECTED!
        lambda_class=1.0,
        grid_size=13,
        num_boxes=2,
        num_classes=num_classes,
        class_weights=class_weights,
        use_focal=use_focal,
        use_giou=use_giou
    )
    print("✓ Loss function initialized with CORRECTED lambda weights")
    
    # Load datasets with augmentation
    print("\n📦 Loading datasets...")
    data_root = Path(data_root)
    
    try:
        train_dataset = PhobiaDataset(
            img_dir=data_root / 'train' / 'images',
            label_dir=data_root / 'train' / 'labels',
            img_size=416,
            grid_size=13,
            num_boxes=2,
            num_classes=num_classes,
            augment=True
        )
        
        val_dataset = PhobiaDataset(
            img_dir=data_root / 'val' / 'images',
            label_dir=data_root / 'val' / 'labels',
            img_size=416,
            grid_size=13,
            num_boxes=2,
            num_classes=num_classes,
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=PhobiaDataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=PhobiaDataset.collate_fn
        )
        
        print(f"✓ Train: {len(train_dataset)} images (augmentation: ON)")
        print(f"✓ Val:   {len(val_dataset)} images (augmentation: OFF)")
        print(f"✓ Batch size: {batch_size}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    # Initialize optimizer
    print("\n⚙️  Initializing optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    print(f"✓ Optimizer: AdamW (base_lr={lr}, weight_decay=1e-4)")
    
    # Warmup + Cosine Annealing Scheduler
    print(f"\n📈 Initializing scheduler...")
    print(f"   Warmup: {warmup_epochs} epochs (0 → {lr})")
    print(f"   Cosine: T_0={cosine_t0}, T_mult=2, min_lr=1e-6")
    
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
        base_lr=lr,
        min_lr=1e-6,
        warmup_start_lr=0.0,
        cosine_t0=cosine_t0,
        cosine_t_mult=2
    )
    print(f"✓ Scheduler: WarmupCosine")
    
    # Create output directory
    output_dir = Path('outputs/checkpoints')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume:
        print(f"\n🔄 Resuming from checkpoint: {resume}")
        try:
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
            
            scheduler.current_epoch = start_epoch
            
            print(f"✓ Resumed from epoch {checkpoint['epoch']}")
            print(f"✓ Best val_loss so far: {best_val_loss:.2f}")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            start_epoch = 0
            best_val_loss = float('inf')
    
    # Early stopping setup
    print(f"\n🛑 Early stopping enabled (patience={early_stop_patience})")
    epochs_no_improve = 0
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"🚀 Starting training from epoch {start_epoch} to {epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, epochs):
        # Update learning rate
        scheduler.step(epoch)
        current_lr = scheduler.get_last_lr()[0]
        
        # Train phase
        model.train()
        train_loss = 0
        train_metrics = {'coord': 0, 'obj': 0, 'noobj': 0, 'class': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            loss, metrics = loss_fn(outputs, targets)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            # Accumulate
            train_loss += metrics['total_loss']
            train_metrics['coord'] += metrics['coord_loss']
            train_metrics['obj'] += metrics['conf_loss_obj']
            train_metrics['noobj'] += metrics['conf_loss_noobj']
            train_metrics['class'] += metrics['class_loss']
            
            # Update progress
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.2f}",
                'lr': f"{current_lr:.6f}"
            })
        
        # Average train metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {'coord': 0, 'obj': 0, 'noobj': 0, 'class': 0}
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
            
            for images, targets in pbar_val:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss, metrics = loss_fn(outputs, targets)
                
                val_loss += metrics['total_loss']
                val_metrics['coord'] += metrics['coord_loss']
                val_metrics['obj'] += metrics['conf_loss_obj']
                val_metrics['noobj'] += metrics['conf_loss_noobj']
                val_metrics['class'] += metrics['class_loss']
                
                pbar_val.set_postfix({'val_loss': f"{metrics['total_loss']:.2f}"})
        
        # Average val metrics
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   Train Loss: {train_loss:.2f} (coord={train_metrics['coord']:.2f}, obj={train_metrics['obj']:.2f}, noobj={train_metrics['noobj']:.2f})")
        print(f"   Val Loss:   {val_loss:.2f} (coord={val_metrics['coord']:.2f}, obj={val_metrics['obj']:.2f}, noobj={val_metrics['noobj']:.2f})")
        print(f"   LR: {current_lr:.6f}")
        
        # Indicate warmup phase
        if epoch < warmup_epochs:
            print(f"   🔥 Warmup phase ({epoch+1}/{warmup_epochs})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"   💾 Checkpoint saved (epoch {epoch+1})")
        
        # Early stopping check
        if val_loss < best_val_loss:
            # New best model
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            
            if epoch > 0:
                print(f"   🏆 New best model! Val Loss: {val_loss:.2f} (improved by {improvement:.2f})\n")
            else:
                print(f"   🏆 Best model saved (val_loss: {val_loss:.2f})\n")
        else:
            # No improvement
            epochs_no_improve += 1
            print(f"   ⚠️  No improvement for {epochs_no_improve} epoch(s) (best: {best_val_loss:.2f})")
            
            # Check early stopping
            if epochs_no_improve >= early_stop_patience:
                print(f"\n🛑 EARLY STOPPING triggered!")
                print(f"   No improvement for {early_stop_patience} epochs")
                print(f"   Best Val Loss: {best_val_loss:.2f} (Epoch {epoch+1-early_stop_patience})")
                print(f"   Stopping training at epoch {epoch+1}")
                break
            
            print()
    
    # Training completed
    print(f"{'='*60}")
    print(f"✅ Training complete!")
    if epochs_no_improve >= early_stop_patience:
        print(f"🛑 Stopped early (saved {epochs - epoch - 1} epochs)")
    print(f"{'='*60}")
    print(f"📈 Best validation loss: {best_val_loss:.2f}")
    print(f"💾 Model saved: {output_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PhobiaShield - CORRECTED')
    parser.add_argument('--data', type=str, default='data/phobiashield_final', help='Dataset root')
    parser.add_argument('--config', type=str, default='cfg/model/tiny_yolo_5class.yaml', help='Model config')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00005, help='Base learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Optimization flags (defaults: OFF for focal/giou, ON for resnet/cbam)
    parser.add_argument('--use-focal', action='store_true', help='Use Focal Loss (default: OFF)')
    parser.add_argument('--use-giou', action='store_true', help='Use GIoU Loss (default: OFF)')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--cosine-t0', type=int, default=10, help='Cosine restart period')
    parser.add_argument('--no-residual', action='store_true', help='Disable ResNet blocks')
    parser.add_argument('--no-attention', action='store_true', help='Disable CBAM attention')
    
    args = parser.parse_args()
    
    train_phobiashield(
        data_root=args.data,
        model_config=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        resume=args.resume,
        early_stop_patience=args.patience,
        use_focal=args.use_focal,
        use_giou=args.use_giou,
        warmup_epochs=args.warmup_epochs,
        cosine_t0=args.cosine_t0,
        use_residual=not args.no_residual,
        use_attention=not args.no_attention
    )